"""Live detection server (FastAPI).

Ingestion:  UDP datagrams  (default 127.0.0.1:9999, one JSON record per datagram)
            TCP NDJSON     (port udp_port + 1)
            HTTP POST      /api/ingest  (JSON list of records)
            MQTT           optional: env BRIDGEIDS_MQTT=host:port  -> subscribes to bridgeids/packets
Outputs:    http://127.0.0.1:8000           live dashboard (WebSocket push)
            /api/state, /api/alerts, /api/drift, /api/feedback, /api/export/powerbi, /api/reset
            SQLite results/stream/stream.db (events, alerts, drift, metrics)  -> Power BI via ODBC / CSV
            optional Power BI streaming-dataset push (config stream.powerbi_push_url)
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
import time
import urllib.request
from pathlib import Path
from typing import List

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from ..utils import ROOT, get_logger, load_config
from .artifacts import load_artifacts
from .engine import DetectionEngine

log = get_logger("server")
STATIC = Path(__file__).parent / "static"


class Sink:
    """SQLite persistence (+ optional Power BI push) — written from a background thread."""

    def __init__(self, path: Path, push_url: str = ""):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.push_url = push_url
        c = self.db.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS events (seq INTEGER, ts REAL, dataset TEXT, proto TEXT, service TEXT, "
                  "score REAL, p_supervised REAL, anomaly_pct REAL, pred INTEGER, family TEXT, truth_y INTEGER, "
                  "truth_super TEXT, truth_fine TEXT, latency_ms REAL)")
        c.execute("CREATE TABLE IF NOT EXISTS alerts (seq INTEGER, ts REAL, dataset TEXT, family TEXT, score REAL, "
                  "anomaly_pct REAL, truth_fine TEXT, explanation TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS drift (seq INTEGER, t REAL, reason TEXT, action TEXT, psi REAL)")
        c.execute("CREATE TABLE IF NOT EXISTS metrics (ts REAL, n INTEGER, alerts INTEGER, throughput REAL, "
                  "f1 REAL, mcc REAL, fpr REAL, w_online REAL, psi REAL, latency_p95 REAL)")
        self.db.commit()
        self.lock = threading.Lock()

    def write(self, results: List[dict], snap: dict, drift: List[dict]):
        with self.lock:
            c = self.db.cursor()
            c.executemany("INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                          [(r["seq"], r["ts"], r["dataset"], r["proto"], r["service"], r["score"], r["p_supervised"],
                            r["anomaly_pct"], r["pred"], r["family"], r["truth_y"], r["truth_super"], r["truth_fine"],
                            r["latency_ms"]) for r in results])
            c.executemany("INSERT INTO alerts VALUES (?,?,?,?,?,?,?,?)",
                          [(r["seq"], r["ts"], r["dataset"], r["family"], r["score"], r["anomaly_pct"],
                            r["truth_fine"], r.get("explanation")) for r in results if r["pred"]])
            c.executemany("INSERT INTO drift VALUES (?,?,?,?,?)",
                          [(d["seq"], d["time"], d["reason"], d.get("action"), d.get("psi")) for d in drift])
            c.execute("INSERT INTO metrics VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (time.time(), snap["n"], snap["alerts"], snap["throughput"], snap.get("f1"), snap.get("mcc"),
                       snap.get("fpr"), snap["w_online"], snap["psi"], snap.get("latency_ms_p95")))
            self.db.commit()
        if self.push_url:
            try:
                row = [{"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "packets": snap["n"],
                        "alerts": snap["alerts"], "throughput": snap["throughput"], "f1": snap.get("f1") or 0,
                        "fpr": snap.get("fpr") or 0, "drift_events": snap["drift_events"]}]
                req = urllib.request.Request(self.push_url, data=json.dumps(row).encode(),
                                             headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=3).read()
            except Exception as e:  # pragma: no cover
                log.debug(f"Power BI push failed: {e}")


def create_app(cfg=None, artifacts: str = "stream_artifacts"):
    cfg = cfg or load_config()
    sc = cfg["stream"]
    engine = DetectionEngine(load_artifacts(artifacts, cfg=cfg), cfg, feedback_rate=0.05, feedback_delay=200)
    sink = Sink(ROOT / sc["sqlite"], sc.get("powerbi_push_url", ""))
    queue: asyncio.Queue = asyncio.Queue(maxsize=200_000)
    clients: set = set()
    app = FastAPI(title="BridgeIDS live detector")
    app.state.engine = engine

    class UDPProto(asyncio.DatagramProtocol):
        def datagram_received(self, data, addr):
            try:
                queue.put_nowait(json.loads(data))
            except Exception:
                pass

    async def tcp_handler(reader, writer):
        while not reader.at_eof():
            line = await reader.readline()
            if line.strip():
                try:
                    await queue.put(json.loads(line))
                except Exception:
                    pass
        writer.close()

    async def worker():
        loop = asyncio.get_running_loop()
        last_push = 0.0
        while True:
            await asyncio.sleep(sc["batch_ms"] / 1000)
            batch = []
            while not queue.empty() and len(batch) < sc["max_batch"]:
                batch.append(queue.get_nowait())
            if not batch:
                continue
            results = await loop.run_in_executor(None, engine.process, batch)
            snap = engine.snapshot()
            drift = engine.events_for_sink
            engine.events_for_sink = []
            loop.run_in_executor(None, sink.write, results, snap, drift)
            if clients and time.time() - last_push > 0.4:
                last_push = time.time()
                msg = json.dumps(snap, default=str)
                for ws in list(clients):
                    try:
                        await ws.send_text(msg)
                    except Exception:
                        clients.discard(ws)

    @app.on_event("startup")
    async def _start():
        loop = asyncio.get_running_loop()
        import socket as _socket
        usock = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
        try:
            usock.setsockopt(_socket.SOL_SOCKET, _socket.SO_RCVBUF, 16 * 1024 * 1024)  # absorb bursts
        except OSError:
            pass
        usock.bind((sc["host"], sc["udp_port"]))
        await loop.create_datagram_endpoint(UDPProto, sock=usock)
        await asyncio.start_server(tcp_handler, sc["host"], sc["udp_port"] + 1)
        asyncio.create_task(worker())
        mq = os.environ.get("BRIDGEIDS_MQTT")
        if mq:
            try:
                import paho.mqtt.client as mqtt
                h, p = mq.split(":")
                try:
                    cli = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
                except Exception:
                    cli = mqtt.Client()
                cli.on_message = lambda c, u, m: loop.call_soon_threadsafe(queue.put_nowait, json.loads(m.payload))
                cli.connect(h, int(p))
                cli.subscribe("bridgeids/packets")
                cli.loop_start()
                log.info(f"MQTT ingestion from {mq}")
            except Exception as e:
                log.warning(f"MQTT disabled: {e}")
        log.info(f"listening UDP {sc['host']}:{sc['udp_port']}  TCP {sc['udp_port'] + 1}  "
                 f"dashboard http://{sc['host']}:{sc['http_port']}")

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "dashboard.html")

    @app.get("/api/state")
    async def state():
        return JSONResponse(json.loads(json.dumps(engine.snapshot(), default=str)))

    @app.get("/api/alerts")
    async def alerts(n: int = 100):
        return list(engine.alerts)[-n:][::-1]

    @app.get("/api/drift")
    async def drift():
        return engine.drift_events

    @app.post("/api/ingest")
    async def ingest(request: Request):
        records = await request.json()
        records = records if isinstance(records, list) else [records]
        for r in records:
            await queue.put(r)
        return {"queued": len(records)}

    @app.post("/api/feedback")
    async def feedback(request: Request):
        """{"fields": {...raw record...}, "label": 0|1} — analyst confirmation (drives online adaptation)."""
        item = await request.json()
        import pandas as pd
        from ..features.views import canonical_view, pbs_from_view, detect_schema
        df = pd.DataFrame([item["fields"]])
        X = pbs_from_view(canonical_view(df, detect_schema(df))).reindex(columns=engine.a["pbs_cols"], fill_value=0)
        engine.feedback(X.values[0].astype(float), int(item["label"]), 0.5, engine.a["pbs_cols"])
        return {"ok": True, "online_seen": engine.online_seen}

    @app.post("/api/reset")
    async def reset():
        engine.reset()
        return {"ok": True}

    @app.get("/api/export/powerbi")
    async def export():
        from ..reporting.powerbi import export_stream_tables
        paths = export_stream_tables(sink.db, cfg)
        return {"written": [str(p) for p in paths]}

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        clients.add(ws)
        try:
            await ws.send_text(json.dumps(engine.snapshot(), default=str))
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            clients.discard(ws)

    return app


def serve(host=None, port=None, udp_port=None, artifacts: str = "stream_artifacts"):
    import uvicorn
    cfg = load_config()
    if udp_port:
        cfg["stream"]["udp_port"] = udp_port
    if host:
        cfg["stream"]["host"] = host
    app = create_app(cfg, artifacts)
    uvicorn.run(app, host=cfg["stream"]["host"], port=port or cfg["stream"]["http_port"], log_level="warning")
