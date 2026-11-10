"""Packet replayer — turns dataset records into a live stream of network messages.

Each record is serialised as one JSON message (one UDP datagram / one NDJSON line / one MQTT publish):

    {"seq": 17, "ts": 1731250000.12, "src": "edge", "schema": "edge",
     "fields": {"tcp.flags": 2.0, "tcp.dstport": 80.0, ...},          # raw tshark / KDD fields only
     "truth": {"y": 1, "super": "DOS", "fine": "DDoS_TCP"}}             # optional, for live evaluation

Scenarios
  shuffle      – random mix of one dataset
  campaign     – kill-chain: benign → recon → spoofing → access → injection → DoS → malware, attack bursts on
                 top of benign background (60 % benign)
  domain_shift – ForgeIIoT traffic, then the network "changes" to Edge-IIoT traffic (drift detection demo)
  zero_day     – benign + attacks incl. families withheld from the live model (XSS, Ransomware)
"""
from __future__ import annotations

import json
import socket
import time
import urllib.request
from typing import Iterator, List, Optional

import numpy as np
import pandas as pd

from ..data.loaders import load
from ..utils import get_logger, load_config

log = get_logger("replay")

_META = {"label_fine", "super", "y", "split", "difficulty", "label_cat"}
PHASES = ["NORMAL", "RECON", "MITM_SPOOF", "UNAUTH_ACCESS", "INJECTION", "DOS", "MALWARE_PRIVESC"]


def _records(df: pd.DataFrame, src: str) -> List[dict]:
    cols = [c for c in df.columns if c not in _META and not c.startswith("_")]
    vals = df[cols].to_dict("records")
    out = []
    for i, (row, (_, m)) in enumerate(zip(vals, df[["y", "super", "label_fine"]].iterrows())):
        fields = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()
                  if v is not None and not (isinstance(v, float) and np.isnan(v)) and v != ""}
        out.append({"src": src, "schema": src, "fields": fields,
                    "truth": {"y": int(m["y"]), "super": str(m["super"]), "fine": str(m["label_fine"])}})
    return out


def build_stream(dataset: str = "edge", scenario: str = "campaign", n: int = 20000, seed: int = 0,
                 split: str = "test") -> List[dict]:
    rng = np.random.default_rng(seed)
    if scenario == "domain_shift":
        a = build_stream("forge", "shuffle", n // 2, seed)
        b = build_stream("edge", "shuffle", n - n // 2, seed + 1)
        return a + b
    ds = "edge" if dataset == "mixed" else dataset
    from ..experiments.common import _split_frame
    tr, te = _split_frame(ds, load_config()["seed"])
    df = te if split == "test" else tr                 # held-out rows never seen by the live models
    if scenario == "shuffle":
        part = df.sample(min(n, len(df)), random_state=seed)
        return _records(part, ds)
    if scenario == "zero_day":
        from .artifacts import ZERO_DAY_FAMILIES
        normal = df[df.y == 0].sample(int(n * 0.6), replace=True, random_state=seed)
        zd = df[df.label_fine.isin(ZERO_DAY_FAMILIES)]
        other = df[(df.y == 1) & ~df.label_fine.isin(ZERO_DAY_FAMILIES)]
        att = pd.concat([zd.sample(min(len(zd), int(n * 0.25)), random_state=seed),
                         other.sample(int(n * 0.15), random_state=seed)])
        return _records(pd.concat([normal, att]).sample(frac=1, random_state=seed), ds)
    # campaign: phases of equal length; each phase = benign background + bursts of that phase's attacks
    phases = [p for p in PHASES if p == "NORMAL" or (df.super == p).any()]
    per = n // len(phases)
    chunks = []
    for ph in phases:
        normal = df[df.y == 0].sample(per if ph == "NORMAL" else int(per * 0.6), replace=True,
                                      random_state=int(rng.integers(1e9)))
        if ph == "NORMAL":
            chunks.append(normal)
            continue
        att = df[df.super == ph].sample(per - len(normal), replace=True, random_state=int(rng.integers(1e9)))
        # burstiness: attacks arrive in clumps
        mix = pd.concat([normal, att])
        order = np.argsort(np.r_[rng.random(len(normal)), np.sort(rng.random(len(att))) * 0.6 + 0.2])
        chunks.append(mix.iloc[order])
    return _records(pd.concat(chunks), ds)


def _sender(transport: str, host: str, port: int):
    if transport == "udp":
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        return lambda msgs: [sock.sendto(m.encode(), (host, port)) for m in msgs]
    if transport == "tcp":
        sock = socket.create_connection((host, port))
        return lambda msgs: sock.sendall(("\n".join(msgs) + "\n").encode())
    if transport == "http":
        url = f"http://{host}:{port}/api/ingest"

        def post(msgs):
            body = ("[" + ",".join(msgs) + "]").encode()
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10).read()
        return post
    if transport == "mqtt":
        import paho.mqtt.client as mqtt
        try:
            cli = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        except Exception:
            cli = mqtt.Client()
        cli.connect(host, port)
        cli.loop_start()
        return lambda msgs: [cli.publish("bridgeids/packets", m) for m in msgs]
    raise ValueError(transport)


def replay(dataset="edge", scenario="campaign", rate=300.0, n=20000, transport="udp", host="127.0.0.1",
           port: Optional[int] = None, seed: int = 0):
    cfg = load_config()
    if port is None:
        port = {"udp": cfg["stream"]["udp_port"], "tcp": cfg["stream"]["udp_port"] + 1,
                "http": cfg["stream"]["http_port"], "mqtt": 1883}[transport]
    stream = build_stream(dataset, scenario, n, seed)
    send = _sender(transport, host, port)
    log.info(f"replaying {len(stream)} records ({dataset}/{scenario}) via {transport}://{host}:{port} at {rate}/s")
    batch = max(1, int(rate / 20))          # 20 sends per second
    t0 = time.time()
    for i in range(0, len(stream), batch):
        msgs = []
        for j, rec in enumerate(stream[i:i + batch]):
            rec = dict(rec, seq=i + j, ts=time.time())
            msgs.append(json.dumps(rec, default=str))
        send(msgs)
        target = t0 + (i + batch) / rate
        time.sleep(max(0.0, target - time.time()))
        if (i // batch) % 100 == 0:
            log.info(f"sent {i + len(msgs):,}/{len(stream):,}")
    log.info(f"done: {len(stream):,} records in {time.time() - t0:.1f}s")
