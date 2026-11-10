"""Command-line entry point.

    python -m bridgeids.run prepare            # build data/processed/*.parquet from raw data
    python -m bridgeids.run exp all            # run every experiment (01..12)
    python -m bridgeids.run exp 03 05          # run selected experiments
    python -m bridgeids.run report             # paper tables + HTML results dashboard + Power BI export
    python -m bridgeids.run serve              # start the live detector (UDP 9999 + http://127.0.0.1:8000)
    python -m bridgeids.run replay --dataset edge --scenario campaign --rate 300
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
import traceback

from .utils import get_logger, load_config, save_json, seed_everything

EXPERIMENTS = {
    "01": "e01_dataset_study", "02": "e02_baselines", "03": "e03_cross_dataset", "04": "e04_transfer",
    "05": "e05_domain_adaptation", "06": "e06_continual", "07": "e07_nlp", "08": "e08_zeroshot",
    "09": "e09_xai", "10": "e10_drift", "11": "e11_stream_benchmark", "12": "e12_artifacts",
}
log = get_logger("run")


def run_experiments(ids):
    cfg = load_config()
    seed_everything(cfg["seed"])
    ids = list(EXPERIMENTS) if ids == ["all"] else [i.zfill(2) for i in ids]
    summary = {}
    for i in ids:
        mod = importlib.import_module(f".experiments.{EXPERIMENTS[i]}", __package__)
        log.info(f"===== Experiment {i}: {EXPERIMENTS[i]} =====")
        t0 = time.time()
        try:
            mod.run(cfg)
            summary[i] = {"status": "ok", "seconds": round(time.time() - t0, 1)}
        except Exception as e:  # keep going, report at the end
            traceback.print_exc()
            summary[i] = {"status": f"failed: {e!r}", "seconds": round(time.time() - t0, 1)}
        log.info(f"Experiment {i} -> {summary[i]}")
    save_json(summary, cfg["paths"]["results"] / "logs" / f"run_summary_{'_'.join(ids)}.json")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bridgeids")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--force", action="store_true")
    e = sub.add_parser("exp")
    e.add_argument("ids", nargs="+")
    sub.add_parser("report")
    s = sub.add_parser("serve")
    s.add_argument("--host", default=None)
    s.add_argument("--port", type=int, default=None)
    s.add_argument("--udp-port", type=int, default=None)
    s.add_argument("--artifacts", default="stream_artifacts",
                   help="stream_artifacts (Forge+Edge, default) or bench_forge_only (legacy deployment -> drift demo)")
    r = sub.add_parser("replay")
    r.add_argument("--dataset", default="edge", choices=["nsl", "forge", "edge", "mixed"])
    r.add_argument("--scenario", default="campaign", choices=["shuffle", "campaign", "domain_shift", "zero_day"])
    r.add_argument("--rate", type=float, default=300.0, help="packets per second")
    r.add_argument("--n", type=int, default=20000)
    r.add_argument("--transport", default="udp", choices=["udp", "tcp", "http", "mqtt"])
    r.add_argument("--host", default="127.0.0.1")
    r.add_argument("--port", type=int, default=None)
    args = ap.parse_args(argv)

    if args.cmd == "prepare":
        from .data.loaders import prepare_all
        prepare_all(force=args.force)
    elif args.cmd == "exp":
        res = run_experiments(args.ids)
        bad = [k for k, v in res.items() if v["status"] != "ok"]
        sys.exit(1 if bad else 0)
    elif args.cmd == "report":
        from .reporting.report import build_all
        build_all()
    elif args.cmd == "serve":
        from .stream.server import serve
        serve(host=args.host, port=args.port, udp_port=args.udp_port, artifacts=args.artifacts)
    elif args.cmd == "replay":
        from .stream.replayer import replay
        replay(dataset=args.dataset, scenario=args.scenario, rate=args.rate, n=args.n, transport=args.transport,
               host=args.host, port=args.port)


if __name__ == "__main__":
    main()
