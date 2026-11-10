"""E11 — Static vs dynamic detection on a live stream (prequential, test-then-train).

Deployment story: a detector trained on ForgeIIoT (the bridge) is deployed; half-way through the stream
the plant traffic becomes Edge-IIoT traffic (new devices, new attacks). We compare
  static supervised | static hybrid (+ benign-calibrated anomaly) | dynamic hybrid with 2 % and 10 %
  delayed analyst feedback (online challenger + Hedge weighting + drift-triggered re-calibration)
Metrics: per-phase F1/MCC/FPR, rolling timeline, drift-detection delay, throughput.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..reporting.plots import PALETTE, plt, savefig
from ..stream.artifacts import build_artifacts, load_artifacts
from ..stream.engine import DetectionEngine
from ..stream.replayer import build_stream
from ..utils import binary_metrics, get_logger, save_table

log = get_logger("e11")


def run(cfg):
    seed = cfg["seed"]
    n_total = 30000
    build_artifacts(train_on=("forge",), holdout=[], name="bench_forge_only", rows_per_ds=60000, cfg=cfg)
    stream = build_stream(scenario="domain_shift", n=n_total, seed=seed)
    switch = n_total // 2
    for i, r in enumerate(stream):
        r["seq"] = i
    configs = [("Static supervised", dict(adaptive=False, use_anomaly=False, feedback_rate=0.0)),
               ("Static hybrid", dict(adaptive=False, use_anomaly=True, feedback_rate=0.0)),
               ("Dynamic hybrid (2% feedback)", dict(adaptive=True, use_anomaly=True, feedback_rate=0.02)),
               ("Dynamic hybrid (10% feedback)", dict(adaptive=True, use_anomaly=True, feedback_rate=0.10))]
    rows, timeline = [], []
    batch = 256
    for name, kw in configs:
        art = load_artifacts("bench_forge_only", cfg=cfg, build_if_missing=False)
        eng = DetectionEngine(art, cfg, feedback_delay=300, explain=False, seed=seed, **kw)
        preds, scores, truth = [], [], []
        t0 = time.time()
        for i in range(0, len(stream), batch):
            res = eng.process(stream[i:i + batch])
            preds += [r["pred"] for r in res]
            scores += [r["score"] for r in res]
            truth += [r["truth_y"] for r in res]
            m = eng.metrics()
            timeline.append({"config": name, "n": eng.n, "f1": m.get("f1"), "mcc": m.get("mcc"), "fpr": m.get("fpr"),
                             "w_online": eng.w_online, "psi": eng.last_psi})
        el = time.time() - t0
        p, s, y = np.array(preds), np.array(scores), np.array(truth)
        first_drift = next((d["seq"] for d in eng.drift_events if d["seq"] >= switch), None)
        for phase, sl in (("phase 1: ForgeIIoT", slice(0, switch)), ("phase 2: Edge-IIoTset", slice(switch, None)),
                          ("phase 2, last 5k", slice(len(y) - 5000, None)), ("whole stream", slice(None))):
            r = binary_metrics(y[sl], p[sl], s[sl])
            rows.append({"config": name, "phase": phase, **{k: r[k] for k in ("f1", "mcc", "fpr", "recall",
                                                                               "precision", "roc_auc")},
                         "throughput_pkt_s": len(y) / el, "drift_events": len(eng.drift_events),
                         "drift_detection_delay_pkts": (first_drift - switch) if first_drift is not None else None,
                         "feedback_labels_used": eng.online_seen})
        log.info(f"{name:32s} phase2 MCC={rows[-3]['mcc']:.3f} F1={rows[-3]['f1']:.3f} drift_events="
                 f"{len(eng.drift_events)} first_after_switch={first_drift} thr={len(y) / el:.0f} pkt/s")
    res = pd.DataFrame(rows)
    tl = pd.DataFrame(timeline)
    save_table(res, "e11_stream_benchmark")
    save_table(tl, "e11_stream_benchmark_timeline")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    for i, (name, _) in enumerate(configs):
        g = tl[tl.config == name]
        axes[0].plot(g.n, g.mcc, label=name, color=PALETTE[i], lw=1.6)
        axes[1].plot(g.n, g.fpr, color=PALETTE[i], lw=1.4)
    for ax in axes:
        ax.axvline(switch, ls="--", c="k", lw=1)
    axes[0].text(switch, 0.04, "  traffic switches to Edge-IIoT →", transform=axes[0].get_xaxis_transform(),
                 va="bottom", fontsize=9)
    axes[0].set_ylabel(f"rolling MCC (window {cfg['stream']['window']})")
    axes[1].set_ylabel("rolling false-positive rate")
    axes[1].set_xlabel("packets processed")
    axes[0].legend(fontsize=8, loc="lower left")
    axes[0].set_title("Static vs dynamic detection under domain shift (prequential evaluation)")
    savefig(fig, "e11_stream_static_vs_dynamic")
    return {"results": res}
