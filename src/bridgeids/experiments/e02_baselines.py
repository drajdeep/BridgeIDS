"""E02 — In-domain baselines (Team-1 #3) + leakage audit of the earlier notebooks.

* Six classifiers (DT, RF, AdaBoost, XGBoost, LightGBM, MLP) on every dataset (native features).
* Multi-class super-class detection with LightGBM.
* Leakage audit: ForgeIIoT with capture-order identifiers (frame.number / epoch / tcp.stream) vs without,
  random vs capture-temporal split; Edge-IIoT with IP / timestamp / checksum identifiers vs without.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from ..models.zoo import BASELINES, PRETTY, make_model, proba_pos
from ..reporting.plots import grouped_bar, heatmap
from ..utils import binary_metrics, get_logger, multiclass_metrics, save_table
from .common import DS, get_split

log = get_logger("e02")


def run(cfg):
    seed = cfg["seed"]
    rows = []
    for name in DS:
        sp = get_split(name, "native", seed)
        for m in BASELINES:
            t0 = time.time()
            model = make_model(m, seed)
            model.fit(sp.Xtr.values, sp.ytr)
            p = proba_pos(model, sp.Xte.values)
            r = binary_metrics(sp.yte, p >= 0.5, p)
            r.update(dataset=sp.pretty, model=PRETTY[m], train_s=round(time.time() - t0, 1),
                     n_train=len(sp.ytr), n_test=len(sp.yte), n_features=sp.Xtr.shape[1])
            rows.append(r)
            log.info(f"{sp.pretty:13s} {PRETTY[m]:13s} F1={r['f1']:.4f} AUC={r['roc_auc']:.4f} ({r['train_s']}s)")
    res = pd.DataFrame(rows)
    cols = ["dataset", "model", "accuracy", "precision", "recall", "f1", "balanced_acc", "mcc", "roc_auc", "pr_auc",
            "fpr", "train_s",
            "n_train", "n_test", "n_features"]
    save_table(res[cols], "e02_baselines_binary")
    grouped_bar(res, "model", "dataset", "f1", "In-domain binary F1 (native leak-free features)", "e02_baselines_f1",
                ylim=(0.5, 1.01))

    # ---------------- multi-class (super-class) with LightGBM
    mc = []
    for name in DS:
        sp = get_split(name, "native", seed)
        model = make_model("lgbm", seed)
        model.fit(sp.Xtr.values, sp.mtr["super"].values)
        pred = model.predict(sp.Xte.values)
        r = multiclass_metrics(sp.mte["super"].values, pred)
        r["dataset"] = sp.pretty
        rep = classification_report(sp.mte["super"].values, pred, output_dict=True, zero_division=0)
        for cls, d in rep.items():
            if isinstance(d, dict) and cls not in ("macro avg", "weighted avg"):
                r[f"f1_{cls}"] = d["f1-score"]
        mc.append(r)
    mc = pd.DataFrame(mc)
    save_table(mc, "e02_baselines_superclass")

    # ---------------- leakage audit
    audit = []
    for name, variants in (("forge", [("leak-free, random split", False, False),
                                      ("+ frame.number/epoch/tcp.stream, random split", True, False),
                                      ("leak-free, capture-temporal split", False, True),
                                      ("+ identifiers, capture-temporal split", True, True)]),
                           ("edge", [("leak-free", False, False),
                                     ("+ IP / frame.time / checksums", True, False)])):
        for label, leaky, temporal in variants:
            sp = get_split(name, "native", seed, temporal=temporal, include_leaky=leaky)
            model = make_model("lgbm", seed)
            model.fit(sp.Xtr.values, sp.mtr["label_fine"].values)
            pred = model.predict(sp.Xte.values)
            mcm = multiclass_metrics(sp.mte["label_fine"].values, pred)
            mb = make_model("lgbm", seed).fit(sp.Xtr.values, sp.ytr)
            pb = proba_pos(mb, sp.Xte.values)
            bm = binary_metrics(sp.yte, pb >= 0.5, pb)
            imp = pd.Series(mb.feature_importances_, index=sp.Xtr.columns).sort_values(ascending=False)
            audit.append({"dataset": sp.pretty, "setting": label, "binary_f1": bm["f1"], "binary_auc": bm["roc_auc"],
                          "fine_macro_f1": mcm["macro_f1"], "fine_accuracy": mcm["accuracy"],
                          "top3_features": ", ".join(imp.index[:3])})
            log.info(f"audit {sp.pretty} | {label}: fine-acc={mcm['accuracy']:.4f} top={list(imp.index[:3])}")
    audit = pd.DataFrame(audit)
    save_table(audit, "e02_leakage_audit")
    heatmap(audit.set_index(audit.dataset + " | " + audit.setting)[["binary_f1", "fine_macro_f1", "fine_accuracy"]],
            "Leakage audit (LightGBM)", "e02_leakage_audit", fmt="{:.3f}", cmap="Purples", figsize=(7.5, 4))
    return {"binary": res, "superclass": mc, "audit": audit}
