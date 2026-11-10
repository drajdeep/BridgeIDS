"""E03 — Cross-dataset testing / domain-shift report (Team-1 #4, RQ1, RQ2).

* CBS (all three datasets): train on A, test on B for every pair (+ two-source pooling).
* PBS (packet bridge, Forge <-> Edge): richer shared packet space.
* Super-class transfer on the classes the two datasets share.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from ..data.loaders import DATASET_NAMES
from ..models.zoo import make_model, proba_pos
from ..reporting.plots import heatmap
from ..utils import binary_metrics, bootstrap_ci, get_logger, multiclass_metrics, save_table
from .common import DS, get_split

log = get_logger("e03")


def _eval(model, Xte, yte, boot, seed):
    p = proba_pos(model, Xte)
    r = binary_metrics(yte, p >= 0.5, p)
    r["f1_lo"], r["f1_hi"] = bootstrap_ci(yte, (p >= 0.5).astype(int), f1_score, n=boot, seed=seed)
    return r


def run(cfg):
    seed, boot = cfg["seed"], cfg["experiments"]["bootstrap"]
    rows = []
    for kind, names in (("cbs", DS), ("pbs", ["forge", "edge"])):
        splits = {n: get_split(n, kind, seed) for n in names}
        for model_name in ("lgbm", "rf"):
            # single source
            for src in names:
                m = make_model(model_name, seed).fit(splits[src].Xtr.values, splits[src].ytr)
                for tgt in names:
                    r = _eval(m, splits[tgt].Xte.values, splits[tgt].yte, boot, seed)
                    r.update(space=kind.upper(), model=model_name, source=DATASET_NAMES[src], target=DATASET_NAMES[tgt],
                             in_domain=src == tgt)
                    rows.append(r)
            # two-source pooling -> held-out third (CBS only)
            if kind == "cbs":
                for tgt in names:
                    srcs = [s for s in names if s != tgt]
                    X = np.vstack([splits[s].Xtr.values for s in srcs])
                    y = np.concatenate([splits[s].ytr for s in srcs])
                    m = make_model(model_name, seed).fit(X, y)
                    r = _eval(m, splits[tgt].Xte.values, splits[tgt].yte, boot, seed)
                    r.update(space="CBS", model=model_name, source=" + ".join(DATASET_NAMES[s] for s in srcs),
                             target=DATASET_NAMES[tgt], in_domain=False)
                    rows.append(r)
    res = pd.DataFrame(rows)
    save_table(res, "e03_cross_dataset_binary")

    # domain-shift report: degradation relative to in-domain
    rep = []
    for (space, model), g in res.groupby(["space", "model"]):
        ind = g[g.in_domain].set_index("target")
        for _, r in g[~g.in_domain].iterrows():
            base = ind.loc[r.target]
            rep.append({"space": space, "model": model, "source": r.source, "target": r.target,
                        "in_domain_f1": base.f1, "cross_f1": r.f1, "f1_drop": base.f1 - r.f1,
                        "in_domain_auc": base.roc_auc, "cross_auc": r.roc_auc, "auc_drop": base.roc_auc - r.roc_auc})
    rep = pd.DataFrame(rep)
    save_table(rep, "e03_domain_shift_report")
    for space in ("CBS", "PBS"):
        g = res[(res.space == space) & (res.model == "lgbm") & (~res.source.str.contains(r"\+"))]
        for metric in ("f1", "roc_auc"):
            mat = g.pivot(index="source", columns="target", values=metric)
            order = [DATASET_NAMES[n] for n in DS if DATASET_NAMES[n] in mat.index]
            heatmap(mat.loc[order, order], f"{space} cross-dataset {metric.upper()} (LightGBM, rows=train)",
                    f"e03_{space.lower()}_matrix_{metric}", fmt="{:.3f}", cmap="RdYlGn", vmin=0, vmax=1,
                    xlabel="test dataset", ylabel="train dataset")

    # super-class transfer restricted to shared classes
    sc_rows = []
    splits = {n: get_split(n, "cbs", seed) for n in DS}
    for src in DS:
        m = make_model("lgbm", seed).fit(splits[src].Xtr.values, splits[src].mtr["super"].values)
        for tgt in DS:
            if src == tgt:
                continue
            shared = sorted(set(splits[src].mtr["super"]) & set(splits[tgt].mte["super"]))
            mask = splits[tgt].mte["super"].isin(shared).values
            pred = m.predict(splits[tgt].Xte.values[mask])
            r = multiclass_metrics(splits[tgt].mte["super"].values[mask], pred)
            r.update(source=DATASET_NAMES[src], target=DATASET_NAMES[tgt], shared_classes=", ".join(shared))
            sc_rows.append(r)
    save_table(pd.DataFrame(sc_rows), "e03_cross_dataset_superclass")
    log.info("\n" + rep[rep.model == "lgbm"].round(3).to_string(index=False))
    return {"binary": res, "report": rep}
