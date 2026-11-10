"""E10 — Dataset drift analysis: feature / concept / attack drift (Team-2 #6)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.loaders import DATASET_NAMES
from ..drift import conditional_label_shift, domain_classifier_auc, feature_drift, label_js, mmd_rbf
from ..models.zoo import make_model, proba_pos
from ..reporting.plots import heatmap, plt, savefig
from ..utils import get_logger, save_table
from .common import DS, get_split, subsample

log = get_logger("e10")


def run(cfg):
    seed = cfg["seed"]
    sp = {n: get_split(n, "cbs", seed) for n in DS}
    cols = list(sp["nsl"].Xtr.columns)
    pairs = [("nsl", "forge"), ("nsl", "edge"), ("forge", "edge")]
    feat_rows, glob_rows = [], []
    for a, b in pairs:
        A, yA = subsample(sp[a].Xtr.values, sp[a].ytr, 20000, seed)
        B, yB = subsample(sp[b].Xtr.values, sp[b].ytr, 20000, seed)
        fd = feature_drift(A, B)
        cl = conditional_label_shift(A, yA, B, yB)
        pair = f"{DATASET_NAMES[a]} vs {DATASET_NAMES[b]}"
        for j, c in enumerate(cols):
            feat_rows.append({"pair": pair, "feature": c, "kl": fd["kl"][j], "js": fd["js"][j],
                              "wasserstein": fd["wasserstein"][j], "psi": fd["psi"][j], "concept_shift": cl[j]})
        auc, pad = domain_classifier_auc(A, B, seed=seed)
        # concept drift: P(y|x) disagreement of models trained on each domain, evaluated on the union
        ma = make_model("lgbm", seed).fit(A, yA)
        mb = make_model("lgbm", seed).fit(B, yB)
        U = np.vstack([A[:5000], B[:5000]])
        disagree = float(((proba_pos(ma, U) >= 0.5) != (proba_pos(mb, U) >= 0.5)).mean())
        la = pd.Series(sp[a].mtr["super"]).value_counts(normalize=True).to_dict()
        lb = pd.Series(sp[b].mtr["super"]).value_counts(normalize=True).to_dict()
        glob_rows.append({"pair": pair, "mean_JS": fd["js"].mean(), "mean_KL": fd["kl"].mean(),
                          "mean_Wasserstein": fd["wasserstein"].mean(), "mean_PSI": fd["psi"].mean(),
                          "MMD_rbf": mmd_rbf(A, B, seed=seed), "domain_clf_AUC": auc, "proxy_A_distance": pad,
                          "concept_disagreement": disagree, "mean_conditional_label_shift": cl.mean(),
                          "attack_label_JS": label_js(la, lb),
                          "attack_rate_A": yA.mean(), "attack_rate_B": yB.mean()})
    feat = pd.DataFrame(feat_rows)
    glob = pd.DataFrame(glob_rows)
    save_table(feat, "e10_feature_drift")
    save_table(glob, "e10_drift_summary")
    for metric in ("js", "concept_shift"):
        piv = feat.pivot(index="feature", columns="pair", values=metric)
        piv = piv.loc[piv.max(1).sort_values(ascending=False).index[:18]]
        heatmap(piv, f"Per-feature {'Jensen-Shannon distance' if metric == 'js' else 'concept shift |ΔP(y|x)|'}",
                f"e10_feature_{metric}", fmt="{:.2f}", cmap="Reds", vmin=0, figsize=(7, 7.5))
    g = glob.set_index("pair")[["mean_JS", "MMD_rbf", "proxy_A_distance", "concept_disagreement",
                                "mean_conditional_label_shift", "attack_label_JS"]]
    heatmap(g, "Drift summary (higher = more drift)", "e10_drift_summary", fmt="{:.3f}", cmap="Reds",
            figsize=(9, 3.2))
    # radar-like bar figure
    fig, ax = plt.subplots(figsize=(9, 3.8))
    gg = g / g.max()
    w = 0.27
    for i, (pair, row) in enumerate(gg.iterrows()):
        ax.bar(np.arange(gg.shape[1]) + (i - 1) * w, row.values, w, label=pair)
    ax.set_xticks(range(gg.shape[1]), [c.replace("_", " ") for c in gg.columns], rotation=20, ha="right")
    ax.set_ylabel("relative to max")
    ax.set_title("Feature, concept and attack drift between datasets")
    ax.legend(fontsize=8)
    savefig(fig, "e10_drift_bars")
    log.info("\n" + glob.round(3).to_string(index=False))
    return {"summary": glob}
