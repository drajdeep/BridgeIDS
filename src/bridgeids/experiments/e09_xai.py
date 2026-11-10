"""E09 — Explainable AI: SHAP + LIME, explanation drift across datasets (Team-2 #5, RQ4)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.loaders import DATASET_NAMES
from ..models.zoo import make_model, proba_pos
from ..reporting.plots import PALETTE, heatmap, plt, savefig
from ..utils import get_logger, save_json, save_table
from ..xai.explain import (_clean, lime_explain, narrative, rank_agreement, shap_importance,
                           shap_values_binary)
from .common import DS, get_split, subsample

log = get_logger("e09")


def _bar(imp: pd.Series, title: str, name: str, k: int = 15, color=PALETTE[0]):
    top = imp.head(k)[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * k + 1))
    ax.barh([_clean(i) for i in top.index], top.values, color=color)
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title(title)
    savefig(fig, name)


def run(cfg):
    seed, n_shap = cfg["seed"], cfg["experiments"]["shap_samples"]
    out = {}
    # 1) native SHAP per dataset
    top_rows = []
    natives = {}
    for i, n in enumerate(DS):
        sp = get_split(n, "native", seed)
        m = make_model("lgbm", seed).fit(sp.Xtr.values, sp.ytr)
        natives[n] = (sp, m)
        Xs, _ = subsample(sp.Xte.values, sp.yte, n_shap, seed)
        imp = shap_importance(m, Xs, sp.Xtr.columns)
        _bar(imp, f"SHAP global importance — {DATASET_NAMES[n]} (native)", f"e09_shap_native_{n}", color=PALETTE[i])
        for r, (f, v) in enumerate(imp.head(15).items()):
            top_rows.append({"dataset": DATASET_NAMES[n], "rank": r + 1, "feature": _clean(f), "mean_abs_shap": v})
    save_table(pd.DataFrame(top_rows), "e09_shap_top_features_native")

    # 2) explanation drift in the shared CBS space
    imps = {}
    for n in DS:
        sp = get_split(n, "cbs", seed)
        m = make_model("lgbm", seed).fit(sp.Xtr.values, sp.ytr)
        Xs, _ = subsample(sp.Xte.values, sp.yte, n_shap, seed)
        imps[DATASET_NAMES[n]] = shap_importance(m, Xs, sp.Xtr.columns)
    M = pd.DataFrame(imps).fillna(0)
    Mn = M / M.sum()
    save_table(Mn.reset_index().rename(columns={"index": "feature"}), "e09_shap_cbs_importance_by_dataset")
    top = Mn.max(1).sort_values(ascending=False).index[:16]
    heatmap(Mn.loc[top], "Normalised SHAP importance in the shared CBS space", "e09_shap_cbs_heatmap",
            fmt="{:.2f}", cmap="YlOrBr", figsize=(6.5, 7))
    agree = []
    names = list(imps)
    for a in range(3):
        for b in range(a + 1, 3):
            agree.append({"dataset_a": names[a], "dataset_b": names[b], **rank_agreement(imps[names[a]],
                                                                                          imps[names[b]])})
    save_table(pd.DataFrame(agree), "e09_explanation_drift_rank_agreement")

    # 3) same model, different data: Forge-trained PBS model explained on Forge vs Edge
    pf, pe = get_split("forge", "pbs", seed), get_split("edge", "pbs", seed)
    m = make_model("lgbm", seed).fit(pf.Xtr.values, pf.ytr)
    i_f = shap_importance(m, subsample(pf.Xte.values, pf.yte, n_shap, seed)[0], pf.Xtr.columns)
    i_e = shap_importance(m, subsample(pe.Xte.values, pe.yte, n_shap, seed)[0], pe.Xtr.columns)
    shift = pd.DataFrame({"on_ForgeIIoT": i_f / i_f.sum(), "on_Edge-IIoTset": i_e / i_e.sum()}).fillna(0)
    shift["shift"] = shift["on_Edge-IIoTset"] - shift["on_ForgeIIoT"]
    shift = shift.reindex(shift["shift"].abs().sort_values(ascending=False).index)
    save_table(shift.reset_index().rename(columns={"index": "feature"}), "e09_shap_shift_forge_model_on_edge")
    s_top = shift.head(14)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([_clean(i) for i in s_top.index], s_top["shift"], color=np.where(s_top["shift"] > 0, PALETTE[3], PALETTE[0]))
    ax.axvline(0, c="k", lw=0.8)
    ax.set_xlabel("Δ normalised mean|SHAP| (Edge − Forge)")
    ax.set_title("How importance changes when a Forge model meets Edge traffic")
    savefig(fig, "e09_shap_shift")
    agree_same = rank_agreement(i_f, i_e)

    # 4) LIME on Edge alerts + LIME/SHAP agreement + narratives
    sp, m = natives["edge"]
    names_ = list(sp.Xtr.columns)
    Xbg, _ = subsample(sp.Xtr.values, sp.ytr, 5000, seed)
    p = proba_pos(m, sp.Xte.values)
    rng = np.random.default_rng(seed)
    cand = np.where(p >= 0.5)[0]
    picks = rng.choice(cand, min(12, len(cand)), replace=False)
    sv = shap_values_binary(m, sp.Xte.values[picks])
    lime_rows, jacc, stories = [], [], []
    for j, idx in enumerate(picks):
        lst = lime_explain(m, Xbg, sp.Xte.values[idx], names_, num_features=6, num_samples=1200, seed=seed)
        lime_feats = []
        for cond, w in lst:
            feat = next((f for f in sorted(names_, key=len, reverse=True) if f in cond), cond)
            lime_feats.append(feat)
            lime_rows.append({"record": int(idx), "true_family": sp.mte["label_fine"].iloc[idx], "condition": cond,
                              "weight": w})
        shap_top = [names_[k] for k in np.argsort(-np.abs(sv[j]))[:6]]
        jacc.append(len(set(shap_top) & set(lime_feats)) / len(set(shap_top) | set(lime_feats)))
        order = np.argsort(-np.abs(sv[j]))[:4]
        stories.append({"record": int(idx), "true_family": sp.mte["label_fine"].iloc[idx],
                        "narrative": narrative(float(p[idx]), [(names_[k], float(sp.Xte.values[idx, k]),
                                                                float(sv[j, k])) for k in order], record_id=int(idx))})
    save_table(pd.DataFrame(lime_rows), "e09_lime_explanations_edge")
    save_json({"lime_shap_top6_jaccard_mean": float(np.mean(jacc)), "per_record": jacc,
               "forge_model_on_forge_vs_edge_rank_agreement": agree_same, "narratives": stories},
              cfg["paths"]["tables"] / "e09_xai_summary.json")
    log.info(f"LIME/SHAP top-6 agreement (Jaccard) = {np.mean(jacc):.3f}")
    log.info("\n" + pd.DataFrame(agree).round(3).to_string(index=False))
    log.info(stories[0]["narrative"])
    return out
