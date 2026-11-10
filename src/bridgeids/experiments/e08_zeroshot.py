"""E08 — Zero-shot / unseen-attack detection (Team-2 #4, RQ5).

A. Leave-one-family-out (in-domain, Edge & Forge native features): supervised vs one-class vs hybrid.
B. Cross-dataset zero-shot: train on NSL-KDD + ForgeIIoT, detect Edge-IIoT attack families that do not
   exist in the sources (only benign Edge traffic may be used for calibration — no attack labels).
C. Semantic zero-shot *classification* of unseen Edge classes:
   DAP (attribute prediction, Lampert et al.) and text-embedding ZSL (DeViSE/ConSE-style projection).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, normalize

from ..data.loaders import DATASET_NAMES
from ..data.taxonomy import ATTRIBUTES, EDGE_FINE_ATTRIBUTES, EDGE_FINE_DESCRIPTIONS
from ..models.hybrid import HybridDetector, NormalCalibratedAnomaly
from ..models.zoo import make_model, proba_pos
from ..nlp.encoders import LocalEncoder, get_encoder
from ..nlp.flow2text import records_to_text
from ..reporting.plots import heatmap, plt, savefig
from ..utils import get_logger, save_table
from .common import get_frames, get_split

log = get_logger("e08")


def _thr_at_fpr(scores_normal, fpr=0.01):
    return float(np.quantile(scores_normal, 1 - fpr))


def lofo(name, cfg, max_families=None):
    seed = cfg["seed"]
    sp = get_split(name, "native", seed)
    Xtr, Xte = sp.Xtr.values.astype(float), sp.Xte.values.astype(float)
    ftr, fte = sp.mtr["label_fine"].values, sp.mte["label_fine"].values
    normal_label = "Normal" if name == "edge" else "normal"
    fams = [f for f in pd.Series(fte).value_counts().index if f != normal_label][:max_families]
    rows = []
    anom = NormalCalibratedAnomaly(seed=seed).fit(Xtr[ftr == normal_label])
    a_te = anom.percentile(Xte)
    a_tr_norm = anom.percentile(Xtr[ftr == normal_label])
    for fam in fams:
        keep = ftr != fam
        hyb = HybridDetector(seed=seed).fit(Xtr[keep], sp.ytr[keep])
        p_sup, _ = hyb.components(Xte)
        s_h = hyb.score(Xte)
        # thresholds chosen on *training* normals (no test peeking): 1 % FPR
        p_sup_trn = proba_pos(hyb.sup, Xtr[ftr == normal_label])
        s_h_trn = hyb.score(Xtr[ftr == normal_label])
        t_sup, t_an, t_h = _thr_at_fpr(p_sup_trn), _thr_at_fpr(a_tr_norm), _thr_at_fpr(s_h_trn)
        m_f, m_n = fte == fam, fte == normal_label
        rows.append({"dataset": DATASET_NAMES[name], "unseen_family": fam, "n_test": int(m_f.sum()),
                     "supervised@0.5": float((p_sup[m_f] >= 0.5).mean()),
                     "supervised@1%FPR": float((p_sup[m_f] >= t_sup).mean()),
                     "one-class IF@1%FPR": float((a_te[m_f] >= t_an).mean()),
                     "hybrid@1%FPR": float((s_h[m_f] >= t_h).mean()),
                     "fpr_hybrid_test": float((s_h[m_n] >= t_h).mean()),
                     "fpr_supervised_test": float((p_sup[m_n] >= t_sup).mean())})
        log.info(f"{name} unseen={fam:22s} sup={rows[-1]['supervised@1%FPR']:.3f} "
                 f"IF={rows[-1]['one-class IF@1%FPR']:.3f} hybrid={rows[-1]['hybrid@1%FPR']:.3f}")
    return pd.DataFrame(rows)


def cross_dataset_zero_shot(cfg):
    """Sources NSL+Forge (CBS + text). Target Edge: only benign samples are used (threshold / one-class)."""
    seed = cfg["seed"]
    fr = {n: get_frames(n, seed) for n in ("nsl", "forge", "edge")}
    txt = {n: (records_to_text(a, n), records_to_text(b, n)) for n, (a, b) in fr.items()}
    enc = get_encoder(cfg, np.concatenate([np.concatenate(v) for v in txt.values()]))
    feats = {}
    for n in fr:
        c = get_split(n, "cbs", seed)
        feats[n] = (np.hstack([c.Xtr.values, enc.encode(txt[n][0])]), np.hstack([c.Xte.values, enc.encode(txt[n][1])]))
    Xs = np.vstack([feats["nsl"][0], feats["forge"][0]])
    ys = np.concatenate([fr["nsl"][0]["y"].to_numpy(), fr["forge"][0]["y"].to_numpy()])
    src_labels = set(fr["nsl"][0]["super"]) | set(fr["forge"][0]["super"])
    Xe_tr, Xe_te = feats["edge"]
    e_tr, e_te = fr["edge"][0], fr["edge"][1]
    benign_all = Xe_tr[e_tr["y"].to_numpy() == 0][:4000]   # assumption: a benign capture of the target site only
    fit_part, benign = benign_all[:2500], benign_all[2500:]  # disjoint fit / threshold-calibration samples
    sup = make_model("lgbm", seed).fit(Xs, ys)
    an = NormalCalibratedAnomaly(seed=seed).fit(fit_part)
    p, a = proba_pos(sup, Xe_te), an.percentile(Xe_te)
    p_b, a_b = proba_pos(sup, benign), an.percentile(benign)
    t_p, t_a = _thr_at_fpr(p_b, 0.05), _thr_at_fpr(a_b, 0.05)
    hyb = np.maximum(p / max(t_p, 1e-9), a / max(t_a, 1e-9))     # each normalised to its own 5 %-FPR threshold
    t_h = _thr_at_fpr(np.maximum(p_b / max(t_p, 1e-9), a_b / max(t_a, 1e-9)), 0.05)
    rows = []
    for fam, g in e_te.groupby("label_fine"):
        m = (e_te["label_fine"] == fam).to_numpy()
        rows.append({"edge_family": fam, "super": g["super"].iloc[0],
                     "family_seen_in_sources": g["super"].iloc[0] in src_labels and fam != "Normal",
                     "n": int(m.sum()), "source-supervised@0.5": float((p[m] >= 0.5).mean()),
                     "source-supervised@5%FPR": float((p[m] >= t_p).mean()),
                     "target one-class@5%FPR": float((a[m] >= t_a).mean()),
                     "hybrid@5%FPR": float((hyb[m] >= t_h).mean())})
    return pd.DataFrame(rows)


def semantic_zsl(cfg):
    seed = cfg["seed"]
    holdout = cfg["experiments"]["zsl_holdout_edge"]
    tr, te = get_frames("edge", seed)
    sp = get_split("edge", "native", seed)
    sc = StandardScaler().fit(sp.Xtr.values)
    Xtr, Xte = sc.transform(sp.Xtr.values), sc.transform(sp.Xte.values)
    ftr, fte = tr["label_fine"].to_numpy(), te["label_fine"].to_numpy()
    seen = ~np.isin(ftr, holdout)
    unseen_te = np.isin(fte, holdout)
    A = np.array([EDGE_FINE_ATTRIBUTES[f] for f in ftr])
    rows = []
    # ---- DAP
    P = np.zeros((unseen_te.sum(), len(ATTRIBUTES)))
    for j, att in enumerate(ATTRIBUTES):
        yj = A[seen, j]
        if yj.min() == yj.max():
            P[:, j] = yj[0]
            continue
        m = make_model("lgbm", seed).fit(sp.Xtr.values[seen], yj)
        P[:, j] = proba_pos(m, sp.Xte.values[unseen_te])
    P = np.clip(P, 1e-3, 1 - 1e-3)
    sig = np.array([EDGE_FINE_ATTRIBUTES[c] for c in holdout])
    ll = np.log(P) @ sig.T + np.log(1 - P) @ (1 - sig).T
    pred_dap = np.array(holdout)[ll.argmax(1)]
    # ---- text-embedding ZSL: ridge projection features -> class-description embedding
    desc = {c: EDGE_FINE_DESCRIPTIONS[c] for c in EDGE_FINE_DESCRIPTIONS}
    corpus = list(desc.values()) + list(records_to_text(tr.sample(min(len(tr), 20000), random_state=seed), "edge"))
    enc = get_encoder(cfg, corpus)
    if isinstance(enc, LocalEncoder):  # LSA needs enough vocabulary overlap; use full TF-IDF space
        enc = LocalEncoder(dim=128, seed=seed).fit(corpus)
    C = {c: enc.encode([d])[0] for c, d in desc.items()}
    Ytr = np.vstack([C[f] for f in ftr[seen]])
    reg = Ridge(alpha=10.0).fit(Xtr[seen], Ytr)
    proj = normalize(reg.predict(Xte[unseen_te]))
    Cu = normalize(np.vstack([C[c] for c in holdout]))
    pred_txt = np.array(holdout)[(proj @ Cu.T).argmax(1)]
    y_u = fte[unseen_te]
    for meth, pred in (("DAP (attribute)", pred_dap), ("Text-embedding ZSL", pred_txt)):
        rows.append({"method": meth, "unseen_classes": ", ".join(holdout), "top1_accuracy": float((pred == y_u).mean()),
                     "chance": 1.0 / len(holdout),
                     **{f"acc_{c}": float((pred[y_u == c] == c).mean()) for c in holdout}})
    conf = pd.crosstab(pd.Series(y_u, name="true"), pd.Series(pred_dap, name="DAP predicted"))
    return pd.DataFrame(rows), conf


def run(cfg):
    a = pd.concat([lofo("edge", cfg), lofo("forge", cfg)], ignore_index=True)
    save_table(a, "e08_zeroshot_lofo")
    for ds, g in a.groupby("dataset"):
        heatmap(g.set_index("unseen_family")[["supervised@1%FPR", "one-class IF@1%FPR", "hybrid@1%FPR"]],
                f"Unseen-family detection rate — {ds} (leave-one-family-out)",
                f"e08_lofo_{ds.split('-')[0].lower()}", fmt="{:.2f}", cmap="Greens", vmin=0, vmax=1,
                figsize=(6.5, 0.45 * len(g) + 1.5))
    b = cross_dataset_zero_shot(cfg)
    save_table(b, "e08_zeroshot_cross_dataset")
    heatmap(b.set_index("edge_family")[["source-supervised@5%FPR", "target one-class@5%FPR", "hybrid@5%FPR"]],
            "Cross-dataset zero-shot (train NSL-KDD+ForgeIIoT, test Edge-IIoT)", "e08_cross_dataset_zeroshot",
            fmt="{:.2f}", cmap="Greens", vmin=0, vmax=1, figsize=(6.5, 7))
    c, conf = semantic_zsl(cfg)
    save_table(c, "e08_semantic_zsl")
    save_table(conf.reset_index(), "e08_semantic_zsl_confusion_dap")
    log.info("\n" + a.groupby("dataset")[["supervised@1%FPR", "one-class IF@1%FPR", "hybrid@1%FPR"]].mean().round(3)
             .to_string())
    log.info("\n" + b.round(3).to_string(index=False))
    log.info("\n" + c.round(3).to_string(index=False))
    return {"lofo": a, "cross": b, "zsl": c}
