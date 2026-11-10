"""E07 — NLP-based representation (Team-2 #3).

(a) cross-dataset: text embeddings of flow sentences vs CBS numeric features (same train/test rows)
(b) in-domain: native vs text-only vs native + text (NLP-augmented)
(c) embedding map: do the three datasets share one semantic space?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from ..data.loaders import DATASET_NAMES
from ..models.zoo import make_model, proba_pos
from ..nlp.encoders import get_encoder
from ..nlp.flow2text import records_to_text
from ..reporting.plots import DS_COLORS, PALETTE, plt, savefig
from ..utils import binary_metrics, get_logger, save_json, save_table
from .common import DS, get_frames, get_split

log = get_logger("e07")


def run(cfg):
    seed = cfg["seed"]
    frames = {n: get_frames(n, seed) for n in DS}
    text = {n: (records_to_text(tr, n), records_to_text(te, n)) for n, (tr, te) in frames.items()}
    corpus = np.concatenate([np.concatenate(t) for t in text.values()])
    enc = get_encoder(cfg, corpus)
    emb = {n: (enc.encode(a), enc.encode(b)) for n, (a, b) in text.items()}
    examples = {DATASET_NAMES[n]: list(pd.Series(text[n][0]).drop_duplicates().head(6)) for n in DS}
    save_json({"encoder": enc.name, "unique_sentences": int(len(np.unique(corpus))), "examples": examples},
              cfg["paths"]["tables"] / "e07_text_examples.json")
    log.info(f"encoder={enc.name}, unique sentences={len(np.unique(corpus))}")

    # (a) cross-dataset
    rows = []
    for src in DS:
        Xs_txt = emb[src][0]
        ys = frames[src][0]["y"].to_numpy()
        cbs_s = get_split(src, "cbs", seed)
        m_txt = make_model("lgbm", seed).fit(Xs_txt, ys)
        m_cbs = make_model("lgbm", seed).fit(cbs_s.Xtr.values, cbs_s.ytr)
        m_both = make_model("lgbm", seed).fit(np.hstack([cbs_s.Xtr.values, Xs_txt]), ys)
        for tgt in DS:
            yt = frames[tgt][1]["y"].to_numpy()
            cbs_t = get_split(tgt, "cbs", seed)
            for rep, p in (("text embedding", proba_pos(m_txt, emb[tgt][1])),
                           ("CBS numeric", proba_pos(m_cbs, cbs_t.Xte.values)),
                           ("CBS + text", proba_pos(m_both, np.hstack([cbs_t.Xte.values, emb[tgt][1]])))):
                r = binary_metrics(yt, p >= 0.5, p)
                rows.append({"source": DATASET_NAMES[src], "target": DATASET_NAMES[tgt], "representation": rep,
                             "in_domain": src == tgt, **{k: r[k] for k in ("roc_auc", "mcc", "f1", "balanced_acc")}})
    cross = pd.DataFrame(rows)
    save_table(cross, "e07_nlp_cross_dataset")

    # (b) in-domain augmentation (with dataset-specific context phrases)
    rows = []
    for n in DS:
        tr, te = frames[n]
        nat = get_split(n, "native", seed)
        t_tr, t_te = records_to_text(tr, n, with_context=True), records_to_text(te, n, with_context=True)
        enc_n = get_encoder({**cfg, "nlp": {**cfg["nlp"], "prefer_sbert": cfg["nlp"]["prefer_sbert"]}},
                            np.concatenate([t_tr, t_te]))
        E_tr, E_te = enc_n.encode(t_tr), enc_n.encode(t_te)
        for rep, (A, B) in {"native": (nat.Xtr.values, nat.Xte.values), "text only": (E_tr, E_te),
                            "native + text (NLP-augmented)": (np.hstack([nat.Xtr.values, E_tr]),
                                                              np.hstack([nat.Xte.values, E_te]))}.items():
            m = make_model("lgbm", seed).fit(A, nat.ytr)
            p = proba_pos(m, B)
            r = binary_metrics(nat.yte, p >= 0.5, p)
            mc = make_model("lgbm", seed).fit(A, nat.mtr["super"].values)
            from sklearn.metrics import f1_score
            rows.append({"dataset": DATASET_NAMES[n], "representation": rep, "binary_f1": r["f1"],
                         "binary_auc": r["roc_auc"], "binary_mcc": r["mcc"],
                         "superclass_macro_f1": f1_score(nat.mte["super"].values, mc.predict(B), average="macro")})
    ind = pd.DataFrame(rows)
    save_table(ind, "e07_nlp_in_domain")

    # (c) shared semantic map
    rng = np.random.default_rng(seed)
    pts, ds, sup = [], [], []
    for n in DS:
        idx = rng.choice(len(emb[n][1]), min(2500, len(emb[n][1])), replace=False)
        pts.append(emb[n][1][idx])
        ds += [DATASET_NAMES[n]] * len(idx)
        sup += list(frames[n][1]["super"].to_numpy()[idx])
    P = PCA(2, random_state=seed).fit_transform(np.vstack(pts))
    P = P + rng.normal(0, 0.01, P.shape)  # jitter (many identical sentences)
    ds, sup = np.array(ds), np.array(sup)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for d, c in DS_COLORS.items():
        m = ds == d
        axes[0].scatter(P[m, 0], P[m, 1], s=5, alpha=0.35, c=c, label=d)
    axes[0].set_title("Flow-sentence embeddings by dataset")
    axes[0].legend(markerscale=3)
    for i, s in enumerate(sorted(set(sup))):
        m = sup == s
        axes[1].scatter(P[m, 0], P[m, 1], s=5, alpha=0.35, c=PALETTE[i % len(PALETTE)], label=s)
    axes[1].set_title("... by unified super-class")
    axes[1].legend(markerscale=3, fontsize=8)
    savefig(fig, "e07_text_embedding_map")
    log.info("\n" + cross[~cross.in_domain].pivot_table(index=["source", "target"], columns="representation",
                                                         values="roc_auc").round(3).to_string())
    log.info("\n" + ind.round(4).to_string(index=False))
    return {"cross": cross, "in_domain": ind}
