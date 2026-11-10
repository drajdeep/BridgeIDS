"""E05 — Domain adaptation comparison study: DomainNorm, CORAL, TCA, IW, DANN (Team-1 #6, RQ3).

Zero target labels. Classifier for feature-space methods: LightGBM (and logistic regression for TCA).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..adaptation.methods import TCA, DomainScaler, coral, importance_weights, self_training
from ..models.npnet import DANN, MLP, fit_classifier
from ..models.zoo import make_model, proba_pos
from ..reporting.plots import plt, savefig
from ..utils import binary_metrics, get_logger, save_table
from .common import get_split, subsample

log = get_logger("e05")

PAIRS = [  # (label, space, sources, target)
    ("NSL-KDD → Edge-IIoTset", "cbs", ["nsl"], "edge"),
    ("NSL-KDD → ForgeIIoT", "cbs", ["nsl"], "forge"),
    ("NSL-KDD + ForgeIIoT → Edge-IIoTset", "cbs", ["nsl", "forge"], "edge"),
    ("ForgeIIoT → Edge-IIoTset", "pbs", ["forge"], "edge"),
    ("Edge-IIoTset → ForgeIIoT", "pbs", ["edge"], "forge"),
]


def _m(y, p, thr=0.5):
    r = binary_metrics(y, p >= thr, p)
    return {k: r[k] for k in ("roc_auc", "mcc", "f1", "balanced_acc", "fpr")}


def run(cfg):
    seed = cfg["seed"]
    tca_n = cfg["experiments"]["tca_rows"]
    epochs = cfg["experiments"]["dann_epochs"]
    rows = []
    for label, kind, srcs, tgt in PAIRS:
        S = [get_split(s, kind, seed) for s in srcs]
        T = get_split(tgt, kind, seed)
        Xs = np.vstack([s.Xtr.values for s in S]).astype(float)
        ys = np.concatenate([s.ytr for s in S])
        Xs, ys = subsample(Xs, ys, 40000, seed)
        Xt_u = T.Xtr.values.astype(float)            # unlabelled target pool
        Xte, yte = T.Xte.values.astype(float), T.yte
        res = {}
        # 0) no adaptation
        res["No adaptation (LightGBM)"] = proba_pos(make_model("lgbm", seed).fit(Xs, ys), Xte)
        # 1) DomainNorm
        ss, st = DomainScaler().fit(Xs), DomainScaler().fit(Xt_u)
        Zs, Zt, Zte = ss.transform(Xs), st.transform(Xt_u), st.transform(Xte)
        res["DomainNorm"] = proba_pos(make_model("lgbm", seed).fit(Zs, ys), Zte)
        # 2) CORAL (on domain-normalised features)
        Zs_c = coral(Zs, Zt)
        res["CORAL"] = proba_pos(make_model("lgbm", seed).fit(Zs_c, ys), Zte)
        # 3) TCA (subsampled kernel problem)
        a, ya = subsample(Zs, ys, tca_n, seed)
        b = Zt[np.random.default_rng(seed).choice(len(Zt), min(tca_n, len(Zt)), replace=False)]
        tca = TCA(dim=min(20, Zs.shape[1]), kernel="linear").fit(a, b)
        Ts, Tte = tca.transform(Zs), tca.transform(Zte)
        res["TCA (+LightGBM)"] = proba_pos(make_model("lgbm", seed).fit(Ts, ys), Tte)
        res["TCA (+LogReg)"] = LogisticRegression(max_iter=1000).fit(Ts, ys).predict_proba(Tte)[:, 1]
        # 4) importance weighting
        w = importance_weights(Zs, Zt[:20000], seed=seed)
        res["Importance weighting"] = proba_pos(make_model("lgbm", seed).fit(Zs, ys, sample_weight=w), Zte)
        # 5) pseudo-label self-training on domain-normalised features
        st_clf = self_training(lambda: make_model("lgbm", seed), Zs, ys, Zt[:30000], seed=seed)
        res["DomainNorm + self-training"] = proba_pos(st_clf, Zte)
        # 6) MLP without / with adversarial adaptation (same architecture)
        for s in cfg["experiments"]["seeds"][:2]:
            mlp = fit_classifier(MLP([Zs.shape[1], 64, 32, 1], seed=s), Zs, ys, epochs=max(5, epochs // 3), seed=s)
            res.setdefault("MLP (no adaptation)", []).append(mlp.predict_proba(Zte)[:, 1])
            dann = DANN(Zs.shape[1], hidden=(64, 32), seed=s).fit(Zs, ys, Zt, epochs=max(5, epochs // 2), seed=s)
            res.setdefault("DANN", []).append(dann.predict_proba(Zte)[:, 1])
        for meth, p in res.items():
            ps = p if isinstance(p, list) else [p]
            ms = [_m(yte, q) for q in ps]
            r = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
            r["auc_std"] = float(np.std([m["roc_auc"] for m in ms])) if len(ms) > 1 else 0.0
            rows.append({"pair": label, "space": kind.upper(), "method": meth, **r})
        g = pd.DataFrame(rows)
        log.info(f"{label}\n{g[g.pair == label][['method', 'roc_auc', 'mcc', 'f1']].round(4).to_string(index=False)}")
    res = pd.DataFrame(rows)
    save_table(res, "e05_domain_adaptation")
    base = res[res.method == "No adaptation (LightGBM)"].set_index("pair")["roc_auc"]
    res["auc_gain_vs_noadapt"] = res.roc_auc - res.pair.map(base)
    save_table(res, "e05_domain_adaptation")

    piv = res.pivot(index="method", columns="pair", values="roc_auc")
    fig, ax = plt.subplots(figsize=(11, 4.8))
    n = piv.shape[1]
    w = 0.8 / n
    for i, c in enumerate(piv.columns):
        ax.bar(np.arange(len(piv)) + i * w - 0.4 + w / 2, piv[c].values, w, label=c)
    ax.axhline(0.5, ls=":", c="gray")
    ax.set_xticks(range(len(piv)), piv.index, rotation=25, ha="right")
    ax.set_ylabel("Target ROC-AUC (no target labels)")
    ax.set_ylim(0, 1)
    ax.set_title("Unsupervised domain adaptation")
    ax.legend(fontsize=7.5, ncol=2)
    savefig(fig, "e05_domain_adaptation_auc")
    return {"results": res}
