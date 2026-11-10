"""E04 — Transfer learning with few labelled target samples (Team-1 #5, RQ2).

Target = Edge-IIoTset. Sources = ForgeIIoT (PBS), NSL-KDD (CBS), NSL-KDD + ForgeIIoT (CBS).
Labelled target budget: 0.1 %, 1 %, 5 %, 10 % of the Edge training split (3 random draws each).

Methods
  target-only     LightGBM on the k % target labels only
  source-only     LightGBM trained on the source, no target labels (k = 0 reference)
  pooled          LightGBM on source + k % target (target up-weighted)
  continued-GBM   LightGBM warm-started from the source booster, +150 trees on target (tree "fine-tuning")
  MLP-scratch     numpy MLP on k % target
  MLP-finetune    MLP pre-trained on source then fine-tuned on k % target
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler

from ..models.npnet import MLP, fit_classifier
from ..models.zoo import make_model, proba_pos
from ..reporting.plots import plt, savefig
from ..utils import binary_metrics, get_logger, save_table
from .common import get_split

log = get_logger("e04")


def _draw(y, frac, seed, min_per_class=5):
    rng = np.random.default_rng(seed)
    n = max(int(round(len(y) * frac)), 2 * min_per_class)
    idx = []
    for c in np.unique(y):
        ci = np.where(y == c)[0]
        k = max(min_per_class, int(round(n * len(ci) / len(y))))
        idx.append(rng.choice(ci, min(k, len(ci)), replace=False))
    return np.concatenate(idx)


def _metrics(y, p):
    r = binary_metrics(y, p >= 0.5, p)
    return {k: r[k] for k in ("f1", "roc_auc", "mcc", "balanced_acc", "fpr")}


def run(cfg):
    seed = cfg["seed"]
    fracs = cfg["experiments"]["transfer_fractions"]
    seeds = cfg["experiments"]["seeds"]
    setups = []
    pf, pe = get_split("forge", "pbs", seed), get_split("edge", "pbs", seed)
    setups.append(("ForgeIIoT (PBS)", pf.Xtr.values, pf.ytr, pe))
    cn, cf, ce = get_split("nsl", "cbs", seed), get_split("forge", "cbs", seed), get_split("edge", "cbs", seed)
    setups.append(("NSL-KDD (CBS)", cn.Xtr.values, cn.ytr, ce))
    setups.append(("NSL-KDD + ForgeIIoT (CBS)", np.vstack([cn.Xtr.values, cf.Xtr.values]),
                   np.concatenate([cn.ytr, cf.ytr]), ce))
    rows = []
    for src_name, Xs, ys, tgt in setups:
        Xt_all, yt_all, Xte, yte = tgt.Xtr.values, tgt.ytr, tgt.Xte.values, tgt.yte
        src_gbm = make_model("lgbm", seed).fit(Xs, ys)
        r = _metrics(yte, proba_pos(src_gbm, Xte))
        rows.append({"source": src_name, "method": "source-only", "frac": 0.0, "n_target": 0, "seed": seed, **r})
        scaler = StandardScaler().fit(np.vstack([Xs, Xt_all]))  # unsupervised statistics only
        base_mlp = fit_classifier(MLP([Xs.shape[1], 64, 32, 1], seed=seed), scaler.transform(Xs), ys,
                                  epochs=6, seed=seed)
        r = _metrics(yte, base_mlp.predict_proba(scaler.transform(Xte))[:, 1])
        rows.append({"source": src_name, "method": "MLP source-only", "frac": 0.0, "n_target": 0, "seed": seed, **r})
        for frac in fracs:
            for s in seeds:
                idx = _draw(yt_all, frac, s)
                Xt, yt = Xt_all[idx], yt_all[idx]
                res = {}
                m = make_model("lgbm", s)
                m.set_params(min_child_samples=max(2, min(20, len(yt) // 20)))
                res["target-only"] = proba_pos(m.fit(Xt, yt), Xte)
                w = np.concatenate([np.ones(len(ys)), np.full(len(yt), min(10.0, len(ys) / len(yt)))])
                res["pooled"] = proba_pos(make_model("lgbm", s).fit(np.vstack([Xs, Xt]), np.concatenate([ys, yt]),
                                                                    sample_weight=w), Xte)
                cont = LGBMClassifier(n_estimators=150, num_leaves=31, learning_rate=0.05, verbose=-1, random_state=s,
                                      min_child_samples=max(2, min(20, len(yt) // 20)))
                cont.fit(Xt, yt, init_model=src_gbm.booster_)
                res["continued-GBM"] = proba_pos(cont, Xte)
                mlp_s = fit_classifier(MLP([Xs.shape[1], 64, 32, 1], seed=s), scaler.transform(Xt), yt,
                                       epochs=max(10, min(60, 20000 // len(yt))), seed=s)
                res["MLP-scratch"] = mlp_s.predict_proba(scaler.transform(Xte))[:, 1]
                mlp_f = base_mlp.clone()        # keep the pre-trained body, re-initialise the output head
                mlp_f.W[-1] = np.random.default_rng(s).normal(0, np.sqrt(2.0 / mlp_f.W[-1].shape[0]), mlp_f.W[-1].shape)
                mlp_f.b[-1] = np.zeros_like(mlp_f.b[-1])
                mlp_f.reset_optimizer()
                mlp_f = fit_classifier(mlp_f, scaler.transform(Xt), yt, epochs=max(10, min(60, 20000 // len(yt))),
                                       seed=s)  # identical schedule to MLP-scratch -> fair comparison
                res["MLP-finetune"] = mlp_f.predict_proba(scaler.transform(Xte))[:, 1]
                for meth, p in res.items():
                    rows.append({"source": src_name, "method": meth, "frac": frac, "n_target": len(yt), "seed": s,
                                 **_metrics(yte, p)})
            g = pd.DataFrame(rows)
            g = g[(g.source == src_name) & (g.frac == frac)].groupby("method")[["f1", "roc_auc", "mcc"]].mean()
            log.info(f"{src_name} | {frac:.3%}\n{g.round(4).to_string()}")
    res = pd.DataFrame(rows)
    save_table(res, "e04_transfer_raw")
    summ = res.groupby(["source", "method", "frac"]).agg(
        n_target=("n_target", "mean"), f1=("f1", "mean"), f1_std=("f1", "std"), auc=("roc_auc", "mean"),
        auc_std=("roc_auc", "std"), mcc=("mcc", "mean"), mcc_std=("mcc", "std")).reset_index()
    save_table(summ, "e04_transfer_learning_results")

    fig, axes = plt.subplots(1, len(setups), figsize=(5 * len(setups), 4), sharey=True)
    for ax, (src_name, *_ ) in zip(np.atleast_1d(axes), setups):
        g = summ[(summ.source == src_name) & (summ.frac > 0)]
        for meth in ["target-only", "pooled", "continued-GBM", "MLP-scratch", "MLP-finetune"]:
            h = g[g.method == meth].sort_values("frac")
            ax.errorbar(h.frac * 100, h.mcc, yerr=h.mcc_std.fillna(0), marker="o", capsize=3, label=meth)
        so = summ[(summ.source == src_name) & (summ.method == "source-only")].mcc.mean()
        ax.axhline(so, ls="--", c="gray", lw=1, label="source-only")
        ax.set_xscale("log")
        ax.set_xlabel("% labelled Edge-IIoT training data")
        ax.set_title(f"Source: {src_name}")
    np.atleast_1d(axes)[0].set_ylabel("MCC on Edge-IIoT test")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    savefig(fig, "e04_transfer_curves")
    return {"summary": summ}
