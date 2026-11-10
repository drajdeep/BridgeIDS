"""E06 — Continual (domain-incremental) learning NSL-KDD -> ForgeIIoT -> Edge-IIoTset (Team-1 #7).

Strategies: naive fine-tuning, EWC, experience replay, EWC+replay, LightGBM continued boosting,
joint training (upper bound). Metrics: accuracy matrix R[t, i] (balanced accuracy and AUC on task i
after training task t), average final performance, backward transfer (BWT) and forgetting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from ..data.loaders import DATASET_NAMES
from ..models.npnet import MLP, fisher_diag, fit_classifier
from ..reporting.plots import heatmap, plt, savefig
from ..utils import binary_metrics, get_logger, save_table
from .common import get_split, subsample

log = get_logger("e06")


def _prep(X):
    X = np.asarray(X, float).copy()
    return X  # CBS is binary/one-hot except log_bytes (scaled below)


def _score(model, X, y):
    p = model.predict_proba(X)[:, 1]
    r = binary_metrics(y, p >= 0.5, p)
    return r["balanced_acc"], r["roc_auc"]


def _cl_metrics(R):
    T = R.shape[0]
    final = R[-1].mean()
    bwt = np.mean([R[-1, i] - R[i, i] for i in range(T - 1)])
    forg = np.mean([R[:-1, i].max() - R[-1, i] for i in range(T - 1)])
    fwt = np.mean([R[i - 1, i] for i in range(1, T)])  # zero-shot performance on the next task before training it
    return final, bwt, forg, fwt


def run(cfg):
    seed = cfg["seed"]
    ep = cfg["experiments"]["cl_epochs"]
    orders = [["nsl", "forge", "edge"], ["edge", "forge", "nsl"]]
    rows, mats = [], {}
    for order in orders:
        tasks = []
        for n in order:
            sp = get_split(n, "cbs", seed)
            Xtr, ytr = subsample(sp.Xtr.values, sp.ytr, 30000, seed)
            col = list(sp.Xtr.columns).index("log_bytes")
            Xtr = _prep(Xtr)
            Xte = _prep(sp.Xte.values)
            Xtr[:, col] /= 10.0
            Xte[:, col] /= 10.0
            tasks.append((n, Xtr, ytr, Xte, sp.yte))
        d = tasks[0][1].shape[1]
        oname = " → ".join(DATASET_NAMES[n] for n in order)
        strategies = ["Naive fine-tuning", "EWC", "Replay", "EWC + Replay", "LightGBM continued", "Joint (offline)"]
        for strat in strategies:
          Rbs, Ras = [], []
          for seed in cfg["experiments"]["seeds"]:
            Rb, Ra = np.zeros((3, 3)), np.zeros((3, 3))
            if strat == "Joint (offline)":
                for t in range(3):
                    X = np.vstack([tasks[i][1] for i in range(t + 1)])
                    y = np.concatenate([tasks[i][2] for i in range(t + 1)])
                    w = np.concatenate([np.full(len(tasks[i][2]), len(y) / ((t + 1) * len(tasks[i][2])))
                                        for i in range(t + 1)])  # task-balanced loss
                    m = fit_classifier(MLP([d, 128, 64, 1], seed=seed), X, y, epochs=ep, seed=seed, sample_weight=w)
                    for i in range(3):
                        Rb[t, i], Ra[t, i] = _score(m, tasks[i][3], tasks[i][4])
            elif strat == "LightGBM continued":
                booster = None
                for t in range(3):
                    m = LGBMClassifier(n_estimators=150, num_leaves=31, learning_rate=0.08, verbose=-1, random_state=seed)
                    m.fit(tasks[t][1], tasks[t][2], init_model=booster)
                    booster = m.booster_
                    for i in range(3):
                        Rb[t, i], Ra[t, i] = _score(m, tasks[i][3], tasks[i][4])
            else:
                m = MLP([d, 128, 64, 1], seed=seed)
                ewc_terms, mem_X, mem_y = [], [], []
                use_ewc = "EWC" in strat
                use_rep = "Replay" in strat
                rng = np.random.default_rng(seed)
                for t in range(3):
                    _, Xtr, ytr, _, _ = tasks[t]
                    replay = (np.vstack(mem_X), np.concatenate(mem_y)) if (use_rep and mem_X) else None
                    m.reset_optimizer()
                    fit_classifier(m, Xtr, ytr, epochs=ep, seed=seed + t, ewc=ewc_terms if use_ewc else None,
                                   replay=replay, replay_ratio=0.5)
                    for i in range(3):
                        Rb[t, i], Ra[t, i] = _score(m, tasks[i][3], tasks[i][4])
                    if use_ewc:
                        idx = rng.choice(len(Xtr), min(5000, len(Xtr)), replace=False)
                        F = fisher_diag(m, Xtr[idx], ytr[idx])
                        F = [f / (max(fi.max() for fi in F) + 1e-12) for f in F]  # normalise Fisher scale
                        ewc_terms.append((F, [p.copy() for p in m.params], 50.0))
                    if use_rep:
                        idx = rng.choice(len(Xtr), min(2000, len(Xtr)), replace=False)
                        mem_X.append(Xtr[idx])
                        mem_y.append(ytr[idx])
            Rbs.append(Rb)
            Ras.append(Ra)
          for metric, Rs in (("balanced_acc", Rbs), ("roc_auc", Ras)):
                ms = np.array([_cl_metrics(R) for R in Rs])
                R = np.mean(Rs, axis=0)
                rows.append({"order": oname, "strategy": strat, "metric": metric, "avg_final": ms[:, 0].mean(),
                             "avg_final_std": ms[:, 0].std(), "BWT": ms[:, 1].mean(), "forgetting": ms[:, 2].mean(),
                             "forward_zero_shot": ms[:, 3].mean(), "R": np.round(R, 4).tolist()})
          Rb = np.mean(Rbs, axis=0)
          mats[(oname, strat)] = Rb
          log.info(f"{oname} | {strat:20s} final-BA={_cl_metrics(Rb)[0]:.4f} BWT={_cl_metrics(Rb)[1]:+.4f} "
                     f"forgetting={_cl_metrics(Rb)[2]:.4f}")
    res = pd.DataFrame(rows)
    save_table(res.drop(columns=["R"]), "e06_continual_learning")
    save_table(res, "e06_continual_learning_with_matrices")
    # accuracy-matrix heatmaps for the plan's order
    oname = " → ".join(DATASET_NAMES[n] for n in orders[0])
    labels = [DATASET_NAMES[n] for n in orders[0]]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, strat in zip(axes, ["Naive fine-tuning", "EWC", "Replay", "EWC + Replay"]):
        R = mats[(oname, strat)]
        ax.imshow(R, cmap="RdYlGn", vmin=0.3, vmax=1)
        ax.grid(False)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{R[i, j]:.2f}", ha="center", va="center", fontsize=9)
        ax.set_xticks(range(3), labels, rotation=25, ha="right")
        ax.set_yticks(range(3), [f"after {l}" for l in labels])
        ax.set_title(strat)
    fig.suptitle("Balanced accuracy matrix R[t,i] (row = after training task t, column = evaluated task i)")
    savefig(fig, "e06_continual_matrices")
    s = res[(res.metric == "balanced_acc") & (res.order == oname)].set_index("strategy")[
        ["avg_final", "BWT", "forgetting"]]
    heatmap(s, "Continual learning summary (balanced accuracy)", "e06_continual_summary", fmt="{:+.3f}", cmap="PuOr")
    return {"results": res}
