"""Dataset-drift measures (Team-2 #6): feature, concept and attack drift.

feature drift : KL divergence, Jensen-Shannon distance, Wasserstein-1, PSI per feature
global shift  : RBF-MMD, domain-classifier AUC and proxy-A-distance (Ben-David et al.)
concept drift : change of P(y | x) — cross-domain vs in-domain performance gap, conditional label-rate
                shift per feature bin, prediction disagreement of source/target models
attack drift  : Jensen-Shannon distance between attack (super-class) label distributions
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict


def _hist(a, b, bins=20):
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    if hi <= lo:
        hi = lo + 1e-6
    edges = np.unique(np.linspace(lo, hi, bins + 1))
    if len(np.unique(np.concatenate([a, b]))) <= 3:   # binary / categorical
        vals = np.unique(np.concatenate([a, b]))
        pa = np.array([(a == v).mean() for v in vals])
        pb = np.array([(b == v).mean() for v in vals])
        return pa, pb
    pa, _ = np.histogram(a, edges)
    pb, _ = np.histogram(b, edges)
    return pa / max(pa.sum(), 1), pb / max(pb.sum(), 1)


def feature_drift(A: np.ndarray, B: np.ndarray, eps: float = 1e-4):
    """Return dict of arrays (one value per feature)."""
    out = {"kl": [], "js": [], "wasserstein": [], "psi": []}
    for j in range(A.shape[1]):
        a, b = A[:, j].astype(float), B[:, j].astype(float)
        pa, pb = _hist(a, b)
        pa_, pb_ = pa + eps, pb + eps
        pa_, pb_ = pa_ / pa_.sum(), pb_ / pb_.sum()
        out["kl"].append(float(np.sum(pa_ * np.log(pa_ / pb_))))
        out["js"].append(float(jensenshannon(pa_, pb_, base=2)))
        sd = np.std(np.concatenate([a, b])) + 1e-9
        out["wasserstein"].append(float(wasserstein_distance(a / sd, b / sd)))
        out["psi"].append(float(np.sum((pa_ - pb_) * np.log(pa_ / pb_))))
    return {k: np.array(v) for k, v in out.items()}


def mmd_rbf(A, B, n: int = 1500, seed: int = 0):
    rng = np.random.default_rng(seed)
    A = A[rng.choice(len(A), min(n, len(A)), replace=False)]
    B = B[rng.choice(len(B), min(n, len(B)), replace=False)]
    Z = np.vstack([A, B])
    d = (Z ** 2).sum(1)[:, None] + (Z ** 2).sum(1)[None] - 2 * Z @ Z.T
    gamma = 1.0 / (np.median(d[d > 0]) + 1e-9)
    K = np.exp(-gamma * np.clip(d, 0, None))
    na = len(A)
    return float(K[:na, :na].mean() + K[na:, na:].mean() - 2 * K[:na, na:].mean())


def domain_classifier_auc(A, B, n: int = 5000, seed: int = 0):
    rng = np.random.default_rng(seed)
    A = A[rng.choice(len(A), min(n, len(A)), replace=False)]
    B = B[rng.choice(len(B), min(n, len(B)), replace=False)]
    X = np.vstack([A, B])
    y = np.r_[np.zeros(len(A)), np.ones(len(B))]
    from lightgbm import LGBMClassifier
    p = cross_val_predict(LGBMClassifier(n_estimators=100, verbose=-1, random_state=seed), X, y, cv=3,
                          method="predict_proba")[:, 1]
    auc = roc_auc_score(y, p)
    err = 1 - ((p >= 0.5) == y).mean()
    return float(auc), float(2 * (1 - 2 * min(err, 0.5)))   # (AUC, proxy-A-distance)


def conditional_label_shift(A, yA, B, yB, bins: int = 5):
    """Mean |P(y=1 | feature bin)_A - P(y=1 | feature bin)_B| per feature (concept drift proxy)."""
    res = []
    for j in range(A.shape[1]):
        a, b = A[:, j], B[:, j]
        edges = np.unique(np.quantile(np.concatenate([a, b]), np.linspace(0, 1, bins + 1)))
        if len(edges) < 2:
            res.append(0.0)
            continue
        ia, ib = np.digitize(a, edges[1:-1]), np.digitize(b, edges[1:-1])
        diffs = []
        for k in np.unique(np.concatenate([ia, ib])):
            ma, mb = ia == k, ib == k
            if ma.sum() > 30 and mb.sum() > 30:
                diffs.append(abs(yA[ma].mean() - yB[mb].mean()))
        res.append(float(np.mean(diffs)) if diffs else 0.0)
    return np.array(res)


def label_js(pA: dict, pB: dict):
    keys = sorted(set(pA) | set(pB))
    a = np.array([pA.get(k, 0) for k in keys], float) + 1e-9
    b = np.array([pB.get(k, 0) for k in keys], float) + 1e-9
    return float(jensenshannon(a / a.sum(), b / b.sum(), base=2))


class StreamPSI:
    """Windowed PSI of a feature stream against a reference sample (used by the live detector)."""

    def __init__(self, reference: np.ndarray, bins: int = 10):
        self.edges = [np.unique(np.quantile(reference[:, j], np.linspace(0, 1, bins + 1)))
                      for j in range(reference.shape[1])]
        self.ref = [self._p(reference[:, j], j) for j in range(reference.shape[1])]

    def _p(self, x, j):
        e = self.edges[j]
        if len(e) < 2:
            return np.array([1.0])
        c = np.bincount(np.clip(np.digitize(x, e[1:-1]), 0, len(e) - 2), minlength=len(e) - 1).astype(float)
        return (c + 1e-3) / (c.sum() + 1e-3 * len(c))

    def psi(self, W: np.ndarray) -> np.ndarray:
        out = []
        for j in range(W.shape[1]):
            p, q = self._p(W[:, j], j), self.ref[j]
            if len(p) != len(q):
                out.append(0.0)
                continue
            out.append(float(np.sum((p - q) * np.log(p / q))))
        return np.array(out)
