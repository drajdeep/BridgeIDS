"""Unsupervised domain adaptation for tabular IDS features (Team-1 #6, RQ3).

All methods use labelled SOURCE data and *unlabelled* TARGET data only.

* DomainNorm  – per-domain standardisation (simple, strong baseline)
* CORAL       – CORrelation ALignment (Sun, Feng & Saenko, AAAI 2016)
* TCA         – Transfer Component Analysis (Pan et al., IEEE TNN 2011)
* IW          – importance weighting with a domain classifier (density-ratio estimation)
* DANN        – see models.npnet.DANN
"""
from __future__ import annotations

import numpy as np
from scipy import linalg
from sklearn.linear_model import LogisticRegression


class DomainScaler:
    """z-score with statistics of the domain itself (no labels needed)."""

    def fit(self, X):
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-6
        return self

    def transform(self, X):
        return (X - self.mu) / self.sd


def coral(Xs: np.ndarray, Xt: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """Return source features re-coloured with the target covariance (inputs should be standardised)."""
    Cs = np.cov(Xs, rowvar=False) + eps * np.eye(Xs.shape[1])
    Ct = np.cov(Xt, rowvar=False) + eps * np.eye(Xt.shape[1])
    Ws = linalg.fractional_matrix_power(Cs, -0.5)
    Wt = linalg.fractional_matrix_power(Ct, 0.5)
    return np.real(Xs @ Ws @ Wt)


class TCA:
    """Transfer Component Analysis with a linear or RBF kernel (closed form, eigen-decomposition)."""

    def __init__(self, dim: int = 20, kernel: str = "linear", mu: float = 1.0, gamma: float | None = None):
        self.dim, self.kernel, self.mu, self.gamma = dim, kernel, mu, gamma

    def _k(self, A, B):
        if self.kernel == "linear":
            return A @ B.T
        g = self.gamma or 1.0 / A.shape[1]
        d = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
        return np.exp(-g * np.clip(d, 0, None))

    def fit(self, Xs, Xt):
        self.X = np.vstack([Xs, Xt])
        ns, nt = len(Xs), len(Xt)
        n = ns + nt
        e = np.vstack([np.full((ns, 1), 1.0 / ns), np.full((nt, 1), -1.0 / nt)])
        M = e @ e.T
        M /= np.linalg.norm(M, "fro")
        H = np.eye(n) - np.full((n, n), 1.0 / n)
        K = self._k(self.X, self.X)
        A = np.linalg.solve(K @ M @ K + self.mu * np.eye(n), K @ H @ K)
        w, V = np.linalg.eig(A)
        order = np.argsort(-np.real(w))[: self.dim]
        self.W = np.real(V[:, order])
        return self

    def transform(self, X):
        return self._k(X, self.X) @ self.W


def importance_weights(Xs, Xt, clip: float = 20.0, seed: int = 0) -> np.ndarray:
    """w(x) = p_t(x)/p_s(x) estimated with a logistic domain classifier."""
    X = np.vstack([Xs, Xt])
    d = np.concatenate([np.zeros(len(Xs)), np.ones(len(Xt))])
    clf = LogisticRegression(max_iter=500, C=1.0, random_state=seed).fit(X, d)
    p = np.clip(clf.predict_proba(Xs)[:, 1], 1e-4, 1 - 1e-4)
    w = (p / (1 - p)) * (len(Xs) / len(Xt))
    w = np.clip(w, 1.0 / clip, clip)
    return w / w.mean()


def self_training(make_clf, Xs, ys, Xt, rounds: int = 3, conf: float = 0.95, max_add: int = 20000, seed: int = 0):
    """Pseudo-label self-training: iteratively add confidently-labelled target samples to the training set."""
    rng = np.random.default_rng(seed)
    clf = make_clf().fit(Xs, ys)
    X_aug, y_aug = Xs, ys
    for _ in range(rounds):
        p = clf.predict_proba(Xt)[:, 1]
        sel = np.where((p >= conf) | (p <= 1 - conf))[0]
        if len(sel) == 0:
            break
        # class-balanced selection so the majority class does not swamp the pseudo-labels
        pos, neg = sel[p[sel] >= 0.5], sel[p[sel] < 0.5]
        k = min(max_add // 2, max(len(pos), 1), max(len(neg), 1)) if len(pos) and len(neg) else min(max_add, len(sel))
        pick = np.concatenate([rng.choice(pos, min(k, len(pos)), replace=False) if len(pos) else [],
                               rng.choice(neg, min(k, len(neg)), replace=False) if len(neg) else []]).astype(int)
        X_aug = np.vstack([Xs, Xt[pick]])
        y_aug = np.concatenate([ys, (p[pick] >= 0.5).astype(int)])
        clf = make_clf().fit(X_aug, y_aug)
    return clf
