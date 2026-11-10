"""Hybrid supervised + one-class anomaly detector (zero-day aware).

score = max( p_supervised(attack | x) ,  w-scaled ECDF_normal(anomaly(x)) )

* supervised part: LightGBM (knows the attack families it was trained on)
* anomaly part: IsolationForest fitted on benign traffic only; its raw score is converted to a
  calibrated percentile with the empirical CDF of held-out benign scores, so "0.99" means
  "more anomalous than 99 % of normal traffic". Unseen attack families therefore still raise alerts.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from .zoo import make_model, proba_pos


class NormalCalibratedAnomaly:
    def __init__(self, n_estimators: int = 200, seed: int = 0, max_samples: int = 2048):
        self.iforest = IsolationForest(n_estimators=n_estimators, max_samples=max_samples, random_state=seed,
                                       n_jobs=-1)
        self.seed = seed

    def fit(self, X_normal: np.ndarray):
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(X_normal))
        cut = max(1, int(0.7 * len(idx)))
        self.iforest.fit(X_normal[idx[:cut]])
        cal = X_normal[idx[cut:]] if len(idx) - cut > 50 else X_normal
        self.ref = np.sort(-self.iforest.score_samples(cal))
        return self

    def raw(self, X):
        return -self.iforest.score_samples(X)

    def percentile(self, X):
        """ECDF of benign anomaly scores -> value in [0, 1]."""
        return np.searchsorted(self.ref, self.raw(X), side="right") / len(self.ref)


class HybridDetector:
    def __init__(self, seed: int = 0, anomaly_weight: float = 1.0, anomaly_gate: float = 0.995):
        self.seed, self.w, self.gate = seed, anomaly_weight, anomaly_gate

    def fit(self, X, y, X_normal_extra=None):
        X = np.asarray(X, float)
        self.sup = make_model("lgbm", self.seed).fit(X, y)
        Xn = X[np.asarray(y) == 0]
        if X_normal_extra is not None:
            Xn = np.vstack([Xn, X_normal_extra])
        self.anom = NormalCalibratedAnomaly(seed=self.seed).fit(Xn)
        return self

    def components(self, X):
        X = np.asarray(X, float)
        return proba_pos(self.sup, X), self.anom.percentile(X)

    def score(self, X):
        p, a = self.components(X)
        # anomaly percentile only contributes above the gate (keeps the false-positive rate bounded)
        a_term = np.where(a >= self.gate, self.w * (0.5 + 0.5 * (a - self.gate) / (1 - self.gate + 1e-9)), 0.0)
        return np.maximum(p, a_term)
