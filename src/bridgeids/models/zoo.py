"""Baseline model zoo (Team-1 deliverable 3): DT, RF, AdaBoost, XGBoost, LightGBM, MLP."""
from __future__ import annotations

import warnings
from typing import Dict

import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None


def make_model(name: str, seed: int = 42, n_jobs: int = -1, n_classes: int = 2):
    name = name.lower()
    if name in ("dt", "decision_tree"):
        return DecisionTreeClassifier(max_depth=None, min_samples_leaf=2, random_state=seed)
    if name in ("rf", "random_forest"):
        return RandomForestClassifier(n_estimators=150, min_samples_leaf=1, n_jobs=n_jobs, random_state=seed)
    if name in ("ada", "adaboost"):
        return AdaBoostClassifier(n_estimators=150, learning_rate=0.5, random_state=seed)
    if name in ("xgb", "xgboost"):
        return XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
                             tree_method="hist", n_jobs=n_jobs, random_state=seed, verbosity=0,
                             eval_metric="logloss" if n_classes == 2 else "mlogloss")
    if name in ("lgbm", "lightgbm"):
        return LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.08, subsample=0.9, subsample_freq=1,
                              colsample_bytree=0.9, n_jobs=n_jobs, random_state=seed, verbose=-1)
    if name == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), early_stopping=True,
                                                             max_iter=60, random_state=seed))
    raise ValueError(name)


BASELINES = ["dt", "rf", "ada", "xgb", "lgbm", "mlp"]
PRETTY = {"dt": "Decision Tree", "rf": "Random Forest", "ada": "AdaBoost", "xgb": "XGBoost", "lgbm": "LightGBM",
          "mlp": "MLP"}


def proba_pos(model, X) -> np.ndarray:
    p = model.predict_proba(X)
    return p[:, 1] if p.ndim == 2 and p.shape[1] > 1 else p.ravel()
