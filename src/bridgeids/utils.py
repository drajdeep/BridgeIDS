"""Shared helpers: config, logging, seeding, metrics, I/O."""
from __future__ import annotations

import json
import logging
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parents[2]
_CFG: Optional[Dict[str, Any]] = None


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load configs/default.yaml (or `path`) once and resolve paths against the project root."""
    global _CFG
    if _CFG is not None and path is None:
        return _CFG
    cfg_path = Path(path) if path else ROOT / "configs" / "default.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for k, v in cfg["paths"].items():
        p = Path(v)
        cfg["paths"][k] = p if p.is_absolute() else ROOT / p
    for k in ("tables", "figures", "models", "processed", "reports", "powerbi", "weka"):
        Path(cfg["paths"][k]).mkdir(parents=True, exist_ok=True)
    if path is None:
        _CFG = cfg
    return cfg


def get_logger(name: str = "bridgeids") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:  # torch is optional for most of the pipeline
        pass


@contextmanager
def timer(label: str, logger: Optional[logging.Logger] = None):
    t0 = time.time()
    yield
    (logger or get_logger()).info(f"{label} finished in {time.time() - t0:.1f}s")


def binary_metrics(y_true, y_pred, y_score=None) -> Dict[str, float]:
    """Accuracy / precision / recall / F1 / ROC-AUC / PR-AUC for attack(1) vs normal(0)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan,
        "mcc": matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan,
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
        out["pr_auc"] = average_precision_score(y_true, y_score)
    else:
        out["roc_auc"] = np.nan
        out["pr_auc"] = np.nan
    # false-positive rate on normal traffic – the number operators care about
    neg = y_true == 0
    out["fpr"] = float((y_pred[neg] == 1).mean()) if neg.any() else np.nan
    return {k: float(v) for k, v in out.items()}


def multiclass_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def bootstrap_ci(y_true, y_pred, metric=f1_score, n: int = 300, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI for a metric (default F1)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        vals.append(metric(y_true[idx], y_pred[idx], zero_division=0) if metric in (f1_score, precision_score, recall_score)
                    else metric(y_true[idx], y_pred[idx]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def save_table(df: pd.DataFrame, name: str, cfg=None, index: bool = False) -> Path:
    cfg = cfg or load_config()
    p = Path(cfg["paths"]["tables"]) / f"{name}.csv"
    df.to_csv(p, index=index)
    try:
        md = df.to_markdown(index=index, floatfmt=".4f")
    except Exception:
        md = df.to_string(index=index)
    (Path(cfg["paths"]["tables"]) / f"{name}.md").write_text(md, encoding="utf-8")
    return p


def save_json(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return str(o)

    path.write_text(json.dumps(obj, indent=2, default=_default), encoding="utf-8")


def stratified_cap(df: pd.DataFrame, label_col: str, n: Optional[int], seed: int = 42) -> pd.DataFrame:
    """Down-sample to at most n rows while keeping every class (proportional, min 1% of n per class)."""
    if n is None or len(df) <= n:
        return df
    frac = n / len(df)
    rng = np.random.default_rng(seed)
    parts = []
    for _, g in df.groupby(label_col, sort=False):
        k = max(int(round(len(g) * frac)), min(len(g), max(1, n // 100)))
        parts.append(g.iloc[rng.choice(len(g), size=min(k, len(g)), replace=False)])
    return pd.concat(parts).sample(frac=1.0, random_state=seed)
