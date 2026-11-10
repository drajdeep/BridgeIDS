"""Shared split / feature-space helpers used by every experiment."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..data.loaders import DATASET_NAMES, load
from ..features.views import native_features, space
from ..utils import load_config, stratified_cap

DS = ["nsl", "forge", "edge"]


@dataclass
class Split:
    name: str
    space: str
    Xtr: pd.DataFrame
    ytr: np.ndarray
    Xte: pd.DataFrame
    yte: np.ndarray
    mtr: pd.DataFrame   # meta: label_fine, super (aligned with Xtr rows)
    mte: pd.DataFrame

    @property
    def pretty(self):
        return DATASET_NAMES[self.name]


def _split_frame(name: str, seed: int, temporal: bool = False):
    df = load(name)
    if name == "nsl":
        return df[df.split == "train"].reset_index(drop=True), df[df.split == "test"].reset_index(drop=True)
    if temporal and "_ts" in df.columns:
        # capture-temporal split: first 70 % (by time) of every capture file -> train, last 30 % -> test
        tr, te = [], []
        for _, g in df.groupby("label_fine"):
            g = g.sort_values("_ts")
            k = int(len(g) * 0.7)
            tr.append(g.iloc[:k])
            te.append(g.iloc[k:])
        return pd.concat(tr).sample(frac=1, random_state=seed).reset_index(drop=True), \
            pd.concat(te).sample(frac=1, random_state=seed).reset_index(drop=True)
    tr, te = train_test_split(df, test_size=0.3, stratify=df["label_fine"], random_state=seed)
    return tr.reset_index(drop=True), te.reset_index(drop=True)


@lru_cache(maxsize=16)
def get_frames(name: str, seed: int = 42, cap_train: Optional[int] = -1, cap_test: Optional[int] = -1,
               temporal: bool = False):
    """Raw (capped) train/test frames — the exact rows behind get_split()."""
    cfg = load_config()
    cap_train = cfg["experiments"]["max_train_rows"] if cap_train == -1 else cap_train
    cap_test = cfg["experiments"]["max_test_rows"] if cap_test == -1 else cap_test
    tr, te = _split_frame(name, seed, temporal)
    tr = stratified_cap(tr, "label_fine", cap_train, seed).reset_index(drop=True)
    te = stratified_cap(te, "label_fine", cap_test, seed + 1).reset_index(drop=True)
    return tr, te


@lru_cache(maxsize=32)
def get_split(name: str, kind: str = "cbs", seed: int = 42, cap_train: Optional[int] = -1,
              cap_test: Optional[int] = -1, temporal: bool = False, include_leaky: bool = False) -> Split:
    """kind in {'native','cbs','pbs'}. cap=-1 -> use config caps, None -> no cap."""
    tr, te = get_frames(name, seed, cap_train, cap_test, temporal)
    if kind == "native":
        Xtr, cats = native_features(tr, name, include_leaky=include_leaky)
        Xte, _ = native_features(te, name, categories=cats, include_leaky=include_leaky)
        Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)
    else:
        Xtr, Xte = space(tr, name, kind), space(te, name, kind)
    meta_cols = ["label_fine", "super"]
    return Split(name, kind, Xtr, tr["y"].to_numpy(), Xte, te["y"].to_numpy(),
                 tr[meta_cols].reset_index(drop=True), te[meta_cols].reset_index(drop=True))


def subsample(X, y, n, seed=0, meta=None):
    if n is None or len(X) <= n:
        return (X, y) if meta is None else (X, y, meta)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), n, replace=False)
    Xs = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
    if meta is None:
        return Xs, y[idx]
    return Xs, y[idx], meta.iloc[idx]
