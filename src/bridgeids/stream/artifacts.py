"""Train / load the models used by the live detector."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..data.loaders import load
from ..features.views import native_features, space
from ..models.hybrid import NormalCalibratedAnomaly
from ..models.zoo import make_model
from ..utils import get_logger, load_config, stratified_cap

log = get_logger("artifacts")

# Edge families withheld from the live model so the "zero_day" replay scenario has genuinely unseen attacks
ZERO_DAY_FAMILIES = ["XSS", "Ransomware"]


def artifact_path(name: str = "stream_artifacts", cfg=None) -> Path:
    cfg = cfg or load_config()
    return Path(cfg["paths"]["models"]) / f"{name}.joblib"


def build_artifacts(train_on=("forge", "edge"), holdout=ZERO_DAY_FAMILIES, name="stream_artifacts",
                    rows_per_ds=60000, cfg=None):
    cfg = cfg or load_config()
    seed = cfg["seed"]
    frames = []
    from ..experiments.common import _split_frame
    for n in train_on:
        df = _split_frame(n, seed)[0]          # TRAIN split only – the replayer streams the held-out TEST split
        df = df[~df["label_fine"].isin(holdout)]
        frames.append((n, stratified_cap(df, "label_fine", rows_per_ds, seed)))
    X = pd.concat([space(df, n, "pbs") for n, df in frames], ignore_index=True)
    y = np.concatenate([df["y"].to_numpy() for _, df in frames])
    fam = np.concatenate([df["super"].to_numpy() for _, df in frames])
    binm = make_model("lgbm", seed).fit(X.values, y)
    famm = make_model("lgbm", seed).fit(X.values, fam)
    anom = NormalCalibratedAnomaly(seed=seed).fit(X.values[y == 0])
    rng = np.random.default_rng(seed)
    ref = X.values[rng.choice(len(X), min(20000, len(X)), replace=False)]
    ridx = rng.choice(len(X), min(2000, len(X)), replace=False)
    art = {"pbs_cols": list(X.columns), "pbs_binary": binm, "pbs_family": famm, "anomaly": anom, "psi_ref": ref,
           "replay_X": X.values[ridx].astype(np.float32), "replay_y": y[ridx],
           "meta": {"trained_on": list(train_on), "zero_day_holdout": list(holdout), "n_train": int(len(y))}}
    # NSL connection records -> native NSL model
    nsl = load("nsl")
    nsl_tr = nsl[nsl.split == "train"]
    Xn, cats = native_features(nsl_tr, "nsl")
    art["nsl_cats"] = cats
    art["nsl_cols"] = list(Xn.columns)
    art["nsl_binary"] = make_model("lgbm", seed).fit(Xn.values, nsl_tr["y"].to_numpy())
    art["nsl_family"] = make_model("lgbm", seed).fit(Xn.values, nsl_tr["super"].to_numpy())
    art["nsl_anomaly"] = NormalCalibratedAnomaly(seed=seed).fit(Xn.values[nsl_tr["y"].to_numpy() == 0])
    p = artifact_path(name, cfg)
    joblib.dump(art, p, compress=3)
    log.info(f"saved {p} ({p.stat().st_size / 1e6:.1f} MB) trained on {train_on}, zero-day holdout={holdout}")
    return art


def load_artifacts(name="stream_artifacts", cfg=None, build_if_missing=True):
    p = artifact_path(name, cfg)
    if not p.exists():
        if not build_if_missing:
            raise FileNotFoundError(p)
        log.info("stream artifacts not found – training them now (one-off, ~1 min)")
        return build_artifacts(name=name, cfg=cfg)
    return joblib.load(p)
