"""E12 — Deployment artifacts + Weka experiments + Power BI export.

1. live-detector models (ForgeIIoT + Edge-IIoT train splits; XSS & Ransomware withheld = zero-day demo)
2. Weka: ARFF files (native + CBS for every dataset) and — if a weka.jar is found — real Weka runs of
   J48, RandomForest, AdaBoostM1 (in-domain and cross-dataset CBS) parsed into a results table
3. Power BI star-schema CSV export
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.loaders import DATASET_NAMES
from ..reporting.powerbi import export_experiment_tables
from ..stream.artifacts import build_artifacts
from ..utils import ROOT, get_logger, save_table
from .common import DS, get_split

log = get_logger("e12")

WEKA_CLASSIFIERS = {"J48": "weka.classifiers.trees.J48",
                    "RandomForest": "weka.classifiers.trees.RandomForest -I 100",
                    "AdaBoostM1": "weka.classifiers.meta.AdaBoostM1 -I 50 -W weka.classifiers.trees.DecisionStump"}


def write_arff(path: Path, X: pd.DataFrame, y: np.ndarray, relation: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [re.sub(r"[^A-Za-z0-9_]", "_", c) for c in X.columns]
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"@RELATION {relation}\n\n")
        for c in cols:
            f.write(f"@ATTRIBUTE {c} NUMERIC\n")
        f.write("@ATTRIBUTE class {normal,attack}\n\n@DATA\n")
        vals = X.values.astype(float)
        lab = np.where(np.asarray(y).astype(int) == 1, "attack", "normal")
        for row, l in zip(vals, lab):
            f.write(",".join(f"{v:.6g}" for v in row) + f",{l}\n")


def find_weka_jar():
    for p in [os.environ.get("WEKA_JAR", ""), str(ROOT / "weka" / "weka.jar"), "/tmp/weka.jar",
              r"C:\Program Files\Weka-3-8-6\weka.jar", r"C:\Program Files\Weka-3-9-7\weka.jar",
              os.path.expanduser("~/weka/weka.jar")]:
        if p and Path(p).exists():
            return p
    try:  # python-weka-wrapper3 ships a weka.jar
        import weka
        p = Path(weka.__file__).parent / "lib" / "weka.jar"
        if p.exists():
            return str(p)
    except Exception:
        pass
    return None


def run_weka(jar, train, test, clf):
    cmd = ["java", "-Xmx2g", "-cp", jar] + WEKA_CLASSIFIERS[clf].split()[:1] + \
          ["-t", str(train), "-T", str(test), "-o"] + WEKA_CLASSIFIERS[clf].split()[1:]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=1800).stdout
    sec = out.split("=== Error on test data ===")[-1] if "Error on test data" in out else out
    acc = re.search(r"Correctly Classified Instances\s+\d+\s+([\d.]+)\s*%", sec)
    wavg = re.search(r"Weighted Avg\.\s+([\d.?]+)\s+([\d.?]+)\s+([\d.?]+)\s+([\d.?]+)\s+([\d.?]+)\s+([\d.?-]+)\s+"
                     r"([\d.?]+)", sec)
    attack = re.search(r"\n\s+([\d.?]+)\s+([\d.?]+)\s+([\d.?]+)\s+([\d.?]+)\s+([\d.?]+)\s+([\d.?-]+)\s+([\d.?]+)\s+"
                       r"([\d.?]+)\s+attack", sec)
    f = lambda m, i: float(m.group(i)) if m and m.group(i) not in ("?",) else np.nan  # noqa: E731
    return {"accuracy": f(acc, 1) / 100 if acc else np.nan, "attack_precision": f(attack, 3),
            "attack_recall": f(attack, 4), "attack_f1": f(attack, 5), "mcc": f(attack, 6), "roc_auc": f(attack, 7),
            "weighted_f1": f(wavg, 5)}


def run(cfg):
    seed = cfg["seed"]
    build_artifacts(cfg=cfg)
    wdir = Path(cfg["paths"]["weka"]) / "data"
    files = {}
    for n in DS:
        for kind in ("native", "cbs"):
            sp = get_split(n, kind, seed, cap_train=20000, cap_test=10000)
            tr, te = wdir / f"{n}_{kind}_train.arff", wdir / f"{n}_{kind}_test.arff"
            write_arff(tr, sp.Xtr, sp.ytr, f"{n}_{kind}")
            write_arff(te, sp.Xte, sp.yte, f"{n}_{kind}")
            files[(n, kind)] = (tr, te)
    log.info(f"ARFF files written to {wdir}")
    jar = find_weka_jar()
    if not jar or shutil.which("java") is None:
        log.warning("weka.jar or java not found – ARFFs exported, Weka runs skipped (see weka/README.md)")
    else:
        rows = []
        for clf in WEKA_CLASSIFIERS:
            for n in DS:
                r = run_weka(jar, *files[(n, "native")], clf)
                rows.append({"setting": "in-domain (native)", "train": DATASET_NAMES[n], "test": DATASET_NAMES[n],
                             "classifier": f"Weka {clf}", **r})
                log.info(f"Weka {clf} {n} native: {r}")
            for a, b in (("nsl", "edge"), ("forge", "edge"), ("nsl", "forge")):
                r = run_weka(jar, files[(a, "cbs")][0], files[(b, "cbs")][1], clf)
                rows.append({"setting": "cross-dataset (CBS)", "train": DATASET_NAMES[a], "test": DATASET_NAMES[b],
                             "classifier": f"Weka {clf}", **r})
        save_table(pd.DataFrame(rows), "e12_weka_results")
    export_experiment_tables(cfg)
    return {}
