"""Power BI export — star schema CSVs from experiment results and from the live stream database.

Tables (powerbi/data/*.csv)
  dim_dataset, dim_attack (taxonomy), dim_feature
  fact_model_performance (baselines, cross-dataset, DA, transfer, continual, zero-shot, NLP)
  fact_feature_importance (SHAP), fact_drift_feature, fact_drift_summary, fact_label_distribution
  stream_events / stream_alerts / stream_drift / stream_metrics (live detector)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..data.taxonomy import taxonomy_table
from ..utils import get_logger, load_config

log = get_logger("powerbi")


def _t(cfg, name):
    p = Path(cfg["paths"]["tables"]) / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def export_experiment_tables(cfg=None):
    cfg = cfg or load_config()
    out = Path(cfg["paths"]["powerbi"]) / "data"
    out.mkdir(parents=True, exist_ok=True)
    written = []

    def w(df, name):
        if df is not None and len(df):
            p = out / f"{name}.csv"
            df.to_csv(p, index=False)
            written.append(p)

    w(pd.DataFrame({"dataset_key": ["nsl", "forge", "edge"], "dataset": ["NSL-KDD", "ForgeIIoT", "Edge-IIoTset"],
                    "era": ["Legacy (1999/2009)", "Bridge IIoT (2023)", "Modern IIoT (2022)"],
                    "granularity": ["connection", "packet", "packet"]}), "dim_dataset")
    w(taxonomy_table(), "dim_attack")
    perf = []
    b = _t(cfg, "e02_baselines_binary")
    if b is not None:
        perf.append(b.assign(experiment="Baseline (in-domain)", source=b.dataset, target=b.dataset, method=b.model))
    c = _t(cfg, "e03_cross_dataset_binary")
    if c is not None:
        perf.append(c.assign(experiment="Cross-dataset " + c.space, method=c.model))
    d = _t(cfg, "e05_domain_adaptation")
    if d is not None:
        sp = d.pair.str.split(" → ", expand=True)
        perf.append(d.assign(experiment="Domain adaptation", source=sp[0], target=sp[1]))
    t = _t(cfg, "e04_transfer_learning_results")
    if t is not None:
        perf.append(t.rename(columns={"auc": "roc_auc"}).assign(experiment="Transfer learning", target="Edge-IIoTset",
                                                                 label_fraction=t.frac))
    n = _t(cfg, "e07_nlp_cross_dataset")
    if n is not None:
        perf.append(n.assign(experiment="NLP representation", method=n.representation))
    if perf:
        cols = ["experiment", "source", "target", "method", "accuracy", "precision", "recall", "f1", "roc_auc", "mcc",
                "balanced_acc", "fpr", "label_fraction"]
        allp = pd.concat(perf, ignore_index=True)
        w(allp.reindex(columns=cols), "fact_model_performance")
    w(_t(cfg, "e06_continual_learning"), "fact_continual_learning")
    w(_t(cfg, "e08_zeroshot_lofo"), "fact_zeroshot_lofo")
    w(_t(cfg, "e08_zeroshot_cross_dataset"), "fact_zeroshot_cross_dataset")
    s = _t(cfg, "e09_shap_cbs_importance_by_dataset")
    if s is not None:
        w(s.melt(id_vars="feature", var_name="dataset", value_name="importance"), "fact_feature_importance_cbs")
    w(_t(cfg, "e09_shap_top_features_native"), "fact_feature_importance_native")
    w(_t(cfg, "e10_feature_drift"), "fact_drift_feature")
    w(_t(cfg, "e10_drift_summary"), "fact_drift_summary")
    w(_t(cfg, "e01_label_distribution"), "fact_label_distribution")
    w(_t(cfg, "e11_stream_benchmark_timeline"), "fact_stream_benchmark_timeline")
    w(_t(cfg, "e11_stream_benchmark"), "fact_stream_benchmark")
    log.info(f"Power BI: wrote {len(written)} CSVs to {out}")
    return written


def export_stream_tables(db, cfg=None):
    cfg = cfg or load_config()
    out = Path(cfg["paths"]["powerbi"]) / "data"
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for tbl in ("events", "alerts", "drift", "metrics"):
        df = pd.read_sql_query(f"SELECT * FROM {tbl}", db)
        p = out / f"stream_{tbl}.csv"
        df.to_csv(p, index=False)
        paths.append(p)
    return paths
