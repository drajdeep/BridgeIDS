"""Explainability helpers: SHAP (TreeExplainer), LIME, explanation-drift metrics, incident narratives."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

FEATURE_PHRASES = {
    "syn_only": "half-open SYN without ACK", "flag_rst": "TCP reset", "flag_syn": "SYN flag",
    "flag_ack": "ACK flag", "flag_psh": "PUSH flag", "flag_fin": "FIN flag", "log_bytes": "payload volume",
    "has_payload": "presence of payload", "svc_mqtt": "MQTT broker port", "svc_http": "HTTP port",
    "app_mqtt": "MQTT application traffic", "app_http": "HTTP application traffic", "land": "source = destination",
    "proto_icmp": "ICMP protocol", "proto_udp": "UDP protocol", "proto_tcp": "TCP protocol", "proto_arp": "ARP frame",
    "svc_ephemeral": "ephemeral port", "svc_registered": "registered high port", "log_tcp_len": "TCP segment length",
    "log_tcp_seq": "TCP sequence offset", "log_tcp_ack": "TCP ACK offset", "http_get": "HTTP GET",
    "http_post": "HTTP POST", "log_http_clen": "HTTP content length", "mqtt_publish": "MQTT PUBLISH",
    "mqtt_connect": "MQTT CONNECT", "serror_rate": "SYN-error rate", "same_srv_rate": "same-service rate",
    "count": "connections to same host (2 s)", "dst_host_srv_count": "connections to same service",
    "src_bytes": "bytes from source", "dst_bytes": "bytes to source", "tcp.time_delta": "inter-packet time",
}


def _clean(name: str) -> str:
    return name.replace("pbs_", "")


def phrase(name: str) -> str:
    n = _clean(name)
    return FEATURE_PHRASES.get(n, n)


def shap_values_binary(model, X: np.ndarray) -> np.ndarray:
    import shap
    ex = shap.TreeExplainer(model)
    sv = ex.shap_values(X, check_additivity=False)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[..., 1]
    return sv


def shap_importance(model, X: np.ndarray, names: Sequence[str]) -> pd.Series:
    sv = shap_values_binary(model, X)
    return pd.Series(np.abs(sv).mean(0), index=list(names)).sort_values(ascending=False)


def rank_agreement(a: pd.Series, b: pd.Series, k: int = 10) -> Dict[str, float]:
    common = a.index.intersection(b.index)
    a, b = a[common], b[common]
    rho = spearmanr(a.values, b.values).correlation
    tau = kendalltau(a.values, b.values).correlation
    ta, tb = set(a.sort_values(ascending=False).index[:k]), set(b.sort_values(ascending=False).index[:k])
    return {"spearman": float(np.nan_to_num(rho)), "kendall": float(np.nan_to_num(tau)),
            f"top{k}_jaccard": len(ta & tb) / len(ta | tb)}


def lime_explain(model, X_train: np.ndarray, x: np.ndarray, names: Sequence[str], num_features: int = 6,
                 num_samples: int = 1500, seed: int = 0) -> List[tuple]:
    from lime.lime_tabular import LimeTabularExplainer
    ex = LimeTabularExplainer(X_train, feature_names=list(names), class_names=["normal", "attack"],
                              discretize_continuous=False, random_state=seed, mode="classification")
    e = ex.explain_instance(x, model.predict_proba, num_features=num_features, num_samples=num_samples)
    return e.as_list()


def narrative(prob: float, contribs: Sequence[tuple], label: str | None = None, record_id=None,
              anomaly: float | None = None) -> str:
    """Human-readable incident text from (feature, value, shap) triples."""
    verdict = "ATTACK" if prob >= 0.5 else "normal"
    head = f"Record {record_id}: " if record_id is not None else ""
    lab = f" (predicted family: {label})" if label else ""
    drivers = []
    for f, val, s in contribs:
        direction = "towards attack" if s > 0 else "towards normal"
        is_num = isinstance(val, (int, float, np.floating))
        if is_num and float(val) in (0.0, 1.0) and _clean(f).split("_")[0] in ("svc", "proto", "flag", "app", "mqtt",
                                                                                 "http", "arp", "has", "syn", "land"):
            txt = ("presence of " if float(val) == 1.0 else "absence of ") + phrase(f)
        else:
            txt = f"{phrase(f)} = {val:.3g}" if is_num else f"{phrase(f)} = {val}"
        drivers.append(f"{txt} ({direction}, {s:+.2f})")
    an = f" Anomaly percentile vs benign traffic: {anomaly:.1%}." if anomaly is not None else ""
    return f"{head}classified as {verdict}{lab} with confidence {prob:.0%}.{an} Main drivers: " + "; ".join(drivers) + "."
