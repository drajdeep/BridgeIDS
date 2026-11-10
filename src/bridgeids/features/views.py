"""Feature alignment (Team-1 deliverable 2) — the core of the "bridge".

NSL-KDD describes *connections* (41 KDD features); ForgeIIoT and Edge-IIoTset describe *packets*
(tshark fields). We align them in three tiers:

1. ``canonical_view``  — a semantic, schema-free view (proto / service / TCP-flag semantics / bytes /
   land / application layer) that every record of every dataset can be mapped into. It feeds
   * CBS  – Canonical Behavioural Space   (all 3 datasets, ~26 numeric features)
   * the flow->text serializer used by the NLP branch
   * the streaming featurizer (schema auto-detection)
2. PBS — Packet Bridge Space (Forge <-> Edge only, ~50 features): CBS + packet-level fields that exist
   in both tshark exports (TCP seq/ack/len, ARP opcode, HTTP method/length, MQTT message types ...).
3. native — the richest leakage-free feature set of each dataset on its own (in-domain baselines).

Identifiers (IP/MAC addresses, checksums, raw ports, stream indices, frame numbers, epoch time) are
never features: the leakage audit (exp. 02) shows that they inflate ForgeIIoT accuracy to ~100 %.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

SERVICES = ["http", "mqtt", "dns", "ftp", "smtp", "remote_shell", "modbus", "icmp_echo", "wellknown_other",
            "registered", "ephemeral", "none"]
PROTOS = ["tcp", "udp", "icmp", "arp"]
MQTT_TYPES = {1: "connect", 2: "connack", 3: "publish", 4: "puback", 8: "subscribe", 9: "suback",
              12: "pingreq", 13: "pingresp", 14: "disconnect"}

# ---------------------------------------------------------------------------- helpers
_PORT_SERVICE = {80: "http", 8080: "http", 443: "http", 8000: "http", 8001: "http", 8443: "http",
                 1883: "mqtt", 8883: "mqtt", 53: "dns", 5353: "dns", 21: "ftp", 20: "ftp", 69: "ftp",
                 25: "smtp", 587: "smtp", 110: "smtp", 143: "smtp", 22: "remote_shell", 23: "remote_shell",
                 513: "remote_shell", 514: "remote_shell", 502: "modbus"}

_NSL_SERVICE = {
    **{s: "http" for s in ["http", "http_443", "http_8001", "http_2784", "harvest", "www"]},
    **{s: "dns" for s in ["domain", "domain_u"]},
    **{s: "ftp" for s in ["ftp", "ftp_data", "tftp_u"]},
    **{s: "smtp" for s in ["smtp", "pop_2", "pop_3", "imap4", "nntp"]},
    **{s: "remote_shell" for s in ["telnet", "ssh", "login", "shell", "exec", "klogin", "kshell", "remote_job"]},
    **{s: "icmp_echo" for s in ["ecr_i", "eco_i", "tim_i", "urp_i", "red_i"]},
    "private": "ephemeral", "other": "registered",
}

_NSL_FLAGS = {  # flag -> (syn, ack, fin, rst, syn_only)
    "SF": (1, 1, 1, 0, 0), "S0": (1, 0, 0, 0, 1), "REJ": (1, 1, 0, 1, 0), "RSTO": (1, 1, 0, 1, 0),
    "RSTR": (1, 1, 0, 1, 0), "RSTOS0": (1, 0, 0, 1, 1), "SH": (1, 0, 1, 0, 1), "S1": (1, 1, 0, 0, 0),
    "S2": (1, 1, 0, 0, 0), "S3": (1, 1, 0, 0, 0), "OTH": (0, 0, 0, 0, 0),
}


def _num(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").fillna(0).to_numpy(dtype=float)


def _col(df: pd.DataFrame, name: str, default=0.0) -> pd.Series:
    return df[name] if name in df.columns else pd.Series(default, index=df.index)


def _first_and_count(s: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """tshark multi-value cells such as '3,3,3' -> (first value, number of values)."""
    st = s.fillna("").astype(str).where(s.notna(), "")
    st = st.replace({"nan": "", "None": "", "0": "", "0.0": "", "<NA>": ""})
    first = pd.to_numeric(st.str.split(",").str[0], errors="coerce").fillna(0).to_numpy()
    count = np.where(st.str.len() > 0, st.str.count(",") + 1, 0)
    return first, count


def port_service(port: np.ndarray) -> np.ndarray:
    port = np.nan_to_num(port.astype(float)).astype(int)
    out = np.full(port.shape, "none", dtype=object)
    out[(port > 0) & (port < 1024)] = "wellknown_other"
    out[(port >= 1024) & (port < 49152)] = "registered"
    out[port >= 49152] = "ephemeral"
    for p, s in _PORT_SERVICE.items():
        out[port == p] = s
    return out


def _service_from_ports(dport: np.ndarray, sport: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Service = well-known side of the conversation (like NSL's 'service'); to_server = 1 if dst is that side."""
    ds, ss = port_service(dport), port_service(sport)
    named = set(_PORT_SERVICE.values())
    d_named = np.isin(ds, list(named))
    s_named = np.isin(ss, list(named))
    svc = np.where(d_named, ds, np.where(s_named, ss, np.where(dport > 0, ds, ss)))
    to_server = (d_named | (~s_named & (dport < np.where(sport > 0, sport, 1e9)))).astype(int)
    return svc, to_server


# ---------------------------------------------------------------------------- schema detection
def detect_schema(df: pd.DataFrame) -> str:
    cols = set(df.columns)
    if {"protocol_type", "service", "flag"} <= cols:
        return "nsl"
    if "tcp.flags.syn" in cols or "frame.protocols" in cols or "udp.dstport" in cols:
        return "forge"
    if "tcp.flags" in cols or "udp.port" in cols or "mqtt.hdrflags" in cols:
        return "edge"
    raise ValueError("Unknown record schema")


# ---------------------------------------------------------------------------- canonical view
def canonical_view(df: pd.DataFrame, schema: str | None = None) -> pd.DataFrame:
    """Map any record (connection or packet) to a common semantic view."""
    schema = schema or detect_schema(df)
    n = len(df)
    v = pd.DataFrame(index=df.index)
    if schema == "nsl":
        v["proto"] = df["protocol_type"].fillna("").astype(str).str.lower().values
        v["service"] = df["service"].fillna("").astype(str).map(lambda s: _NSL_SERVICE.get(s, "wellknown_other")).values
        fl = np.array([_NSL_FLAGS.get(str(f), (0, 0, 0, 0, 0)) for f in df["flag"]])
        fl = fl * (v["proto"].to_numpy() == "tcp")[:, None]  # TCP flag semantics only exist for TCP
        v["syn"], v["ack"], v["fin"], v["rst"], v["syn_only"] = fl.T
        v["psh"] = (_num(df["src_bytes"]) > 0).astype(int) & v["ack"].values
        v["bytes"] = _num(df["src_bytes"]) + _num(df["dst_bytes"])
        v["land"] = _num(df["land"]).astype(int)
        v["to_server"] = 1
        v["app_http"] = (v["service"] == "http").astype(int)
        v["app_mqtt"] = 0
        v["http_method"] = np.where(v["app_http"] == 1, "GET", "")
        v["mqtt_type"] = 0
        v["arp_op"] = 0
        return v

    if schema == "forge":
        tcp_d, tcp_s = _num(_col(df, "tcp.dstport")), _num(_col(df, "tcp.srcport"))
        udp_d, udp_s = _num(_col(df, "udp.dstport")), _num(_col(df, "udp.srcport"))
        is_tcp = _col(df, "tcp.dstport", np.nan).notna().to_numpy() | (tcp_d > 0)
        is_udp = _col(df, "udp.dstport", np.nan).notna().to_numpy() | (udp_d > 0)
        is_icmp = (_col(df, "icmp.type", np.nan).notna() | _col(df, "icmpv6.type", np.nan).notna()).to_numpy()
        arp_op = _num(_col(df, "arp.opcode"))
        syn, ack = _num(_col(df, "tcp.flags.syn")), _num(_col(df, "tcp.flags.ack"))
        fin, rst = _num(_col(df, "tcp.flags.fin")), _num(_col(df, "tcp.flags.reset"))
        psh = _num(_col(df, "tcp.flags.push"))
        tcp_len = _num(_col(df, "tcp.len"))
        udp_len = np.clip(_num(_col(df, "udp.length")) - 8, 0, None)
        meth = _col(df, "http.request.method", "").fillna("").astype(str).replace({"nan": "", "<NA>": "", "None": ""})
        mq_type, mq_n = _first_and_count(_col(df, "mqtt.msgtype", ""))
        src, dst = _col(df, "ip.src_host", "").fillna("").astype(str), _col(df, "ip.dst_host", "").fillna("").astype(str)
        land = ((src == dst) & (src.str.len() > 3) & ~src.isin(["nan", "<NA>"])).astype(int).to_numpy()
        dport = np.where(is_tcp, tcp_d, udp_d)
        sport = np.where(is_tcp, tcp_s, udp_s)
        icmp_type = _num(_col(df, "icmp.type"))
        payload = tcp_len + np.where(is_tcp, 0, udp_len)
    elif schema == "edge":
        tcp_d, tcp_s = _num(_col(df, "tcp.dstport")), _num(_col(df, "tcp.srcport"))
        flags = _num(_col(df, "tcp.flags")).astype(int)
        is_tcp = (tcp_d > 0) | (tcp_s > 0) | (flags > 0)
        udp_p = _num(_col(df, "udp.port"))
        is_udp = (udp_p > 0) & ~is_tcp
        is_icmp = ((_num(_col(df, "icmp.checksum")) > 0) | (_num(_col(df, "icmp.seq_le")) > 0)) & ~is_tcp
        arp_op = _num(_col(df, "arp.opcode"))
        fin, syn, rst, psh, ack = [((flags >> b) & 1).astype(float) for b in (0, 1, 2, 3, 4)]
        ack = np.maximum(ack, _num(_col(df, "tcp.flags.ack")))
        tcp_len = _num(_col(df, "tcp.len"))
        meth = _col(df, "http.request.method", "").fillna("").astype(str).replace({"0": "", "0.0": "", "nan": "", "<NA>": ""})
        mq_type = _num(_col(df, "mqtt.msgtype"))
        mq_n = (mq_type > 0).astype(int)
        src, dst = _col(df, "ip.src_host", "").fillna("").astype(str), _col(df, "ip.dst_host", "").fillna("").astype(str)
        land = ((src == dst) & (src.str.count(r"\.") == 3)).astype(int).to_numpy()
        dport = np.where(is_tcp, tcp_d, udp_p)
        sport = np.where(is_tcp, tcp_s, 0)
        icmp_type = np.zeros(n)
        payload = tcp_len.copy()
    else:
        raise ValueError(schema)

    proto = np.where(arp_op > 0, "arp", np.where(is_tcp, "tcp", np.where(is_udp, "udp",
                                                                          np.where(is_icmp, "icmp", "other"))))
    svc, to_server = _service_from_ports(dport, sport)
    svc = np.where(proto == "icmp", "icmp_echo", np.where(np.isin(proto, ["arp", "other"]), "none", svc))
    http_clen = _num(_col(df, "http.content_length"))
    mq_len, _ = _first_and_count(_col(df, "mqtt.len", "")) if schema == "forge" else (_num(_col(df, "mqtt.len")), None)
    v["proto"] = proto
    v["service"] = svc
    v["syn"], v["ack"], v["fin"], v["rst"], v["psh"] = syn, ack, fin, rst, psh
    v["syn_only"] = ((syn == 1) & (ack == 0)).astype(int)
    v["bytes"] = payload + http_clen
    v["land"] = land
    v["to_server"] = to_server
    v["app_http"] = ((meth.str.len() > 0).to_numpy() | (svc == "http")).astype(int)
    v["app_mqtt"] = ((mq_type > 0) | (svc == "mqtt")).astype(int)
    v["http_method"] = meth.values
    v["mqtt_type"] = mq_type
    v["mqtt_n"] = mq_n
    v["mqtt_len"] = mq_len
    v["arp_op"] = arp_op
    v["tcp_len"] = tcp_len
    v["tcp_seq"] = _num(_col(df, "tcp.seq"))
    v["tcp_ack"] = _num(_col(df, "tcp.ack"))
    v["http_clen"] = http_clen
    v["http_resp"] = (_num(_col(df, "http.response")) > 0).astype(int) if schema == "edge" else \
        (_num(_col(df, "http.response.code")) > 0).astype(int)
    v["icmp_type"] = icmp_type
    v["synack"] = ((syn == 1) & (ack == 1)).astype(int)
    v["mqtt_topic_len"] = _first_and_count(_col(df, "mqtt.topic_len", ""))[0] if schema == "forge" else \
        _num(_col(df, "mqtt.topic_len"))
    return v


# ---------------------------------------------------------------------------- numeric spaces
def _onehot(values: np.ndarray, cats: List[str], prefix: str) -> pd.DataFrame:
    return pd.DataFrame({f"{prefix}_{c}": (values == c).astype(np.float32) for c in cats})


def cbs_from_view(v: pd.DataFrame) -> pd.DataFrame:
    """Canonical Behavioural Space — available for all three datasets."""
    out = [_onehot(v["proto"].to_numpy(), PROTOS, "proto"), _onehot(v["service"].to_numpy(), SERVICES, "svc")]
    base = pd.DataFrame({
        "flag_syn": v["syn"], "flag_ack": v["ack"], "flag_fin": v["fin"], "flag_rst": v["rst"],
        "flag_psh": v["psh"], "syn_only": v["syn_only"],
        "log_bytes": np.log1p(v["bytes"].astype(float).clip(lower=0)),
        "has_payload": (v["bytes"].astype(float) > 0).astype(float),
        "land": v["land"], "app_http": v["app_http"], "app_mqtt": v["app_mqtt"],
    }).reset_index(drop=True).astype(np.float32)
    return pd.concat([o.reset_index(drop=True) for o in out] + [base], axis=1)


def pbs_from_view(v: pd.DataFrame) -> pd.DataFrame:
    """Packet Bridge Space — CBS + tshark fields shared by ForgeIIoT and Edge-IIoTset."""
    cbs = cbs_from_view(v)
    mt = v["mqtt_type"].astype(float).to_numpy()
    meth = v["http_method"].fillna("").astype(str).str.upper().to_numpy()
    extra = pd.DataFrame({
        "log_tcp_len": np.log1p(v["tcp_len"].astype(float).clip(lower=0)),
        "log_tcp_seq": np.log1p(v["tcp_seq"].astype(float).clip(lower=0)),
        "log_tcp_ack": np.log1p(v["tcp_ack"].astype(float).clip(lower=0)),
        "synack": v["synack"], "to_server": v["to_server"],
        "arp_request": (v["arp_op"] == 1).astype(float), "arp_reply": (v["arp_op"] == 2).astype(float),
        "http_get": (meth == "GET").astype(float), "http_post": (meth == "POST").astype(float),
        "http_other_method": ((np.char.str_len(meth.astype(str)) > 0) & ~np.isin(meth, ["GET", "POST"])).astype(float),
        "http_resp": v["http_resp"], "log_http_clen": np.log1p(v["http_clen"].astype(float).clip(lower=0)),
        "log_mqtt_len": np.log1p(v["mqtt_len"].astype(float).clip(lower=0)),
        "mqtt_topic_len": np.log1p(v["mqtt_topic_len"].astype(float).clip(lower=0)),
        "mqtt_multi": (v["mqtt_n"].astype(float) > 1).astype(float),
        **{f"mqtt_{name}": (mt == code).astype(float) for code, name in MQTT_TYPES.items()},
    }).reset_index(drop=True).astype(np.float32)
    return pd.concat([cbs, extra], axis=1)


# ---------------------------------------------------------------------------- native spaces
NSL_CATS = ["protocol_type", "service", "flag"]
FORGE_LEAKY = ["_ts", "_frame_no", "_tcp_stream"]                 # capture-order identifiers
EDGE_LEAKY = ["tcp.checksum", "icmp.checksum", "tcp.ack_raw", "udp.stream", "icmp.transmit_timestamp"]
_FORGE_DROP = ["ip.src_host", "ip.dst_host", "frame.protocols", "tcp.srcport", "tcp.dstport", "udp.srcport",
               "udp.dstport", "mqtt.msgtype", "mqtt.len", "mqtt.topic_len", "mqtt.protoname", "http.request.method",
               "eth.type", "frame.time_delta", "tcp.seq", "tcp.ack"]
_EDGE_DROP = ["ip.src_host", "ip.dst_host", "tcp.srcport", "tcp.dstport", "udp.port", "tcp.seq", "tcp.ack"]


def native_features(df: pd.DataFrame, name: str, categories: dict | None = None,
                    include_leaky: bool = False) -> Tuple[pd.DataFrame, dict]:
    """Richest in-domain feature matrix. Returns (X, categories) so test sets reuse train encodings."""
    meta = {"label_fine", "super", "y", "split", "difficulty", "label_cat"}
    if name == "nsl":
        X = df[[c for c in df.columns if c not in meta]].copy()
        categories = categories or {c: sorted(X[c].fillna("").astype(str).unique()) for c in NSL_CATS}
        for c in NSL_CATS:
            X[c] = pd.Categorical(X[c].fillna("").astype(str), categories=categories[c]).codes.astype(np.float32)
        return X.astype(np.float32).reset_index(drop=True), categories
    v = canonical_view(df, name)
    pbs = pbs_from_view(v)
    skip = set(meta) | {c for c in df.columns if c.startswith("_")}
    drop = _FORGE_DROP if name == "forge" else _EDGE_DROP
    num = df[[c for c in df.columns if c not in skip and c not in drop
              and (include_leaky or c not in EDGE_LEAKY)]]
    num = num.apply(pd.to_numeric, errors="coerce")
    num = num.loc[:, num.notna().any()].fillna(0).clip(-3e38, 3e38).reset_index(drop=True)
    if categories is None:
        keep = [c for c in num.columns if num[c].nunique() > 1]
        categories = {"cols": keep}
    num = num.reindex(columns=categories["cols"], fill_value=0)
    X = pd.concat([num.astype(np.float32), pbs.add_prefix("pbs_")], axis=1)
    if include_leaky and name == "forge":
        for c in FORGE_LEAKY:
            if c in df.columns:
                X[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy()
    if include_leaky and name == "edge":
        for c in ("ip.src_host", "ip.dst_host", "_frame_time"):
            # stable hash encoding -> identical codes in train and test
            X["enc_" + c] = (pd.util.hash_array(df[c].fillna("").astype(str).to_numpy()) % 1_000_003).astype(np.float32)
    return X, categories


def space(df: pd.DataFrame, name: str, kind: str) -> pd.DataFrame:
    """kind in {'cbs','pbs'}"""
    v = canonical_view(df, name)
    return cbs_from_view(v) if kind == "cbs" else pbs_from_view(v)
