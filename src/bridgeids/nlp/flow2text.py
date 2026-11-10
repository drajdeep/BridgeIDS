"""Flow/packet -> natural-language serialisation (Team-2 #3: NLP-based representation).

Language is used as a *dataset-agnostic interlingua*: a connection from NSL-KDD and a packet from an
IIoT capture are both described with the same vocabulary, e.g.

    "tcp traffic to mqtt service port, flags ack push, small payload, mqtt publish message"
    "tcp traffic to http service port, flags syn only half open, no payload"

Canonical phrases come from features.views.canonical_view (identical semantics for every dataset).
Optional context phrases add dataset-specific knowledge (NSL window statistics, packet timing).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.views import MQTT_TYPES, canonical_view

_SVC_TEXT = {"http": "http web service port", "mqtt": "mqtt broker service port", "dns": "dns name service port",
             "ftp": "ftp file transfer port", "smtp": "mail service port", "remote_shell": "remote shell login port",
             "modbus": "modbus industrial control port", "icmp_echo": "icmp echo", "wellknown_other":
             "other well known port", "registered": "registered high port", "ephemeral": "ephemeral private port",
             "none": "no port"}


def _bytes_phrase(b: float) -> str:
    if b <= 0:
        return "no payload"
    if b < 64:
        return "tiny payload"
    if b < 512:
        return "small payload"
    if b < 4096:
        return "medium payload"
    return "large payload"


def canonical_sentences(v: pd.DataFrame) -> np.ndarray:
    proto = v["proto"].fillna("").astype(str).to_numpy()
    svc = v["service"].fillna("").astype(str).to_numpy()
    syn, ack, fin, rst, psh = (v[c].astype(int).to_numpy() for c in ("syn", "ack", "fin", "rst", "psh"))
    synonly = v["syn_only"].astype(int).to_numpy()
    b = v["bytes"].astype(float).to_numpy()
    land = v["land"].astype(int).to_numpy()
    http = v["app_http"].astype(int).to_numpy()
    meth = v["http_method"].fillna("").astype(str).str.upper().to_numpy()
    mq = v["mqtt_type"].astype(float).to_numpy()
    arp = v["arp_op"].astype(float).to_numpy()
    out = []
    for i in range(len(v)):
        parts = []
        if proto[i] == "arp":
            parts.append("arp " + ("request" if arp[i] == 1 else "reply" if arp[i] == 2 else "frame"))
        else:
            parts.append(f"{proto[i]} traffic to {_SVC_TEXT.get(svc[i], svc[i])}")
        if proto[i] == "tcp":
            fl = [n for n, f in (("syn", syn[i]), ("ack", ack[i]), ("fin", fin[i]), ("reset", rst[i]),
                                 ("push", psh[i])) if f]
            parts.append("flags " + (" ".join(fl) if fl else "none"))
            if synonly[i]:
                parts.append("half open connection attempt")
            if rst[i]:
                parts.append("connection reset or rejected")
        parts.append(_bytes_phrase(b[i]))
        if land[i]:
            parts.append("source equals destination land")
        if http[i] and meth[i] not in ("", "NAN"):
            parts.append(f"http {meth[i].lower()} request")
        elif http[i]:
            parts.append("http exchange")
        if mq[i] > 0:
            parts.append(f"mqtt {MQTT_TYPES.get(int(mq[i]), 'control')} message")
        out.append(", ".join(parts))
    return np.array(out, dtype=object)


def context_sentences(df: pd.DataFrame, name: str) -> np.ndarray:
    """Dataset-specific extra phrases (used for in-domain NLP augmentation only)."""
    n = len(df)
    ctx = [[] for _ in range(n)]
    if name == "nsl":
        def add(mask, text):
            for i in np.where(mask)[0]:
                ctx[i].append(text)
        add(df["serror_rate"].to_numpy() > 0.5, "high syn error rate")
        add(df["rerror_rate"].to_numpy() > 0.5, "high rejection rate")
        add(df["count"].to_numpy() > 100, "many recent connections to the same host")
        add(df["same_srv_rate"].to_numpy() < 0.2, "many different services")
        add(df["num_failed_logins"].to_numpy() > 0, "failed login attempts")
        add(df["root_shell"].to_numpy() > 0, "root shell obtained")
        add(df["logged_in"].to_numpy() == 1, "logged in session")
        add(df["duration"].to_numpy() > 60, "long connection")
    else:
        td = pd.to_numeric(df.get("tcp.time_delta", pd.Series(0, index=df.index)), errors="coerce").fillna(0).to_numpy()
        for i in np.where((td > 0) & (td < 1e-4))[0]:
            ctx[i].append("burst of back to back packets")
        for i in np.where(td > 1.0)[0]:
            ctx[i].append("slow idle stream")
    return np.array([", ".join(c) for c in ctx], dtype=object)


def records_to_text(df: pd.DataFrame, name: str, with_context: bool = False) -> np.ndarray:
    s = canonical_sentences(canonical_view(df, name))
    if with_context:
        c = context_sentences(df, name)
        s = np.array([a if not b else f"{a}, {b}" for a, b in zip(s, c)], dtype=object)
    return s
