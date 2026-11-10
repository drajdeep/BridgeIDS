"""Dataset loaders + one-time preparation into compact parquet files.

* NSL-KDD   — KDDTrain+/KDDTest+ (.txt has the fine attack name; .arff only normal/anomaly).
* ForgeIIoT — rebuilt from the *Unprocessed* per-attack captures with a stratified random sample
              per capture file (falls back to the Preprocessed CSV when raw captures are absent).
* Edge-IIoT — ML-EdgeIIoT-dataset.csv.

Meta columns used everywhere:  label_fine | super | y  (y = 1 attack, 0 normal) | split (NSL only)
Columns starting with "_" are bookkeeping only (timestamps / frame numbers) and are NEVER features.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..utils import get_logger, load_config
from .taxonomy import EDGE_TO_SUPER, FORGE_TO_SUPER, NSL_CATEGORY, nsl_super

log = get_logger("data")

NSL_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]

# tshark fields kept from the Forge raw captures (superset of what the feature extractors need)
FORGE_KEEP = [
    "ip.src_host", "ip.dst_host", "ip.proto", "ip.ttl", "ip.len", "ip.flags.df", "ip.hdr_len", "ip.version",
    "eth.type", "arp.opcode", "icmp.type", "icmp.code", "icmpv6.type",
    "http.request.method", "http.content_length", "http.response.code",
    "tcp.dstport", "tcp.srcport", "tcp.flags.ack", "tcp.flags.fin", "tcp.flags.push", "tcp.flags.reset",
    "tcp.flags.syn", "tcp.len", "tcp.seq", "tcp.ack", "tcp.hdr_len", "tcp.window_size", "tcp.time_delta",
    "tcp.connection.syn", "tcp.connection.synack", "tcp.connection.rst", "tcp.connection.fin",
    "tcp.analysis.retransmission", "tcp.analysis.duplicate_ack", "tcp.stream",
    "udp.dstport", "udp.srcport", "udp.length", "udp.time_delta",
    "mqtt.msgtype", "mqtt.len", "mqtt.topic_len", "mqtt.conflag.cleansess", "mqtt.protoname", "mqtt.retain",
    "mqtt.hdr_reserved", "mqtt.conack.val",
    "frame.len", "frame.protocols", "frame.time_delta", "frame.time_epoch", "frame.number",
    "attack_label", "attack_type",
]

EDGE_KEEP = [
    "frame.time", "ip.src_host", "ip.dst_host", "arp.opcode", "arp.hw.size", "icmp.checksum", "icmp.seq_le",
    "icmp.transmit_timestamp", "http.content_length", "http.request.method", "http.referer",
    "http.request.version", "http.response", "http.request.uri.query", "http.file_data", "http.request.full_uri",
    "tcp.ack", "tcp.ack_raw", "tcp.checksum", "tcp.connection.fin", "tcp.connection.rst", "tcp.connection.syn",
    "tcp.connection.synack", "tcp.dstport", "tcp.flags", "tcp.flags.ack", "tcp.len", "tcp.seq", "tcp.srcport",
    "tcp.options", "tcp.payload", "udp.port", "udp.stream", "udp.time_delta", "dns.qry.name", "dns.qry.name.len",
    "dns.qry.qu", "dns.qry.type", "dns.retransmission", "dns.retransmit_request", "dns.retransmit_request_in",
    "mqtt.conack.flags", "mqtt.conflag.cleansess", "mqtt.conflags", "mqtt.hdrflags", "mqtt.len",
    "mqtt.msg_decoded_as", "mqtt.msg", "mqtt.msgtype", "mqtt.proto_len", "mqtt.protoname", "mqtt.topic",
    "mqtt.topic_len", "mqtt.ver", "mbtcp.len", "mbtcp.trans_id", "mbtcp.unit_id", "Attack_label", "Attack_type",
]
EDGE_STRING_COLS = ["frame.time", "ip.src_host", "ip.dst_host", "arp.dst.proto_ipv4", "arp.src.proto_ipv4",
                    "http.request.method", "http.referer", "http.request.version", "http.request.uri.query",
                    "http.file_data", "http.request.full_uri", "tcp.options", "tcp.payload", "dns.qry.name",
                    "dns.qry.name.len", "mqtt.conack.flags", "mqtt.msg", "mqtt.protoname", "mqtt.topic",
                    "Attack_type"]


# ============================================================================ NSL-KDD
def _read_nsl_txt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    df = df.iloc[:, :43]
    df.columns = NSL_COLUMNS + ["label_fine", "difficulty"]
    return df


def _read_nsl_arff(path: Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    start = next(i for i, l in enumerate(lines) if l.strip().lower() == "@data") + 1
    df = pd.read_csv(path, skiprows=start, header=None)
    df.columns = NSL_COLUMNS + ["label_fine"]
    df["difficulty"] = np.nan
    return df


def prepare_nsl(cfg=None) -> pd.DataFrame:
    cfg = cfg or load_config()
    raw = Path(cfg["paths"]["raw_nsl"])
    parts = []
    for split, stem in (("train", "KDDTrain+"), ("test", "KDDTest+")):
        txt, arff = raw / f"{stem}.txt", raw / f"{stem}.arff"
        if txt.exists():
            d = _read_nsl_txt(txt)
        elif arff.exists():
            log.warning(f"{stem}.txt not found – using .arff (binary labels only; taxonomy experiments degrade)")
            d = _read_nsl_arff(arff)
        else:
            raise FileNotFoundError(f"Put {stem}.txt (or .arff) into {raw}")
        d["split"] = split
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df["label_fine"] = df["label_fine"].astype(str).str.strip().str.lower()
    df.loc[df.label_fine == "anomaly", "label_fine"] = "unknown_attack"
    df["label_cat"] = df["label_fine"].map(lambda a: NSL_CATEGORY.get(a, "Normal" if a == "normal" else "R2L"))
    df["super"] = df["label_fine"].map(nsl_super)
    df["y"] = (df["label_fine"] != "normal").astype(int)
    for c in ("protocol_type", "service", "flag"):
        df[c] = df[c].astype(str).str.strip()
    return df


# ============================================================================ ForgeIIoT
def _forge_dirs(raw: Path):
    unproc, pre = None, None
    for dirpath, dirnames, filenames in os.walk(raw, followlinks=True):  # follows symlinked folders too
        if Path(dirpath).name == "Unprocessed Dataset" and any(f.endswith(".csv") for f in filenames):
            unproc = unproc or Path(dirpath)
        if "Forge_IIOT_Pro.csv" in filenames:
            pre = pre or Path(dirpath) / "Forge_IIOT_Pro.csv"
    return unproc, pre


def _count_lines(path: Path) -> int:
    with open(path, "rb") as f:
        return sum(buf.count(b"\n") for buf in iter(lambda: f.read(1 << 22), b"")) - 1


def prepare_forge(cfg=None) -> pd.DataFrame:
    cfg = cfg or load_config()
    raw = Path(cfg["paths"]["raw_forge"])
    seed = cfg["seed"]
    per_class = cfg["prepare"]["forge_rows_per_class"]
    n_normal = cfg["prepare"]["forge_normal_rows"]
    unproc, pre = _forge_dirs(raw)
    rng = np.random.default_rng(seed)
    parts = []
    if unproc is not None:
        files = sorted(glob.glob(str(unproc / "*.csv")))
        log.info(f"Forge: sampling {len(files)} raw capture files from {unproc}")
        for fp in files:
            name = Path(fp).stem
            n_rows = _count_lines(Path(fp))
            want = n_normal if name == "normal" else per_class
            keep_p = min(1.0, want / max(n_rows, 1))
            header = pd.read_csv(fp, nrows=0).columns
            usecols = [c for c in FORGE_KEEP if c in header]
            chunks = []
            for ch in pd.read_csv(fp, usecols=usecols, chunksize=200_000, low_memory=False, dtype=str):
                if keep_p < 1.0:
                    ch = ch[rng.random(len(ch)) < keep_p]
                chunks.append(ch)
            d = pd.concat(chunks, ignore_index=True)
            if len(d) > want:
                d = d.sample(want, random_state=seed)
            d["attack_type"] = name
            parts.append(d)
            log.info(f"  {name:12s} {n_rows:>9,} rows -> kept {len(d):,}")
        df = pd.concat(parts, ignore_index=True)
    elif pre is not None:
        log.warning("Forge raw captures not found – using the Preprocessed CSV (fewer packet fields).")
        chunks = []
        for ch in pd.read_csv(pre, chunksize=200_000, low_memory=False, dtype=str):
            chunks.append(ch.sample(frac=min(1.0, 200_000 / 3_300_000), random_state=seed))
        df = pd.concat(chunks, ignore_index=True)
    else:
        raise FileNotFoundError(f"Put the Forge_IIOT_Pro_Dataset folder into {raw}")

    num_cols = [c for c in df.columns if c not in ("ip.src_host", "ip.dst_host", "frame.protocols", "attack_type",
                                                   "mqtt.msgtype", "mqtt.len", "mqtt.topic_len", "mqtt.protoname",
                                                   "http.request.method", "eth.type")]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["label_fine"] = df["attack_type"].astype(str)
    df["super"] = df["label_fine"].map(FORGE_TO_SUPER).fillna("DOS")
    df["y"] = (df["label_fine"] != "normal").astype(int)
    df = df.rename(columns={"frame.time_epoch": "_ts", "frame.number": "_frame_no", "tcp.stream": "_tcp_stream"})
    df = df.drop(columns=["attack_label", "attack_type"], errors="ignore")
    return df.reset_index(drop=True)


# ============================================================================ Edge-IIoT
def _find_edge_csv(raw: Path) -> Path:
    cands = sorted([Path(d) / f for d, _, fs in os.walk(raw, followlinks=True) for f in fs if f.lower().endswith(".csv")],
                   key=lambda p: os.path.getsize(p), reverse=True)
    if not cands:
        raise FileNotFoundError(f"Put ML-EdgeIIoT-dataset.csv into {raw}")
    return cands[0]


def prepare_edge(cfg=None) -> pd.DataFrame:
    cfg = cfg or load_config()
    path = _find_edge_csv(Path(cfg["paths"]["raw_edge"]))
    log.info(f"Edge-IIoT: reading {path}")
    df = pd.read_csv(path, low_memory=False, dtype=str)
    df = df[[c for c in EDGE_KEEP if c in df.columns]]
    for c in df.columns:
        if c not in EDGE_STRING_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["label_fine"] = df["Attack_type"].astype(str).str.strip()
    df["super"] = df["label_fine"].map(EDGE_TO_SUPER).fillna("NORMAL")
    df["y"] = pd.to_numeric(df["Attack_label"], errors="coerce").fillna(0).astype(int)
    df = df.drop(columns=["Attack_label", "Attack_type"])
    df = df.rename(columns={"frame.time": "_frame_time"})
    mx = cfg["prepare"].get("edge_max_rows")
    if mx and len(df) > mx:
        df = df.sample(mx, random_state=cfg["seed"])
    return df.reset_index(drop=True)


# ============================================================================ public API
_PREP = {"nsl": prepare_nsl, "forge": prepare_forge, "edge": prepare_edge}
_CACHE = {}


def processed_path(name: str, cfg=None) -> Path:
    cfg = cfg or load_config()
    return Path(cfg["paths"]["processed"]) / f"{name}.parquet"


def prepare_all(cfg=None, force: bool = False) -> None:
    cfg = cfg or load_config()
    for name, fn in _PREP.items():
        p = processed_path(name, cfg)
        if p.exists() and not force:
            log.info(f"{name}: {p.name} already exists (use --force to rebuild)")
            continue
        df = fn(cfg)
        for c in df.columns:  # parquet needs homogeneous object columns
            if df[c].dtype == object:
                df[c] = df[c].astype("string")
        df.to_parquet(p, index=False)
        log.info(f"{name}: wrote {p} {df.shape}")


def load(name: str, cfg=None) -> pd.DataFrame:
    """Load a prepared dataset ('nsl' | 'forge' | 'edge'); prepares it on first use."""
    if name in _CACHE:
        return _CACHE[name].copy()
    cfg = cfg or load_config()
    p = processed_path(name, cfg)
    if not p.exists():
        df = _PREP[name](cfg)
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype("string")
        df.to_parquet(p, index=False)
    df = pd.read_parquet(p)
    for c in df.columns:
        if str(df[c].dtype) == "string":
            df[c] = df[c].astype(object)
    _CACHE[name] = df
    return df.copy()


DATASET_NAMES = {"nsl": "NSL-KDD", "forge": "ForgeIIoT", "edge": "Edge-IIoTset"}
