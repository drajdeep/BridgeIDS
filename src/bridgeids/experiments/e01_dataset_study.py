"""E01 — Dataset study, feature-family comparison, attack taxonomy (Team-1 #1, Team-2 #2)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.loaders import DATASET_NAMES, load
from ..data.taxonomy import SUPER_CLASSES, relationship_matrix, taxonomy_table
from ..features.views import canonical_view, cbs_from_view, native_features, pbs_from_view
from ..reporting.plots import DS_COLORS, heatmap, plt, savefig
from ..utils import get_logger, save_table

log = get_logger("e01")

FAMILIES = {  # feature family -> how each dataset exposes it
    "Connection duration": ("duration", "—", "—"),
    "Byte volume": ("src_bytes, dst_bytes", "frame.len, ip.len, tcp.len, udp.length", "tcp.len, http.content_length"),
    "Protocol": ("protocol_type", "ip.proto, frame.protocols", "derived from tcp/udp/icmp/arp fields"),
    "Service / port": ("service (70 names)", "tcp/udp src+dst ports", "tcp.srcport, tcp.dstport, udp.port"),
    "TCP flag semantics": ("flag (SF,S0,REJ…)", "tcp.flags.syn/ack/fin/reset/push", "tcp.flags (bitmask), tcp.connection.*"),
    "Time-window statistics": ("count, srv_count, *_rate (19 features)", "— (computable from timestamps)", "—"),
    "Content / host features": ("hot, num_failed_logins, root_shell…", "—", "—"),
    "ARP / L2": ("—", "arp.opcode, eth.*", "arp.opcode, arp.hw.size"),
    "ICMP": ("service ecr_i/eco_i", "icmp.type, icmp.code, icmpv6.type", "icmp.checksum, icmp.seq_le"),
    "HTTP": ("service http", "http.request.method, http.content_length", "http.request.method, uri, referer"),
    "MQTT": ("—", "mqtt.msgtype, mqtt.len, mqtt.topic_len…", "mqtt.msgtype, mqtt.topic, mqtt.protoname…"),
    "DNS": ("service domain_u", "udp port 53/5353", "dns.qry.*"),
    "Modbus/TCP": ("—", "—", "mbtcp.* (constant in ML subset)"),
    "Granularity": ("connection record", "packet", "packet"),
}


def run(cfg):
    stats, dist = [], []
    for name in ("nsl", "forge", "edge"):
        df = load(name)
        Xn, _ = native_features(df.sample(min(len(df), 20000), random_state=0), name)
        v = canonical_view(df, name)
        stats.append({
            "dataset": DATASET_NAMES[name], "records (used)": len(df), "native features (leak-free)": Xn.shape[1],
            "CBS features": cbs_from_view(v.head(5)).shape[1],
            "PBS features": pbs_from_view(v.head(5)).shape[1] if name != "nsl" else np.nan,
            "fine labels": df["label_fine"].nunique(), "super-classes": df["super"].nunique(),
            "attack %": round(100 * df["y"].mean(), 2),
            "TCP %": round(100 * (v.proto == "tcp").mean(), 1), "UDP %": round(100 * (v.proto == "udp").mean(), 1),
            "ICMP %": round(100 * (v.proto == "icmp").mean(), 1), "ARP %": round(100 * (v.proto == "arp").mean(), 1),
            "MQTT %": round(100 * v["app_mqtt"].mean(), 1), "HTTP %": round(100 * v["app_http"].mean(), 1),
        })
        for (fine, sup), n in df.groupby(["label_fine", "super"]).size().items():
            dist.append({"dataset": DATASET_NAMES[name], "label_fine": fine, "super": sup, "count": int(n)})
    stats = pd.DataFrame(stats)
    save_table(stats, "e01_dataset_statistics")
    dist = pd.DataFrame(dist)
    save_table(dist, "e01_label_distribution")

    fam = pd.DataFrame([(k, *v) for k, v in FAMILIES.items()], columns=["feature family", "NSL-KDD", "ForgeIIoT",
                                                                         "Edge-IIoTset"])
    save_table(fam, "e01_feature_family_comparison")
    save_table(taxonomy_table(), "e01_attack_taxonomy")
    save_table(relationship_matrix(), "e01_attack_relationship_matrix")

    # super-class distribution figure
    piv = dist.pivot_table(index="super", columns="dataset", values="count", aggfunc="sum").reindex(SUPER_CLASSES)
    share = (piv / piv.sum()).fillna(0) * 100
    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.27
    for i, c in enumerate(["NSL-KDD", "ForgeIIoT", "Edge-IIoTset"]):
        ax.bar(np.arange(len(share)) + (i - 1) * w, share[c].values, w, label=c, color=DS_COLORS[c])
    ax.set_xticks(range(len(share)), share.index, rotation=20, ha="right")
    ax.set_ylabel("% of records")
    ax.set_title("Unified super-class distribution per dataset (attack drift at label level)")
    ax.legend()
    savefig(fig, "e01_superclass_distribution")
    heatmap(share.round(1), "Super-class share (%)", "e01_superclass_share_heatmap", fmt="{:.1f}", cmap="Oranges")
    log.info("\n" + stats.to_string(index=False))
    return {"stats": stats}
