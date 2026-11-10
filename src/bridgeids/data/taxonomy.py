"""Unified cross-dataset attack taxonomy (Team-2 deliverable: Attack Taxonomy Mapping).

Every fine-grained label of NSL-KDD, ForgeIIoT and Edge-IIoTset is mapped to one of seven
super-classes so that models trained on one dataset can be scored on another:

    NORMAL | DOS | RECON | MITM_SPOOF | UNAUTH_ACCESS | INJECTION | MALWARE_PRIVESC

Each super-class also has a natural-language description and a binary semantic attribute
signature; both are used by the zero-shot detectors (attribute-based DAP and text-embedding ZSL).
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

SUPER_CLASSES: List[str] = ["NORMAL", "DOS", "RECON", "MITM_SPOOF", "UNAUTH_ACCESS", "INJECTION", "MALWARE_PRIVESC"]

# ---------------------------------------------------------------- NSL-KDD (legacy)
NSL_CATEGORY = {  # standard NSL-KDD attack -> legacy category
    **{a: "DoS" for a in ["back", "land", "neptune", "pod", "smurf", "teardrop", "apache2", "udpstorm",
                          "processtable", "worm", "mailbomb"]},
    **{a: "Probe" for a in ["satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"]},
    **{a: "R2L" for a in ["guess_passwd", "ftp_write", "imap", "phf", "multihop", "warezmaster", "warezclient",
                          "spy", "xlock", "xsnoop", "snmpguess", "snmpgetattack", "httptunnel", "sendmail", "named"]},
    **{a: "U2R" for a in ["buffer_overflow", "loadmodule", "rootkit", "perl", "sqlattack", "xterm", "ps"]},
    "normal": "Normal",
}
NSL_TO_SUPER = {"Normal": "NORMAL", "DoS": "DOS", "Probe": "RECON", "R2L": "UNAUTH_ACCESS", "U2R": "MALWARE_PRIVESC"}
NSL_FINE_OVERRIDE = {"sqlattack": "INJECTION"}  # the one U2R attack that is really an injection

# ---------------------------------------------------------------- ForgeIIoT (bridge)
FORGE_TO_SUPER: Dict[str, str] = {
    "normal": "NORMAL",
    "ddos_tcp": "DOS", "ddos_mp": "DOS", "dos_ipv4": "DOS", "dos_icmp": "DOS", "dos_udp": "DOS",
    "dos_land": "DOS", "dos_deauth": "DOS",
    "port_scan": "RECON", "vuln_scan": "RECON", "os_finger": "RECON",
    "mitm_arp": "MITM_SPOOF", "mitm_dhcp": "MITM_SPOOF", "mitm_ndp": "MITM_SPOOF", "sybil": "MITM_SPOOF",
    "wild_card": "UNAUTH_ACCESS",          # MQTT wildcard-topic subscription abuse
    "malw_back": "MALWARE_PRIVESC",        # malware back-door (reverse shell on 4444)
}
FORGE_DESCRIPTION = {
    "ddos_tcp": "distributed TCP SYN flood", "ddos_mp": "distributed MQTT publish flood",
    "dos_ipv4": "raw IPv4 packet flood", "dos_icmp": "ICMP echo flood", "dos_udp": "UDP flood",
    "dos_land": "LAND attack (source = destination)", "dos_deauth": "Wi-Fi de-authentication DoS",
    "port_scan": "TCP port scan", "vuln_scan": "vulnerability scan", "os_finger": "OS fingerprinting",
    "mitm_arp": "ARP-spoofing man-in-the-middle", "mitm_dhcp": "rogue DHCP man-in-the-middle",
    "mitm_ndp": "IPv6 NDP-spoofing man-in-the-middle", "sybil": "Sybil (forged MQTT client identities)",
    "wild_card": "MQTT wildcard (#) topic subscription abuse", "malw_back": "malware back-door / reverse shell",
    "normal": "benign MQTT/IIoT traffic",
}

# ---------------------------------------------------------------- Edge-IIoTset (modern)
EDGE_TO_SUPER: Dict[str, str] = {
    "Normal": "NORMAL",
    "DDoS_UDP": "DOS", "DDoS_ICMP": "DOS", "DDoS_TCP": "DOS", "DDoS_HTTP": "DOS",
    "Port_Scanning": "RECON", "Vulnerability_scanner": "RECON", "Fingerprinting": "RECON",
    "MITM": "MITM_SPOOF",
    "Password": "UNAUTH_ACCESS",
    "SQL_injection": "INJECTION", "XSS": "INJECTION", "Uploading": "INJECTION",
    "Backdoor": "MALWARE_PRIVESC", "Ransomware": "MALWARE_PRIVESC",
}

# ---------------------------------------------------------------- semantics for zero-shot
SUPER_DESCRIPTIONS = {
    "NORMAL": "benign industrial IoT traffic, regular sensor telemetry, MQTT publish and acknowledgements, "
              "complete TCP handshakes with moderate payload",
    "DOS": "denial of service flood, very high packet rate of SYN, UDP, ICMP or HTTP requests, half open "
           "connections, small or empty payloads exhausting the target",
    "RECON": "reconnaissance scanning, probing many ports and hosts, SYN packets without payload, connection "
             "rejected or reset, operating system fingerprinting and vulnerability scanning",
    "MITM_SPOOF": "man in the middle and spoofing, forged ARP replies, rogue DHCP or IPv6 neighbour "
                  "advertisements, forged identities intercepting traffic",
    "UNAUTH_ACCESS": "unauthorized access, password guessing and brute force logins, credential abuse, "
                     "subscribing to all MQTT topics with wildcards, remote to local intrusion",
    "INJECTION": "web injection attacks, SQL injection, cross site scripting, malicious file upload through HTTP "
                 "requests with crafted query strings and payload",
    "MALWARE_PRIVESC": "malware, backdoor and ransomware, reverse shell on unusual port, privilege escalation to "
                       "root, persistent command and control sessions with payload",
}

ATTRIBUTES = ["flooding", "half_open", "scanning", "spoofing", "credential", "web_http", "payload_heavy",
              "unusual_port", "mqtt_related", "icmp_or_udp"]
SUPER_ATTRIBUTES = {   # class-attribute matrix (domain-expert prior, used for DAP zero-shot)
    "NORMAL":          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    "DOS":             [1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    "RECON":           [0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
    "MITM_SPOOF":      [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    "UNAUTH_ACCESS":   [0, 0, 0, 0, 1, 1, 0, 0, 1, 0],
    "INJECTION":       [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    "MALWARE_PRIVESC": [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
}
# fine-grained Edge attributes (more specific than the super-class prior)
EDGE_FINE_ATTRIBUTES = {
    "Normal":                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    "DDoS_UDP":              [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    "DDoS_ICMP":             [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    "DDoS_TCP":              [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "DDoS_HTTP":             [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    "Port_Scanning":         [0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
    "Vulnerability_scanner": [0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    "Fingerprinting":        [0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
    "MITM":                  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    "Password":              [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    "SQL_injection":         [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    "XSS":                   [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    "Uploading":             [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    "Backdoor":              [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    "Ransomware":            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
}
EDGE_FINE_DESCRIPTIONS = {
    "Normal": "benign sensor telemetry published over MQTT to the broker",
    "DDoS_UDP": "UDP flood denial of service with many datagrams",
    "DDoS_ICMP": "ICMP echo ping flood denial of service",
    "DDoS_TCP": "TCP SYN flood with half open connections",
    "DDoS_HTTP": "HTTP GET request flood against the web server",
    "Port_Scanning": "port scanning with SYN probes to many ports, connections reset",
    "Vulnerability_scanner": "vulnerability scanner sending many HTTP probes for known weaknesses",
    "Fingerprinting": "operating system fingerprinting with crafted ICMP and TCP probes",
    "MITM": "ARP spoofing man in the middle interception",
    "Password": "password brute force guessing of login credentials over HTTP",
    "SQL_injection": "SQL injection with crafted HTTP query strings against the database",
    "XSS": "cross site scripting injecting script payload in HTTP requests",
    "Uploading": "malicious file upload over HTTP POST with large payload",
    "Backdoor": "backdoor command and control session on an unusual port",
    "Ransomware": "ransomware sending encrypted payload over a command channel on an unusual port",
}


def nsl_super(fine: str) -> str:
    fine = str(fine).strip().lower()
    if fine in NSL_FINE_OVERRIDE:
        return NSL_FINE_OVERRIDE[fine]
    return NSL_TO_SUPER.get(NSL_CATEGORY.get(fine, "R2L" if fine != "normal" else "Normal"), "UNAUTH_ACCESS")


def taxonomy_table() -> pd.DataFrame:
    """Long-format mapping table (dataset, fine label, legacy category, super-class)."""
    rows = []
    for a, cat in NSL_CATEGORY.items():
        rows.append(("NSL-KDD", a, cat, nsl_super(a), ""))
    for a, s in FORGE_TO_SUPER.items():
        rows.append(("ForgeIIoT", a, "", s, FORGE_DESCRIPTION.get(a, "")))
    for a, s in EDGE_TO_SUPER.items():
        rows.append(("Edge-IIoTset", a, "", s, EDGE_FINE_DESCRIPTIONS.get(a, "")))
    return pd.DataFrame(rows, columns=["dataset", "label", "legacy_category", "super_class", "description"])


def relationship_matrix() -> pd.DataFrame:
    """Plan's 4-row matrix (NSL category <-> Forge <-> Edge) generated from the mapping."""
    t = taxonomy_table()
    out = []
    for sc in SUPER_CLASSES:
        sub = t[t.super_class == sc]
        out.append({
            "super_class": sc,
            "NSL-KDD": ", ".join(sorted(set(sub[sub.dataset == "NSL-KDD"].legacy_category))) or "—",
            "NSL-KDD examples": ", ".join(sorted(sub[sub.dataset == "NSL-KDD"].label)[:6]) or "—",
            "ForgeIIoT": ", ".join(sorted(sub[sub.dataset == "ForgeIIoT"].label)) or "—",
            "Edge-IIoTset": ", ".join(sorted(sub[sub.dataset == "Edge-IIoTset"].label)) or "—",
        })
    return pd.DataFrame(out)
