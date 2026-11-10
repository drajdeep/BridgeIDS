"""Dynamic detection engine — turns the static models into an adaptive, explainable stream detector.

Per micro-batch of packet/connection records:
  1. schema auto-detection (NSL connection record vs tshark packet from ForgeIIoT / Edge-IIoT)
  2. featurisation into PBS (packets) or native NSL features (connections)
  3. static hybrid score = max(supervised P(attack), gated benign-ECDF anomaly percentile)
  4. online challenger: a sliding-window LightGBM re-trained every few hundred analyst labels on recent
     feedback + a replay sample of the original training data (continual-learning style), combined with
     the static champion by Hedge (exponential-weights) on prequential log-loss
  5. drift detection: ADWIN on the feedback error stream + windowed PSI of the feature stream; on drift
     the anomaly reference is re-calibrated with recent confirmed-benign traffic and the online learner's
     Hedge weight is boosted
  6. SHAP explanation + narrative for alerts, family (super-class) prediction
  7. rolling metrics (if ground truth is present), protocol / MQTT analytics, alert log, drift log
"""
from __future__ import annotations

import collections
import math
import threading
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..drift import StreamPSI
from ..features.views import canonical_view, detect_schema, native_features, pbs_from_view
from ..models.zoo import proba_pos
from ..utils import get_logger
from ..xai.explain import narrative, shap_values_binary

log = get_logger("engine")

try:
    from river import drift as river_drift
    HAVE_RIVER = True
except Exception:  # pragma: no cover
    HAVE_RIVER = False


class _ADWINFallback:
    """Tiny two-window mean-shift detector used if river is not installed."""

    def __init__(self, delta=0.002, w=500):
        self.buf = collections.deque(maxlen=2 * w)
        self.w, self.delta = w, delta
        self.drift_detected = False

    def update(self, x):
        self.buf.append(float(x))
        self.drift_detected = False
        if len(self.buf) == self.buf.maxlen:
            a = np.array(self.buf)
            m1, m2 = a[: self.w].mean(), a[self.w:].mean()
            eps = math.sqrt(math.log(4 / self.delta) / (2 * self.w))
            if abs(m1 - m2) > 2 * eps:
                self.drift_detected = True
                self.buf.clear()


class DetectionEngine:
    def __init__(self, artifacts: dict, cfg: dict, feedback_rate: float = 0.05, feedback_delay: int = 200,
                 adaptive: bool = True, explain: bool = True, seed: int = 0, use_anomaly: bool = True):
        self.a = artifacts
        self.cfg = cfg
        sc = cfg["stream"]
        self.threshold = sc["alert_threshold"]
        self.top_k = sc["explain_top_k"]
        self.window = sc["window"]
        self.feedback_rate, self.feedback_delay = feedback_rate, feedback_delay
        self.adaptive, self.explain, self.use_anomaly = adaptive, explain, use_anomaly
        self.rng = np.random.default_rng(seed)
        self.lock = threading.Lock()
        self._explainer = None
        self.psi = StreamPSI(artifacts["psi_ref"])
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        self.t0 = time.time()
        self.n = 0
        self.n_alerts = 0
        self.recent = collections.deque(maxlen=self.window)          # (pred, truth, score, latency)
        self.timeline = collections.deque(maxlen=600)                  # per-batch snapshots
        self.alerts = collections.deque(maxlen=500)
        self.drift_events: List[dict] = []
        self.family_counts = collections.Counter()
        self.truth_counts = collections.Counter()
        self.proto_counts = collections.Counter()
        self.service_counts = collections.Counter()
        self.mqtt_counts = collections.Counter()
        self.dataset_counts = collections.Counter()
        self.pending_feedback = collections.deque()
        self.psi_buf = collections.deque(maxlen=1000)
        self.benign_buf = collections.deque(maxlen=4000)
        self.online = None                       # sliding-window LightGBM challenger
        self.fb_X = collections.deque(maxlen=4000)
        self.fb_y = collections.deque(maxlen=4000)
        self.fb_since_fit = 0
        self.online_seen = 0
        self.err_detector = river_drift.ADWIN(delta=0.002) if HAVE_RIVER else _ADWINFallback()
        self.loss_static, self.loss_online = 0.3, 0.7                # EWMA log-loss (Hedge)
        self.w_online = 0.0
        self.last_psi = 0.0
        self.events_for_sink: List[dict] = []
        self.anomaly_ref_version = 0

    # ------------------------------------------------------------------ featurisation
    def _featurize(self, df: pd.DataFrame, schema: str):
        if schema == "nsl":
            X, _ = native_features(df, "nsl", categories=self.a["nsl_cats"])
            X = X.reindex(columns=self.a["nsl_cols"], fill_value=0)
            v = canonical_view(df, "nsl")
            return X.values.astype(float), v
        v = canonical_view(df, schema)
        X = pbs_from_view(v).reindex(columns=self.a["pbs_cols"], fill_value=0)
        return X.values.astype(float), v

    # ------------------------------------------------------------------ main entry
    def process(self, records: List[dict]) -> List[dict]:
        if not records:
            return []
        t_start = time.time()
        with self.lock:
            out = []
            df_all = pd.DataFrame([r.get("fields", r) for r in records])
            groups: Dict[str, list] = collections.defaultdict(list)
            for i, r in enumerate(records):
                f = r.get("fields", r)
                schema = r.get("schema") or ("nsl" if "protocol_type" in f else
                                             "forge" if ("tcp.flags.syn" in f or "frame.protocols" in f or
                                                         "udp.dstport" in f or "icmp.type" in f) else "edge")
                groups[schema].append(i)
            results = [None] * len(records)
            for schema, idx in groups.items():
                df = df_all.iloc[idx].reset_index(drop=True)
                df = df.dropna(axis=1, how="all")
                X, v = self._featurize(df, schema)
                res = self._score(X, v, schema, [records[i] for i in idx])
                for j, i in enumerate(idx):
                    results[i] = res[j]
            lat = (time.time() - t_start) * 1000 / len(records)
            for r, res in zip(records, results):
                res["latency_ms"] = lat
                out.append(res)
            self._update_stats(out)
            return out

    def _score(self, X, v, schema, recs):
        if schema == "nsl":
            p_sup = proba_pos(self.a["nsl_binary"], X)
            an = self.a["nsl_anomaly"].percentile(X)
            fam = self.a["nsl_family"].predict(X)
            model = self.a["nsl_binary"]
            cols = self.a["nsl_cols"]
        else:
            p_sup = proba_pos(self.a["pbs_binary"], X)
            an = self.a["anomaly"].percentile(X)
            fam = self.a["pbs_family"].predict(X)
            model = self.a["pbs_binary"]
            cols = self.a["pbs_cols"]
            self.psi_buf.extend(X)              # sliding window of the most recent packets
        gate = 0.995
        a_term = np.where(an >= gate, 0.5 + 0.5 * (an - gate) / (1 - gate), 0.0) if self.use_anomaly else 0.0 * an
        s_static = np.maximum(p_sup, a_term)
        # online challenger
        p_on = np.full(len(X), np.nan)
        if self.adaptive and self.online is not None and schema != "nsl":
            p_on = proba_pos(self.online, X)
            score = (1 - self.w_online) * s_static + self.w_online * p_on
        else:
            score = s_static
        pred = (score >= self.threshold).astype(int)
        # never let the online model alone suppress a confident zero-day anomaly
        if self.use_anomaly:
            pred = np.maximum(pred, (an >= 0.999).astype(int))
        fam = np.where(pred == 1, np.where(fam == "NORMAL", "UNKNOWN_ANOMALY", fam), "NORMAL")
        res = []
        alert_idx = np.where(pred == 1)[0]
        expl = {}
        if self.explain and len(alert_idx):
            pick = alert_idx[-8:]          # newest alerts of the batch are the ones operators see first
            try:
                if self._explainer is None or self._explainer[0] is not model:
                    import shap
                    self._explainer = (model, shap.TreeExplainer(model))
                sv = self._explainer[1].shap_values(X[pick], check_additivity=False)
                sv = np.asarray(sv[1] if isinstance(sv, list) else sv)
                if sv.ndim == 3:
                    sv = sv[..., 1]
                for k, i in enumerate(pick):
                    order = np.argsort(-np.abs(sv[k]))[: self.top_k]
                    expl[i] = [(cols[o], float(X[i, o]), float(sv[k, o])) for o in order]
            except Exception as e:  # explanation must never break detection
                log.debug(f"shap failed: {e}")
        for i, r in enumerate(recs):
            truth = r.get("truth") or {}
            item = {"seq": r.get("seq", self.n + i), "ts": r.get("ts", time.time()), "dataset": r.get("src", schema),
                    "schema": schema, "proto": str(v["proto"].iloc[i]), "service": str(v["service"].iloc[i]),
                    "mqtt_type": float(v["mqtt_type"].iloc[i]) if "mqtt_type" in v else 0.0,
                    "score": float(score[i]), "p_supervised": float(p_sup[i]), "anomaly_pct": float(an[i]),
                    "p_online": None if np.isnan(p_on[i]) else float(p_on[i]), "pred": int(pred[i]),
                    "family": str(fam[i]), "truth_y": truth.get("y"), "truth_super": truth.get("super"),
                    "truth_fine": truth.get("fine")}
            if i in expl:
                item["explanation"] = narrative(float(score[i]), expl[i], label=str(fam[i]), record_id=item["seq"],
                                                anomaly=float(an[i]))
                item["top_features"] = [{"feature": f, "value": val, "shap": s} for f, val, s in expl[i]]
            elif pred[i] == 1:
                src = "benign-traffic anomaly model" if p_sup[i] < 0.5 else "supervised model"
                item["explanation"] = (f"Record {item['seq']}: flagged by the {src} — supervised P(attack)={p_sup[i]:.2f}, "
                                       f"anomaly percentile vs benign traffic {an[i]:.2%}; {item['proto']} to "
                                       f"{item['service']} (SHAP budget used by earlier alerts of this batch).")
            res.append(item)
            # schedule simulated analyst feedback (label budget + delay) when truth is present
            if truth.get("y") is not None and self.rng.random() < self.feedback_rate and schema != "nsl":
                self.pending_feedback.append((self.n + i + self.feedback_delay, X[i], int(truth["y"]),
                                              float(s_static[i]), cols))
        return res

    # ------------------------------------------------------------------ feedback / adaptation
    def _refit_challenger(self):
        y = np.array(self.fb_y)
        if len(y) < 150 or y.min() == y.max():
            return
        from lightgbm import LGBMClassifier
        X = np.vstack([np.array(self.fb_X), self.a["replay_X"]]) if "replay_X" in self.a else np.array(self.fb_X)
        yy = np.concatenate([y, self.a["replay_y"]]) if "replay_y" in self.a else y
        w = np.concatenate([np.ones(len(y)), np.full(len(yy) - len(y), 0.3)])   # recent feedback dominates
        m = LGBMClassifier(n_estimators=120, num_leaves=31, learning_rate=0.1, min_child_samples=5, verbose=-1,
                           random_state=0, n_jobs=1)
        self.online = m.fit(X, yy, sample_weight=w)
        self.fb_since_fit = 0

    def feedback(self, x_row, y: int, s_static: float, cols):
        """Analyst label for one record (also used by POST /api/feedback)."""
        if not self.adaptive:
            return
        p_on = float(proba_pos(self.online, x_row[None, :])[0]) if self.online is not None else 0.5
        eps = 1e-4
        ls = -(y * math.log(max(s_static, eps)) + (1 - y) * math.log(max(1 - s_static, eps)))
        lo = -(y * math.log(max(p_on, eps)) + (1 - y) * math.log(max(1 - p_on, eps)))
        self.loss_static = 0.98 * self.loss_static + 0.02 * ls
        self.loss_online = 0.98 * self.loss_online + 0.02 * lo
        eta = 4.0
        if self.online is not None:
            ws, wo = math.exp(-eta * self.loss_static), math.exp(-eta * self.loss_online)
            self.w_online = wo / (ws + wo)
        self.fb_X.append(np.asarray(x_row, dtype=np.float32))
        self.fb_y.append(int(y))
        self.online_seen += 1
        self.fb_since_fit += 1
        if self.fb_since_fit >= 200:
            self._refit_challenger()
        if y == 0:
            self.benign_buf.append(x_row)
        err = int((s_static >= self.threshold) != y)
        self.err_detector.update(err)
        if self.err_detector.drift_detected:
            self._on_drift("ADWIN on feedback error stream")

    def _on_drift(self, reason: str):
        ev = {"seq": self.n, "time": time.time() - self.t0, "reason": reason, "psi": self.last_psi}
        if self.adaptive and len(self.benign_buf) > 300:
            # re-calibrate benign anomaly reference with confirmed-benign recent traffic
            B = np.array(self.benign_buf)
            an = self.a["anomaly"]
            new_ref = np.sort(-an.iforest.score_samples(B))
            an.ref = np.sort(np.concatenate([an.ref[:: max(1, len(an.ref) // 2000)], new_ref]))
            self.anomaly_ref_version += 1
            ev["action"] = f"anomaly reference re-calibrated (v{self.anomaly_ref_version})"
        elif self.adaptive:
            ev["action"] = "anomaly reference kept (not enough confirmed-benign samples yet)"
        else:
            ev["action"] = "logged only (static mode, adaptation disabled)"
        if self.adaptive:
            # forget pre-drift feedback (ADWIN-style window shrink); challenger refits on post-drift labels
            keep = min(len(self.fb_y), 100)
            self.fb_X = collections.deque(list(self.fb_X)[-keep:], maxlen=4000)
            self.fb_y = collections.deque(list(self.fb_y)[-keep:], maxlen=4000)
            self.fb_since_fit = 150
            self.loss_online = min(self.loss_online, self.loss_static)   # give the challenger a fair restart
            ev["action"] += "; feedback window reset, challenger will re-train on post-drift labels"
        self.drift_events.append(ev)
        self.events_for_sink.append({"type": "drift", **ev})
        log.info(f"DRIFT @ {self.n}: {reason} -> {ev['action']}")

    # ------------------------------------------------------------------ stats
    def _update_stats(self, out: List[dict]):
        for r in out:
            self.recent.append((r["pred"], r["truth_y"], r["score"], r["latency_ms"]))
            self.proto_counts[r["proto"]] += 1
            self.service_counts[r["service"]] += 1
            self.dataset_counts[r["dataset"]] += 1
            if r["mqtt_type"]:
                self.mqtt_counts[int(r["mqtt_type"])] += 1
            if r["pred"]:
                self.n_alerts += 1
                self.family_counts[r["family"]] += 1
                self.alerts.append({k: r.get(k) for k in ("seq", "ts", "dataset", "proto", "service", "score",
                                                          "family", "truth_fine", "anomaly_pct", "explanation")})
            if r["truth_super"]:
                self.truth_counts[r["truth_super"]] += 1
        self.n += len(out)
        # due feedback
        while self.pending_feedback and self.pending_feedback[0][0] <= self.n:
            _, x, y, s, cols = self.pending_feedback.popleft()
            self.feedback(x, y, s, cols)
        # PSI of the last 1000 packets vs the training reference, checked every 500 packets
        if len(self.psi_buf) >= 1000 and (self.n // 500) != ((self.n - len(out)) // 500):
            psi = self.psi.psi(np.array(self.psi_buf))
            prev = self.last_psi
            self.last_psi = float(np.mean(np.sort(psi)[-10:]))
            # alarm on a jump of the shift level (not on a persistently shifted but stable regime)
            if self.last_psi > 0.2 and self.last_psi > 1.5 * max(prev, 0.05) and \
                    (not self.drift_events or self.n - self.drift_events[-1]["seq"] > 3000):
                self._on_drift(f"PSI feature drift (top-10 mean PSI={self.last_psi:.2f})")
        self.timeline.append(self.snapshot(light=True))

    def metrics(self) -> dict:
        arr = [(p, t) for p, t, _, _ in self.recent if t is not None]
        lat = [l for *_, l in self.recent]
        m = {"window": len(self.recent), "labelled": len(arr),
             "latency_ms_p50": float(np.percentile(lat, 50)) if lat else 0.0,
             "latency_ms_p95": float(np.percentile(lat, 95)) if lat else 0.0}
        if arr:
            p, t = np.array(arr).T.astype(int)
            tp, fp = int(((p == 1) & (t == 1)).sum()), int(((p == 1) & (t == 0)).sum())
            fn, tn = int(((p == 0) & (t == 1)).sum()), int(((p == 0) & (t == 0)).sum())
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) or 1.0
            m.update({"accuracy": (tp + tn) / len(p), "precision": prec, "recall": rec,
                      "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "mcc": (tp * tn - fp * fn) / den,
                      "fpr": fp / (fp + tn) if fp + tn else 0.0, "tp": tp, "fp": fp, "tn": tn, "fn": fn})
        return m

    def snapshot(self, light: bool = False) -> dict:
        el = max(time.time() - self.t0, 1e-6)
        s = {"t": round(el, 2), "n": self.n, "alerts": self.n_alerts, "throughput": self.n / el,
             "w_online": self.w_online, "psi": self.last_psi, "online_seen": self.online_seen,
             "drift_events": len(self.drift_events), **self.metrics()}
        if light:
            return s
        s.update({"family_counts": dict(self.family_counts), "truth_counts": dict(self.truth_counts),
                  "proto_counts": dict(self.proto_counts), "service_counts": dict(self.service_counts.most_common(10)),
                  "mqtt_counts": {str(k): v for k, v in self.mqtt_counts.items()},
                  "dataset_counts": dict(self.dataset_counts), "drift_log": self.drift_events[-20:],
                  "recent_alerts": list(self.alerts)[-25:][::-1], "timeline": list(self.timeline)[-300:],
                  "meta": self.a.get("meta", {})})
        return s
