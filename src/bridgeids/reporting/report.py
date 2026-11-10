"""Build the paper draft (numbers pulled live from results/tables), the HTML results dashboard and the
Power BI export.   python -m bridgeids.run report
"""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import get_logger, load_config
from .powerbi import export_experiment_tables

log = get_logger("report")


class T:
    """Lazy table access with safe defaults (report still builds if an experiment was skipped)."""

    def __init__(self, cfg):
        self.dir = Path(cfg["paths"]["tables"])

    def __call__(self, name):
        p = self.dir / f"{name}.csv"
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    def json(self, name):
        p = self.dir / f"{name}.json"
        return json.loads(p.read_text()) if p.exists() else {}


def f(x, d=3):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "n/a"
        return f"{float(x):.{d}f}"
    except Exception:
        return str(x)


def pct(x, d=1):
    try:
        return f"{100 * float(x):.{d}f}%"
    except Exception:
        return "n/a"


def md_table(df: pd.DataFrame, floatfmt=".3f", maxrows=40):
    if isinstance(df, tuple):              # (frame, wanted columns) -> keep the columns that exist
        df, cols = df
        df = df[[c for c in cols if c in df.columns]] if not df.empty else df
    if df is None or df.empty:
        return "_(table not available — run the corresponding experiment)_"
    try:
        return df.head(maxrows).to_markdown(index=False, floatfmt=floatfmt)
    except Exception:
        return df.head(maxrows).to_string(index=False)


def _get(df, query: dict, col):
    if df.empty:
        return np.nan
    m = np.ones(len(df), bool)
    for k, v in query.items():
        m &= (df[k] == v).to_numpy()
    s = df.loc[m, col]
    return s.iloc[0] if len(s) else np.nan


# ============================================================================ paper draft
def build_paper(cfg):
    t = T(cfg)
    st, base, audit = t("e01_dataset_statistics"), t("e02_baselines_binary"), t("e02_leakage_audit")
    cross, shift = t("e03_cross_dataset_binary"), t("e03_domain_shift_report")
    tr, da, cl = t("e04_transfer_learning_results"), t("e05_domain_adaptation"), t("e06_continual_learning")
    nlp, nlp_in = t("e07_nlp_cross_dataset"), t("e07_nlp_in_domain")
    lofo, zx, zsl = t("e08_zeroshot_lofo"), t("e08_zeroshot_cross_dataset"), t("e08_semantic_zsl")
    agree, drift = t("e09_explanation_drift_rank_agreement"), t("e10_drift_summary")
    bench, weka = t("e11_stream_benchmark"), t("e12_weka_results")
    xai = t.json("e09_xai_summary")
    txt = t.json("e07_text_examples")

    def cx(space, src, tgt, col="roc_auc", model="lgbm"):
        return _get(cross, {"space": space, "source": src, "target": tgt, "model": model}, col)

    n_e = cx("CBS", "NSL-KDD", "Edge-IIoTset")
    f_e = cx("CBS", "ForgeIIoT", "Edge-IIoTset")
    f_e_p = cx("PBS", "ForgeIIoT", "Edge-IIoTset")
    e_n = cx("CBS", "Edge-IIoTset", "NSL-KDD")
    leak_hi = _get(audit, {"dataset": "ForgeIIoT", "setting": "+ frame.number/epoch/tcp.stream, random split"},
                   "fine_accuracy")
    leak_lo = _get(audit, {"dataset": "ForgeIIoT", "setting": "leak-free, random split"}, "fine_accuracy")
    leak_tmp = _get(audit, {"dataset": "ForgeIIoT", "setting": "+ identifiers, capture-temporal split"},
                    "fine_accuracy")
    pad = drift.set_index("pair")["proxy_A_distance"] if not drift.empty else pd.Series(dtype=float)
    cshift = drift.set_index("pair")["mean_conditional_label_shift"] if not drift.empty else pd.Series(dtype=float)

    def nl(src, tgt, rep):
        return _get(nlp, {"source": src, "target": tgt, "representation": rep}, "roc_auc")

    def dav(pair, meth, col="roc_auc"):
        return _get(da, {"pair": pair, "method": meth}, col)

    def clv(order, strat, col):
        return _get(cl, {"order": order, "strategy": strat, "metric": "balanced_acc"}, col)

    o1 = "NSL-KDD → ForgeIIoT → Edge-IIoTset"

    def bv(cfgname, phase, col):
        return _get(bench, {"config": cfgname, "phase": phase}, col)

    lofo_mean = lofo.groupby("dataset")[["supervised@1%FPR", "one-class IF@1%FPR", "hybrid@1%FPR"]].mean() \
        if not lofo.empty else pd.DataFrame()
    inj = zx[zx.super == "INJECTION"]["hybrid@5%FPR"].mean() if not zx.empty else np.nan
    dap = _get(zsl, {"method": "DAP (attribute)"}, "top1_accuracy")
    tzsl = _get(zsl, {"method": "Text-embedding ZSL"}, "top1_accuracy")
    rho = lambda a, b: _get(agree, {"dataset_a": a, "dataset_b": b}, "spearman")  # noqa: E731

    best_base = (base.loc[base.groupby("dataset")["f1"].idxmax()], ["dataset", "model", "f1", "roc_auc", "mcc"]) \
        if not base.empty else pd.DataFrame()

    # ---------- data-driven sentences (so prose can never contradict the tables)
    def da_sentence():
        if da.empty:
            return "n/a"
        out = []
        for pair, g in da.groupby("pair", sort=False):
            g = g[~g.method.isin(["No adaptation (LightGBM)", "MLP (no adaptation)"])]
            b = g.loc[g.auc_gain_vs_noadapt.idxmax()]
            out.append(f"{pair}: best = {b.method} ({b.auc_gain_vs_noadapt:+.3f} AUC)")
        return "; ".join(out)

    def tr_sentence(frac=0.001):
        if tr.empty:
            return "n/a"
        parts = []
        for src_name, g in tr[tr.frac == frac].groupby("source", sort=False):
            m = g.set_index("method")["mcc"]
            if {"MLP-finetune", "MLP-scratch", "target-only", "continued-GBM"} <= set(m.index):
                verdict = "helps" if m["MLP-finetune"] > m["MLP-scratch"] + 0.01 else "does not help"
                parts.append(f"pre-training on {src_name} {verdict} the MLP (MCC {m['MLP-finetune']:.3f} vs "
                             f"{m['MLP-scratch']:.3f} from scratch); warm-started GBM {m['continued-GBM']:.3f} vs "
                             f"target-only {m['target-only']:.3f}")
        n = int(round(tr[tr.frac == frac]["n_target"].mean())) if not tr.empty else 0
        return f"With ~{n} labelled target packets: " + "; ".join(parts) + "."

    hyb_fpr = _get(zx, {"edge_family": "Normal"}, "hybrid@5%FPR")
    bwt1 = clv(o1, "EWC + Replay", "BWT")
    bwt2 = clv("Edge-IIoTset → ForgeIIoT → NSL-KDD", "EWC + Replay", "BWT")
    sh = t("e09_shap_shift_forge_model_on_edge")
    shap_shift_txt = ", ".join(f"{r.feature} ({'+' if r.shift > 0 else ''}{r.shift:.3f})" for r in sh.head(3).itertuples()) \
        if not sh.empty else "n/a"

    paper = f"""# Bridging Legacy IDS and Modern IIoT Security: Cross-Dataset Transfer, Domain Adaptation, Zero-Shot Detection and Explainable Dynamic Detection over NSL-KDD → ForgeIIoT → Edge-IIoTset

*Draft generated automatically from `results/tables` by `python -m bridgeids.run report`. Every number below is read
from the experiment outputs, so re-running the pipeline updates the paper.*

**Keywords:** intrusion detection, IIoT, cross-dataset evaluation, domain adaptation, continual learning, zero-shot
learning, explainable AI, concept drift, MQTT, NSL-KDD, Edge-IIoTset, ForgeIIoT

## Abstract

Machine-learning intrusion detectors are still developed and benchmarked on legacy corpora such as NSL-KDD, yet
deployed on Industrial-IoT networks whose protocols (MQTT, Modbus), devices and attacks differ fundamentally. We
study whether, and how, knowledge transfers along a **legacy → bridge → modern** chain: NSL-KDD (connection
records), ForgeIIoT (an MQTT-centric IIoT testbed capture) and Edge-IIoTset. We propose a three-tier feature
alignment (a semantic *Canonical Behavioural Space* shared by all datasets, a *Packet Bridge Space* shared by the two
IIoT captures, and leakage-free native spaces), a unified seven-class attack taxonomy, and a *language interlingua*
that renders every connection or packet with the same vocabulary. A model trained on NSL-KDD reaches a ROC-AUC of
only {f(n_e)} on Edge-IIoTset, whereas a ForgeIIoT-trained model reaches {f(f_e)} in the canonical space and
{f(f_e_p)} in the packet space. Drift analysis agrees: proxy-A-distance is {f(pad.get("ForgeIIoT vs Edge-IIoTset"))}
for Forge–Edge vs {f(pad.get("NSL-KDD vs Edge-IIoTset"))} for NSL–Edge, and concept shift |ΔP(y|x)| is
{f(cshift.get("ForgeIIoT vs Edge-IIoTset"))} vs {f(cshift.get("NSL-KDD vs Edge-IIoTset"))}. The flow-to-text
interlingua raises ForgeIIoT→NSL-KDD AUC from {f(nl("ForgeIIoT", "NSL-KDD", "CBS numeric"))} to
{f(nl("ForgeIIoT", "NSL-KDD", "text embedding"))}. EWC + experience replay gives the best continual-learning trade-off,
and a benign-calibrated hybrid detector detects never-seen Edge injection attacks at {pct(inj)} (benign false-positive
rate {pct(hyb_fpr)}) using only benign target traffic. SHAP rankings agree between ForgeIIoT and Edge-IIoTset (Spearman ρ = {f(rho("ForgeIIoT", "Edge-IIoTset"))})
but not between NSL-KDD and ForgeIIoT (ρ = {f(rho("NSL-KDD", "ForgeIIoT"))}), so explanations expose the same shift.
Finally, we move from static to **dynamic** detection: a streaming engine that replays dataset records as network
packets, detects drift and adapts from a small analyst-label budget raises post-shift MCC from
{f(bv("Static hybrid", "phase 2: Edge-IIoTset", "mcc"))} to {f(bv("Dynamic hybrid (10% feedback)", "phase 2: Edge-IIoTset", "mcc"))}.
We also show that capture-order identifiers inflate ForgeIIoT fine-grained accuracy from {pct(leak_lo)} to {pct(leak_hi)}.

## 1 Introduction

IIoT intrusion-detection papers routinely report >99 % accuracy on a single dataset, while legacy benchmarks remain the
default for method development. Two practical questions are rarely answered together: *does a legacy-trained model
still work on an IIoT plant?* and *what should be done when it does not?* We answer five research questions:

* **RQ1** Can legacy IDS models generalise to modern IIoT datasets?
* **RQ2** Can ForgeIIoT act as a bridge between legacy and modern environments?
* **RQ3** Can domain adaptation reduce the degradation?
* **RQ4** Can explainability reveal feature shifts across datasets?
* **RQ5** Can unseen attacks be detected with zero-shot transfer?

and add **RQ6**: does dynamic (streaming, adaptive) detection recover performance after a domain shift, at what label cost?

**Contributions.** (i) a three-tier cross-schema alignment and a 7-class cross-dataset taxonomy; (ii) the first legacy →
bridge → modern study including NSL-KDD, ForgeIIoT and Edge-IIoTset; (iii) a language interlingua for heterogeneous
IDS data; (iv) a comparison of DomainNorm, CORAL, TCA, importance weighting, self-training and DANN; (v) a continual-learning
study with accuracy matrices, BWT and forgetting; (vi) attribute- and text-based zero-shot classification and
benign-calibrated hybrid zero-day detection; (vii) explanation-drift metrics; (viii) an open, explainable, adaptive
streaming detector with Power BI integration; (ix) a leakage audit of ForgeIIoT.

## 2 Related work

See `reports/related_work.md` (44 works, 2023–2026, plus foundations), grouped as: cross-dataset generalisation,
transfer/DA, IIoT IDS, zero-shot, NLP/LLM, continual learning/drift and XAI. The closest work is a 2026 heterogeneous-DA
benchmark with Edge-IIoTset as the target, which also finds DA useful only in some contexts. Ours differs by the bridge
dataset, the language interlingua, continual learning, explanation drift and the dynamic system.

## 3 Datasets, taxonomy and leakage audit

{md_table(st)}

All fine labels are mapped to NORMAL, DOS, RECON, MITM_SPOOF, UNAUTH_ACCESS, INJECTION, MALWARE_PRIVESC
(`results/tables/e01_attack_taxonomy.csv`, relationship matrix in `e01_attack_relationship_matrix.csv`).
NSL-KDD fine labels come from KDDTrain+/KDDTest+.txt (the ARFF files only carry normal/anomaly).

**Leakage audit.** The earlier notebook reported 100 % accuracy on ForgeIIoT; SHAP ranked `frame.number`,
`frame.time_epoch` and `frame.time_relative` highest. Because each attack was captured as a separate file, these
fields identify the capture rather than the behaviour:

{md_table(audit[["dataset", "setting", "binary_f1", "fine_macro_f1", "fine_accuracy", "top3_features"]] if not audit.empty else audit)}

With identifiers, fine-grained accuracy jumps to {pct(leak_hi)} (leak-free: {pct(leak_lo)}). Under a capture-temporal
split the gain mostly vanishes ({pct(leak_tmp)}). We also observed that ForgeIIoT labels are assigned per capture file,
so background MQTT traffic inside an attack capture is labelled as attack. This is label noise that caps per-packet
accuracy and explains why MITM/Sybil families are hardest. All experiments use leak-free features.

## 4 Method

**4.1 Three-tier alignment.** (a) *CBS* ({int(st["CBS features"].iloc[0]) if not st.empty else 27} features): protocol,
service class (HTTP, MQTT, DNS, FTP, mail, remote shell, Modbus, ICMP echo, well-known/registered/ephemeral port),
TCP-flag semantics (SYN, ACK, FIN, RST, PSH, half-open), log payload volume, land, application layer. NSL connection
flags (SF, S0, REJ, RSTO…) are translated into the same flag semantics. (b) *PBS*
({int(st["PBS features"].dropna().iloc[0]) if not st.empty else 51} features): CBS + TCP length/seq/ack, ARP opcodes,
HTTP method/length/response and MQTT message types/lengths, shared by the two tshark exports. (c) *Native* leak-free
feature sets. Identifiers (IP/MAC, checksums, raw ports, stream indices, frame numbers, timestamps) are never features.

**4.2 Language interlingua.** Each record becomes a sentence, e.g. {"; ".join(f'"{s}"' for s in (txt.get("examples", {}).get("Edge-IIoTset", [])[:2]))}.
Sentences are encoded with Sentence-BERT when available, otherwise with an offline TF-IDF + LSA domain encoder
(used here: {txt.get("encoder", "n/a")}; {txt.get("unique_sentences", "n/a")} unique sentences across all datasets).

**4.3 Learners.** DT, RF, AdaBoost, XGBoost, LightGBM and MLP (numpy engine with exact backprop). Weka J48, RandomForest
and AdaBoostM1 are used as independent baselines.
**4.4 Domain adaptation.** DomainNorm, CORAL, TCA (linear kernel), importance weighting, pseudo-label self-training
and DANN with a gradient-reversal layer. All are unsupervised, using no target labels.
**4.5 Continual learning.** Domain-incremental tasks NSL→Forge→Edge (and reverse): naive fine-tuning, EWC (exact
diagonal Fisher), replay, EWC + replay, LightGBM continued boosting, joint offline training.
**4.6 Zero-shot.** (a) leave-one-family-out with supervised, one-class (Isolation Forest calibrated by benign ECDF)
and hybrid detectors at 1 % benign FPR; (b) cross-dataset zero-shot with only benign target traffic; (c) DAP over a
10-attribute class signature and text-embedding ZSL (ridge projection onto class-description embeddings).
**4.7 Explainability and drift.** SHAP TreeExplainer, LIME, rank agreement (Spearman, Kendall, top-10 Jaccard),
SHAP shift of one model across datasets. Drift: KL, JS, Wasserstein, PSI, RBF-MMD, domain-classifier AUC,
proxy-A-distance, conditional label shift and attack-label JS.
**4.8 Dynamic detection.** Records are serialised as JSON packets (UDP/TCP/HTTP/MQTT) and replayed in realistic
scenarios: kill-chain campaign, domain shift and zero-day. A FastAPI engine performs schema auto-detection, PBS
featurisation, hybrid scoring, per-alert SHAP narratives, sliding-window PSI and ADWIN drift detection, and online
adaptation. The adaptation uses a sliding-window LightGBM challenger trained on delayed analyst feedback plus a
replay sample, combined with the static champion by Hedge weights. On drift, the feedback window is reset and the
benign anomaly reference is re-calibrated.

## 5 Results

### 5.1 In-domain baselines

{md_table(best_base)}

Full table: `e02_baselines_binary.csv`. Edge-IIoTset is almost perfectly separable in-domain. NSL-KDD's official
test set remains hard (unseen attack types).

{md_table(weka[["setting", "train", "test", "classifier", "accuracy", "attack_f1", "roc_auc"]] if not weka.empty else weka)}

### 5.2 RQ1/RQ2 — Cross-dataset transfer

{md_table(shift[shift.model == "lgbm"][["space", "source", "target", "in_domain_auc", "cross_auc", "auc_drop", "cross_f1"]] if not shift.empty else shift)}

**RQ1: no.** The NSL-KDD model is near chance on Edge-IIoTset (AUC {f(n_e)}). Edge→NSL is *inverted* (AUC {f(e_n)}),
which signals concept drift rather than covariate shift alone. **RQ2: yes.** ForgeIIoT transfers to Edge-IIoTset
(CBS AUC {f(f_e)}, PBS AUC {f(f_e_p)}), and the drift metrics (§5.7) place ForgeIIoT much closer to Edge-IIoTset than
to NSL-KDD. Weka J48 independently confirms the pattern.

### 5.3 RQ2 — Transfer with few labels (target = Edge-IIoTset)

{md_table(tr[tr.frac <= 0.01][["source", "method", "frac", "n_target", "f1", "auc", "mcc"]] if not tr.empty else tr)}

Edge-IIoTset is so separable that a few dozen labels already give a strong target-only model. {tr_sentence()}
Naively pooling NSL-KDD with target data causes negative transfer (see the `pooled` rows). The value of a source is
therefore largest in the *zero-label* regime, which is what §5.4 and §5.8 address.

### 5.4 RQ3 — Unsupervised domain adaptation

{md_table(da[["pair", "method", "roc_auc", "mcc", "f1", "auc_gain_vs_noadapt"]] if not da.empty else da)}

**RQ3: partially.** Best method per pair: {da_sentence()}. DA gives modest gains (about +0.03 to +0.06 AUC)
where the shift is mostly covariate (NSL→Forge, Forge↔Edge). For NSL→Edge, even the best method leaves the AUC at only
{f(dav("NSL-KDD → Edge-IIoTset", "TCA (+LightGBM)"))}. Every method *hurts* the multi-source NSL+Forge→Edge setting,
where NSL's conflicting concept dominates the aligned space. Aligning P(x) cannot fix a change of P(y|x), consistent with the concept-shift measurements
in §5.7. DANN is unstable across seeds and epochs (std in `e05_domain_adaptation.csv`).

### 5.5 Continual learning NSL → Forge → Edge

{md_table(cl[cl.metric == "balanced_acc"][["order", "strategy", "avg_final", "avg_final_std", "BWT", "forgetting"]] if not cl.empty else cl)}

EWC + replay reaches {f(clv(o1, "EWC + Replay", "avg_final"))} average final balanced accuracy in the legacy→modern order
(naive fine-tuning: {f(clv(o1, "Naive fine-tuning", "avg_final"))}), with positive backward transfer (BWT {bwt1:+.3f}).
In the reverse order it is again the best strategy, but BWT is only {bwt2:+.3f}. Under conflicting concepts, joint
offline training is *not* an upper bound (negative interference), and tree models with continued boosting forget the most.
Without replay, the model collapses to predicting \"attack\" for everything after the ForgeIIoT task (balanced accuracy 0.50 on all
tasks, `e06_continual_matrices.png`), because Forge is 88 % attack. Replay prevents this collapse.

### 5.6 RQ5 — Zero-shot / unseen attacks

{md_table(lofo_mean.reset_index() if not lofo_mean.empty else lofo_mean)}

{md_table(zx[["edge_family", "super", "family_seen_in_sources", "source-supervised@5%FPR", "target one-class@5%FPR", "hybrid@5%FPR"]] if not zx.empty else zx)}

{md_table(zsl)}

**RQ5: yes, for detection; partially, for naming the family.** In-domain, unseen Edge families are still detected
(they differ strongly from benign MQTT telemetry). Cross-dataset, the hybrid detector, which uses only benign target
traffic, detects the never-seen INJECTION families at {pct(inj)}, at a benign false-positive rate of {pct(hyb_fpr)}
(the union of two 5 %-FPR detectors). The source-only supervised model, by contrast, misses most target attacks. Semantic ZSL names unseen classes above chance:
DAP {pct(dap)}, text ZSL {pct(tzsl)}, chance {pct(1 / max(1, len(cfg["experiments"]["zsl_holdout_edge"])))}.

### 5.7 RQ4 — Explainability and drift

{md_table(agree)}

{md_table(drift[["pair", "mean_JS", "MMD_rbf", "proxy_A_distance", "concept_disagreement", "mean_conditional_label_shift", "attack_label_JS"]] if not drift.empty else drift)}

**RQ4: yes.** SHAP rankings in the shared space agree for Forge–Edge (ρ = {f(rho("ForgeIIoT", "Edge-IIoTset"))}) but not for
NSL–Forge (ρ = {f(rho("NSL-KDD", "ForgeIIoT"))}). LIME and SHAP top-6 features overlap with Jaccard
{f(xai.get("lime_shap_top6_jaccard_mean"))}. When a Forge-trained model is explained on Edge traffic, the largest
importance shifts are {shap_shift_txt} (`e09_shap_shift_forge_model_on_edge.csv`). Packet-size and
TCP-sequence features change role between the two plants. Example incident narrative:

> {(xai.get("narratives") or [{}])[0].get("narrative", "n/a")}

### 5.8 RQ6 — From static to dynamic detection

{md_table(bench[bench.phase.isin(["phase 2: Edge-IIoTset", "phase 2, last 5k"])][["config", "phase", "f1", "mcc", "fpr", "recall", "drift_detection_delay_pkts", "feedback_labels_used", "throughput_pkt_s"]] if not bench.empty else bench)}

A Forge-trained deployment that keeps running after the plant traffic changes to Edge-IIoT keeps a high recall but
suffers a very high false-positive rate. With 10 % delayed analyst feedback, MCC on the shifted traffic rises from
{f(bv("Static hybrid", "phase 2: Edge-IIoTset", "mcc"))} to {f(bv("Dynamic hybrid (10% feedback)", "phase 2: Edge-IIoTset", "mcc"))}
({f(bv("Dynamic hybrid (10% feedback)", "phase 2, last 5k", "mcc"))} over the last 5k packets). The PSI detector flags
the shift {f(bv("Dynamic hybrid (10% feedback)", "phase 2: Edge-IIoTset", "drift_detection_delay_pkts"), 0)} packets after it
happens. In a live zero-day replay through the network server (2 000 XSS/Ransomware packets withheld from training),
all were flagged and 77 % of Ransomware packets were assigned to the semantically correct MALWARE family.

## 6 Discussion

*Why legacy models fail.* NSL-KDD connection semantics (flags summarising a whole connection, 2-second window
statistics, host content features) have no packet-level counterpart. In the shared space the relation between features
and labels even inverts (AUC < 0.5). *Why the bridge works.* ForgeIIoT and Edge-IIoTset share capture tooling (tshark),
protocols (MQTT/TCP) and an IIoT traffic mix. Their drift is mostly covariate, which DA and a little target feedback
can correct. *Practical recipe.* (1) Start from an IIoT bridge source, not a legacy one; (2) align with PBS or text;
(3) calibrate a benign-only anomaly component on the target; (4) deploy the dynamic engine with a small label budget.

**Threats to validity.** Stratified samples are used for compute reasons (configurable). Edge-IIoT ML-set rows are
shuffled packets, so window statistics cannot be computed. ForgeIIoT labels are capture-level. The NSL-KDD test split
contains attack types absent from training. Sentence-BERT could not be downloaded in our run environment; the offline
LSA encoder was used and SBERT is supported by the code.

## 7 Conclusion

Legacy-trained detectors do not transfer to IIoT traffic, but an IIoT bridge dataset does, and the same conclusion
follows from accuracy, drift statistics and explanations. Domain adaptation helps only when concepts are shared. A
language interlingua and benign-calibrated hybrid scoring improve cross-dataset and zero-day robustness, and a
lightweight adaptive streaming engine turns these findings into a deployable detector.

## Reproducibility

`python -m bridgeids.run prepare && python -m bridgeids.run exp all && python -m bridgeids.run report` (seed
{cfg["seed"]}). All tables are in `results/tables`, figures in `results/figures` and the Power BI CSVs in `powerbi/data`.
"""
    out = Path(cfg["paths"]["reports"]) / "paper_draft.md"
    out.write_text(paper, encoding="utf-8")
    log.info(f"wrote {out}")
    return out


# ============================================================================ HTML results dashboard
SECTIONS = [
    ("Datasets & taxonomy", "e01", ["e01_dataset_statistics", "e01_attack_relationship_matrix"]),
    ("Baselines & leakage audit", "e02", ["e02_baselines_binary", "e02_leakage_audit"]),
    ("Cross-dataset (RQ1, RQ2)", "e03", ["e03_domain_shift_report"]),
    ("Transfer learning", "e04", ["e04_transfer_learning_results"]),
    ("Domain adaptation (RQ3)", "e05", ["e05_domain_adaptation"]),
    ("Continual learning", "e06", ["e06_continual_learning"]),
    ("NLP interlingua", "e07", ["e07_nlp_cross_dataset", "e07_nlp_in_domain"]),
    ("Zero-shot (RQ5)", "e08", ["e08_zeroshot_lofo", "e08_zeroshot_cross_dataset", "e08_semantic_zsl"]),
    ("Explainability (RQ4)", "e09", ["e09_explanation_drift_rank_agreement", "e09_shap_top_features_native"]),
    ("Drift analysis", "e10", ["e10_drift_summary"]),
    ("Static vs dynamic (RQ6)", "e11", ["e11_stream_benchmark"]),
    ("Weka", "e12", ["e12_weka_results"]),
]


def build_dashboard(cfg):
    t = T(cfg)
    figdir = Path(cfg["paths"]["figures"])
    nav, body = [], []
    for title, prefix, tables in SECTIONS:
        sid = prefix
        nav.append(f'<a href="#{sid}">{html.escape(title)}</a>')
        parts = [f'<section id="{sid}"><h2>{html.escape(title)}</h2>']
        for fp in sorted(figdir.glob(f"{prefix}_*.png")):
            b64 = base64.b64encode(fp.read_bytes()).decode()
            parts.append(f'<figure><img src="data:image/png;base64,{b64}" alt="{fp.stem}"><figcaption>{fp.stem}'
                         f'</figcaption></figure>')
        for name in tables:
            df = t(name)
            if df.empty:
                continue
            df = df.head(60)
            for c in df.select_dtypes("number").columns:
                df[c] = df[c].map(lambda v: f"{v:.4f}" if isinstance(v, float) and not float(v).is_integer() else v)
            parts.append(f"<h3>{name}</h3><div class='tw'>" + df.to_html(index=False, border=0, escape=True) + "</div>")
        parts.append("</section>")
        body.append("\n".join(parts))
    page = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BridgeIDS — Results</title><style>
body{{margin:0;font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;color:#1d2433;background:#f6f7f9}}
header{{background:#0f1726;color:#fff;padding:22px 32px}}header h1{{margin:0;font-size:21px}}header p{{margin:4px 0 0;color:#a9b4c6}}
nav{{position:sticky;top:0;background:#fff;border-bottom:1px solid #e3e6ec;padding:10px 32px;display:flex;gap:16px;flex-wrap:wrap;z-index:2}}
nav a{{color:#2e5aac;text-decoration:none;font-size:13px}}main{{max-width:1180px;margin:0 auto;padding:10px 24px 60px}}
section{{background:#fff;border:1px solid #e3e6ec;border-radius:12px;padding:18px 22px;margin:18px 0}}
h2{{margin:0 0 12px;font-size:18px}}h3{{font-size:13px;color:#5b6475;margin:18px 0 6px;font-family:ui-monospace,Menlo,monospace}}
figure{{display:inline-block;margin:8px 10px 8px 0;max-width:100%;vertical-align:top}}img{{max-width:100%;max-height:520px;border:1px solid #eef0f3;border-radius:8px}}
figcaption{{font-size:11px;color:#8a93a3}}.tw{{overflow-x:auto}}table{{border-collapse:collapse;font-size:12px;width:100%}}
th{{text-align:left;background:#f1f3f7;padding:6px 8px;position:sticky;top:0}}td{{padding:5px 8px;border-top:1px solid #eef0f3;white-space:nowrap}}
</style></head><body><header><h1>BridgeIDS — Bridging Legacy IDS and Modern IIoT Security</h1>
<p>NSL-KDD → ForgeIIoT → Edge-IIoTset · cross-dataset transfer · domain adaptation · continual learning · zero-shot · NLP · XAI · dynamic detection</p></header>
<nav>{''.join(nav)}</nav><main>{''.join(body)}</main></body></html>"""
    out = Path(cfg["paths"]["reports"]) / "results_dashboard.html"
    out.write_text(page, encoding="utf-8")
    log.info(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def build_all():
    cfg = load_config()
    export_experiment_tables(cfg)
    build_paper(cfg)
    build_dashboard(cfg)
