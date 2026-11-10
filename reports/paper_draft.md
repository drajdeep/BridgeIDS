# Bridging Legacy IDS and Modern IIoT Security: Cross-Dataset Transfer, Domain Adaptation, Zero-Shot Detection and Explainable Dynamic Detection over NSL-KDD → ForgeIIoT → Edge-IIoTset

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
only 0.585 on Edge-IIoTset, whereas a ForgeIIoT-trained model reaches 0.853 in the canonical space and
0.842 in the packet space. Drift analysis agrees: proxy-A-distance is 1.436
for Forge–Edge vs 1.969 for NSL–Edge, and concept shift |ΔP(y|x)| is
0.035 vs 0.386. The flow-to-text
interlingua raises ForgeIIoT→NSL-KDD AUC from 0.340 to
0.789. EWC + experience replay gives the best continual-learning trade-off,
and a benign-calibrated hybrid detector detects never-seen Edge injection attacks at 99.0% (benign false-positive
rate 7.3%) using only benign target traffic. SHAP rankings agree between ForgeIIoT and Edge-IIoTset (Spearman ρ = 0.619)
but not between NSL-KDD and ForgeIIoT (ρ = -0.028), so explanations expose the same shift.
Finally, we move from static to **dynamic** detection: a streaming engine that replays dataset records as network
packets, detects drift and adapts from a small analyst-label budget raises post-shift MCC from
0.618 to 0.864.
We also show that capture-order identifiers inflate ForgeIIoT fine-grained accuracy from 69.4% to 99.9%.

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

| dataset      |   records (used) |   native features (leak-free) |   CBS features |   PBS features |   fine labels |   super-classes |   attack % |   TCP % |   UDP % |   ICMP % |   ARP % |   MQTT % |   HTTP % |
|:-------------|-----------------:|------------------------------:|---------------:|---------------:|--------------:|----------------:|-----------:|--------:|--------:|---------:|--------:|---------:|---------:|
| NSL-KDD      |           148517 |                            41 |             27 |        nan     |            40 |               6 |     48.120 |  81.900 |  11.900 |    6.300 |   0.000 |    0.000 |   32.800 |
| ForgeIIoT    |           170573 |                            83 |             27 |         51.000 |            17 |               6 |     88.270 |  74.500 |   7.700 |    9.600 |   1.200 |   43.600 |    1.000 |
| Edge-IIoTset |           157800 |                            78 |             27 |         51.000 |            15 |               7 |     84.600 |  86.800 |   0.300 |   10.200 |   1.000 |   12.700 |   48.200 |

All fine labels are mapped to NORMAL, DOS, RECON, MITM_SPOOF, UNAUTH_ACCESS, INJECTION, MALWARE_PRIVESC
(`results/tables/e01_attack_taxonomy.csv`, relationship matrix in `e01_attack_relationship_matrix.csv`).
NSL-KDD fine labels come from KDDTrain+/KDDTest+.txt (the ARFF files only carry normal/anomaly).

**Leakage audit.** The earlier notebook reported 100 % accuracy on ForgeIIoT; SHAP ranked `frame.number`,
`frame.time_epoch` and `frame.time_relative` highest. Because each attack was captured as a separate file, these
fields identify the capture rather than the behaviour:

| dataset      | setting                                       |   binary_f1 |   fine_macro_f1 |   fine_accuracy | top3_features                                     |
|:-------------|:----------------------------------------------|------------:|----------------:|----------------:|:--------------------------------------------------|
| ForgeIIoT    | leak-free, random split                       |       0.965 |           0.660 |           0.694 | tcp.time_delta, pbs_log_tcp_seq, pbs_log_tcp_ack  |
| ForgeIIoT    | + frame.number/epoch/tcp.stream, random split |       1.000 |           0.999 |           0.999 | _ts, _frame_no, _tcp_stream                       |
| ForgeIIoT    | leak-free, capture-temporal split             |       0.966 |           0.600 |           0.675 | tcp.time_delta, pbs_log_tcp_seq, pbs_log_tcp_ack  |
| ForgeIIoT    | + identifiers, capture-temporal split         |       0.946 |           0.646 |           0.706 | _ts, _frame_no, _tcp_stream                       |
| Edge-IIoTset | leak-free                                     |       1.000 |           0.882 |           0.908 | pbs_log_tcp_ack, pbs_log_tcp_seq, tcp.flags       |
| Edge-IIoTset | + IP / frame.time / checksums                 |       0.999 |           0.893 |           0.904 | enc__frame_time, enc_ip.dst_host, enc_ip.src_host |

With identifiers, fine-grained accuracy jumps to 99.9% (leak-free: 69.4%). Under a capture-temporal
split the gain mostly vanishes (70.6%). We also observed that ForgeIIoT labels are assigned per capture file,
so background MQTT traffic inside an attack capture is labelled as attack. This is label noise that caps per-packet
accuracy and explains why MITM/Sybil families are hardest. All experiments use leak-free features.

## 4 Method

**4.1 Three-tier alignment.** (a) *CBS* (27 features): protocol,
service class (HTTP, MQTT, DNS, FTP, mail, remote shell, Modbus, ICMP echo, well-known/registered/ephemeral port),
TCP-flag semantics (SYN, ACK, FIN, RST, PSH, half-open), log payload volume, land, application layer. NSL connection
flags (SF, S0, REJ, RSTO…) are translated into the same flag semantics. (b) *PBS*
(51 features): CBS + TCP length/seq/ack, ARP opcodes,
HTTP method/length/response and MQTT message types/lengths, shared by the two tshark exports. (c) *Native* leak-free
feature sets. Identifiers (IP/MAC, checksums, raw ports, stream indices, frame numbers, timestamps) are never features.

**4.2 Language interlingua.** Each record becomes a sentence, e.g. "tcp traffic to registered high port, flags ack, no payload"; "tcp traffic to http web service port, flags ack fin, no payload, http exchange".
Sentences are encoded with Sentence-BERT when available, otherwise with an offline TF-IDF + LSA domain encoder
(used here: Local TF-IDF + LSA (offline); 279 unique sentences across all datasets).

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

| dataset      | model         |    f1 |   roc_auc |   mcc |
|:-------------|:--------------|------:|----------:|------:|
| Edge-IIoTset | Random Forest | 1.000 |     1.000 | 0.997 |
| ForgeIIoT    | XGBoost       | 0.966 |     0.969 | 0.673 |
| NSL-KDD      | Decision Tree | 0.850 |     0.859 | 0.722 |

Full table: `e02_baselines_binary.csv`. Edge-IIoTset is almost perfectly separable in-domain. NSL-KDD's official
test set remains hard (unseen attack types).

| setting             | train        | test         | classifier        |   accuracy |   attack_f1 |   roc_auc |
|:--------------------|:-------------|:-------------|:------------------|-----------:|------------:|----------:|
| in-domain (native)  | NSL-KDD      | NSL-KDD      | Weka J48          |      0.791 |       0.788 |     0.821 |
| in-domain (native)  | ForgeIIoT    | ForgeIIoT    | Weka J48          |      0.936 |       0.964 |     0.940 |
| in-domain (native)  | Edge-IIoTset | Edge-IIoTset | Weka J48          |      0.998 |       0.999 |     0.996 |
| cross-dataset (CBS) | NSL-KDD      | Edge-IIoTset | Weka J48          |      0.408 |       0.560 |     0.328 |
| cross-dataset (CBS) | ForgeIIoT    | Edge-IIoTset | Weka J48          |      0.847 |       0.917 |     0.920 |
| cross-dataset (CBS) | NSL-KDD      | ForgeIIoT    | Weka J48          |      0.737 |       0.838 |     0.629 |
| in-domain (native)  | NSL-KDD      | NSL-KDD      | Weka RandomForest |      0.776 |       0.767 |     0.965 |
| in-domain (native)  | ForgeIIoT    | ForgeIIoT    | Weka RandomForest |      0.929 |       0.960 |     0.956 |
| in-domain (native)  | Edge-IIoTset | Edge-IIoTset | Weka RandomForest |      0.999 |       0.999 |     1.000 |
| cross-dataset (CBS) | NSL-KDD      | Edge-IIoTset | Weka RandomForest |      0.416 |       0.569 |     0.379 |
| cross-dataset (CBS) | ForgeIIoT    | Edge-IIoTset | Weka RandomForest |      0.847 |       0.917 |     0.737 |
| cross-dataset (CBS) | NSL-KDD      | ForgeIIoT    | Weka RandomForest |      0.730 |       0.833 |     0.780 |
| in-domain (native)  | NSL-KDD      | NSL-KDD      | Weka AdaBoostM1   |      0.788 |       0.785 |     0.939 |
| in-domain (native)  | ForgeIIoT    | ForgeIIoT    | Weka AdaBoostM1   |      0.901 |       0.947 |     0.941 |
| in-domain (native)  | Edge-IIoTset | Edge-IIoTset | Weka AdaBoostM1   |      0.997 |       0.998 |     0.999 |
| cross-dataset (CBS) | NSL-KDD      | Edge-IIoTset | Weka AdaBoostM1   |      0.380 |       0.546 |     0.323 |
| cross-dataset (CBS) | ForgeIIoT    | Edge-IIoTset | Weka AdaBoostM1   |      0.847 |       0.917 |     0.900 |
| cross-dataset (CBS) | NSL-KDD      | ForgeIIoT    | Weka AdaBoostM1   |      0.736 |       0.838 |     0.707 |

### 5.2 RQ1/RQ2 — Cross-dataset transfer

| space   | source                   | target       |   in_domain_auc |   cross_auc |   auc_drop |   cross_f1 |
|:--------|:-------------------------|:-------------|----------------:|------------:|-----------:|-----------:|
| CBS     | NSL-KDD                  | ForgeIIoT    |           0.877 |       0.778 |      0.099 |      0.759 |
| CBS     | NSL-KDD                  | Edge-IIoTset |           0.998 |       0.585 |      0.413 |      0.523 |
| CBS     | ForgeIIoT                | NSL-KDD      |           0.951 |       0.340 |      0.612 |      0.725 |
| CBS     | ForgeIIoT                | Edge-IIoTset |           0.998 |       0.853 |      0.145 |      0.917 |
| CBS     | Edge-IIoTset             | NSL-KDD      |           0.951 |       0.253 |      0.698 |      0.634 |
| CBS     | Edge-IIoTset             | ForgeIIoT    |           0.877 |       0.846 |      0.030 |      0.644 |
| CBS     | ForgeIIoT + Edge-IIoTset | NSL-KDD      |           0.951 |       0.330 |      0.621 |      0.693 |
| CBS     | NSL-KDD + Edge-IIoTset   | ForgeIIoT    |           0.877 |       0.853 |      0.024 |      0.705 |
| CBS     | NSL-KDD + ForgeIIoT      | Edge-IIoTset |           0.998 |       0.779 |      0.219 |      0.917 |
| PBS     | ForgeIIoT                | Edge-IIoTset |           1.000 |       0.842 |      0.158 |      0.950 |
| PBS     | Edge-IIoTset             | ForgeIIoT    |           0.945 |       0.816 |      0.129 |      0.696 |

**RQ1: no.** The NSL-KDD model is near chance on Edge-IIoTset (AUC 0.585). Edge→NSL is *inverted* (AUC 0.253),
which signals concept drift rather than covariate shift alone. **RQ2: yes.** ForgeIIoT transfers to Edge-IIoTset
(CBS AUC 0.853, PBS AUC 0.842), and the drift metrics (§5.7) place ForgeIIoT much closer to Edge-IIoTset than
to NSL-KDD. Weka J48 independently confirms the pattern.

### 5.3 RQ2 — Transfer with few labels (target = Edge-IIoTset)

| source                    | method          |   frac |   n_target |    f1 |   auc |    mcc |
|:--------------------------|:----------------|-------:|-----------:|------:|------:|-------:|
| ForgeIIoT (PBS)           | MLP source-only |  0.000 |      0.000 | 0.926 | 0.766 |  0.321 |
| ForgeIIoT (PBS)           | MLP-finetune    |  0.000 |     15.000 | 0.894 | 0.858 |  0.558 |
| ForgeIIoT (PBS)           | MLP-finetune    |  0.001 |     60.000 | 0.958 | 0.955 |  0.669 |
| ForgeIIoT (PBS)           | MLP-finetune    |  0.010 |    604.000 | 0.993 | 0.996 |  0.950 |
| ForgeIIoT (PBS)           | MLP-scratch     |  0.000 |     15.000 | 0.923 | 0.873 |  0.606 |
| ForgeIIoT (PBS)           | MLP-scratch     |  0.001 |     60.000 | 0.972 | 0.933 |  0.799 |
| ForgeIIoT (PBS)           | MLP-scratch     |  0.010 |    604.000 | 0.994 | 0.996 |  0.960 |
| ForgeIIoT (PBS)           | continued-GBM   |  0.000 |     15.000 | 0.937 | 0.867 |  0.655 |
| ForgeIIoT (PBS)           | continued-GBM   |  0.001 |     60.000 | 0.989 | 0.945 |  0.924 |
| ForgeIIoT (PBS)           | continued-GBM   |  0.010 |    604.000 | 0.998 | 0.996 |  0.988 |
| ForgeIIoT (PBS)           | pooled          |  0.000 |     15.000 | 0.960 | 0.850 |  0.703 |
| ForgeIIoT (PBS)           | pooled          |  0.001 |     60.000 | 0.964 | 0.930 |  0.739 |
| ForgeIIoT (PBS)           | pooled          |  0.010 |    604.000 | 0.987 | 0.996 |  0.915 |
| ForgeIIoT (PBS)           | source-only     |  0.000 |      0.000 | 0.950 | 0.842 |  0.612 |
| ForgeIIoT (PBS)           | target-only     |  0.000 |     15.000 | 0.978 | 0.899 |  0.870 |
| ForgeIIoT (PBS)           | target-only     |  0.001 |     60.000 | 0.989 | 0.986 |  0.928 |
| ForgeIIoT (PBS)           | target-only     |  0.010 |    604.000 | 0.998 | 0.997 |  0.985 |
| NSL-KDD (CBS)             | MLP source-only |  0.000 |      0.000 | 0.541 | 0.132 | -0.393 |
| NSL-KDD (CBS)             | MLP-finetune    |  0.000 |     15.000 | 0.958 | 0.920 |  0.767 |
| NSL-KDD (CBS)             | MLP-finetune    |  0.001 |     60.000 | 0.987 | 0.949 |  0.914 |
| NSL-KDD (CBS)             | MLP-finetune    |  0.010 |    604.000 | 0.988 | 0.991 |  0.917 |
| NSL-KDD (CBS)             | MLP-scratch     |  0.000 |     15.000 | 0.949 | 0.933 |  0.754 |
| NSL-KDD (CBS)             | MLP-scratch     |  0.001 |     60.000 | 0.979 | 0.968 |  0.855 |
| NSL-KDD (CBS)             | MLP-scratch     |  0.010 |    604.000 | 0.991 | 0.990 |  0.940 |
| NSL-KDD (CBS)             | continued-GBM   |  0.000 |     15.000 | 0.954 | 0.931 |  0.709 |
| NSL-KDD (CBS)             | continued-GBM   |  0.001 |     60.000 | 0.977 | 0.947 |  0.855 |
| NSL-KDD (CBS)             | continued-GBM   |  0.010 |    604.000 | 0.989 | 0.985 |  0.925 |
| NSL-KDD (CBS)             | pooled          |  0.000 |     15.000 | 0.809 | 0.790 |  0.363 |
| NSL-KDD (CBS)             | pooled          |  0.001 |     60.000 | 0.953 | 0.929 |  0.718 |
| NSL-KDD (CBS)             | pooled          |  0.010 |    604.000 | 0.992 | 0.992 |  0.948 |
| NSL-KDD (CBS)             | source-only     |  0.000 |      0.000 | 0.523 | 0.585 |  0.045 |
| NSL-KDD (CBS)             | target-only     |  0.000 |     15.000 | 0.982 | 0.913 |  0.882 |
| NSL-KDD (CBS)             | target-only     |  0.001 |     60.000 | 0.981 | 0.962 |  0.880 |
| NSL-KDD (CBS)             | target-only     |  0.010 |    604.000 | 0.994 | 0.994 |  0.960 |
| NSL-KDD + ForgeIIoT (CBS) | MLP source-only |  0.000 |      0.000 | 0.878 | 0.661 | -0.112 |
| NSL-KDD + ForgeIIoT (CBS) | MLP-finetune    |  0.000 |     15.000 | 0.961 | 0.896 |  0.771 |
| NSL-KDD + ForgeIIoT (CBS) | MLP-finetune    |  0.001 |     60.000 | 0.977 | 0.943 |  0.842 |
| NSL-KDD + ForgeIIoT (CBS) | MLP-finetune    |  0.010 |    604.000 | 0.986 | 0.993 |  0.907 |
| NSL-KDD + ForgeIIoT (CBS) | MLP-scratch     |  0.000 |     15.000 | 0.949 | 0.935 |  0.747 |
| NSL-KDD + ForgeIIoT (CBS) | MLP-scratch     |  0.001 |     60.000 | 0.979 | 0.969 |  0.854 |

Edge-IIoTset is so separable that a few dozen labels already give a strong target-only model. With ~60 labelled target packets: pre-training on ForgeIIoT (PBS) does not help the MLP (MCC 0.669 vs 0.799 from scratch); warm-started GBM 0.924 vs target-only 0.928; pre-training on NSL-KDD (CBS) helps the MLP (MCC 0.914 vs 0.855 from scratch); warm-started GBM 0.855 vs target-only 0.880; pre-training on NSL-KDD + ForgeIIoT (CBS) does not help the MLP (MCC 0.842 vs 0.854 from scratch); warm-started GBM 0.895 vs target-only 0.880.
Naively pooling NSL-KDD with target data causes negative transfer (see the `pooled` rows). The value of a source is
therefore largest in the *zero-label* regime, which is what §5.4 and §5.8 address.

### 5.4 RQ3 — Unsupervised domain adaptation

| pair                               | method                     |   roc_auc |    mcc |    f1 |   auc_gain_vs_noadapt |
|:-----------------------------------|:---------------------------|----------:|-------:|------:|----------------------:|
| NSL-KDD → Edge-IIoTset             | No adaptation (LightGBM)   |     0.487 |  0.053 | 0.550 |                 0.000 |
| NSL-KDD → Edge-IIoTset             | DomainNorm                 |     0.547 | -0.127 | 0.613 |                 0.060 |
| NSL-KDD → Edge-IIoTset             | CORAL                      |     0.490 |  0.012 | 0.792 |                 0.003 |
| NSL-KDD → Edge-IIoTset             | TCA (+LightGBM)            |     0.595 |  0.093 | 0.534 |                 0.108 |
| NSL-KDD → Edge-IIoTset             | TCA (+LogReg)              |     0.049 | -0.538 | 0.327 |                -0.439 |
| NSL-KDD → Edge-IIoTset             | Importance weighting       |     0.556 |  0.131 | 0.632 |                 0.069 |
| NSL-KDD → Edge-IIoTset             | DomainNorm + self-training |     0.498 | -0.224 | 0.491 |                 0.011 |
| NSL-KDD → Edge-IIoTset             | MLP (no adaptation)        |     0.471 | -0.138 | 0.640 |                -0.017 |
| NSL-KDD → Edge-IIoTset             | DANN                       |     0.381 | -0.067 | 0.633 |                -0.106 |
| NSL-KDD → ForgeIIoT                | No adaptation (LightGBM)   |     0.753 |  0.380 | 0.759 |                 0.000 |
| NSL-KDD → ForgeIIoT                | DomainNorm                 |     0.804 |  0.197 | 0.838 |                 0.051 |
| NSL-KDD → ForgeIIoT                | CORAL                      |     0.807 |  0.384 | 0.759 |                 0.055 |
| NSL-KDD → ForgeIIoT                | TCA (+LightGBM)            |     0.798 |  0.387 | 0.763 |                 0.046 |
| NSL-KDD → ForgeIIoT                | TCA (+LogReg)              |     0.206 | -0.286 | 0.655 |                -0.546 |
| NSL-KDD → ForgeIIoT                | Importance weighting       |     0.804 |  0.395 | 0.771 |                 0.052 |
| NSL-KDD → ForgeIIoT                | DomainNorm + self-training |     0.787 |  0.197 | 0.837 |                 0.035 |
| NSL-KDD → ForgeIIoT                | MLP (no adaptation)        |     0.816 |  0.402 | 0.777 |                 0.063 |
| NSL-KDD → ForgeIIoT                | DANN                       |     0.726 |  0.319 | 0.666 |                -0.027 |
| NSL-KDD + ForgeIIoT → Edge-IIoTset | No adaptation (LightGBM)   |     0.781 | -0.086 | 0.893 |                 0.000 |
| NSL-KDD + ForgeIIoT → Edge-IIoTset | DomainNorm                 |     0.656 |  0.053 | 0.894 |                -0.125 |
| NSL-KDD + ForgeIIoT → Edge-IIoTset | CORAL                      |     0.355 |  0.000 | 0.917 |                -0.426 |
| NSL-KDD + ForgeIIoT → Edge-IIoTset | TCA (+LightGBM)            |     0.502 | -0.027 | 0.915 |                -0.279 |
| NSL-KDD + ForgeIIoT → Edge-IIoTset | TCA (+LogReg)              |     0.721 |  0.044 | 0.864 |                -0.059 |
| NSL-KDD + ForgeIIoT → Edge-IIoTset | Importance weighting       |     0.638 |  0.060 | 0.896 |                -0.143 |
| NSL-KDD + ForgeIIoT → Edge-IIoTset | DomainNorm + self-training |     0.417 | -0.086 | 0.894 |                -0.363 |
| NSL-KDD + ForgeIIoT → Edge-IIoTset | MLP (no adaptation)        |     0.522 | -0.125 | 0.824 |                -0.259 |
| NSL-KDD + ForgeIIoT → Edge-IIoTset | DANN                       |     0.434 | -0.154 | 0.731 |                -0.347 |
| ForgeIIoT → Edge-IIoTset           | No adaptation (LightGBM)   |     0.846 |  0.613 | 0.950 |                 0.000 |
| ForgeIIoT → Edge-IIoTset           | DomainNorm                 |     0.883 |  0.612 | 0.950 |                 0.037 |
| ForgeIIoT → Edge-IIoTset           | CORAL                      |     0.878 |  0.000 | 0.917 |                 0.032 |
| ForgeIIoT → Edge-IIoTset           | TCA (+LightGBM)            |     0.716 |  0.430 | 0.933 |                -0.129 |
| ForgeIIoT → Edge-IIoTset           | TCA (+LogReg)              |     0.808 |  0.451 | 0.824 |                -0.038 |
| ForgeIIoT → Edge-IIoTset           | Importance weighting       |     0.874 |  0.612 | 0.950 |                 0.028 |
| ForgeIIoT → Edge-IIoTset           | DomainNorm + self-training |     0.618 |  0.000 | 0.917 |                -0.227 |
| ForgeIIoT → Edge-IIoTset           | MLP (no adaptation)        |     0.690 |  0.182 | 0.923 |                -0.155 |
| ForgeIIoT → Edge-IIoTset           | DANN                       |     0.723 |  0.140 | 0.907 |                -0.122 |
| Edge-IIoTset → ForgeIIoT           | No adaptation (LightGBM)   |     0.806 |  0.333 | 0.696 |                 0.000 |
| Edge-IIoTset → ForgeIIoT           | DomainNorm                 |     0.790 |  0.333 | 0.696 |                -0.016 |
| Edge-IIoTset → ForgeIIoT           | CORAL                      |     0.781 |  0.392 | 0.774 |                -0.025 |
| Edge-IIoTset → ForgeIIoT           | TCA (+LightGBM)            |     0.689 |  0.178 | 0.914 |                -0.117 |

**RQ3: partially.** Best method per pair: NSL-KDD → Edge-IIoTset: best = TCA (+LightGBM) (+0.108 AUC); NSL-KDD → ForgeIIoT: best = CORAL (+0.055 AUC); NSL-KDD + ForgeIIoT → Edge-IIoTset: best = TCA (+LogReg) (-0.059 AUC); ForgeIIoT → Edge-IIoTset: best = DomainNorm (+0.037 AUC); Edge-IIoTset → ForgeIIoT: best = DANN (+0.026 AUC). DA gives modest gains (about +0.03 to +0.06 AUC)
where the shift is mostly covariate (NSL→Forge, Forge↔Edge). For NSL→Edge, even the best method leaves the AUC at only
0.595. Every method *hurts* the multi-source NSL+Forge→Edge setting,
where NSL's conflicting concept dominates the aligned space. Aligning P(x) cannot fix a change of P(y|x), consistent with the concept-shift measurements
in §5.7. DANN is unstable across seeds and epochs (std in `e05_domain_adaptation.csv`).

### 5.5 Continual learning NSL → Forge → Edge

| order                              | strategy           |   avg_final |   avg_final_std |    BWT |   forgetting |
|:-----------------------------------|:-------------------|------------:|----------------:|-------:|-------------:|
| NSL-KDD → ForgeIIoT → Edge-IIoTset | Naive fine-tuning  |       0.744 |           0.006 | -0.034 |        0.127 |
| NSL-KDD → ForgeIIoT → Edge-IIoTset | EWC                |       0.761 |           0.014 | -0.005 |        0.098 |
| NSL-KDD → ForgeIIoT → Edge-IIoTset | Replay             |       0.840 |           0.015 |  0.123 |       -0.029 |
| NSL-KDD → ForgeIIoT → Edge-IIoTset | EWC + Replay       |       0.857 |           0.029 |  0.148 |       -0.050 |
| NSL-KDD → ForgeIIoT → Edge-IIoTset | LightGBM continued |       0.735 |           0.000 |  0.010 |        0.134 |
| NSL-KDD → ForgeIIoT → Edge-IIoTset | Joint (offline)    |       0.733 |           0.061 |  0.032 |        0.068 |
| Edge-IIoTset → ForgeIIoT → NSL-KDD | Naive fine-tuning  |       0.680 |           0.055 | -0.113 |        0.245 |
| Edge-IIoTset → ForgeIIoT → NSL-KDD | EWC                |       0.746 |           0.007 | -0.012 |        0.143 |
| Edge-IIoTset → ForgeIIoT → NSL-KDD | Replay             |       0.711 |           0.061 | -0.066 |        0.197 |
| Edge-IIoTset → ForgeIIoT → NSL-KDD | EWC + Replay       |       0.756 |           0.058 |  0.004 |        0.129 |
| Edge-IIoTset → ForgeIIoT → NSL-KDD | LightGBM continued |       0.452 |           0.000 | -0.238 |        0.371 |
| Edge-IIoTset → ForgeIIoT → NSL-KDD | Joint (offline)    |       0.688 |           0.006 | -0.186 |        0.245 |

EWC + replay reaches 0.857 average final balanced accuracy in the legacy→modern order
(naive fine-tuning: 0.744), with positive backward transfer (BWT +0.148).
In the reverse order it is again the best strategy, but BWT is only +0.004. Under conflicting concepts, joint
offline training is *not* an upper bound (negative interference), and tree models with continued boosting forget the most.
Without replay, the model collapses to predicting "attack" for everything after the ForgeIIoT task (balanced accuracy 0.50 on all
tasks, `e06_continual_matrices.png`), because Forge is 88 % attack. Replay prevents this collapse.

### 5.6 RQ5 — Zero-shot / unseen attacks

| dataset      |   supervised@1%FPR |   one-class IF@1%FPR |   hybrid@1%FPR |
|:-------------|-------------------:|---------------------:|---------------:|
| Edge-IIoTset |              1.000 |                0.536 |          1.000 |
| ForgeIIoT    |              0.796 |                0.049 |          0.790 |

| edge_family           | super           | family_seen_in_sources   |   source-supervised@5%FPR |   target one-class@5%FPR |   hybrid@5%FPR |
|:----------------------|:----------------|:-------------------------|--------------------------:|-------------------------:|---------------:|
| Backdoor              | MALWARE_PRIVESC | True                     |                     0.200 |                    0.998 |          0.998 |
| DDoS_HTTP             | DOS             | True                     |                     0.584 |                    1.000 |          1.000 |
| DDoS_ICMP             | DOS             | True                     |                     0.000 |                    1.000 |          1.000 |
| DDoS_TCP              | DOS             | True                     |                     0.000 |                    1.000 |          1.000 |
| DDoS_UDP              | DOS             | True                     |                     0.912 |                    1.000 |          1.000 |
| Fingerprinting        | RECON           | True                     |                     0.343 |                    1.000 |          1.000 |
| MITM                  | MITM_SPOOF      | True                     |                     0.637 |                    1.000 |          1.000 |
| Normal                | NORMAL          | False                    |                     0.084 |                    0.052 |          0.073 |
| Password              | UNAUTH_ACCESS   | True                     |                     0.493 |                    1.000 |          1.000 |
| Port_Scanning         | RECON           | True                     |                     0.045 |                    1.000 |          1.000 |
| Ransomware            | MALWARE_PRIVESC | True                     |                     0.580 |                    0.588 |          1.000 |
| SQL_injection         | INJECTION       | False                    |                     0.455 |                    1.000 |          1.000 |
| Uploading             | INJECTION       | False                    |                     0.691 |                    0.940 |          0.969 |
| Vulnerability_scanner | RECON           | True                     |                     0.113 |                    1.000 |          1.000 |
| XSS                   | INJECTION       | False                    |                     0.533 |                    1.000 |          1.000 |

| method             | unseen_classes                             |   top1_accuracy |   chance |   acc_DDoS_HTTP |   acc_XSS |   acc_Fingerprinting |   acc_Ransomware |
|:-------------------|:-------------------------------------------|----------------:|---------:|----------------:|----------:|---------------------:|-----------------:|
| DAP (attribute)    | DDoS_HTTP, XSS, Fingerprinting, Ransomware |           0.584 |    0.250 |           0.068 |     0.927 |                0.000 |            0.853 |
| Text-embedding ZSL | DDoS_HTTP, XSS, Fingerprinting, Ransomware |           0.359 |    0.250 |           0.124 |     0.000 |                0.077 |            0.956 |

**RQ5: yes, for detection; partially, for naming the family.** In-domain, unseen Edge families are still detected
(they differ strongly from benign MQTT telemetry). Cross-dataset, the hybrid detector, which uses only benign target
traffic, detects the never-seen INJECTION families at 99.0%, at a benign false-positive rate of 7.3%
(the union of two 5 %-FPR detectors). The source-only supervised model, by contrast, misses most target attacks. Semantic ZSL names unseen classes above chance:
DAP 58.4%, text ZSL 35.9%, chance 25.0%.

### 5.7 RQ4 — Explainability and drift

| dataset_a   | dataset_b    |   spearman |   kendall |   top10_jaccard |
|:------------|:-------------|-----------:|----------:|----------------:|
| NSL-KDD     | ForgeIIoT    |     -0.028 |    -0.032 |           0.111 |
| NSL-KDD     | Edge-IIoTset |      0.259 |     0.174 |           0.333 |
| ForgeIIoT   | Edge-IIoTset |      0.619 |     0.473 |           0.538 |

| pair                      |   mean_JS |   MMD_rbf |   proxy_A_distance |   concept_disagreement |   mean_conditional_label_shift |   attack_label_JS |
|:--------------------------|----------:|----------:|-------------------:|-----------------------:|-------------------------------:|------------------:|
| NSL-KDD vs ForgeIIoT      |     0.219 |     0.267 |              1.955 |                  0.399 |                          0.418 |             0.494 |
| NSL-KDD vs Edge-IIoTset   |     0.178 |     0.213 |              1.969 |                  0.538 |                          0.386 |             0.512 |
| ForgeIIoT vs Edge-IIoTset |     0.118 |     0.094 |              1.436 |                  0.288 |                          0.035 |             0.429 |

**RQ4: yes.** SHAP rankings in the shared space agree for Forge–Edge (ρ = 0.619) but not for
NSL–Forge (ρ = -0.028). LIME and SHAP top-6 features overlap with Jaccard
0.319. When a Forge-trained model is explained on Edge traffic, the largest
importance shifts are log_tcp_len (+0.038), log_tcp_seq (-0.032), log_bytes (-0.030) (`e09_shap_shift_forge_model_on_edge.csv`). Packet-size and
TCP-sequence features change role between the two plants. Example incident narrative:

> Record 13222: classified as ATTACK with confidence 100%. Main drivers: absence of MQTT broker port (towards attack, +2.85); tcp.flags = 0 (towards attack, +1.76); absence of HTTP port (towards normal, -1.23); presence of registered high port (towards attack, +0.81).

### 5.8 RQ6 — From static to dynamic detection

| config                        | phase                 |    f1 |   mcc |   fpr |   recall |   drift_detection_delay_pkts |   feedback_labels_used |   throughput_pkt_s |
|:------------------------------|:----------------------|------:|------:|------:|---------:|-----------------------------:|-----------------------:|-------------------:|
| Static supervised             | phase 2: Edge-IIoTset | 0.950 | 0.618 | 0.578 |    1.000 |                         1128 |                      0 |           2904.925 |
| Static supervised             | phase 2, last 5k      | 0.951 | 0.618 | 0.579 |    1.000 |                         1128 |                      0 |           2904.925 |
| Static hybrid                 | phase 2: Edge-IIoTset | 0.950 | 0.618 | 0.578 |    1.000 |                         1128 |                      0 |           2897.183 |
| Static hybrid                 | phase 2, last 5k      | 0.951 | 0.618 | 0.579 |    1.000 |                         1128 |                      0 |           2897.183 |
| Dynamic hybrid (2% feedback)  | phase 2: Edge-IIoTset | 0.958 | 0.687 | 0.486 |    1.000 |                         1128 |                    598 |           3145.803 |
| Dynamic hybrid (2% feedback)  | phase 2, last 5k      | 0.971 | 0.796 | 0.329 |    1.000 |                         1128 |                    598 |           3145.803 |
| Dynamic hybrid (10% feedback) | phase 2: Edge-IIoTset | 0.980 | 0.864 | 0.220 |    1.000 |                         1128 |                   2896 |           2822.596 |
| Dynamic hybrid (10% feedback) | phase 2, last 5k      | 0.984 | 0.893 | 0.170 |    0.999 |                         1128 |                   2896 |           2822.596 |

A Forge-trained deployment that keeps running after the plant traffic changes to Edge-IIoT keeps a high recall but
suffers a very high false-positive rate. With 10 % delayed analyst feedback, MCC on the shifted traffic rises from
0.618 to 0.864
(0.893 over the last 5k packets). The PSI detector flags
the shift 1128 packets after it
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
42). All tables are in `results/tables`, figures in `results/figures` and the Power BI CSVs in `powerbi/data`.
