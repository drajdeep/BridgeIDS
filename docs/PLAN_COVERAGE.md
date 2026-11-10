# Plan coverage — audit of the hand-over and what BridgeIDS adds

## 1. What was handed over (audit of the three notebooks + data)

| Hand-over item | What it does | Findings |
|---|---|---|
| `NSL_KDD_XAI_NIDS_Kernel_Weka_Colab_Stronger_Accuracy.ipynb` | NSL-KDD binary: RF, XGBoost, tuned XGBoost, RBF-SVM, Weka RF/J48, SHAP | Solid baseline (test accuracy ≈ 0.80, AUC ≈ 0.97). It uses the ARFF files, which only contain **normal/anomaly**, so the attack-category mapping (DoS/Probe/R2L/U2R) from the plan was impossible. Fixed by loading `KDDTrain+.txt/KDDTest+.txt`, which carry the attack names (same rows). |
| `ML_EdgeIIoT_XAI_NIDS_Colab_Fast_Optimized.ipynb` | Edge-IIoT binary: RF, XGBoost, SVM, soft/weighted voting, SHAP | Correct in-domain work (≈ 0.999). Binary only, no multi-class / taxonomy, no cross-dataset. |
| `Project_Network_Intrusion_Detection_Data.ipynb` | ForgeIIoT sample (30 k rows): stacking RF+XGB+DNN, SHAP, SBERT branch, HTML dashboard, Excel | **Reported 100 % accuracy because of label leakage.** SHAP's top features were `frame.number`, `frame.time_epoch`, `frame.time_relative`, which identify the capture file, not the behaviour. The NSL column mapping was broken (sentences like "7 protocol, 0 service"), drift and EDA were placeholders, and the dashboard was static. |
| `Forge_IIOT_Pro_Dataset.zip` | 17 raw capture CSVs (~3.3 M packets) + a preprocessed CSV | Rebuilt from the raw captures with a stratified sample per file. Labels are per capture file, so background MQTT inside an attack capture is labelled as attack (label noise, documented). |
| `ML-EdgeIIoT-dataset.csv`, `KDD*.arff` | data | used as-is; NSL `.txt` added |

## 2. Plan items → implementation → evidence

Legend: ✅ done here · ♻️ redone / fixed from hand-over · ➕ beyond the plan

### Team 1 — ML / Dataset / Engineering

| Plan item | Status | Code | Results |
|---|---|---|---|
| 1 Dataset study (features, attack categories, statistics table) | ✅ | `experiments/e01_dataset_study.py`, `data/loaders.py` | `e01_dataset_statistics`, `e01_feature_family_comparison`, `e01_label_distribution`, `e01_superclass_*.png` |
| 2 Feature alignment / mapping matrix | ✅ ➕ three tiers (CBS / PBS / native) | `features/views.py` | `e01_feature_family_comparison`, docs/ARCHITECTURE.md |
| 3 Baselines DT, RF, AdaBoost, XGBoost, LightGBM, MLP (Weka + Python) | ✅ ♻️ leak-free | `models/zoo.py`, `e02_baselines.py`, `e12_artifacts.py` (Weka J48/RF/AdaBoostM1) | `e02_baselines_binary`, `e02_baselines_superclass`, `e12_weka_results` |
| ➕ Leakage audit | ➕ | `e02_baselines.py` | `e02_leakage_audit`, `e02_leakage_audit.png` |
| 4 Cross-dataset testing (train NSL → test Forge + Edge, domain-shift report) | ✅ ➕ full 3×3 matrix, two-source pooling, bootstrap CIs | `e03_cross_dataset.py` | `e03_cross_dataset_binary`, `e03_domain_shift_report`, `e03_*_matrix_*.png` |
| 5 Transfer learning 1/5/10 % Forge fine-tune → test Edge | ✅ ➕ also 0.02 % / 0.1 %, 3 seeds, 6 methods incl. warm-started GBM and MLP fine-tune | `e04_transfer.py` | `e04_transfer_learning_results`, `e04_transfer_curves.png` |
| 6 Domain adaptation CORAL, TCA, DANN | ✅ ➕ DomainNorm, importance weighting, self-training | `adaptation/methods.py`, `models/npnet.py` (DANN + GRL) | `e05_domain_adaptation`, `e05_domain_adaptation_auc.png` |
| 7 Continual learning NSL → Forge → Edge, retention, forgetting | ✅ ➕ EWC (exact Fisher), replay, EWC+replay, GBM-continued, reverse order, BWT/FWT | `e06_continual.py` | `e06_continual_learning`, `e06_continual_matrices.png` |

### Team 2 — NLP / Explainability / Research

| Plan item | Status | Code | Results |
|---|---|---|---|
| 1 Literature review 2023–2026, related-work table (30–50) | ✅ 44 works + foundations + gap analysis | — | `reports/related_work.md` |
| 2 Attack taxonomy mapping / relationship matrix | ✅ 7 super-classes, all fine labels | `data/taxonomy.py` | `e01_attack_taxonomy`, `e01_attack_relationship_matrix` |
| 3 NLP-based representation (flows → text, BERT/SBERT) | ✅ ♻️ ➕ dataset-agnostic *interlingua* | `nlp/flow2text.py`, `nlp/encoders.py` (SBERT or offline LSA) | `e07_nlp_cross_dataset`, `e07_nlp_in_domain`, `e07_text_embedding_map.png` |
| 4 Zero-shot detection (train NSL+Forge, test unseen Edge) | ✅ ➕ leave-one-family-out, hybrid zero-day detector, DAP + text ZSL | `e08_zeroshot.py`, `models/hybrid.py` | `e08_zeroshot_*`, `e08_semantic_zsl` |
| 5 Explainable AI SHAP + LIME, importance change | ✅ ➕ explanation-drift metrics, incident narratives | `xai/explain.py`, `e09_xai.py` | `e09_*` |
| 6 Dataset drift (feature/concept/attack; KL, JS, Wasserstein) | ✅ ➕ PSI, MMD, proxy-A-distance, conditional label shift | `drift.py`, `e10_drift.py` | `e10_drift_summary`, `e10_feature_drift` |
| 7 Visualization — Power BI (attack distribution, real-time, importance, drift, protocol, MQTT) | ✅ data + DAX + theme + guide; ➕ live web dashboard, HTML results dashboard, streaming push | `reporting/powerbi.py`, `powerbi/`, `stream/static/dashboard.html`, `reporting/report.py` | `powerbi/data/*.csv`, `reports/results_dashboard.html` |
| Final deliverable: results + analysis + visualisations + paper draft | ✅ | `reporting/report.py` | `reports/paper_draft.md` |

### Requested in the hand-over message: static → dynamic detection

| Item | Code | Evidence |
|---|---|---|
| Records sent as network packets (UDP / TCP / HTTP / MQTT) | `stream/replayer.py` | `python -m bridgeids.run replay …` |
| Live detector: schema auto-detect, hybrid scoring, SHAP per alert, family | `stream/engine.py`, `stream/server.py` | `docs/img/dashboard_live.png` |
| Drift detection (PSI window, ADWIN) + adaptation (online challenger, Hedge, re-calibration) | `stream/engine.py` | `e11_stream_benchmark`, `e11_stream_static_vs_dynamic.png` |
| Scenarios: kill-chain campaign, domain shift, zero-day | `stream/replayer.py` | live tests |
| Persistence + BI: SQLite, Power BI push URL, CSV export | `stream/server.py`, `reporting/powerbi.py` | `results/stream/stream.db` |

## 3. Improvements to the original plan (and why)

1. **Leak-free evaluation first.** Identifiers are banned from features, and a capture-temporal split is added. Without this, every ForgeIIoT number is meaningless.
2. **Three-tier alignment instead of one mapping matrix.** NSL (connection) and IIoT (packet) data cannot share raw features, so the plan's "Feature A ↔ Feature A" matrix becomes CBS (semantic, all three), PBS (packet, Forge↔Edge) and native.
3. **Language as interlingua.** The plan's NLP branch becomes the main cross-dataset tool rather than an in-domain add-on, and it improves cross-dataset AUC.
4. **Hybrid zero-day detection.** Supervised models cannot flag what they never saw. A benign-ECDF-calibrated one-class component with a bounded FPR complements them.
5. **Dynamic detection (RQ6).** Streaming replay, drift detection and label-budgeted adaptation measure what the static RQs can only suggest.
6. **Drift metrics to choose the source.** Proxy-A-distance and concept shift predict which source will transfer before any training.
7. **Statistics.** MCC and balanced accuracy (attack prevalence is 85–88 % in the IIoT sets, which makes F1 misleading), bootstrap CIs, 3 seeds for stochastic methods.
8. **Torch-free numpy NN engine.** DANN, EWC and MLPs install in seconds on Windows/Linux laptops without CUDA.

## 4. Revised two-week schedule (if the teams continue)

| | Week 1 | Week 2 |
|---|---|---|
| Team 1 | Re-run `exp all` at full scale (raise caps in `configs/default.yaml`); add the DNN-EdgeIIoT file; tune DANN (reverse validation) | Window features from Forge timestamps; federated variant of the online challenger; edge benchmark (Raspberry Pi) |
| Team 2 | Verify the bibliography, add 2026 venues; SBERT/DistilBERT run with internet; human evaluation of narratives | Build the Power BI `.pbix` from the kit; write Sections 1–2 and 6 of the paper; prepare the artefact appendix |
