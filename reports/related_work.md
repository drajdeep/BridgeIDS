# Related Work Table (Team-2 deliverable 1)

Literature compiled on 10 Nov 2026. Each entry was checked against a public index page (IEEE Xplore/DataPort, arXiv, MDPI, DOAJ, publisher or institutional repository). The **Relation** column says how BridgeIDS uses the work, extends it or differs from it. Foundational pre-2023 references, needed for the methods, are listed separately at the end.

> Note for the authors: before submission, open each link and check page numbers, author order and the final venue. Preprints (arXiv) may since have appeared in journals.

## A. Cross-dataset generalisation and feature alignment

| # | Year | Authors | Title | Venue | Datasets | Relation to BridgeIDS | Link |
|---|---|---|---|---|---|---|---|
| 1 | 2024 | M. Cantone, C. Marrocco, A. Bria | On the Cross-Dataset Generalization of Machine Learning for Network Intrusion Detection | IEEE Access (doi 10.1109/ACCESS.2024.3472907) | CIC-IDS2017, CSE-CIC-IDS2018, LycoS | Reports near-random cross-dataset accuracy for flow datasets. BridgeIDS extends the study to **legacy → IIoT packet** data and shows that a bridge dataset restores transfer (E03). | https://arxiv.org/abs/2402.10974 |
| 2 | 2026 | (U. Nottingham et al.) | Exploring Robust Intrusion Detection: A Benchmark Study of Feature Transferability in IoT Botnet Attack Detection | arXiv 2602.23874 | IoT botnet sets | Shows feature-space design drives cross-domain robustness. Our CBS/PBS/text spaces are a concrete answer to this. | https://arxiv.org/abs/2602.23874 |
| 3 | 2026 | M. Chizari, Q. K. A. Mirza, A. Alam, H. Chizari | Closed-Set Heterogeneous Domain Adaptation for IoT Intrusion Detection: An Anchor-Based Benchmark Across Single- and Multi-Source Transfer | Sensors 26(11):3610 | **Edge-IIoTset (target)**, CICIDS2017, UNSW-NB15, **NSL-KDD** | Closest work. Like us, it finds DA "deployment-competitive only in some contexts". BridgeIDS adds a bridge dataset (ForgeIIoT), a language interlingua, continual learning and streaming. | https://eprints.glos.ac.uk/16365/ |
| 4 | 2021/22 | M. Sarhan, S. Layeghy, N. Moustafa, M. Portmann | Towards a Standard Feature Set for NIDS Datasets (NetFlow v2) | Mobile Netw. Appl. / arXiv 2101.11315 | NF-UNSW-NB15, NF-ToN-IoT, … | A common *flow* feature set. It cannot cover NSL-KDD vs packet-level IIoT, which motivates our semantic CBS tier. | https://arxiv.org/abs/2101.11315 |
| 5 | 2022 | G. Apruzzese et al. | The Cross-evaluation of Machine Learning-based NIDS (XeNIDS) | IEEE TNSM / arXiv 2203.04686 | CTU13, NB15, IDS18, DDOS19 | Cross-evaluation framework (foundational). We adopt the train-on-A/test-on-B protocol. | https://arxiv.org/abs/2203.04686 |

## B. Transfer learning and domain adaptation for IoT/IIoT IDS

| # | Year | Authors | Title | Venue | Datasets | Relation | Link |
|---|---|---|---|---|---|---|---|
| 6 | 2024 | (UTS group) | EMTD-SSC: An Enhanced Malicious Traffic Detection Model Using Transfer Learning Under Small Sample Conditions in IoT | IEEE Internet of Things Journal | Traditional IDS + small IoT set | Merges legacy and IoT domains with MK-MMD. We compare few-shot transfer budgets (E04) and show where pooling legacy data causes **negative transfer**. | https://opus.lib.uts.edu.au/handle/10453/180682 |
| 7 | 2024 | K. Li, W. Ma, H. Duan, H. Xie | Multi-source refined adversarial domain adaptation with transfer complementarity infusion for IoT intrusion detection under limited samples (MRADDA-TC) | Expert Systems with Applications 254 | Multi-source | Multi-source adversarial DA. We test NSL+Forge→Edge multi-source settings with DANN/CORAL/TCA. | https://searchworks.jioinstitute.edu.in/articles/asx__178885681 |
| 8 | 2025 | (Sunderland et al.) | A Domain Adaptive IoT Intrusion Detection Algorithm Based on AEC-GAT Feature Extraction and Joint Domain Adversary | IEEE Trans. Industrial Informatics (accepted Nov 2025) | Traditional NIDS → IoT | Same legacy→IoT motivation. We add an explicit bridge dataset and a drift/explanation analysis of *why* transfer fails. | https://sure.sunderland.ac.uk/id/eprint/19711/ |
| 9 | 2024 | (Future Internet authors) | A Transferable Deep Learning Framework for Improving the Accuracy of IoT Intrusion Detection | Future Internet (Feb 2024) | Several IoT sets | Addresses source-dataset selection. Our drift metrics (proxy-A-distance, concept shift) give a quantitative way to choose the source (E10). | https://doaj.org/article/736093598768455fb94641cbee3d7cc3 |
| 10 | 2025 | (TU/e) | Mind the Shift: A Study on Transfer Learning and Domain Adaptation in Vehicular Intrusion Detection | IEEE BigData 2025, pp. 4315–4324 | CarHacking → CIC-IoV-2024 | Legacy→modern transfer in another domain. Supports our "legacy is not enough" finding. | https://research.tue.nl/en/publications/65915593-69bd-4071-81c8-da7cd0e45f08 |

## C. IIoT intrusion detection on Edge-IIoTset / ForgeIIoT

| # | Year | Authors | Title | Venue | Datasets | Relation | Link |
|---|---|---|---|---|---|---|---|
| 11 | 2022 | M. A. Ferrag, O. Friha, D. Hamouda, L. Maglaras, H. Janicke | Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT and IIoT Applications | IEEE Access 10:40281–40306 | Edge-IIoTset | Modern target dataset (14 attacks, 5 threat groups). | https://doi.org/10.1109/ACCESS.2022.3165809 |
| 12 | 2024 | P. Kumar, S. Mullick, R. Das | IIEST_IoTForge_Pro / ForgeIIoT dataset (IoTForge Pro testbed) | IEEE DataPort, doi 10.21227/c4z1-yc52 | ForgeIIoT (17 capture files, ~3 M packets) | **Bridge dataset** of this project. Our leakage audit shows its capture-order fields must be excluded, and that capture-level labelling adds label noise. | https://ieee-dataport.org/documents/iotforge-pro |
| 13 | 2024 | M. A. Ferrag et al. | Revolutionizing Cyber Threat Detection with LLMs: A Privacy-Preserving BERT-Based Lightweight Model for IoT/IIoT Devices (SecurityBERT) | IEEE Access 12:23733–23750 | Edge-IIoTset | Text/BERT encoding of IIoT traffic (98.2% on 15 classes). We use language for **cross-dataset** alignment rather than in-domain accuracy. | https://doi.org/10.1109/ACCESS.2024.3363469 |
| 14 | 2024 | O. F. Awad, L. R. Hazim, A. A. Jasim, O. Ata | Enhancing IIoT Security with ML and DL for Intrusion Detection | Malaysian J. Computer Science 37(2) | Edge-IIoTset | Reports ~100% DT accuracy. Our E02 confirms Edge is near-separable in-domain, which is why cross-domain evaluation matters. | https://doi.org/10.22452/mjcs.vol37no2.3 |
| 15 | 2024 | A. Gueriani et al. | Adaptive Cyber-Attack Detection in IIoT Using Attention-Based LSTM-CNN Models | ICTIS 2024 | Edge-IIoTset | In-domain DL baseline. | https://arxiv.org/abs/2501.13962 |
| 16 | 2025 | M. A. Yagiz, P. Goktas | LENS-XAI: Lightweight Explainable Network Security via Knowledge Distillation and VAEs | Applied Soft Computing / arXiv 2501.00790 | Edge-IIoTset, NSL-KDD, … | Uses the same legacy and modern datasets, but separately. We study them **jointly**. | https://arxiv.org/abs/2501.00790 |
| 17 | 2025 | S. Rahman | DAE-IDS: A Domain-Aware Ensemble IDS with Explainable AI for Industrial IoT Networks | IJACSA 16(9) | Edge-IIoTset | XAI ensemble for IIoT. | https://thesai.org/Downloads/Volume16No9/Paper_72-DAE_IDS_A_Domain_Aware_Ensemble_Intrusion_Detection_System.pdf |

## D. Zero-shot / unseen-attack detection

| # | Year | Authors | Title | Venue | Datasets | Relation | Link |
|---|---|---|---|---|---|---|---|
| 18 | 2023 | M. Sarhan, S. Layeghy, M. Gallagher, M. Portmann | From Zero-Shot Machine Learning to Zero-Day Attack Detection | Int. J. Information Security 22(4):947–959 | NetFlow NIDS sets | Attribute-learning ZSL + zero-day detection rate. Our DAP experiment (E08-C) follows this attribute paradigm, with a cross-dataset taxonomy. | https://ouci.dntb.gov.ua/works/4VZG1Wk4 |
| 19 | 2026 | P. B. Sudasinghe, K. S. K. Liyanage, H. S. G. Pussewalage | Lightweight LLMs for Network Attack Detection in IoT Networks | arXiv 2601.15269 | CICIoT2023 | Structured-to-text + RAG zero-shot (42.6% on unseen). Our text ZSL is a lightweight, offline alternative. | https://arxiv.org/abs/2601.15269 |
| 20 | 2025 | A. Mirza, S. Arshad et al. | ZDBERTa: Advancing Zero-Day Cyberattack Detection in IoV with Zero-Shot Learning | Computers 14(10):424 | CICIoV2024 | BERT ZSL with held-out attack families. We use the same leave-family-out protocol (E08-A). | https://www.mdpi.com/2073-431X/14/10/424 |
| 21 | 2026 | A. Awasthi, P. Vediya, H. Miranka, R. B. Battula, M. S. Gaur | PANDORA: Lightweight Adversarial Defense for Edge IoT Using Uncertainty-Aware Metric Learning | NDSS 2026 | CICIDS2017 + new IoT set | Zero-shot under domain shift at the edge. | https://www.ndss-symposium.org/wp-content/uploads/2026-f713-paper.pdf |
| 22 | 2026 | A. A. Abualhassan, L. Al-Awami, M. Hamdan Mohamed | SSL-FSL Framework for Detecting Unseen IoT Attacks under Limited Labeled Data | IFIP Networking 2026 | **Edge-IIoTset**, CICIoT2023 | Few-shot / unseen attacks on Edge-IIoTset. Complements our label-budget study (E04, E11). | https://norma.ncirl.ie/9482/ |
| 23 | 2025 | S. Ustebay | Enhancing Zero-Day Attack Detection in IoT Networks via Isolation Forest and Ensemble Tree Models | (journal article, 2025) | CICEVSE2024, CICIoT2023, RT-IoT2022 | Two-layer supervised + Isolation Forest (30–62% zero-day). Our **benign-ECDF-calibrated hybrid** is a principled version with a bounded FPR. | https://earsiv.medeniyet.edu.tr/items/4f429c7e-941f-422b-ba15-48737556b8d2/full |
| 24 | 2024 | S. A. Raza, M. Shaikh, R. Akhtar, A. Anwar | A Novel Approach to Intrusion Detection Using Zero-Shot Learning Hybrid Partial Labels | Mehran Univ. Res. J. Eng. Tech. 43(1) | Image-based NIDS set | ZSL for NIDS. | https://doi.org/10.22581/muet1982.2401.2945 |

## E. NLP / language-model representations of traffic

| # | Year | Authors | Title | Venue | Datasets | Relation | Link |
|---|---|---|---|---|---|---|---|
| 25 | 2025 | (EURASIP authors) | BERT-based Network for Intrusion Detection System | EURASIP J. Information Security (doi 10.1186/s13635-025-00191-w) | several | Traffic → natural-language-like sequence → BERT. Our flow-to-text uses the **same vocabulary for all datasets**, so the text works as an interlingua. | https://doi.org/10.1186/s13635-025-00191-w |
| 26 | 2025 | L. Gutiérrez-Galeano, J.-J. Domínguez-Jiménez, J. Schäfer, I. Medina-Bulo | LLM-Based Cyberattack Detection Using Network Flow Statistics | Applied Sciences 15(12):6529 | 3 datasets | Flow statistics turned into an "artificial language" for T5. | https://www.mdpi.com/2076-3417/15/12/6529 |
| 27 | 2024 | P. R. B. Houssel et al. | Towards Explainable Network Intrusion Detection using Large Language Models | arXiv 2408.04342 | NetFlow sets | LLMs for detection + explanation. Our template narratives are a lightweight, deterministic counterpart. | https://arxiv.org/abs/2408.04342 |
| 28 | 2025 | A. Rehman et al. | Prompt-only LLM intrusion detection with a flow-to-text protocol (UNSW-NB15) | arXiv 2510.17883 | UNSW-NB15 | Flow-to-text with interpretable cues + calibration. | https://arxiv.org/abs/2510.17883 |
| 29 | 2023 | (authors of arXiv 2306.04920) | Flow-based Network Intrusion Detection Based on BERT Masked Language Model | arXiv 2306.04920 | CIC sets | Flow-as-word, sequence-as-sentence. Notes domain-adaptation benefits. | https://arxiv.org/abs/2306.04920 |
| 30 | 2023 | (authors of arXiv 2310.17127) | A Method for Network Intrusion Detection Using Flow Sequence and BERT Framework | arXiv 2310.17127 | CIC sets | BERT feature extractor + MLP. | https://arxiv.org/abs/2310.17127 |

## F. Continual learning and concept drift

| # | Year | Authors | Title | Venue | Datasets | Relation | Link |
|---|---|---|---|---|---|---|---|
| 31 | 2025 | X. Zhang, R. Zhao, Z. Jiang, H. Chen, Y. Ding, E. C. H. Ngai, S.-H. Yang | Continual Learning with Strategic Selection and Forgetting for Network Intrusion Detection (SSF) | IEEE INFOCOM 2025 | NSL-KDD, UNSW-NB15 | Drift detection + memory buffer with strategic forgetting. Our live engine also flushes pre-drift feedback on drift (E11). | https://arxiv.org/abs/2412.16264 |
| 32 | 2024 | J. Talpini, F. Sartori, M. Savi | Hierarchical Multiclass Continual Learning for Network Intrusion Detection | IEEE NetSoft 2024 | NIDS sets | Generative-replay CL for NIDS. | https://boa.unimib.it/bitstream/10281/483239/1/Talpini-2024-NetSoft-AAM.pdf |
| 33 | 2026 | S. Banerjee, D. Bergqvist, S. Toor, C. Rohner, A. Johnsson | Quantifying Catastrophic Forgetting in IoT Intrusion Detection Systems | arXiv 2603.00363 | RPL-IoT (48 domains) | Benchmarks EWC, SI, LwF, Replay; Replay is best overall. Consistent with our E06, where **EWC + Replay** wins *across datasets*. | https://arxiv.org/abs/2603.00363 |
| 34 | 2026 | D. Kassymov, T. K. Zhukabayeva | Adaptive Real-Time Network Intrusion Detection System (HAT + ADWIN) | Smart Technologies Journal 2(1) | UNSW-NB15 stream | Online HAT + ADWIN. In our tests a sliding-window GBM challenger clearly beat HAT under few labels (E11). | https://doi.org/10.62687/STJ.1.2.2026.13 |
| 35 | 2024 | (IEEE Access authors) | UASDAC: Unsupervised Adaptive Scalable DDoS Attack Classification in Large-Scale IoT Network Under Concept Drift | IEEE Access 2024 | NSL-KDD, IoT-23 | Drift-aware IoT stream pipeline. | https://doaj.org/article/870feaeaf0a34f3abd2d06eff4b94a4e |
| 36 | 2024 | Y. K. Beshah, S. L. Abebe, H. M. Melaku | Drift Adaptive Online DDoS Attack Detection Framework for IoT System | Electronics 13(6):1004 | IoT DDoS sets | Accuracy-weighted ensemble under drift. Related to our Hedge-weighted champion/challenger. | https://www.mdpi.com/2079-9292/13/6/1004 |
| 37 | 2024 | V. Agate, A. De Paola, S. Drago, P. Ferraro, G. Lo Re | Enhancing IoT Network Security with Concept Drift-Aware Unsupervised Threat Detection | IEEE ISCC 2024 | IoT malware traffic | Retrains only when drift is detected, as our engine does. | https://iris.unipa.it/handle/10447/664065 |
| 38 | 2026 | (JCSC authors) | CD-AHAL: A Concept Drift-Aware Hybrid Active Learning Framework for NIDS | J. Computer Science & Cybernetics 42 | IoT NIDS | Active learning with a label budget. Our feedback-rate study (2% vs 10%) is the streaming analogue. | https://vjs.ac.vn/jcc/article/download/24229/2543256903/2543292494 |

## G. Explainable AI for IDS

| # | Year | Authors | Title | Venue | Datasets | Relation | Link |
|---|---|---|---|---|---|---|---|
| 39 | 2024 | D. Gaspar, P. Silva, C. Silva | Explainable AI for IDS: LIME and SHAP Applicability on Multi-Layer Perceptron | IEEE Access 12:30164–30175 | IoT IDS | LIME vs SHAP comparison. We report LIME–SHAP top-k agreement and **explanation drift across datasets** (E09). | https://doaj.org/article/645cdc092bf54326a4b64f13212e9901 |
| 40 | 2025 | V. Z. Mohale, I. C. Obagbuwa | A Systematic Review on the Integration of XAI in IDS | Frontiers in AI (Jan 2025) | review | Notes SHAP/LIME cost in real time. We explain only alerts, within a per-batch budget. | https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1526221/pdf |
| 41 | 2024 | (TU Delft et al.) | Explainable AI for Intrusion Detection in IoT Networks (DNN + LIME + SHAP) | Expert Systems with Applications 238:121751 | IoT sets | DL + LIME/SHAP. | https://research.tudelft.nl/en/publications/explainable-artificial-intelligence-for-intrusion-detection-in-io |
| 42 | 2025 | Y. Wang, M. A. Azad, M. Zafar, A. Gul | Enhancing AI Transparency in IoT Intrusion Detection Using XAI Techniques | Internet of Things 33:101714 | IoT sets | SHAP + LIME on several models. | https://www.open-access.bcu.ac.uk/16619/ |
| 43 | 2025 | S. Sadhwani, A. Navare, A. Mohan, R. Muthalagu, P. M. Pawar | IoT-based IDS Using Explainable Multi-class Deep Learning Approaches | (Elsevier journal, 2025) | **NSL-KDD**, UNSW-NB15, TON-IoT, X-IIoTID | SHAP on legacy and IIoT datasets, but separately. We compare explanations **in one shared space**. | https://cris.biu.ac.il/en/publications/iot-based-intrusion-detection-system-using-explainable-multi-clas/ |
| 44 | 2026 | J. Munilla, R. M. Khammas | Evaluation of XAI in IoT IDS Under DeepFool Adversarial Conditions | Preprints 202604.0399 | BoT-IoT, **Edge-IIoT**, N-BaIoT | SHAP is more complete, LIME more rank-stable. Relevant to the robustness of our explanations. | https://www.preprints.org/manuscript/202604.0399/v1 |

## Foundational methods used (pre-2023)

| Method | Reference |
|---|---|
| NSL-KDD | M. Tavallaee, E. Bagheri, W. Lu, A. Ghorbani, "A detailed analysis of the KDD CUP 99 data set," IEEE CISDA 2009 |
| DANN | Y. Ganin et al., "Domain-Adversarial Training of Neural Networks," JMLR 17, 2016 |
| CORAL | B. Sun, J. Feng, K. Saenko, "Return of Frustratingly Easy Domain Adaptation," AAAI 2016 |
| TCA | S. J. Pan, I. W. Tsang, J. T. Kwok, Q. Yang, "Domain Adaptation via Transfer Component Analysis," IEEE TNN 22(2), 2011 |
| EWC | J. Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks," PNAS 114(13), 2017 |
| SHAP | S. Lundberg, S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," NeurIPS 2017 |
| LIME | M. T. Ribeiro, S. Singh, C. Guestrin, "Why Should I Trust You?," KDD 2016 |
| DAP (attribute ZSL) | C. H. Lampert, H. Nickisch, S. Harmeling, "Attribute-based classification for zero-shot visual object categorization," IEEE TPAMI 36(3), 2014 |
| ADWIN | A. Bifet, R. Gavaldà, "Learning from Time-Changing Data with Adaptive Windowing," SDM 2007 |
| Proxy-A-distance | S. Ben-David et al., "A theory of learning from different domains," Machine Learning 79, 2010 |
| Isolation Forest | F. T. Liu, K. M. Ting, Z.-H. Zhou, ICDM 2008 |
| ML pitfalls in security | D. Arp et al., "Dos and Don'ts of Machine Learning in Computer Security," USENIX Security 2022 |

## Gap analysis: what no single prior work does

1. **Legacy → bridge → modern** chain. Prior DA work goes legacy→IoT directly (e.g. #3, #6, #8). None uses an intermediate IIoT dataset or quantifies how much of a bridge it is (proxy-A-distance, concept shift).
2. **Connection-level vs packet-level semantics.** NetFlow standardisation (#4) does not cover KDD-style connection records vs tshark packets. Our three-tier alignment (CBS / PBS / native) and the language interlingua do.
3. **Explanation drift.** XAI papers (#39–#44) explain one dataset at a time. We measure SHAP rank agreement across datasets in a shared space (RQ4).
4. **Static → dynamic.** CL/drift papers (#31–#38) work within a single dataset stream. We combine cross-dataset drift, zero-day hybrid scoring, label-budgeted online adaptation and per-alert explanations in one deployable live system.
5. **Evaluation hygiene.** We show that capture-order identifiers yield ~100% accuracy on ForgeIIoT (cf. Arp et al. 2022), and we release leakage-free splits.
