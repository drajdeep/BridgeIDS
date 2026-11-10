# Power BI Visual Analytics — build guide (Team-2 deliverable 7)

Power BI Desktop files (`.pbix`) are binary and can only be authored inside Power BI Desktop (Windows).
This kit gives you everything you need to build the dashboard in about 20 minutes: the data, the star schema,
the DAX measures, a theme, and a page-by-page layout. The same content is also available in the browser right away:
`reports/results_dashboard.html` (static results) and `http://127.0.0.1:8000` (live detector).

## 1. Data (refresh with `python -m bridgeids.run report`)

`powerbi/data/` (created by the pipeline):

| File | Grain | Use |
|---|---|---|
| `dim_dataset.csv` | dataset | slicer, era / granularity |
| `dim_attack.csv` | dataset × fine label | taxonomy, super-class colours |
| `fact_model_performance.csv` | experiment × source × target × method | baselines, cross-dataset, DA, transfer, NLP |
| `fact_continual_learning.csv` | order × strategy × metric | continual learning |
| `fact_zeroshot_lofo.csv`, `fact_zeroshot_cross_dataset.csv` | unseen family | zero-shot |
| `fact_feature_importance_native.csv`, `fact_feature_importance_cbs.csv` | feature × dataset | SHAP |
| `fact_drift_feature.csv`, `fact_drift_summary.csv` | feature × pair / pair | drift |
| `fact_label_distribution.csv` | dataset × label | attack distribution |
| `fact_stream_benchmark*.csv` | config × phase / time | static vs dynamic |
| `stream_events.csv`, `stream_alerts.csv`, `stream_drift.csv`, `stream_metrics.csv` | packet / alert | live detection (export from the dashboard button or `GET /api/export/powerbi`) |

## 2. Load and model

1. **Get Data → Text/CSV** for each file, or use `load_bridgeids_folder.pq` (Power Query) to load the whole folder.
2. **View → Themes → Browse for themes →** `BridgeIDS_theme.json`.
3. Relationships (Model view):
   * `dim_dataset[dataset]` 1 → * `fact_label_distribution[dataset]`, `fact_feature_importance_native[dataset]`
   * `dim_attack[label]` 1 → * `fact_label_distribution[label_fine]`
   * `dim_attack[label]` 1 → * `stream_events[truth_fine]` (inactive; use `USERELATIONSHIP` if needed)
4. **Modeling → New measure** for each measure in `BridgeIDS_measures.dax`.

## 3. Report pages (modules from the project plan)

| Page | Visuals |
|---|---|
| **1 · Attack Distribution** | Stacked bar `fact_label_distribution` (axis = super, legend = dataset, value = count); treemap of fine labels; card for the number of fine labels per dataset |
| **2 · Model Performance** | Matrix (rows = source, columns = target, values = Avg ROC-AUC, filtered to experiment = "Cross-dataset CBS"), conditional formatting red→green; clustered bar of baselines (method × dataset, Avg F1); card *Generalisation Gap* |
| **3 · Domain Adaptation & Transfer** | Clustered bar (method, Avg ROC-AUC, legend = pair, experiment = "Domain adaptation"); line chart of transfer (label_fraction on a log axis, Avg MCC, legend = method) |
| **4 · Feature Importance (XAI)** | Bar of `fact_feature_importance_native` (top-N filter = 15, slicer = dataset); matrix feature × dataset of `fact_feature_importance_cbs` (heat-map formatting) → explanation drift |
| **5 · Drift** | KPI cards *Proxy A-distance*, *Concept Shift*; matrix feature × pair of `js`; scatter js vs concept_shift (size = psi) |
| **6 · Zero-shot & Continual** | Heat-map matrix unseen_family × detector (`fact_zeroshot_lofo`); bar of `fact_zeroshot_cross_dataset` by family; table of `fact_continual_learning` (avg_final, BWT, forgetting) |
| **7 · Real-time Detection** | Cards *Packets, Alerts, Live F1, Live FPR, Avg Latency, Drift Events*; line chart `stream_metrics` (ts → throughput, f1, w_online); table of `stream_alerts` with the `explanation` column (word-wrap on); donut of `stream_events[family]` |
| **8 · Protocol & MQTT Analytics** | Donut `stream_events[proto]`; bar `stream_events[service]`; MQTT share card; (optional) bar of MQTT message types from the live dashboard export |

## 4. Real-time options

* **A · Streaming dataset (true real time).** Power BI Service → *Workspace → New → Streaming dataset → API*. Create
  fields `timestamp` (DateTime), `packets`, `alerts`, `throughput`, `f1`, `fpr`, `drift_events` (Number), enable
  *Historic data analysis*, copy the **Push URL** into `configs/default.yaml → stream.powerbi_push_url`, restart
  `python -m bridgeids.run serve`. The detector pushes one row per micro-batch. Build a dashboard tile from it.
* **B · DirectQuery-like refresh from SQLite.** The detector writes `results/stream/stream.db`. Install the SQLite
  ODBC driver, **Get Data → ODBC**, select tables `events`, `alerts`, `drift`, `metrics`, then enable
  *Page refresh* (Format → Page refresh → every 5 s; Power BI Desktop with a DirectQuery source).
* **C · CSV snapshot.** Click **Export for Power BI** on the live dashboard, then **Refresh** in Power BI.

## 5. Suggested slicers and interactions
`dim_dataset[dataset]`, `fact_model_performance[experiment]`, `dim_attack[super_class]`; sync slicers across pages 1–6.
