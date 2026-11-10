# Weka experiments

`python -m bridgeids.run exp 12` writes ARFF files to `weka/data/` (native and CBS feature spaces, train/test, for every
dataset). If a `weka.jar` and Java 11+ are found, it also runs **J48, RandomForest and AdaBoostM1** in-domain and
cross-dataset, and stores the parsed results in `results/tables/e12_weka_results.csv`.

The jar is searched in this order: `$WEKA_JAR` → `weka/weka.jar` → the Weka install folder → the jar bundled with
`pip install python-weka-wrapper3`.

Manual run (GUI): open the Weka Explorer → *Open file* `weka/data/edge_cbs_train.arff` → Classify → *Supplied test set*
`weka/data/edge_cbs_test.arff` (or another dataset's test file for cross-dataset testing).

Command line:

```bash
./run_weka.sh /path/to/weka.jar        # Linux / macOS
run_weka.bat C:\path\to\weka.jar       # Windows
```
