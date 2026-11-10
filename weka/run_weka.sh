#!/usr/bin/env bash
# Run Weka J48 / RandomForest / AdaBoostM1 on the exported ARFFs (in-domain + cross-dataset CBS)
JAR=${1:-${WEKA_JAR:-weka.jar}}
cd "$(dirname "$0")/data" || exit 1
for clf in "weka.classifiers.trees.J48" "weka.classifiers.trees.RandomForest -I 100" "weka.classifiers.meta.AdaBoostM1 -I 50 -W weka.classifiers.trees.DecisionStump"; do
  set -- $clf
  for ds in nsl forge edge; do
    echo "=== $1 | $ds native ==="; java -Xmx2g -cp "$JAR" $1 -t ${ds}_native_train.arff -T ${ds}_native_test.arff -o ${@:2} | grep -E "Correctly|Weighted Avg"
  done
  for pair in "nsl edge" "forge edge" "nsl forge"; do
    set -- $pair; a=$1; b=$2; set -- $clf
    echo "=== $1 | $a -> $b (CBS) ==="; java -Xmx2g -cp "$JAR" $1 -t ${a}_cbs_train.arff -T ${b}_cbs_test.arff -o ${@:2} | grep -E "Correctly|Weighted Avg"
  done
done
