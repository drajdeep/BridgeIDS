@echo off
REM Usage: run_weka.bat C:\path\to\weka.jar
set JAR=%1
if "%JAR%"=="" set JAR=weka.jar
cd /d "%~dp0data"
for %%d in (nsl forge edge) do (
  echo === J48 %%d native ===
  java -Xmx2g -cp "%JAR%" weka.classifiers.trees.J48 -t %%d_native_train.arff -T %%d_native_test.arff -o | findstr /C:"Correctly" /C:"Weighted Avg"
  echo === RandomForest %%d native ===
  java -Xmx2g -cp "%JAR%" weka.classifiers.trees.RandomForest -t %%d_native_train.arff -T %%d_native_test.arff -o -I 100 | findstr /C:"Correctly" /C:"Weighted Avg"
)
echo === J48 NSL-KDD -^> Edge-IIoTset (CBS) ===
java -Xmx2g -cp "%JAR%" weka.classifiers.trees.J48 -t nsl_cbs_train.arff -T edge_cbs_test.arff -o | findstr /C:"Correctly" /C:"Weighted Avg"
echo === J48 ForgeIIoT -^> Edge-IIoTset (CBS) ===
java -Xmx2g -cp "%JAR%" weka.classifiers.trees.J48 -t forge_cbs_train.arff -T edge_cbs_test.arff -o | findstr /C:"Correctly" /C:"Weighted Avg"
