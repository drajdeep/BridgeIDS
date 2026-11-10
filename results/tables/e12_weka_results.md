| setting             | train        | test         | classifier        |   accuracy |   attack_precision |   attack_recall |   attack_f1 |      mcc |   roc_auc |   weighted_f1 |
|:--------------------|:-------------|:-------------|:------------------|-----------:|-------------------:|----------------:|------------:|---------:|----------:|--------------:|
| in-domain (native)  | NSL-KDD      | NSL-KDD      | Weka J48          |     0.7905 |             0.9580 |          0.6680 |      0.7880 |   0.6310 |    0.8210 |        0.7900 |
| in-domain (native)  | ForgeIIoT    | ForgeIIoT    | Weka J48          |     0.9355 |             0.9450 |          0.9840 |      0.9640 |   0.6520 |    0.9400 |        0.9300 |
| in-domain (native)  | Edge-IIoTset | Edge-IIoTset | Weka J48          |     0.9984 |             0.9980 |          1.0000 |      0.9990 |   0.9940 |    0.9960 |        0.9980 |
| cross-dataset (CBS) | NSL-KDD      | Edge-IIoTset | Weka J48          |     0.4079 |             0.7550 |          0.4450 |      0.5600 |  -0.2540 |    0.3280 |        0.4890 |
| cross-dataset (CBS) | ForgeIIoT    | Edge-IIoTset | Weka J48          |     0.8469 |             0.8470 |          1.0000 |      0.9170 | nan      |    0.9200 |      nan      |
| cross-dataset (CBS) | NSL-KDD      | ForgeIIoT    | Weka J48          |     0.7367 |             0.9190 |          0.7700 |      0.8380 |   0.1880 |    0.6290 |        0.7750 |
| in-domain (native)  | NSL-KDD      | NSL-KDD      | Weka RandomForest |     0.7757 |             0.9660 |          0.6360 |      0.7670 |   0.6140 |    0.9650 |        0.7740 |
| in-domain (native)  | ForgeIIoT    | ForgeIIoT    | Weka RandomForest |     0.9292 |             0.9510 |          0.9700 |      0.9600 |   0.6370 |    0.9560 |        0.9270 |
| in-domain (native)  | Edge-IIoTset | Edge-IIoTset | Weka RandomForest |     0.9991 |             0.9990 |          1.0000 |      0.9990 |   0.9970 |    1.0000 |        0.9990 |
| cross-dataset (CBS) | NSL-KDD      | Edge-IIoTset | Weka RandomForest |     0.4162 |             0.7590 |          0.4550 |      0.5690 |  -0.2480 |    0.3790 |        0.4970 |
| cross-dataset (CBS) | ForgeIIoT    | Edge-IIoTset | Weka RandomForest |     0.8469 |             0.8470 |          1.0000 |      0.9170 | nan      |    0.7370 |      nan      |
| cross-dataset (CBS) | NSL-KDD      | ForgeIIoT    | Weka RandomForest |     0.7296 |             0.9180 |          0.7620 |      0.8330 |   0.1800 |    0.7800 |        0.7700 |
| in-domain (native)  | NSL-KDD      | NSL-KDD      | Weka AdaBoostM1   |     0.7884 |             0.9590 |          0.6640 |      0.7850 |   0.6290 |    0.9390 |        0.7880 |
| in-domain (native)  | ForgeIIoT    | ForgeIIoT    | Weka AdaBoostM1   |     0.9014 |             0.9020 |          0.9970 |      0.9470 |   0.3730 |    0.9410 |        0.8710 |
| in-domain (native)  | Edge-IIoTset | Edge-IIoTset | Weka AdaBoostM1   |     0.9971 |             0.9970 |          1.0000 |      0.9980 |   0.9890 |    0.9990 |        0.9970 |
| cross-dataset (CBS) | NSL-KDD      | Edge-IIoTset | Weka AdaBoostM1   |     0.3800 |             0.7190 |          0.4400 |      0.5460 |  -0.3680 |    0.3230 |        0.4660 |
| cross-dataset (CBS) | ForgeIIoT    | Edge-IIoTset | Weka AdaBoostM1   |     0.8469 |             0.8470 |          1.0000 |      0.9170 | nan      |    0.9000 |      nan      |
| cross-dataset (CBS) | NSL-KDD      | ForgeIIoT    | Weka AdaBoostM1   |     0.7364 |             0.9190 |          0.7690 |      0.8380 |   0.1880 |    0.7070 |        0.7750 |