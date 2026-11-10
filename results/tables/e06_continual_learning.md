| order                              | strategy           | metric       |   avg_final |   avg_final_std |     BWT |   forgetting |   forward_zero_shot |
|:-----------------------------------|:-------------------|:-------------|------------:|----------------:|--------:|-------------:|--------------------:|
| NSL-KDD → ForgeIIoT → Edge-IIoTset | Naive fine-tuning  | balanced_acc |      0.7442 |          0.0058 | -0.0336 |       0.1269 |              0.5933 |
| NSL-KDD → ForgeIIoT → Edge-IIoTset | Naive fine-tuning  | roc_auc      |      0.7134 |          0.0007 | -0.3270 |       0.3270 |              0.7777 |
| NSL-KDD → ForgeIIoT → Edge-IIoTset | EWC                | balanced_acc |      0.7615 |          0.0144 | -0.0048 |       0.0981 |              0.5933 |
| NSL-KDD → ForgeIIoT → Edge-IIoTset | EWC                | roc_auc      |      0.7310 |          0.0127 | -0.3003 |       0.3003 |              0.7650 |
| NSL-KDD → ForgeIIoT → Edge-IIoTset | Replay             | balanced_acc |      0.8401 |          0.0148 |  0.1228 |      -0.0287 |              0.5820 |
| NSL-KDD → ForgeIIoT → Edge-IIoTset | Replay             | roc_auc      |      0.9110 |          0.0049 | -0.0282 |       0.0282 |              0.6963 |
| NSL-KDD → ForgeIIoT → Edge-IIoTset | EWC + Replay       | balanced_acc |      0.8572 |          0.0285 |  0.1476 |      -0.0495 |              0.5858 |
| NSL-KDD → ForgeIIoT → Edge-IIoTset | EWC + Replay       | roc_auc      |      0.9135 |          0.0040 | -0.0216 |       0.0216 |              0.7109 |
| NSL-KDD → ForgeIIoT → Edge-IIoTset | LightGBM continued | balanced_acc |      0.7352 |          0.0000 |  0.0096 |       0.1339 |              0.6494 |
| NSL-KDD → ForgeIIoT → Edge-IIoTset | LightGBM continued | roc_auc      |      0.7235 |          0.0000 | -0.2132 |       0.2132 |              0.8054 |
| NSL-KDD → ForgeIIoT → Edge-IIoTset | Joint (offline)    | balanced_acc |      0.7334 |          0.0609 |  0.0324 |       0.0677 |              0.5786 |
| NSL-KDD → ForgeIIoT → Edge-IIoTset | Joint (offline)    | roc_auc      |      0.9217 |          0.0027 | -0.0130 |       0.0136 |              0.7216 |
| Edge-IIoTset → ForgeIIoT → NSL-KDD | Naive fine-tuning  | balanced_acc |      0.6803 |          0.0548 | -0.1133 |       0.2447 |              0.6324 |
| Edge-IIoTset → ForgeIIoT → NSL-KDD | Naive fine-tuning  | roc_auc      |      0.7265 |          0.0098 | -0.3006 |       0.3006 |              0.6334 |
| Edge-IIoTset → ForgeIIoT → NSL-KDD | EWC                | balanced_acc |      0.7464 |          0.0070 | -0.0116 |       0.1431 |              0.6324 |
| Edge-IIoTset → ForgeIIoT → NSL-KDD | EWC                | roc_auc      |      0.8035 |          0.0339 | -0.1762 |       0.1762 |              0.5595 |
| Edge-IIoTset → ForgeIIoT → NSL-KDD | Replay             | balanced_acc |      0.7107 |          0.0611 | -0.0658 |       0.1973 |              0.6329 |
| Edge-IIoTset → ForgeIIoT → NSL-KDD | Replay             | roc_auc      |      0.9183 |          0.0008 | -0.0093 |       0.0093 |              0.5649 |
| Edge-IIoTset → ForgeIIoT → NSL-KDD | EWC + Replay       | balanced_acc |      0.7564 |          0.0576 |  0.0038 |       0.1288 |              0.6309 |
| Edge-IIoTset → ForgeIIoT → NSL-KDD | EWC + Replay       | roc_auc      |      0.9129 |          0.0036 | -0.0086 |       0.0086 |              0.5595 |
| Edge-IIoTset → ForgeIIoT → NSL-KDD | LightGBM continued | balanced_acc |      0.4516 |          0.0000 | -0.2383 |       0.3713 |              0.5781 |
| Edge-IIoTset → ForgeIIoT → NSL-KDD | LightGBM continued | roc_auc      |      0.3859 |          0.0000 | -0.2862 |       0.5504 |              0.6330 |
| Edge-IIoTset → ForgeIIoT → NSL-KDD | Joint (offline)    | balanced_acc |      0.6882 |          0.0057 | -0.1863 |       0.2448 |              0.6290 |
| Edge-IIoTset → ForgeIIoT → NSL-KDD | Joint (offline)    | roc_auc      |      0.9224 |          0.0012 | -0.0028 |       0.0028 |              0.5569 |