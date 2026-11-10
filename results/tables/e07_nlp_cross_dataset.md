| source       | target       | representation   | in_domain   |   roc_auc |     mcc |     f1 |   balanced_acc |
|:-------------|:-------------|:-----------------|:------------|----------:|--------:|-------:|---------------:|
| NSL-KDD      | NSL-KDD      | text embedding   | True        |    0.9216 |  0.6701 | 0.8146 |         0.8340 |
| NSL-KDD      | NSL-KDD      | CBS numeric      | True        |    0.9514 |  0.7370 | 0.8674 |         0.8717 |
| NSL-KDD      | NSL-KDD      | CBS + text       | True        |    0.9533 |  0.7132 | 0.8505 |         0.8588 |
| NSL-KDD      | ForgeIIoT    | text embedding   | False       |    0.8183 |  0.3332 | 0.6888 |         0.7587 |
| NSL-KDD      | ForgeIIoT    | CBS numeric      | False       |    0.7778 |  0.3803 | 0.7587 |         0.7948 |
| NSL-KDD      | ForgeIIoT    | CBS + text       | False       |    0.7752 |  0.3864 | 0.7653 |         0.7991 |
| NSL-KDD      | Edge-IIoTset | text embedding   | False       |    0.5156 |  0.0669 | 0.5538 |         0.5453 |
| NSL-KDD      | Edge-IIoTset | CBS numeric      | False       |    0.5850 |  0.0451 | 0.5230 |         0.5302 |
| NSL-KDD      | Edge-IIoTset | CBS + text       | False       |    0.7152 |  0.3427 | 0.8314 |         0.7214 |
| ForgeIIoT    | NSL-KDD      | text embedding   | False       |    0.7892 | -0.0511 | 0.7213 |         0.4959 |
| ForgeIIoT    | NSL-KDD      | CBS numeric      | False       |    0.3397 | -0.0082 | 0.7254 |         0.4999 |
| ForgeIIoT    | NSL-KDD      | CBS + text       | False       |    0.7485 | -0.0508 | 0.7213 |         0.4960 |
| ForgeIIoT    | ForgeIIoT    | text embedding   | True        |    0.8692 |  0.0278 | 0.9378 |         0.5014 |
| ForgeIIoT    | ForgeIIoT    | CBS numeric      | True        |    0.8767 |  0.0991 | 0.9382 |         0.5108 |
| ForgeIIoT    | ForgeIIoT    | CBS + text       | True        |    0.8771 |  0.1010 | 0.9380 |         0.5122 |
| ForgeIIoT    | Edge-IIoTset | text embedding   | False       |    0.8578 |  0.0000 | 0.9171 |         0.5000 |
| ForgeIIoT    | Edge-IIoTset | CBS numeric      | False       |    0.8533 |  0.0000 | 0.9171 |         0.5000 |
| ForgeIIoT    | Edge-IIoTset | CBS + text       | False       |    0.8374 |  0.0000 | 0.9171 |         0.5000 |
| Edge-IIoTset | NSL-KDD      | text embedding   | False       |    0.3585 | -0.0082 | 0.7254 |         0.4999 |
| Edge-IIoTset | NSL-KDD      | CBS numeric      | False       |    0.2530 | -0.1047 | 0.6340 |         0.4576 |
| Edge-IIoTset | NSL-KDD      | CBS + text       | False       |    0.3853 | -0.0776 | 0.7092 |         0.4868 |
| Edge-IIoTset | ForgeIIoT    | text embedding   | False       |    0.8460 |  0.3373 | 0.7114 |         0.7625 |
| Edge-IIoTset | ForgeIIoT    | CBS numeric      | False       |    0.8463 |  0.2950 | 0.6442 |         0.7269 |
| Edge-IIoTset | ForgeIIoT    | CBS + text       | False       |    0.8507 |  0.3401 | 0.7114 |         0.7647 |
| Edge-IIoTset | Edge-IIoTset | text embedding   | True        |    0.9980 |  0.9613 | 0.9942 |         0.9729 |
| Edge-IIoTset | Edge-IIoTset | CBS numeric      | True        |    0.9982 |  0.9689 | 0.9953 |         0.9739 |
| Edge-IIoTset | Edge-IIoTset | CBS + text       | True        |    0.9982 |  0.9689 | 0.9953 |         0.9739 |