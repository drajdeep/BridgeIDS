| space   | model   | source                   | target       |   in_domain_f1 |   cross_f1 |   f1_drop |   in_domain_auc |   cross_auc |   auc_drop |
|:--------|:--------|:-------------------------|:-------------|---------------:|-----------:|----------:|----------------:|------------:|-----------:|
| CBS     | lgbm    | NSL-KDD                  | ForgeIIoT    |         0.9382 |     0.7587 |    0.1794 |          0.8767 |      0.7778 |     0.0989 |
| CBS     | lgbm    | NSL-KDD                  | Edge-IIoTset |         0.9953 |     0.5230 |    0.4723 |          0.9982 |      0.5850 |     0.4132 |
| CBS     | lgbm    | ForgeIIoT                | NSL-KDD      |         0.8674 |     0.7254 |    0.1420 |          0.9514 |      0.3397 |     0.6117 |
| CBS     | lgbm    | ForgeIIoT                | Edge-IIoTset |         0.9953 |     0.9171 |    0.0782 |          0.9982 |      0.8533 |     0.1449 |
| CBS     | lgbm    | Edge-IIoTset             | NSL-KDD      |         0.8674 |     0.6340 |    0.2334 |          0.9514 |      0.2530 |     0.6984 |
| CBS     | lgbm    | Edge-IIoTset             | ForgeIIoT    |         0.9382 |     0.6442 |    0.2940 |          0.8767 |      0.8463 |     0.0305 |
| CBS     | lgbm    | ForgeIIoT + Edge-IIoTset | NSL-KDD      |         0.8674 |     0.6932 |    0.1743 |          0.9514 |      0.3303 |     0.6211 |
| CBS     | lgbm    | NSL-KDD + Edge-IIoTset   | ForgeIIoT    |         0.9382 |     0.7048 |    0.2333 |          0.8767 |      0.8532 |     0.0235 |
| CBS     | lgbm    | NSL-KDD + ForgeIIoT      | Edge-IIoTset |         0.9953 |     0.9167 |    0.0785 |          0.9982 |      0.7788 |     0.2194 |
| CBS     | rf      | NSL-KDD                  | ForgeIIoT    |         0.9381 |     0.7605 |    0.1776 |          0.8769 |      0.7529 |     0.1240 |
| CBS     | rf      | NSL-KDD                  | Edge-IIoTset |         0.9953 |     0.5518 |    0.4435 |          0.9982 |      0.3938 |     0.6044 |
| CBS     | rf      | ForgeIIoT                | NSL-KDD      |         0.8635 |     0.7213 |    0.1422 |          0.9091 |      0.3780 |     0.5311 |
| CBS     | rf      | ForgeIIoT                | Edge-IIoTset |         0.9953 |     0.9171 |    0.0782 |          0.9982 |      0.7934 |     0.2048 |
| CBS     | rf      | Edge-IIoTset             | NSL-KDD      |         0.8635 |     0.7229 |    0.1405 |          0.9091 |      0.4075 |     0.5016 |
| CBS     | rf      | Edge-IIoTset             | ForgeIIoT    |         0.9381 |     0.7079 |    0.2302 |          0.8769 |      0.8489 |     0.0279 |
| CBS     | rf      | ForgeIIoT + Edge-IIoTset | NSL-KDD      |         0.8635 |     0.7213 |    0.1421 |          0.9091 |      0.3795 |     0.5296 |
| CBS     | rf      | NSL-KDD + Edge-IIoTset   | ForgeIIoT    |         0.9381 |     0.7066 |    0.2315 |          0.8769 |      0.8490 |     0.0278 |
| CBS     | rf      | NSL-KDD + ForgeIIoT      | Edge-IIoTset |         0.9953 |     0.8931 |    0.1022 |          0.9982 |      0.6887 |     0.3095 |
| PBS     | lgbm    | ForgeIIoT                | Edge-IIoTset |         0.9995 |     0.9498 |    0.0498 |          0.9999 |      0.8415 |     0.1584 |
| PBS     | lgbm    | Edge-IIoTset             | ForgeIIoT    |         0.9590 |     0.6964 |    0.2626 |          0.9453 |      0.8161 |     0.1292 |
| PBS     | rf      | ForgeIIoT                | Edge-IIoTset |         0.9996 |     0.9416 |    0.0580 |          0.9998 |      0.8579 |     0.1420 |
| PBS     | rf      | Edge-IIoTset             | ForgeIIoT    |         0.9425 |     0.7075 |    0.2351 |          0.8806 |      0.7716 |     0.1090 |