| pair                               | space   | method                     |   roc_auc |     mcc |     f1 |   balanced_acc |    fpr |   auc_std |   auc_gain_vs_noadapt |
|:-----------------------------------|:--------|:---------------------------|----------:|--------:|-------:|---------------:|-------:|----------:|----------------------:|
| NSL-KDD → Edge-IIoTset             | CBS     | No adaptation (LightGBM)   |    0.4872 |  0.0531 | 0.5503 |         0.5360 | 0.3303 |    0.0000 |                0.0000 |
| NSL-KDD → Edge-IIoTset             | CBS     | DomainNorm                 |    0.5474 | -0.1269 | 0.6130 |         0.4120 | 0.6716 |    0.0000 |                0.0602 |
| NSL-KDD → Edge-IIoTset             | CBS     | CORAL                      |    0.4903 |  0.0123 | 0.7919 |         0.5075 | 0.7266 |    0.0000 |                0.0031 |
| NSL-KDD → Edge-IIoTset             | CBS     | TCA (+LightGBM)            |    0.5954 |  0.0927 | 0.5341 |         0.5619 | 0.2576 |    0.0000 |                0.1082 |
| NSL-KDD → Edge-IIoTset             | CBS     | TCA (+LogReg)              |    0.0486 | -0.5376 | 0.3274 |         0.1472 | 0.9344 |    0.0000 |               -0.4385 |
| NSL-KDD → Edge-IIoTset             | CBS     | Importance weighting       |    0.5562 |  0.1306 | 0.6319 |         0.5904 | 0.3067 |    0.0000 |                0.0691 |
| NSL-KDD → Edge-IIoTset             | CBS     | DomainNorm + self-training |    0.4980 | -0.2236 | 0.4915 |         0.3472 | 0.6710 |    0.0000 |                0.0108 |
| NSL-KDD → Edge-IIoTset             | CBS     | MLP (no adaptation)        |    0.4707 | -0.1381 | 0.6398 |         0.4048 | 0.7229 |    0.0648 |               -0.0165 |
| NSL-KDD → Edge-IIoTset             | CBS     | DANN                       |    0.3811 | -0.0674 | 0.6334 |         0.4536 | 0.6065 |    0.1393 |               -0.1061 |
| NSL-KDD → ForgeIIoT                | CBS     | No adaptation (LightGBM)   |    0.7525 |  0.3804 | 0.7588 |         0.7949 | 0.0236 |    0.0000 |                0.0000 |
| NSL-KDD → ForgeIIoT                | CBS     | DomainNorm                 |    0.8038 |  0.1974 | 0.8379 |         0.6353 | 0.4980 |    0.0000 |                0.0513 |
| NSL-KDD → ForgeIIoT                | CBS     | CORAL                      |    0.8073 |  0.3838 | 0.7591 |         0.7975 | 0.0182 |    0.0000 |                0.0548 |
| NSL-KDD → ForgeIIoT                | CBS     | TCA (+LightGBM)            |    0.7985 |  0.3872 | 0.7629 |         0.8000 | 0.0182 |    0.0000 |                0.0460 |
| NSL-KDD → ForgeIIoT                | CBS     | TCA (+LogReg)              |    0.2062 | -0.2857 | 0.6548 |         0.2823 | 0.9858 |    0.0000 |               -0.5463 |
| NSL-KDD → ForgeIIoT                | CBS     | Importance weighting       |    0.8041 |  0.3946 | 0.7706 |         0.8051 | 0.0182 |    0.0000 |                0.0516 |
| NSL-KDD → ForgeIIoT                | CBS     | DomainNorm + self-training |    0.7871 |  0.1965 | 0.8373 |         0.6348 | 0.4980 |    0.0000 |                0.0346 |
| NSL-KDD → ForgeIIoT                | CBS     | MLP (no adaptation)        |    0.8155 |  0.4016 | 0.7772 |         0.8096 | 0.0182 |    0.0023 |                0.0630 |
| NSL-KDD → ForgeIIoT                | CBS     | DANN                       |    0.7258 |  0.3195 | 0.6665 |         0.7468 | 0.0071 |    0.0314 |               -0.0267 |
| NSL-KDD + ForgeIIoT → Edge-IIoTset | CBS     | No adaptation (LightGBM)   |    0.7805 | -0.0864 | 0.8933 |         0.4766 | 1.0000 |    0.0000 |                0.0000 |
| NSL-KDD + ForgeIIoT → Edge-IIoTset | CBS     | DomainNorm                 |    0.6557 |  0.0527 | 0.8943 |         0.5180 | 0.9052 |    0.0000 |               -0.1248 |
| NSL-KDD + ForgeIIoT → Edge-IIoTset | CBS     | CORAL                      |    0.3548 |  0.0000 | 0.9171 |         0.5000 | 1.0000 |    0.0000 |               -0.4257 |
| NSL-KDD + ForgeIIoT → Edge-IIoTset | CBS     | TCA (+LightGBM)            |    0.5017 | -0.0270 | 0.9148 |         0.4976 | 1.0000 |    0.0000 |               -0.2788 |
| NSL-KDD + ForgeIIoT → Edge-IIoTset | CBS     | TCA (+LogReg)              |    0.7214 |  0.0441 | 0.8641 |         0.5207 | 0.8342 |    0.0000 |               -0.0591 |
| NSL-KDD + ForgeIIoT → Edge-IIoTset | CBS     | Importance weighting       |    0.6377 |  0.0602 | 0.8964 |         0.5200 | 0.9050 |    0.0000 |               -0.1429 |
| NSL-KDD + ForgeIIoT → Edge-IIoTset | CBS     | DomainNorm + self-training |    0.4171 | -0.0856 | 0.8938 |         0.4770 | 1.0000 |    0.0000 |               -0.3634 |
| NSL-KDD + ForgeIIoT → Edge-IIoTset | CBS     | MLP (no adaptation)        |    0.5218 | -0.1253 | 0.8241 |         0.4388 | 0.9437 |    0.0874 |               -0.2587 |
| NSL-KDD + ForgeIIoT → Edge-IIoTset | CBS     | DANN                       |    0.4340 | -0.1545 | 0.7314 |         0.4006 | 0.8674 |    0.0351 |               -0.3465 |
| ForgeIIoT → Edge-IIoTset           | PBS     | No adaptation (LightGBM)   |    0.8456 |  0.6127 | 0.9498 |         0.7076 | 0.5848 |    0.0000 |                0.0000 |
| ForgeIIoT → Edge-IIoTset           | PBS     | DomainNorm                 |    0.8829 |  0.6124 | 0.9498 |         0.7074 | 0.5853 |    0.0000 |                0.0373 |
| ForgeIIoT → Edge-IIoTset           | PBS     | CORAL                      |    0.8775 |  0.0000 | 0.9171 |         0.5000 | 1.0000 |    0.0000 |                0.0320 |
| ForgeIIoT → Edge-IIoTset           | PBS     | TCA (+LightGBM)            |    0.7162 |  0.4302 | 0.9335 |         0.6062 | 0.7874 |    0.0000 |               -0.1294 |
| ForgeIIoT → Edge-IIoTset           | PBS     | TCA (+LogReg)              |    0.8078 |  0.4514 | 0.8239 |         0.8041 | 0.1056 |    0.0000 |               -0.0378 |
| ForgeIIoT → Edge-IIoTset           | PBS     | Importance weighting       |    0.8736 |  0.6124 | 0.9498 |         0.7074 | 0.5853 |    0.0000 |                0.0281 |
| ForgeIIoT → Edge-IIoTset           | PBS     | DomainNorm + self-training |    0.6182 |  0.0000 | 0.9171 |         0.5000 | 1.0000 |    0.0000 |               -0.2274 |
| ForgeIIoT → Edge-IIoTset           | PBS     | MLP (no adaptation)        |    0.6901 |  0.1818 | 0.9230 |         0.5381 | 0.9238 |    0.0107 |               -0.1554 |
| ForgeIIoT → Edge-IIoTset           | PBS     | DANN                       |    0.7231 |  0.1403 | 0.9072 |         0.5673 | 0.8179 |    0.0481 |               -0.1224 |
| Edge-IIoTset → ForgeIIoT           | PBS     | No adaptation (LightGBM)   |    0.8065 |  0.3325 | 0.6961 |         0.7585 | 0.0182 |    0.0000 |                0.0000 |
| Edge-IIoTset → ForgeIIoT           | PBS     | DomainNorm                 |    0.7902 |  0.3327 | 0.6964 |         0.7586 | 0.0182 |    0.0000 |               -0.0163 |
| Edge-IIoTset → ForgeIIoT           | PBS     | CORAL                      |    0.7813 |  0.3915 | 0.7740 |         0.8022 | 0.0293 |    0.0000 |               -0.0252 |
| Edge-IIoTset → ForgeIIoT           | PBS     | TCA (+LightGBM)            |    0.6892 |  0.1784 | 0.9136 |         0.5805 | 0.7649 |    0.0000 |               -0.1173 |
| Edge-IIoTset → ForgeIIoT           | PBS     | TCA (+LogReg)              |    0.7835 |  0.3901 | 0.7744 |         0.8011 | 0.0324 |    0.0000 |               -0.0230 |
| Edge-IIoTset → ForgeIIoT           | PBS     | Importance weighting       |    0.7906 |  0.3327 | 0.6964 |         0.7586 | 0.0182 |    0.0000 |               -0.0158 |
| Edge-IIoTset → ForgeIIoT           | PBS     | DomainNorm + self-training |    0.8061 |  0.3293 | 0.7005 |         0.7561 | 0.0290 |    0.0000 |               -0.0004 |
| Edge-IIoTset → ForgeIIoT           | PBS     | MLP (no adaptation)        |    0.8352 |  0.3653 | 0.8132 |         0.7723 | 0.1562 |    0.0129 |                0.0287 |
| Edge-IIoTset → ForgeIIoT           | PBS     | DANN                       |    0.8326 |  0.1777 | 0.8974 |         0.6231 | 0.6423 |    0.0168 |                0.0261 |