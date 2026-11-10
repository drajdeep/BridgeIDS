| dataset      | representation                |   binary_f1 |   binary_auc |   binary_mcc |   superclass_macro_f1 |
|:-------------|:------------------------------|------------:|-------------:|-------------:|----------------------:|
| NSL-KDD      | native                        |      0.7775 |       0.9740 |       0.6325 |                0.1311 |
| NSL-KDD      | text only                     |      0.7732 |       0.9010 |       0.5907 |                0.1688 |
| NSL-KDD      | native + text (NLP-augmented) |      0.7884 |       0.9628 |       0.6463 |                0.3467 |
| ForgeIIoT    | native                        |      0.9652 |       0.9681 |       0.6694 |                0.8263 |
| ForgeIIoT    | text only                     |      0.9380 |       0.8716 |       0.0351 |                0.6613 |
| ForgeIIoT    | native + text (NLP-augmented) |      0.9652 |       0.9677 |       0.6693 |                0.8250 |
| Edge-IIoTset | native                        |      0.9995 |       0.9999 |       0.9968 |                0.8784 |
| Edge-IIoTset | text only                     |      0.9942 |       0.9980 |       0.9613 |                0.6894 |
| Edge-IIoTset | native + text (NLP-augmented) |      0.9996 |       0.9999 |       0.9972 |                0.8784 |