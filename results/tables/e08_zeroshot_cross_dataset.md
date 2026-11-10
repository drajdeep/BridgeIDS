| edge_family           | super           | family_seen_in_sources   |    n |   source-supervised@0.5 |   source-supervised@5%FPR |   target one-class@5%FPR |   hybrid@5%FPR |
|:----------------------|:----------------|:-------------------------|-----:|------------------------:|--------------------------:|-------------------------:|---------------:|
| Backdoor              | MALWARE_PRIVESC | True                     | 1939 |                  1.0000 |                    0.1996 |                   0.9985 |         0.9985 |
| DDoS_HTTP             | DOS             | True                     | 2008 |                  1.0000 |                    0.5842 |                   1.0000 |         1.0000 |
| DDoS_ICMP             | DOS             | True                     | 2679 |                  1.0000 |                    0.0000 |                   1.0000 |         1.0000 |
| DDoS_TCP              | DOS             | True                     | 1948 |                  1.0000 |                    0.0000 |                   1.0000 |         1.0000 |
| DDoS_UDP              | DOS             | True                     | 2756 |                  1.0000 |                    0.9118 |                   1.0000 |         1.0000 |
| Fingerprinting        | RECON           | True                     |  300 |                  1.0000 |                    0.3433 |                   1.0000 |         1.0000 |
| MITM                  | MITM_SPOOF      | True                     |  300 |                  0.6533 |                    0.6367 |                   1.0000 |         1.0000 |
| Normal                | NORMAL          | False                    | 4620 |                  1.0000 |                    0.0840 |                   0.0524 |         0.0725 |
| Password              | UNAUTH_ACCESS   | True                     | 1899 |                  1.0000 |                    0.4929 |                   1.0000 |         1.0000 |
| Port_Scanning         | RECON           | True                     | 1914 |                  1.0000 |                    0.0455 |                   1.0000 |         1.0000 |
| Ransomware            | MALWARE_PRIVESC | True                     | 2077 |                  1.0000 |                    0.5802 |                   0.5883 |         1.0000 |
| SQL_injection         | INJECTION       | False                    | 1960 |                  1.0000 |                    0.4551 |                   1.0000 |         1.0000 |
| Uploading             | INJECTION       | False                    | 1952 |                  1.0000 |                    0.6911 |                   0.9401 |         0.9693 |
| Vulnerability_scanner | RECON           | True                     | 1916 |                  0.9995 |                    0.1127 |                   1.0000 |         1.0000 |
| XSS                   | INJECTION       | False                    | 1911 |                  1.0000 |                    0.5332 |                   1.0000 |         1.0000 |