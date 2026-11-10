| dataset      | label                 | legacy_category   | super_class     | description                                                                    |
|:-------------|:----------------------|:------------------|:----------------|:-------------------------------------------------------------------------------|
| NSL-KDD      | back                  | DoS               | DOS             |                                                                                |
| NSL-KDD      | land                  | DoS               | DOS             |                                                                                |
| NSL-KDD      | neptune               | DoS               | DOS             |                                                                                |
| NSL-KDD      | pod                   | DoS               | DOS             |                                                                                |
| NSL-KDD      | smurf                 | DoS               | DOS             |                                                                                |
| NSL-KDD      | teardrop              | DoS               | DOS             |                                                                                |
| NSL-KDD      | apache2               | DoS               | DOS             |                                                                                |
| NSL-KDD      | udpstorm              | DoS               | DOS             |                                                                                |
| NSL-KDD      | processtable          | DoS               | DOS             |                                                                                |
| NSL-KDD      | worm                  | DoS               | DOS             |                                                                                |
| NSL-KDD      | mailbomb              | DoS               | DOS             |                                                                                |
| NSL-KDD      | satan                 | Probe             | RECON           |                                                                                |
| NSL-KDD      | ipsweep               | Probe             | RECON           |                                                                                |
| NSL-KDD      | nmap                  | Probe             | RECON           |                                                                                |
| NSL-KDD      | portsweep             | Probe             | RECON           |                                                                                |
| NSL-KDD      | mscan                 | Probe             | RECON           |                                                                                |
| NSL-KDD      | saint                 | Probe             | RECON           |                                                                                |
| NSL-KDD      | guess_passwd          | R2L               | UNAUTH_ACCESS   |                                                                                |
| NSL-KDD      | ftp_write             | R2L               | UNAUTH_ACCESS   |                                                                                |
| NSL-KDD      | imap                  | R2L               | UNAUTH_ACCESS   |                                                                                |
| NSL-KDD      | phf                   | R2L               | UNAUTH_ACCESS   |                                                                                |
| NSL-KDD      | multihop              | R2L               | UNAUTH_ACCESS   |                                                                                |
| NSL-KDD      | warezmaster           | R2L               | UNAUTH_ACCESS   |                                                                                |
| NSL-KDD      | warezclient           | R2L               | UNAUTH_ACCESS   |                                                                                |
| NSL-KDD      | spy                   | R2L               | UNAUTH_ACCESS   |                                                                                |
| NSL-KDD      | xlock                 | R2L               | UNAUTH_ACCESS   |                                                                                |
| NSL-KDD      | xsnoop                | R2L               | UNAUTH_ACCESS   |                                                                                |
| NSL-KDD      | snmpguess             | R2L               | UNAUTH_ACCESS   |                                                                                |
| NSL-KDD      | snmpgetattack         | R2L               | UNAUTH_ACCESS   |                                                                                |
| NSL-KDD      | httptunnel            | R2L               | UNAUTH_ACCESS   |                                                                                |
| NSL-KDD      | sendmail              | R2L               | UNAUTH_ACCESS   |                                                                                |
| NSL-KDD      | named                 | R2L               | UNAUTH_ACCESS   |                                                                                |
| NSL-KDD      | buffer_overflow       | U2R               | MALWARE_PRIVESC |                                                                                |
| NSL-KDD      | loadmodule            | U2R               | MALWARE_PRIVESC |                                                                                |
| NSL-KDD      | rootkit               | U2R               | MALWARE_PRIVESC |                                                                                |
| NSL-KDD      | perl                  | U2R               | MALWARE_PRIVESC |                                                                                |
| NSL-KDD      | sqlattack             | U2R               | INJECTION       |                                                                                |
| NSL-KDD      | xterm                 | U2R               | MALWARE_PRIVESC |                                                                                |
| NSL-KDD      | ps                    | U2R               | MALWARE_PRIVESC |                                                                                |
| NSL-KDD      | normal                | Normal            | NORMAL          |                                                                                |
| ForgeIIoT    | normal                |                   | NORMAL          | benign MQTT/IIoT traffic                                                       |
| ForgeIIoT    | ddos_tcp              |                   | DOS             | distributed TCP SYN flood                                                      |
| ForgeIIoT    | ddos_mp               |                   | DOS             | distributed MQTT publish flood                                                 |
| ForgeIIoT    | dos_ipv4              |                   | DOS             | raw IPv4 packet flood                                                          |
| ForgeIIoT    | dos_icmp              |                   | DOS             | ICMP echo flood                                                                |
| ForgeIIoT    | dos_udp               |                   | DOS             | UDP flood                                                                      |
| ForgeIIoT    | dos_land              |                   | DOS             | LAND attack (source = destination)                                             |
| ForgeIIoT    | dos_deauth            |                   | DOS             | Wi-Fi de-authentication DoS                                                    |
| ForgeIIoT    | port_scan             |                   | RECON           | TCP port scan                                                                  |
| ForgeIIoT    | vuln_scan             |                   | RECON           | vulnerability scan                                                             |
| ForgeIIoT    | os_finger             |                   | RECON           | OS fingerprinting                                                              |
| ForgeIIoT    | mitm_arp              |                   | MITM_SPOOF      | ARP-spoofing man-in-the-middle                                                 |
| ForgeIIoT    | mitm_dhcp             |                   | MITM_SPOOF      | rogue DHCP man-in-the-middle                                                   |
| ForgeIIoT    | mitm_ndp              |                   | MITM_SPOOF      | IPv6 NDP-spoofing man-in-the-middle                                            |
| ForgeIIoT    | sybil                 |                   | MITM_SPOOF      | Sybil (forged MQTT client identities)                                          |
| ForgeIIoT    | wild_card             |                   | UNAUTH_ACCESS   | MQTT wildcard (#) topic subscription abuse                                     |
| ForgeIIoT    | malw_back             |                   | MALWARE_PRIVESC | malware back-door / reverse shell                                              |
| Edge-IIoTset | Normal                |                   | NORMAL          | benign sensor telemetry published over MQTT to the broker                      |
| Edge-IIoTset | DDoS_UDP              |                   | DOS             | UDP flood denial of service with many datagrams                                |
| Edge-IIoTset | DDoS_ICMP             |                   | DOS             | ICMP echo ping flood denial of service                                         |
| Edge-IIoTset | DDoS_TCP              |                   | DOS             | TCP SYN flood with half open connections                                       |
| Edge-IIoTset | DDoS_HTTP             |                   | DOS             | HTTP GET request flood against the web server                                  |
| Edge-IIoTset | Port_Scanning         |                   | RECON           | port scanning with SYN probes to many ports, connections reset                 |
| Edge-IIoTset | Vulnerability_scanner |                   | RECON           | vulnerability scanner sending many HTTP probes for known weaknesses            |
| Edge-IIoTset | Fingerprinting        |                   | RECON           | operating system fingerprinting with crafted ICMP and TCP probes               |
| Edge-IIoTset | MITM                  |                   | MITM_SPOOF      | ARP spoofing man in the middle interception                                    |
| Edge-IIoTset | Password              |                   | UNAUTH_ACCESS   | password brute force guessing of login credentials over HTTP                   |
| Edge-IIoTset | SQL_injection         |                   | INJECTION       | SQL injection with crafted HTTP query strings against the database             |
| Edge-IIoTset | XSS                   |                   | INJECTION       | cross site scripting injecting script payload in HTTP requests                 |
| Edge-IIoTset | Uploading             |                   | INJECTION       | malicious file upload over HTTP POST with large payload                        |
| Edge-IIoTset | Backdoor              |                   | MALWARE_PRIVESC | backdoor command and control session on an unusual port                        |
| Edge-IIoTset | Ransomware            |                   | MALWARE_PRIVESC | ransomware sending encrypted payload over a command channel on an unusual port |