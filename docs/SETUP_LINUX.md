# Setup on Linux (Ubuntu/Debian/Fedora/Arch) and macOS

## 1. Prerequisites
```bash
# Ubuntu / Debian
sudo apt update && sudo apt install -y python3 python3-venv python3-pip unzip
# optional: Java for Weka, Mosquitto for MQTT transport
sudo apt install -y default-jre mosquitto
# Fedora: sudo dnf install python3 python3-pip java-latest-openjdk mosquitto
# macOS (Homebrew): brew install python@3.12 openjdk mosquitto libomp   # libomp is needed by LightGBM/XGBoost
```
Python 3.10 – 3.13. No GPU, CUDA or PyTorch needed.

## 2. Install
```bash
unzip BridgeIDS.zip && cd BridgeIDS
./setup_linux.sh            # creates .venv, installs requirements, pip install -e ., runs unit tests
source .venv/bin/activate
```

## 3. Data
Prepared samples (`data/processed/*.parquet`) and NSL-KDD are included, so you can run immediately. To rebuild from raw data:
```bash
mkdir -p data/raw/forge data/raw/edge_iiot
unzip Forge_IIOT_Pro_Dataset.zip -d data/raw/forge/          # folder that contains "Unprocessed Dataset"
cp ML-EdgeIIoT-dataset.csv data/raw/edge_iiot/
python -m bridgeids.run prepare --force
```
Symlinks work too (`ln -s /big/disk/Forge_IIOT_Pro_Dataset data/raw/forge/`).

## 4. Run
```bash
./run_all.sh                         # prepare (skips existing) -> 12 experiments -> report
python -m bridgeids.run exp 07 08    # selected experiments
python -m bridgeids.run report
python -m pytest -q tests            # or: python tests/test_core.py
```
Outputs: `results/tables/*.csv|md`, `results/figures/*.png`, `reports/paper_draft.md`, `reports/results_dashboard.html`, `powerbi/data/*.csv`.

## 5. Live dynamic detection
```bash
./demo_stream.sh campaign edge       # kill-chain campaign
./demo_stream.sh zero_day edge       # unseen families (XSS, Ransomware)
./demo_stream.sh domain_shift        # Forge-trained model meets Edge traffic -> drift + adaptation
```
Manual:
```bash
python -m bridgeids.run serve &                                             # http://127.0.0.1:8000
python -m bridgeids.run replay --scenario campaign --dataset edge --rate 400 # UDP 127.0.0.1:9999
python -m bridgeids.run replay --transport tcp --scenario zero_day           # TCP NDJSON on 10000
BRIDGEIDS_MQTT=127.0.0.1:1883 python -m bridgeids.run serve                 # + MQTT ingestion
python -m bridgeids.run replay --transport mqtt --port 1883
curl -s localhost:8000/api/state | python -m json.tool | head
curl -s localhost:8000/api/export/powerbi                                   # CSVs for Power BI
```
To listen on all interfaces (e.g. for a Raspberry Pi gateway): `python -m bridgeids.run serve --host 0.0.0.0`.

Larger UDP bursts: `sudo sysctl -w net.core.rmem_max=16777216` raises the socket buffer limit the server requests.

## 6. Troubleshooting
| Symptom | Fix |
|---|---|
| `libgomp.so.1` / `libomp` missing | `sudo apt install libgomp1` (Linux) · `brew install libomp` (macOS) |
| `ModuleNotFoundError: bridgeids` | `source .venv/bin/activate && pip install -e .` |
| Address already in use | `serve --port 8080 --udp-port 9998`, `replay --port 9998` |
| Weka skipped | `sudo apt install default-jre && pip install python-weka-wrapper3` or `export WEKA_JAR=/path/weka.jar` |
| SBERT not used | `pip install sentence-transformers` (needs internet once); otherwise the offline encoder is used automatically |
