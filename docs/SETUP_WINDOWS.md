# Setup on Windows 10 / 11

## 1. Prerequisites
* **Python 3.10 – 3.13 (64-bit)** from https://www.python.org/downloads/. Tick **"Add python.exe to PATH"** during install.
* ~4 GB free RAM, ~3 GB disk (more if you keep the raw ForgeIIoT captures).
* *Optional:* Java 11+ (e.g. Eclipse Temurin) for the Weka runs; Power BI Desktop for the BI report.

No GPU, CUDA or PyTorch is needed.

## 2. Install
Unzip `BridgeIDS.zip` to a path **without special characters**, e.g. `C:\BridgeIDS`, then in *Command Prompt*:

```bat
cd C:\BridgeIDS
setup_windows.bat
```

This creates `.venv`, installs `requirements.txt`, installs the package in editable mode (`pip install -e .`) and runs
the unit tests (gradient check, exact Fisher, DANN, feature views). PowerShell users: run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once if activation scripts are blocked, or just use `cmd`.

Manual alternative:
```bat
py -3 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## 3. Data
The zip already contains the prepared samples in `data\processed\*.parquet` and the NSL-KDD files, so **you can run
everything immediately**. To rebuild from the raw data:

| Dataset | Put it here |
|---|---|
| NSL-KDD | `data\raw\nsl_kdd\KDDTrain+.txt`, `KDDTest+.txt` (or the `.arff` files) |
| ForgeIIoT | `data\raw\forge\Forge_IIOT_Pro_Dataset\...` (the unzipped folder that contains `Unprocessed Dataset`) |
| Edge-IIoTset | `data\raw\edge_iiot\ML-EdgeIIoT-dataset.csv` |

```bat
python -m bridgeids.run prepare --force
```
Sample sizes are set in `configs\default.yaml` (`prepare:` and `experiments:`).

## 4. Run
```bat
.venv\Scripts\activate
run_all.bat                                   :: all 12 experiments + paper draft + dashboards (~35-50 min on 2 cores)
python -m bridgeids.run exp 03 05             :: or selected experiments
python -m bridgeids.run report                :: rebuild reports\paper_draft.md, reports\results_dashboard.html, powerbi\data
```

## 5. Live dynamic detection
```bat
demo_stream.bat campaign edge        :: kill-chain attack campaign on Edge-IIoT traffic
demo_stream.bat zero_day edge        :: XSS + Ransomware were never seen in training
demo_stream.bat domain_shift         :: Forge-trained model, traffic switches to Edge -> drift + adaptation
```
Or manually in two terminals:
```bat
python -m bridgeids.run serve                                   :: terminal 1, then open http://127.0.0.1:8000
python -m bridgeids.run replay --scenario campaign --rate 400   :: terminal 2 (UDP packets to 127.0.0.1:9999)
```
Other transports: `--transport tcp` (port 10000), `--transport http`, `--transport mqtt` (needs a broker, e.g.
Mosquitto, and `set BRIDGEIDS_MQTT=127.0.0.1:1883` before `serve`).

**Windows Firewall** may ask to allow Python on first `serve`. Allow *private networks* (only localhost is used by default).

## 6. Troubleshooting
| Symptom | Fix |
|---|---|
| `'python' is not recognized` | reinstall Python with "Add to PATH" or use `py -3` |
| `ModuleNotFoundError: bridgeids` | activate `.venv` and run `pip install -e .` from the project root |
| LightGBM DLL error | install the *Microsoft Visual C++ Redistributable 2015-2022 (x64)* |
| Port 8000 / 9999 in use | `python -m bridgeids.run serve --port 8080 --udp-port 9998` and `replay --port 9998` |
| UDP packets dropped at very high rates | use `--transport tcp` or lower `--rate` (UDP is lossy by design) |
| Weka results missing | install Java and `pip install python-weka-wrapper3`, or set `set WEKA_JAR=C:\path\weka.jar` |
