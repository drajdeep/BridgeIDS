#!/usr/bin/env bash
# BridgeIDS one-shot setup for Linux / macOS
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
$PY -c 'import sys; assert sys.version_info >= (3,10), "Python 3.10+ required"'
$PY -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
pip install -r requirements.txt
pip install -e .
python tests/test_core.py
echo
echo "Setup complete. Activate with:  source .venv/bin/activate"
echo "Next:  ./run_all.sh        (experiments + report)"
echo "       ./demo_stream.sh    (live dynamic detection demo)"
