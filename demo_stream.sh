#!/usr/bin/env bash
# Live dynamic-detection demo: starts the detector, opens the dashboard, replays an attack campaign as UDP packets.
# Usage: ./demo_stream.sh [campaign|zero_day|domain_shift|shuffle] [edge|forge|nsl]
cd "$(dirname "$0")"
[ -d .venv ] && source .venv/bin/activate
SCEN=${1:-campaign}; DS=${2:-edge}
ART=stream_artifacts; [ "$SCEN" = "domain_shift" ] && ART=bench_forge_only
python -m bridgeids.run serve --artifacts $ART & SRV=$!
trap "kill $SRV 2>/dev/null" EXIT
sleep 8
( command -v xdg-open >/dev/null && xdg-open http://127.0.0.1:8000 ) || ( command -v open >/dev/null && open http://127.0.0.1:8000 ) || echo "Open http://127.0.0.1:8000 in your browser"
python -m bridgeids.run replay --scenario "$SCEN" --dataset "$DS" --rate 400 --n 20000
echo "Replay finished - dashboard stays live. Press Ctrl+C to stop the server."
wait $SRV
