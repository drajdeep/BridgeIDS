#!/usr/bin/env bash
# Full pipeline: data preparation -> 12 experiments -> paper draft + HTML dashboard + Power BI CSVs
set -e
cd "$(dirname "$0")"
[ -d .venv ] && source .venv/bin/activate
python -m bridgeids.run prepare          # skips datasets whose parquet already exists
python -m bridgeids.run exp all
python -m bridgeids.run report
echo "Done. Open reports/results_dashboard.html and reports/paper_draft.md"
