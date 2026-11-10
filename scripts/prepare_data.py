"""Build data/processed/*.parquet from the raw datasets (run once)."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bridgeids.data.loaders import prepare_all

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuild even if parquet exists")
    prepare_all(force=ap.parse_args().force)
