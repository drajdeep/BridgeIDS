"""Integration test: replay -> engine (in-process). Skipped if prepared data / models are missing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_engine_end_to_end():
    from bridgeids.utils import load_config
    from bridgeids.stream.artifacts import artifact_path, load_artifacts
    from bridgeids.stream.engine import DetectionEngine
    from bridgeids.stream.replayer import build_stream
    cfg = load_config()
    if not artifact_path(cfg=cfg).exists() or not (Path(cfg["paths"]["processed"]) / "edge.parquet").exists():
        print("SKIP (run `python -m bridgeids.run exp 12` first)")
        return
    eng = DetectionEngine(load_artifacts(cfg=cfg), cfg, explain=True)
    stream = build_stream("edge", "campaign", n=1500, seed=1)
    for i in range(0, len(stream), 256):
        out = eng.process(stream[i:i + 256])
        assert all("score" in r and 0 <= r["score"] <= 1 for r in out)
    s = eng.snapshot()
    assert s["n"] == len(stream) and s["f1"] > 0.8, s


if __name__ == "__main__":
    test_engine_end_to_end()
    print("PASS test_engine_end_to_end")
