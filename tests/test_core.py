"""Smoke + correctness tests.  Run:  python -m pytest -q tests  (or  python tests/test_core.py)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bridgeids.models.npnet import MLP, DANN, _loss_grad, fisher_diag  # noqa: E402
from bridgeids.features.views import canonical_view, cbs_from_view, pbs_from_view, port_service  # noqa: E402


def test_gradient_check():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(16, 5))
    y = (rng.random(16) > 0.5).astype(float)
    m = MLP([5, 7, 4, 1], seed=1)
    z, acts = m.forward(X)
    _, dz = _loss_grad(z, y)
    grads, _ = m.backward(acts, dz)
    eps = 1e-6
    for pi, p in enumerate(m.params):
        it = np.nditer(p, flags=["multi_index"])
        for _ in range(min(6, p.size)):
            ix = it.multi_index
            old = p[ix]
            p[ix] = old + eps
            lp, _ = _loss_grad(m.forward(X)[0], y)
            p[ix] = old - eps
            lm, _ = _loss_grad(m.forward(X)[0], y)
            p[ix] = old
            num = (lp - lm) / (2 * eps)
            assert abs(num - grads[pi][ix]) < 1e-5, (pi, ix, num, grads[pi][ix])
            it.iternext()


def test_fisher_matches_bruteforce():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(10, 4))
    y = (rng.random(10) > 0.5).astype(float)
    m = MLP([4, 6, 1], seed=2)
    F = fisher_diag(m, X, y)
    brute = [np.zeros_like(p) for p in m.params]
    for i in range(10):
        z, a = m.forward(X[i:i + 1])
        _, dz = _loss_grad(z, y[i:i + 1])
        g, _ = m.backward(a, dz)
        for b, gi in zip(brute, g):
            b += gi ** 2 / 10
    for f, b in zip(F, brute):
        assert np.allclose(f, b, atol=1e-10)


def test_dann_runs():
    rng = np.random.default_rng(0)
    Xs, Xt = rng.normal(size=(300, 6)), rng.normal(1, 1, size=(300, 6))
    ys = (Xs[:, 0] > 0).astype(float)
    d = DANN(6, seed=0).fit(Xs, ys, Xt, epochs=3)
    assert d.predict_proba(Xt).shape == (300, 2)


def test_views():
    nsl = pd.DataFrame({"protocol_type": ["tcp", "icmp"], "service": ["http", "ecr_i"], "flag": ["S0", "SF"],
                        "src_bytes": [0, 1032], "dst_bytes": [0, 0], "land": [0, 0]})
    v = canonical_view(nsl, "nsl")
    assert v.loc[0, "syn_only"] == 1 and v.loc[1, "service"] == "icmp_echo"
    edge = pd.DataFrame({"tcp.flags": [2.0, 24.0], "tcp.dstport": [80.0, 1883.0], "tcp.srcport": [5555.0, 40000.0],
                         "tcp.len": [0.0, 39.0], "mqtt.msgtype": [0.0, 3.0]})
    ve = canonical_view(edge, "edge")
    assert ve.loc[0, "syn_only"] == 1 and ve.loc[1, "service"] == "mqtt" and ve.loc[1, "psh"] == 1
    assert cbs_from_view(v).shape[1] == cbs_from_view(ve).shape[1]
    assert pbs_from_view(ve).shape[1] > cbs_from_view(ve).shape[1]
    assert list(port_service(np.array([1883, 80, 60000, 0]))) == ["mqtt", "http", "ephemeral", "none"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
