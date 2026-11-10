"""Tiny dependency-free neural-network engine (numpy only, exact backprop).

Why not PyTorch? The project must install in seconds on Windows/Linux laptops without CUDA, and
every network needed here is small (tabular MLPs). Everything required by the plan is provided:

* ``MLP``            – ReLU MLP with logits output, Adam, mini-batches, early stopping
* ``fit_classifier`` – supervised training (+ optional EWC penalty and replay buffer)
* ``DANN``           – Domain-Adversarial Neural Network with a Gradient Reversal Layer (Ganin 2016)
* ``fisher_diag``    – exact diagonal (empirical) Fisher information for dense nets (used by EWC)
"""
from __future__ import annotations

import copy
from typing import List, Optional, Sequence, Tuple

import numpy as np


def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    """Dense ReLU network.  sizes = [in, h1, ..., out].  Last layer linear (logits)."""

    def __init__(self, sizes: Sequence[int], seed: int = 0, last_relu: bool = False):
        rng = np.random.default_rng(seed)
        self.W = [rng.normal(0, np.sqrt(2.0 / a), (a, b)).astype(np.float64) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.last_relu = last_relu
        self._m = self._v = None
        self.t = 0

    # -- parameters -----------------------------------------------------------
    @property
    def params(self) -> List[np.ndarray]:
        return self.W + self.b

    def set_params(self, ps: List[np.ndarray]):
        k = len(self.W)
        self.W, self.b = [p.copy() for p in ps[:k]], [p.copy() for p in ps[k:]]

    def clone(self) -> "MLP":
        return copy.deepcopy(self)

    # -- forward / backward ---------------------------------------------------
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, list]:
        acts = [X]
        h = X
        L = len(self.W)
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            h = np.maximum(z, 0) if (i < L - 1 or self.last_relu) else z
            acts.append(h)
        return h, acts

    def backward(self, acts: list, dout: np.ndarray, per_sample_sq: bool = False):
        """Returns (grads, dX[, squared per-sample grads for Fisher])."""
        L = len(self.W)
        gW, gb, sqW, sqb = [None] * L, [None] * L, [None] * L, [None] * L
        delta = dout
        for i in reversed(range(L)):
            if i < L - 1 or self.last_relu:
                delta = delta * (acts[i + 1] > 0)
            a = acts[i]
            gW[i] = a.T @ delta
            gb[i] = delta.sum(0)
            if per_sample_sq:
                sqW[i] = (a ** 2).T @ (delta ** 2)
                sqb[i] = (delta ** 2).sum(0)
            delta = delta @ self.W[i].T
        if per_sample_sq:
            return gW + gb, delta, sqW + sqb
        return gW + gb, delta

    def adam_step(self, grads: List[np.ndarray], lr: float = 1e-3, wd: float = 0.0, b1=0.9, b2=0.999, eps=1e-8):
        ps = self.params
        if self._m is None:
            self._m = [np.zeros_like(p) for p in ps]
            self._v = [np.zeros_like(p) for p in ps]
        self.t += 1
        for i, (p, g) in enumerate(zip(ps, grads)):
            if wd:
                g = g + wd * p
            self._m[i] = b1 * self._m[i] + (1 - b1) * g
            self._v[i] = b2 * self._v[i] + (1 - b2) * g * g
            mh = self._m[i] / (1 - b1 ** self.t)
            vh = self._v[i] / (1 - b2 ** self.t)
            p -= lr * mh / (np.sqrt(vh) + eps)

    def reset_optimizer(self):
        self._m = self._v = None
        self.t = 0

    # -- inference ---------------------------------------------------------------
    def predict_proba(self, X: np.ndarray, batch: int = 8192) -> np.ndarray:
        outs = []
        for i in range(0, len(X), batch):
            z, _ = self.forward(np.asarray(X[i:i + batch], dtype=np.float64))
            outs.append(_sigmoid(z).ravel() if z.shape[1] == 1 else _softmax(z))
        p = np.concatenate(outs)
        return np.column_stack([1 - p, p]) if p.ndim == 1 else p

    def predict(self, X):
        p = self.predict_proba(X)
        return p.argmax(1)


def _loss_grad(z: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None):
    """BCE (z: n x 1) or CE (z: n x k) with logits; returns (loss, dL/dz) averaged over the batch."""
    n = len(z)
    w = np.ones(n) if w is None else w
    if z.shape[1] == 1:
        p = _sigmoid(z.ravel())
        loss = -np.mean(w * (y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))
        return loss, ((p - y) * w / n)[:, None]
    p = _softmax(z)
    Y = np.eye(z.shape[1])[y.astype(int)]
    loss = -np.mean(w * np.log((p * Y).sum(1) + 1e-9))
    return loss, (p - Y) * w[:, None] / n


def fisher_diag(model: MLP, X: np.ndarray, y: np.ndarray, batch: int = 2048) -> List[np.ndarray]:
    """Exact empirical diagonal Fisher  E[(d log p(y|x)/d theta)^2]  (per-sample, no approximation)."""
    acc = [np.zeros_like(p) for p in model.params]
    for i in range(0, len(X), batch):
        xb, yb = X[i:i + batch], y[i:i + batch]
        z, acts = model.forward(xb)
        _, dz = _loss_grad(z, yb)
        dz = dz * len(xb)  # undo averaging -> per-sample gradients
        _, _, sq = model.backward(acts, dz, per_sample_sq=True)
        for a, s in zip(acc, sq):
            a += s
    return [a / len(X) for a in acc]


def fit_classifier(model: MLP, X: np.ndarray, y: np.ndarray, epochs: int = 10, batch: int = 256, lr: float = 1e-3,
                   seed: int = 0, sample_weight: Optional[np.ndarray] = None, ewc: Optional[list] = None,
                   replay: Optional[Tuple[np.ndarray, np.ndarray]] = None, replay_ratio: float = 0.5,
                   X_val=None, y_val=None, patience: int = 3, wd: float = 1e-5, verbose: bool = False) -> MLP:
    """Mini-batch Adam training.

    ewc    : list of (fisher, anchor_params, lambda) – Elastic Weight Consolidation penalties
    replay : (X_mem, y_mem) experience-replay memory mixed into every batch
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    best, best_loss, bad = None, np.inf, 0
    for ep in range(epochs):
        idx = rng.permutation(len(X))
        for i in range(0, len(X), batch):
            bi = idx[i:i + batch]
            xb, yb = X[bi], y[bi]
            wb = None if sample_weight is None else sample_weight[bi]
            if replay is not None and len(replay[0]):
                k = max(1, int(len(bi) * replay_ratio))
                ri = rng.integers(0, len(replay[0]), k)
                xb = np.vstack([xb, replay[0][ri]])
                yb = np.concatenate([yb, replay[1][ri]])
                wb = None if wb is None else np.concatenate([wb, np.ones(k)])
            z, acts = model.forward(xb)
            _, dz = _loss_grad(z, yb, wb)
            grads, _ = model.backward(acts, dz)
            if ewc:
                for F, anchor, lam in ewc:
                    grads = [g + lam * f * (p - a) for g, f, p, a in zip(grads, F, model.params, anchor)]
            model.adam_step(grads, lr=lr, wd=wd)
        if X_val is not None:
            zv, _ = model.forward(np.asarray(X_val, dtype=np.float64))
            vl, _ = _loss_grad(zv, np.asarray(y_val))
            if verbose:
                print(f"epoch {ep + 1}: val loss {vl:.4f}")
            if vl < best_loss - 1e-4:
                best_loss, best, bad = vl, [p.copy() for p in model.params], 0
            else:
                bad += 1
                if bad >= patience:
                    break
    if best is not None:
        model.set_params(best)
    return model


class DANN:
    """Domain-Adversarial Neural Network (Ganin et al., JMLR 2016) for tabular IDS features.

    F: feature extractor, C: attack classifier (labelled source), D: domain discriminator.
    The gradient reversal layer multiplies dL_D/dF by -lambda so F learns domain-invariant features.
    """

    def __init__(self, n_in: int, hidden=(64, 32), seed: int = 0):
        self.F = MLP([n_in, *hidden], seed=seed, last_relu=True)
        self.C = MLP([hidden[-1], 1], seed=seed + 1)
        self.D = MLP([hidden[-1], 32, 1], seed=seed + 2)

    def fit(self, Xs, ys, Xt, epochs: int = 20, batch: int = 256, lr: float = 1e-3, seed: int = 0,
            lambda_max: float = 1.0, verbose: bool = False):
        rng = np.random.default_rng(seed)
        Xs, Xt = np.asarray(Xs, float), np.asarray(Xt, float)
        ys = np.asarray(ys, float)
        steps = epochs * int(np.ceil(len(Xs) / batch))
        step = 0
        self.history = []
        for ep in range(epochs):
            idx = rng.permutation(len(Xs))
            for i in range(0, len(Xs), batch):
                p = step / max(steps, 1)
                lam = lambda_max * (2.0 / (1.0 + np.exp(-10 * p)) - 1.0)
                step += 1
                bi = idx[i:i + batch]
                xs, yb = Xs[bi], ys[bi]
                xt = Xt[rng.integers(0, len(Xt), len(bi))]
                X = np.vstack([xs, xt])
                f, fa = self.F.forward(X)
                ns = len(xs)
                # label head on source only
                zc, ca = self.C.forward(f[:ns])
                lc, dzc = _loss_grad(zc, yb)
                gC, dfc = self.C.backward(ca, dzc)
                # domain head on source(1) + target(0)
                d_lab = np.concatenate([np.ones(ns), np.zeros(len(xt))])
                zd, da = self.D.forward(f)
                ld, dzd = _loss_grad(zd, d_lab)
                gD, dfd = self.D.backward(da, dzd)
                # gradient reversal: F minimises Lc and *maximises* Ld
                df = -lam * dfd
                df[:ns] += dfc
                gF, _ = self.F.backward(fa, df)
                self.C.adam_step(gC, lr)
                self.D.adam_step(gD, lr)
                self.F.adam_step(gF, lr)
            self.history.append((ep, float(lc), float(ld), float(lam)))
            if verbose:
                print(f"DANN epoch {ep + 1}: Lc={lc:.4f} Ld={ld:.4f} lambda={lam:.2f}")
        return self

    def predict_proba(self, X):
        f, _ = self.F.forward(np.asarray(X, float))
        p = _sigmoid(self.C.forward(f)[0].ravel())
        return np.column_stack([1 - p, p])

    def features(self, X):
        return self.F.forward(np.asarray(X, float))[0]
