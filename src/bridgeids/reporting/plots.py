"""Consistent, publication-style matplotlib figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..utils import load_config  # noqa: E402

PALETTE = ["#2E5AAC", "#D9822B", "#2F9E72", "#C23B4E", "#7B5EA7", "#8C6D46", "#4A9BB5", "#9AA0A6"]
DS_COLORS = {"NSL-KDD": "#2E5AAC", "ForgeIIoT": "#D9822B", "Edge-IIoTset": "#2F9E72"}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "axes.prop_cycle": matplotlib.cycler(color=PALETTE), "legend.frameon": False,
})


def savefig(fig, name: str, cfg=None) -> Path:
    cfg = cfg or load_config()
    p = Path(cfg["paths"]["figures"]) / f"{name}.png"
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def heatmap(mat: pd.DataFrame, title: str, name: str, fmt: str = "{:.2f}", cmap: str = "Blues",
            vmin=None, vmax=None, figsize=None, cfg=None, xlabel="", ylabel=""):
    figsize = figsize or (1.1 + 0.85 * mat.shape[1], 1.0 + 0.55 * mat.shape[0])
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat.values.astype(float), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(mat.shape[1]), mat.columns, rotation=35, ha="right")
    ax.set_yticks(range(mat.shape[0]), mat.index)
    ax.grid(False)
    vals = mat.values.astype(float)
    finite = vals[np.isfinite(vals)]
    mid = (np.nanmax(finite) + np.nanmin(finite)) / 2 if finite.size else 0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = vals[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8.5,
                        color="white" if (v > mid) == (cmap not in ("Blues_r", "RdYlGn")) and cmap != "RdYlGn" else "black")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, fraction=0.04)
    return savefig(fig, name, cfg)


def grouped_bar(df: pd.DataFrame, x: str, hue: str, yv: str, title: str, name: str, ylim=(0, 1.02), cfg=None,
                figsize=(9, 4.2), ylabel=None):
    piv = df.pivot_table(index=x, columns=hue, values=yv, aggfunc="mean")
    fig, ax = plt.subplots(figsize=figsize)
    n = piv.shape[1]
    w = 0.8 / n
    for i, c in enumerate(piv.columns):
        ax.bar(np.arange(len(piv)) + i * w - 0.4 + w / 2, piv[c].values, w, label=str(c),
               color=DS_COLORS.get(str(c), PALETTE[i % len(PALETTE)]))
    ax.set_xticks(range(len(piv)), piv.index, rotation=20, ha="right")
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel or yv)
    ax.set_title(title)
    ax.legend(ncol=min(n, 4), fontsize=8)
    return savefig(fig, name, cfg)
