#!/usr/bin/env python3
"""Regenerate the paper-facing profile plots in a Julia-like Matplotlib style.

This script replaces the PGFPlots-generated time-sweep PDFs currently included
in main-v15.tex by Python-generated PDFs with:

- one Julia-like blue curve,
- no legend,
- hollow blue circles for global minima,
- hollow blue upward triangles for global maxima.

The source data are read from ``u_shape_numerology_lambda1_profiles.dat``.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

tmpdir = tempfile.gettempdir()
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tmpdir, "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", tmpdir)

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "u_shape_numerology_lambda1_profiles.dat"
JULIA_BLUE = "#4063D8"
JULIA_GREEN = "#389826"
JULIA_RED = "#CB3C33"


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "font.size": 10.5,
        "axes.labelsize": 11.5,
        "axes.linewidth": 1.0,
        "axes.edgecolor": "#222222",
        "xtick.labelsize": 8.7,
        "ytick.labelsize": 9.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "grid.color": "#D6D6D6",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.75,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
    }
)


def load_table(path: Path) -> np.ndarray:
    return np.genfromtxt(path, names=True, dtype=float)


def extrema_mask(values: np.ndarray, mode: str) -> np.ndarray:
    target = np.nanmin(values) if mode == "min" else np.nanmax(values)
    scale = max(1.0, float(np.nanmax(np.abs(values))))
    tol = 1e-9 * scale
    return np.isclose(values, target, atol=tol, rtol=0.0)


def hex_to_rgb(color: str) -> tuple[float, float, float]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{int(round(255.0 * c)):02X}" for c in rgb)


def interpolate_color(color0: str, color1: str, weight: float) -> str:
    rgb0 = hex_to_rgb(color0)
    rgb1 = hex_to_rgb(color1)
    rgb = tuple((1.0 - weight) * c0 + weight * c1 for c0, c1 in zip(rgb0, rgb1))
    return rgb_to_hex(rgb)


def time_label(t_value: float) -> str:
    label = f"{t_value:g}"
    return rf"$t={label}$"


def plot_profile(x: np.ndarray, y: np.ndarray, t_value: float, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.8 / 2.54, 10.8 / 2.54))
    fig.subplots_adjust(left=0.17, right=0.975, bottom=0.19, top=0.91)

    ax.plot(x, y, color=JULIA_BLUE, lw=1.35, solid_capstyle="round")

    min_mask = extrema_mask(y, "min")
    ax.scatter(
        x[min_mask],
        y[min_mask],
        s=42,
        marker="o",
        facecolors="none",
        edgecolors=JULIA_RED,
        linewidths=1.4,
        zorder=4,
        clip_on=False,
    )

    xticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(xticks)
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    span = float(np.nanmax(y) - np.nanmin(y))
    if span <= 0.0:
        span = 1.0
    ax.set_ylim(float(np.nanmin(y) - 0.12 * span), float(np.nanmax(y) + 0.12 * span))
    ax.set_xlabel(r"$\sigma_0$")
    ax.set_title(time_label(t_value), fontsize=10.5, fontweight="bold", pad=6)
    ax.grid(True, which="major")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf")
    plt.close(fig)


def plot_overlay(x: np.ndarray, curves: list[tuple[float, np.ndarray]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.8 / 2.54, 10.8 / 2.54))
    fig.subplots_adjust(left=0.17, right=0.975, bottom=0.19, top=0.985)

    n_curves = len(curves)
    for idx, (_, y) in enumerate(curves):
        weight = idx / max(1, n_curves - 1)
        color = interpolate_color(JULIA_GREEN, JULIA_BLUE, weight)
        alpha = 1.0 if idx in (0, n_curves - 1) else 0.25
        linewidth = 1.35 if idx in (0, n_curves - 1) else 1.05
        ax.plot(x, y, color=color, lw=linewidth, alpha=alpha, solid_capstyle="round")

    xticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(xticks)
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])

    y_all = np.concatenate([y for _, y in curves])
    span = float(np.nanmax(y_all) - np.nanmin(y_all))
    if span <= 0.0:
        span = 1.0
    ax.set_ylim(
        float(np.nanmin(y_all) - 0.12 * span),
        float(np.nanmax(y_all) + 0.12 * span),
    )
    ax.text(
        0.77,
        float(np.interp(0.77, x, curves[0][1]) + 0.06 * span),
        r"$t=1$",
        color=JULIA_GREEN,
        fontsize=10.5,
    )
    ax.text(
        0.15,
        float(np.interp(0.15, x, curves[-1][1]) + 0.06 * span),
        r"$t=2$",
        color=JULIA_BLUE,
        fontsize=10.5,
    )
    ax.set_xlabel(r"$\sigma_0$")
    ax.grid(True, which="major")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf")
    plt.close(fig)


def main() -> None:
    data = load_table(DATA_FILE)
    x = data["s"]

    cases = [
        ("t025_centered", 0.25, ROOT / "fig-v11-t025.pdf"),
        ("t05_centered", 0.5, ROOT / "fig-v11-t05.pdf"),
        ("t1_centered", 1.0, ROOT / "fig-v11-t1.pdf"),
        ("t1p1_centered", 1.1, ROOT / "fig-v11-t1p1.pdf"),
        ("t1p2_centered", 1.2, ROOT / "fig-v11-t1p2.pdf"),
        ("t1p3_centered", 1.3, ROOT / "fig-v11-t1p3.pdf"),
        ("t1p4_centered", 1.4, ROOT / "fig-v11-t1p4.pdf"),
        ("t15_centered", 1.5, ROOT / "fig-v11-t15.pdf"),
        ("t1p6_centered", 1.6, ROOT / "fig-v11-t1p6.pdf"),
        ("t1p7_centered", 1.7, ROOT / "fig-v11-t1p7.pdf"),
        ("t1p8_centered", 1.8, ROOT / "fig-v11-t1p8.pdf"),
        ("t1p9_centered", 1.9, ROOT / "fig-v11-t1p9.pdf"),
        ("t2_centered", 2.0, ROOT / "fig-v11-t2.pdf"),
        ("t5_centered", 5.0, ROOT / "fig-v11-t5.pdf"),
    ]

    for column, t_value, output in cases:
        plot_profile(x, data[column], t_value, output)
        print(f"wrote {output.name}")

    overlay_times = [
        (1.0, data["t1_centered"]),
        (1.1, data["t1p1_centered"]),
        (1.2, data["t1p2_centered"]),
        (1.3, data["t1p3_centered"]),
        (1.4, data["t1p4_centered"]),
        (1.5, data["t15_centered"]),
        (1.6, data["t1p6_centered"]),
        (1.7, data["t1p7_centered"]),
        (1.8, data["t1p8_centered"]),
        (1.9, data["t1p9_centered"]),
        (2.0, data["t2_centered"]),
    ]
    overlay_output = ROOT / "fig-v11-t1-to-t2-overlay.pdf"
    plot_overlay(x, overlay_times, overlay_output)
    print(f"wrote {overlay_output.name}")


if __name__ == "__main__":
    main()
