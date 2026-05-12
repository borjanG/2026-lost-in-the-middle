#!/usr/bin/env python3
"""Simulate one ALiBi particle system and plot circle trajectories.

The figures imitate the black-background trajectory panels in Agazzi et al.,
Figure 1, adapted to the one-dimensional torus model used in the paper.  Token
colors encode the initial cluster membership.
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

tmpdir = tempfile.gettempdir()
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tmpdir, "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", tmpdir)

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np


ROOT = Path(__file__).resolve().parent

plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{upgreek}",
        "font.family": "serif",
    }
)


def build_causal_weights(n_tokens: int, lam: float) -> np.ndarray:
    weights = np.zeros((n_tokens, n_tokens), dtype=float)
    for j in range(1, n_tokens):
        k = np.arange(j)
        coeffs = np.exp(-(lam / n_tokens) * (j - k))
        weights[j, :j] = coeffs / coeffs.sum()
    return weights


def force(theta: np.ndarray, beta: float, weights: np.ndarray) -> np.ndarray:
    n_tokens = theta.size
    out = np.zeros_like(theta)
    for j in range(1, n_tokens):
        diff = theta[j] - theta[:j]
        interaction = -beta * np.sin(diff) * np.exp(beta * np.cos(diff))
        out[j] = np.dot(weights[j, :j], interaction)
    return out


def rk4_step(theta: np.ndarray, dt: float, beta: float, weights: np.ndarray) -> np.ndarray:
    k1 = force(theta, beta, weights)
    k2 = force(theta + 0.5 * dt * k1, beta, weights)
    k3 = force(theta + 0.5 * dt * k2, beta, weights)
    k4 = force(theta + dt * k3, beta, weights)
    return theta + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def initial_configuration(
    n_tokens: int,
    seed: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if mode == "uniform":
        theta = rng.uniform(0.0, 2.0 * math.pi, size=n_tokens)
        labels = np.arange(n_tokens)
        return theta, labels

    if mode == "clusters":
        n_clusters = 4
        labels = np.repeat(np.arange(n_clusters), math.ceil(n_tokens / n_clusters))[:n_tokens]
        centers = np.linspace(0.1, 1.9, n_clusters, endpoint=True) * math.pi
        theta = centers[labels] + rng.normal(0.0, 0.075, size=n_tokens)
        return np.mod(theta, 2.0 * math.pi), labels

    raise ValueError(f"unknown initial configuration mode: {mode}")


def simulate(
    n_tokens: int,
    beta: float,
    lam: float,
    t_max: float,
    dt: float,
    seed: int,
    initial_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta, labels = initial_configuration(n_tokens, seed=seed, mode=initial_mode)
    weights = build_causal_weights(n_tokens, lam)

    n_steps = int(round(t_max / dt))
    times = np.linspace(0.0, t_max, n_steps + 1)
    history = np.empty((n_steps + 1, n_tokens), dtype=float)
    history[0] = theta

    for step in range(1, n_steps + 1):
        theta = rk4_step(theta, dt, beta, weights)
        history[step] = theta

    return times, history, theta, labels


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("#08090D")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    circle = plt.Circle((0.0, 0.0), 1.0, fill=False, color="#353846", lw=0.9, alpha=0.95)
    ax.add_artist(circle)


def position_colors(n_tokens: int) -> np.ndarray:
    return plt.get_cmap("hsv")(np.linspace(0.0, 0.86, n_tokens))


def cluster_colors(labels: np.ndarray) -> np.ndarray:
    base_hex = ["#00A4EF", "#F25022", "#7FBA00", "#FFB900"]
    colors = np.empty((labels.size, 4), dtype=float)
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        base = np.array(mcolors.to_rgb(base_hex[int(label)]))
        for token_index in members:
            colors[token_index, :3] = base
            colors[token_index, 3] = 1.0
    return colors


def plot_snapshot(
    ax: plt.Axes,
    times: np.ndarray,
    history: np.ndarray,
    t_value: float,
    colors: np.ndarray,
) -> None:
    style_axis(ax)
    idx = int(np.argmin(np.abs(times - t_value)))
    stride = max(1, idx // 350)
    theta_initial = history[0]

    initial_colors = np.array(colors, copy=True)
    initial_colors[:, 3] = 0.25
    ax.scatter(
        np.cos(theta_initial),
        np.sin(theta_initial),
        s=18,
        marker="x",
        color=initial_colors,
        linewidths=0.8,
        zorder=2,
    )

    for j, color in enumerate(colors):
        theta_path = history[: idx + 1 : stride, j]
        ax.plot(
            np.cos(theta_path),
            np.sin(theta_path),
            color=color,
            lw=0.85,
            alpha=0.86,
            solid_capstyle="round",
        )

    theta_now = history[idx]
    ax.scatter(
        np.cos(theta_now),
        np.sin(theta_now),
        s=17,
        marker="o",
        facecolors="none",
        edgecolors=colors,
        linewidths=0.85,
        zorder=4,
    )
    for token_index, label, offset, angle_shift in (
        (0, r"$\uptheta_1(t)$", 0.74, 0.08),
        (history.shape[1] - 1, r"$\uptheta_N(t)$", 0.68, -0.07),
    ):
        angle = theta_now[token_index] + angle_shift
        radius = offset
        ax.text(
            radius * np.cos(angle),
            radius * np.sin(angle),
            label,
            color=colors[token_index],
            fontsize=8.5,
            ha="center",
            va="center",
            clip_on=False,
        )
    ax.set_title(rf"$t={t_value:g}$", fontsize=9.5, fontweight="bold", pad=6)


def plot_grid(
    times: np.ndarray,
    history: np.ndarray,
    t_values: list[float],
    output: Path,
    colors: np.ndarray,
) -> None:
    n_cols = 4
    n_rows = math.ceil(len(t_values) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.15 * n_cols, 2.35 * n_rows),
        squeeze=False,
    )
    fig.patch.set_facecolor("white")

    for ax, t_value in zip(axes.flat, t_values):
        plot_snapshot(ax, times, history, t_value, colors)
    for ax in axes.flat[len(t_values) :]:
        ax.axis("off")

    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.92, wspace=0.08, hspace=0.28)
    fig.savefig(output, format="pdf")
    plt.close(fig)


def plot_single(
    times: np.ndarray,
    history: np.ndarray,
    t_value: float,
    output: Path,
    colors: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(2.6, 2.8))
    fig.patch.set_facecolor("white")
    plot_snapshot(ax, times, history, t_value, colors)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.9)
    fig.savefig(output, format="pdf")
    plt.close(fig)


def time_tag(t_value: float) -> str:
    label = f"{t_value:.2f}".rstrip("0").rstrip(".")
    return label.replace(".", "p")


def main() -> None:
    n_tokens = 64
    beta = 1.0
    lam = 1.0
    t_max = 10.0
    dt = 0.0025
    seed = 17
    t_values = [0.2 * k for k in range(51)]

    times, history, _, labels = simulate(
        n_tokens=n_tokens,
        beta=beta,
        lam=lam,
        t_max=t_max,
        dt=dt,
        seed=seed,
        initial_mode="clusters",
    )
    color_array = cluster_colors(labels)

    for t_value in t_values:
        tag = time_tag(t_value)
        output = ROOT / f"fig-alibi-particle-circle-clusters-t{tag}.pdf"
        plot_single(times, history, t_value, output, colors=color_array)
        print(f"wrote {output.name}")

    data_output = ROOT / "alibi_particle_circle_clusters_dynamics.npz"
    np.savez_compressed(
        data_output,
        times=times,
        theta=history,
        beta=beta,
        lam=lam,
        n_tokens=n_tokens,
        seed=seed,
        labels=labels,
    )
    print(f"wrote {data_output.name}")


if __name__ == "__main__":
    main()
