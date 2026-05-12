#!/usr/bin/env python3
"""Generate plain-text data for the fixed-lambda U-shape numerology figures.

The script uses only the Python standard library. It computes:

1. The threshold curve t_crit(beta) at fixed lambda, M, and Fourier truncation.
2. Several sample truncated soft-accuracy profiles S_t^{[n_max]}(s).
3. A common-M multi-head profile exhibiting a double well.

The formulas match the current notation in main-v6/main-v7:

    a_n = n^2 I_n(beta),
    omega_{n,M} = exp(-pi^2 n^2 / (2 M^2)),
    g_{a_n}(t,1;s)
      = [lambda * exp(lambda s) / (exp(lambda)-1)] psi_{a_n t}(Y_lambda(s)),

where

    Y_lambda(s) = log((exp(lambda)-1)/(exp(lambda s)-1)),
    psi_c(y) = sum_{k>=0} c^{k+1} y^k / (k! (k+1)!).
"""

from __future__ import annotations

import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def bessel_i_int(n: int, x: float, tol: float = 1e-15, max_terms: int = 10000) -> float:
    """Modified Bessel I_n(x) for integer n >= 0, via its defining series."""
    term = (x / 2.0) ** n / math.factorial(n)
    total = term
    for k in range(max_terms):
        term *= (x * x / 4.0) / ((k + 1) * (k + n + 1))
        total += term
        if abs(term) <= tol * max(1.0, abs(total)):
            break
    return total


def psi_c(c: float, y: float, tol: float = 1e-15, max_terms: int = 10000) -> float:
    """Series representation of psi_c(y)."""
    term = c
    total = term
    for k in range(max_terms):
        term *= c * y / ((k + 1) * (k + 2))
        total += term
        if abs(term) <= tol * max(1.0, abs(total)):
            break
    return total


def y_lambda(s: float, lam: float) -> float:
    if lam == 0.0:
        return math.log(1.0 / s)
    return math.log(math.expm1(lam) / math.expm1(lam * s))


def omega_nm(n: int, M: float) -> float:
    return math.exp(-0.5 * (math.pi * n / M) ** 2)


def profile_value(s: float, beta: float, t: float, lam: float, M: float, n_max: int) -> float:
    y = y_lambda(s, lam)
    if lam == 0.0:
        coeff = 1.0
    else:
        coeff = lam * math.exp(lam * s) / math.expm1(lam)

    total = 0.0
    for n in range(1, n_max + 1):
        a_n = (n ** 2) * bessel_i_int(n, beta)
        total += omega_nm(n, M) * coeff * psi_c(a_n * t, y)
    return total


def multi_head_profile_value(
    s: float,
    cases: list[tuple[float, float, float]],
    t: float,
    M: float,
    n_max: int,
) -> float:
    total = 0.0
    for weight, beta, lam in cases:
        total += weight * profile_value(s, beta, t, lam, M, n_max)
    return total


def t_crit(beta: float, lam: float, M: float, n_max: int) -> float:
    """Threshold where the right-end slope of the truncated profile changes sign."""
    s1 = 0.0
    s2 = 0.0
    for n in range(1, n_max + 1):
        b_n = (n ** 2) * bessel_i_int(n, beta)
        w_n = omega_nm(n, M)
        s1 += w_n * b_n
        s2 += w_n * b_n * b_n
    return 2.0 * (1.0 - math.exp(-lam)) * s1 / s2


def write_threshold_curve(path: Path, lam: float, M: float, n_max: int) -> None:
    with path.open("w", encoding="ascii") as fh:
        fh.write("beta t_crit\n")
        for j in range(1, 49):
            beta = 0.10 * j
            fh.write(f"{beta:.6f} {t_crit(beta, lam, M, n_max):.12f}\n")


def write_profiles(path: Path, lam: float, M: float, n_max: int) -> None:
    beta = 1.0
    cases = [
        ("t025", beta, 0.25),
        ("t05", beta, 0.5),
        ("t1", beta, 1.0),
        ("t1p1", beta, 1.1),
        ("t1p2", beta, 1.2),
        ("t1p3", beta, 1.3),
        ("t1p4", beta, 1.4),
        ("t15", beta, 1.5),
        ("t1p6", beta, 1.6),
        ("t1p7", beta, 1.7),
        ("t1p8", beta, 1.8),
        ("t1p9", beta, 1.9),
        ("t2", beta, 2.0),
        ("t5", beta, 5.0),
    ]
    grid = [0.02 + j * (0.96 / 799.0) for j in range(800)]
    columns: list[tuple[str, list[float], list[float]]] = []
    for name, beta, t in cases:
        values = [profile_value(s, beta, t, lam, M, n_max) for s in grid]
        offset = min(values)
        centered = [value - offset for value in values]
        columns.append((name, values, centered))

    with path.open("w", encoding="ascii") as fh:
        header_entries = ["s"]
        for name, _, _ in cases:
            header_entries.append(name)
            header_entries.append(f"{name}_centered")
        header = " ".join(header_entries)
        fh.write(f"{header}\n")
        for i, s in enumerate(grid):
            row_entries = [f"{s:.6f}"]
            for _, values, centered in columns:
                row_entries.append(f"{values[i]:.12f}")
                row_entries.append(f"{centered[i]:.12f}")
            fh.write(" ".join(row_entries) + "\n")


def write_common_m_doublewell_profile(path: Path, M: float, n_max: int) -> None:
    t = 1.5
    cases = [
        (1.0, 0.2, 3.0),
        (-27.519314453632965, 0.2, 0.25),
        (-1.5515192155322617, 0.5, 0.25),
        (3.9262326774050965e-06, 3.0, 2.0),
    ]
    grid = [0.02 + j * (0.96 / 799.0) for j in range(800)]
    values = [multi_head_profile_value(s, cases=cases, t=t, M=M, n_max=n_max) for s in grid]
    offset = min(values)

    with path.open("w", encoding="ascii") as fh:
        fh.write("s value centered\n")
        for s, value in zip(grid, values):
            fh.write(f"{s:.6f} {value:.12f} {value - offset:.12f}\n")


def main() -> None:
    lam = 1.0
    M = 8.0
    n_max = 20

    threshold_path = ROOT / "u_shape_numerology_lambda1_tcrit.dat"
    profiles_path = ROOT / "u_shape_numerology_lambda1_profiles.dat"
    doublewell_path = ROOT / "u_shape_numerology_commonM_multhead_doublewell.dat"

    write_threshold_curve(threshold_path, lam=lam, M=M, n_max=n_max)
    write_profiles(profiles_path, lam=lam, M=M, n_max=n_max)
    write_common_m_doublewell_profile(doublewell_path, M=M, n_max=n_max)

    print(f"wrote {threshold_path.name}")
    print(f"wrote {profiles_path.name}")
    print(f"wrote {doublewell_path.name}")
    for beta in (1.0, 2.0, 3.0):
        print(f"beta={beta:.1f}, t_crit={t_crit(beta, lam, M, n_max):.6f}")


if __name__ == "__main__":
    main()
