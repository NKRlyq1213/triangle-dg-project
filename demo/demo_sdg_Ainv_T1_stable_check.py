from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry.sdg_sphere_mapping import (
    sdg_A_from_lambda_theta_patch,
    sdg_Ainv_T1_stable,
    sdg_Ainv_from_A_explicit,
    sdg_mapping_from_xy_patch,
)


def parse_args():
    p = argparse.ArgumentParser(description="Check SDG T1 pole-stable Ainv correction.")
    p.add_argument("--R", type=float, default=1.0)
    p.add_argument("--n", type=int, default=80, help="Grid resolution inside T1.")
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()


def inv_numpy_batch(A: np.ndarray, bad: np.ndarray | None = None) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    out = np.full_like(A, np.nan)

    flat_A = A.reshape((-1, 2, 2))
    flat_out = out.reshape((-1, 2, 2))

    if bad is None:
        flat_bad = np.zeros((flat_A.shape[0],), dtype=bool)
    else:
        flat_bad = np.asarray(bad, dtype=bool).reshape(-1)

    for i in range(flat_A.shape[0]):
        if flat_bad[i]:
            continue
        try:
            flat_out[i] = np.linalg.inv(flat_A[i])
        except np.linalg.LinAlgError:
            pass

    return out


def max_norm_matrix_diff(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.max(np.abs(A - B), axis=(-1, -2))


def build_T1_sample_points(n: int):
    xs = np.linspace(0.0, 1.0, n + 1)
    ys = np.linspace(0.0, 1.0, n + 1)

    pts = []
    for x in xs:
        for y in ys:
            if x + y <= 1.0 and x + y > 1.0e-12:
                pts.append((x, y))

    pts = np.asarray(pts, dtype=float)
    return pts[:, 0], pts[:, 1]


def plot_flat_error(x, y, err, out: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    log_err = np.log10(np.maximum(err, 1.0e-18))

    sc = ax.scatter(x, y, c=log_err, s=18)
    fig.colorbar(sc, ax=ax, label="log10(error)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("T1 stable Ainv vs numpy inv(A)")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_directional_limit(lams, B, out: Path):
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True)

    labels = [
        r"$A^{-1}_{00}$",
        r"$A^{-1}_{01}$",
        r"$A^{-1}_{10}$",
        r"$A^{-1}_{11}$",
    ]

    entries = [
        B[..., 0, 0],
        B[..., 0, 1],
        B[..., 1, 0],
        B[..., 1, 1],
    ]

    for ax, label, vals in zip(axes.ravel(), labels, entries):
        ax.plot(lams, vals)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\lambda$")

    fig.suptitle(r"T1 pole directional limit of stable $A^{-1}$, $\theta\to\pi/2^-$")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    outdir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else ROOT / "outputs" / "sdg_Ainv_T1_stable"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # 1. Non-pole consistency check inside T1
    # ------------------------------------------------------------
    x, y = build_T1_sample_points(args.n)
    pid = np.ones_like(x, dtype=int)

    out = sdg_mapping_from_xy_patch(
        x=x,
        y=y,
        patch_id=pid,
        R=args.R,
    )

    Ainv_stable = sdg_Ainv_T1_stable(out.lambda_, out.theta, R=args.R)
    Ainv_adjugate = sdg_Ainv_from_A_explicit(out.A, R=args.R)
    Ainv_numpy = inv_numpy_batch(out.A, bad=out.bad_mask)

    err_stable_numpy = max_norm_matrix_diff(Ainv_stable, Ainv_numpy)
    err_stable_adjugate = max_norm_matrix_diff(Ainv_stable, Ainv_adjugate)

    regular = (~out.bad_mask) & np.isfinite(err_stable_numpy)

    print("\n=== T1 non-pole consistency ===")
    print(f"n regular samples: {int(np.sum(regular))}")
    print(f"max ||Ainv_stable - inv(A)||inf: {float(np.max(err_stable_numpy[regular])):.6e}")
    print(f"max ||Ainv_stable - adj(A)/(pi R^2)||inf: {float(np.max(err_stable_adjugate[regular])):.6e}")

    plot_flat_error(
        x=x[regular],
        y=y[regular],
        err=err_stable_numpy[regular],
        out=outdir / "01_T1_Ainv_stable_vs_numpy_error.png",
    )

    # ------------------------------------------------------------
    # 2. Directional pole limit check
    # ------------------------------------------------------------
    lams = np.linspace(0.0, 0.5 * np.pi, 300)
    theta_lim = np.full_like(lams, 0.5 * np.pi - 1.0e-12)

    B_lim = sdg_Ainv_T1_stable(lams, theta_lim, R=args.R)

    plot_directional_limit(
        lams=lams,
        B=B_lim,
        out=outdir / "02_T1_Ainv_directional_limit_entries.png",
    )

    print("\n=== T1 directional pole limits ===")
    for lam in [0.0, 0.25 * np.pi, 0.5 * np.pi]:
        B = sdg_Ainv_T1_stable(
            np.array([lam]),
            np.array([0.5 * np.pi - 1.0e-12]),
            R=args.R,
        )[0]
        print(f"\nlambda = {lam:.12f}")
        print(B)

    print("\n=== Output figures ===")
    print(outdir / "01_T1_Ainv_stable_vs_numpy_error.png")
    print(outdir / "02_T1_Ainv_directional_limit_entries.png")


if __name__ == "__main__":
    main()
