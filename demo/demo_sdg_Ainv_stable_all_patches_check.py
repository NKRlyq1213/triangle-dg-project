from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry.sphere_flat_mesh import macro_patch_triangles
from geometry.sdg_sphere_mapping import (
    sdg_Ainv_from_A_explicit,
    sdg_Ainv_stable_from_lambda_theta_patch,
    sdg_mapping_from_xy_patch,
)


def parse_args():
    p = argparse.ArgumentParser(description="Check SDG stable Ainv formulas for all T1--T8 patches.")
    p.add_argument("--R", type=float, default=1.0)
    p.add_argument("--n", type=int, default=100, help="Barycentric sampling resolution per macro patch.")
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


def sample_triangle(vertices: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            a = 1.0 - (i + j) / n
            b = i / n
            c = j / n
            p = a * vertices[0] + b * vertices[1] + c * vertices[2]
            pts.append(p)
    pts = np.asarray(pts, dtype=float)
    return pts[:, 0], pts[:, 1]


def plot_per_patch_bar(summary: dict[int, dict[str, float]], key: str, out: Path, title: str):
    pids = np.arange(1, 9)
    vals = np.array([summary[int(pid)][key] for pid in pids], dtype=float)
    vals = np.maximum(vals, 1.0e-18)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(pids, vals)
    ax.set_yscale("log")
    ax.set_xlabel("patch_id")
    ax.set_ylabel(key)
    ax.set_title(title)
    ax.set_xticks(pids)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def lambda_range_for_patch(pid: int) -> tuple[float, float, str]:
    """
    Return lambda range and pole type according to SDG convention.
    """
    if pid == 1:
        return 0.0, 0.5 * np.pi, "north"
    if pid == 2:
        return 0.0, 0.5 * np.pi, "south"
    if pid == 3:
        return 0.5 * np.pi, np.pi, "north"
    if pid == 4:
        return 0.5 * np.pi, np.pi, "south"
    if pid == 5:
        return np.pi, 1.5 * np.pi, "north"
    if pid == 6:
        return np.pi, 1.5 * np.pi, "south"
    if pid == 7:
        return -0.5 * np.pi, 0.0, "north"
    if pid == 8:
        return -0.5 * np.pi, 0.0, "south"
    raise ValueError("pid must be 1,...,8")


def plot_directional_limit_for_patch(pid: int, R: float, out: Path):
    a, b, pole = lambda_range_for_patch(pid)

    lam = np.linspace(a, b, 300)
    if pole == "north":
        theta = np.full_like(lam, 0.5 * np.pi - 1.0e-12)
    else:
        theta = np.full_like(lam, -0.5 * np.pi + 1.0e-12)

    patch_id = np.full_like(lam, pid, dtype=int)
    B = sdg_Ainv_stable_from_lambda_theta_patch(lam, theta, patch_id, R=R)

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
        ax.plot(lam, vals)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\lambda$")

    fig.suptitle(f"T{pid} stable Ainv directional limit at {pole} pole")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    outdir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else ROOT / "outputs" / "sdg_Ainv_stable_all_patches"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    VX0, VY0, EToV0, patch0 = macro_patch_triangles(R=1.0)

    summary: dict[int, dict[str, float]] = {}

    for local_k, pid in enumerate(patch0):
        verts = np.column_stack([VX0[EToV0[local_k]], VY0[EToV0[local_k]]])
        x, y = sample_triangle(verts, n=args.n)
        patch_id = np.full_like(x, int(pid), dtype=int)

        out = sdg_mapping_from_xy_patch(
            x=x,
            y=y,
            patch_id=patch_id,
            R=args.R,
        )

        Ainv_stable = sdg_Ainv_stable_from_lambda_theta_patch(
            out.lambda_,
            out.theta,
            patch_id,
            R=args.R,
        )
        Ainv_adjugate = sdg_Ainv_from_A_explicit(out.A, R=args.R)
        Ainv_numpy = inv_numpy_batch(out.A, bad=out.bad_mask)

        err_stable_numpy = max_norm_matrix_diff(Ainv_stable, Ainv_numpy)
        err_stable_adjugate = max_norm_matrix_diff(Ainv_stable, Ainv_adjugate)
        err_mapping_stable = max_norm_matrix_diff(out.Ainv, Ainv_stable)

        regular = (
            (~out.bad_mask)
            & np.isfinite(err_stable_numpy)
            & np.isfinite(err_stable_adjugate)
            & np.isfinite(err_mapping_stable)
        )

        if not np.any(regular):
            summary[int(pid)] = {
                "n_regular": 0,
                "stable_vs_numpy": float("nan"),
                "stable_vs_adjugate": float("nan"),
                "mapping_vs_stable": float("nan"),
            }
            continue

        summary[int(pid)] = {
            "n_regular": int(np.sum(regular)),
            "stable_vs_numpy": float(np.max(err_stable_numpy[regular])),
            "stable_vs_adjugate": float(np.max(err_stable_adjugate[regular])),
            "mapping_vs_stable": float(np.max(err_mapping_stable[regular])),
        }

        plot_directional_limit_for_patch(
            pid=int(pid),
            R=args.R,
            out=outdir / f"directional_limit_T{int(pid)}.png",
        )

    print("\n=== SDG stable Ainv all-patch consistency ===")
    for pid in range(1, 9):
        print(f"T{pid}: {summary[pid]}")

    plot_per_patch_bar(
        summary,
        key="stable_vs_numpy",
        out=outdir / "01_per_patch_Ainv_stable_vs_numpy.png",
        title="Per-patch max ||Ainv_stable - inv(A)||inf",
    )

    plot_per_patch_bar(
        summary,
        key="stable_vs_adjugate",
        out=outdir / "02_per_patch_Ainv_stable_vs_adjugate.png",
        title="Per-patch max ||Ainv_stable - adj(A)/(pi R^2)||inf",
    )

    plot_per_patch_bar(
        summary,
        key="mapping_vs_stable",
        out=outdir / "03_per_patch_mapping_Ainv_vs_stable.png",
        title="Per-patch max ||Ainv_mapping - Ainv_stable||inf",
    )

    print("\n=== Output figures ===")
    print(outdir / "01_per_patch_Ainv_stable_vs_numpy.png")
    print(outdir / "02_per_patch_Ainv_stable_vs_adjugate.png")
    print(outdir / "03_per_patch_mapping_Ainv_vs_stable.png")
    for pid in range(1, 9):
        print(outdir / f"directional_limit_T{pid}.png")


if __name__ == "__main__":
    main()
