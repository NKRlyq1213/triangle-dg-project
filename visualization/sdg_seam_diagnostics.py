from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from geometry.sdg_seam_connectivity import (
    map_face_samples_to_sphere,
    seam_pair_xyz_errors,
    sample_face_points,
)


def _ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _mesh_segments(VX: np.ndarray, VY: np.ndarray, EToV: np.ndarray) -> list[np.ndarray]:
    segs = []
    for tri in EToV:
        pts = np.column_stack([VX[tri], VY[tri]])
        segs.append(pts[[0, 1]])
        segs.append(pts[[1, 2]])
        segs.append(pts[[2, 0]])
    return segs


def _face_midpoint(VX, VY, conn, elem: int, face: int) -> np.ndarray:
    pts = sample_face_points(VX, VY, conn, elem, face, n_samples=1, include_endpoints=True)
    return np.mean(pts, axis=0)


def plot_flat_seam_pairing(
    VX: np.ndarray,
    VY: np.ndarray,
    EToV: np.ndarray,
    conn: dict,
    out: str | Path,
) -> Path:
    """
    Plot flat square seam pairs. Each artificial pair is connected by a line
    between face midpoints.
    """
    out = Path(out)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    ax.add_collection(LineCollection(_mesh_segments(VX, VY, EToV), linewidths=0.4, alpha=0.45))

    pairs = list(conn["sphere_seam_pairs"])

    for i, pair in enumerate(pairs):
        pL = _face_midpoint(VX, VY, conn, pair.elem_L, pair.face_L)
        pR = _face_midpoint(VX, VY, conn, pair.elem_R, pair.face_R)

        ax.plot([pL[0], pR[0]], [pL[1], pR[1]], linewidth=1.0)
        ax.scatter([pL[0], pR[0]], [pL[1], pR[1]], s=28)
        c = 0.5 * (pL + pR)
        ax.text(c[0], c[1], str(i), fontsize=7, ha="center", va="center")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("SDG seam pairing on flattened square")
    ax.autoscale()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_sphere_seam_overlap(
    VX: np.ndarray,
    VY: np.ndarray,
    conn: dict,
    elem_patch_id: np.ndarray,
    out: str | Path,
    R: float = 1.0,
    n_samples: int = 9,
    tol: float = 1.0e-12,
) -> Path:
    """
    Plot paired seam samples on the sphere.

    If seam pairing is correct, every pair should overlap visually.
    """
    out = Path(out)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    pairs = list(conn["sphere_seam_pairs"])

    for i, pair in enumerate(pairs):
        _, xyz_L = map_face_samples_to_sphere(
            VX, VY, conn, elem_patch_id,
            elem=pair.elem_L,
            face=pair.face_L,
            n_samples=n_samples,
            R=R,
            tol=tol,
        )
        _, xyz_R = map_face_samples_to_sphere(
            VX, VY, conn, elem_patch_id,
            elem=pair.elem_R,
            face=pair.face_R,
            n_samples=n_samples,
            R=R,
            tol=tol,
        )

        if pair.flip:
            xyz_R = xyz_R[::-1]

        ax.plot(xyz_L[:, 0], xyz_L[:, 1], xyz_L[:, 2], linewidth=1.5)
        ax.scatter(xyz_R[:, 0], xyz_R[:, 1], xyz_R[:, 2], s=12)

        mid = xyz_L[len(xyz_L) // 2]
        ax.text(mid[0], mid[1], mid[2], str(i), fontsize=7)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("SDG seam pairs mapped to sphere")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_seam_xyz_error(
    VX: np.ndarray,
    VY: np.ndarray,
    conn: dict,
    elem_patch_id: np.ndarray,
    out: str | Path,
    R: float = 1.0,
    n_samples: int = 9,
    tol: float = 1.0e-12,
) -> Path:
    out = Path(out)

    errs = seam_pair_xyz_errors(
        VX=VX,
        VY=VY,
        conn=conn,
        elem_patch_id=elem_patch_id,
        n_samples=n_samples,
        R=R,
        tol=tol,
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    idx = np.arange(errs.size)
    vals = np.maximum(errs, 1.0e-18)
    ax.bar(idx, vals)
    ax.set_yscale("log")
    ax.set_xlabel("seam pair index")
    ax.set_ylabel("max sampled xyz mismatch")
    ax.set_title("SDG seam-pair sphere-coordinate mismatch")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_boundary_status(
    VX: np.ndarray,
    VY: np.ndarray,
    EToV: np.ndarray,
    conn: dict,
    out: str | Path,
) -> Path:
    """
    Plot remaining boundary faces after SDG seam augmentation.

    Expected result:
        no highlighted remaining boundary faces.
    """
    out = Path(out)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.add_collection(LineCollection(_mesh_segments(VX, VY, EToV), linewidths=0.4, alpha=0.35))

    boundary = np.asarray(conn["boundary_faces"], dtype=int)
    for k, f in boundary:
        pts = sample_face_points(VX, VY, conn, int(k), int(f), n_samples=2, include_endpoints=True)
        ax.plot(pts[:, 0], pts[:, 1], linewidth=3.0)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Remaining boundary faces after SDG seam pairing: {boundary.shape[0]}")
    ax.autoscale()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def write_all_sdg_seam_diagnostics(
    VX: np.ndarray,
    VY: np.ndarray,
    EToV: np.ndarray,
    elem_patch_id: np.ndarray,
    conn: dict,
    output_dir: str | Path,
    R: float = 1.0,
    n_samples: int = 9,
    tol: float = 1.0e-12,
) -> list[Path]:
    output_dir = _ensure_dir(output_dir)
    paths: list[Path] = []

    paths.append(plot_flat_seam_pairing(
        VX=VX,
        VY=VY,
        EToV=EToV,
        conn=conn,
        out=output_dir / "01_sdg_flat_seam_pairing.png",
    ))

    paths.append(plot_sphere_seam_overlap(
        VX=VX,
        VY=VY,
        conn=conn,
        elem_patch_id=elem_patch_id,
        out=output_dir / "02_sdg_sphere_seam_overlap.png",
        R=R,
        n_samples=n_samples,
        tol=tol,
    ))

    paths.append(plot_seam_xyz_error(
        VX=VX,
        VY=VY,
        conn=conn,
        elem_patch_id=elem_patch_id,
        out=output_dir / "03_sdg_seam_xyz_error.png",
        R=R,
        n_samples=n_samples,
        tol=tol,
    ))

    paths.append(plot_boundary_status(
        VX=VX,
        VY=VY,
        EToV=EToV,
        conn=conn,
        out=output_dir / "04_sdg_remaining_boundary_faces.png",
    ))

    return paths
