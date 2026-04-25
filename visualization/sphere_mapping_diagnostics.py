from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.tri import Triangulation

from geometry.sphere_flat_metrics import SphereFlatGeometryCache
from problems.sphere_advection import sphere_tangent_xyz_velocity


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


def plot_flat_patch_mesh(
    VX: np.ndarray,
    VY: np.ndarray,
    EToV: np.ndarray,
    elem_patch_id: np.ndarray,
    out: str | Path,
    title: str = "Flattened square 8-patch mesh",
) -> Path:
    out = Path(out)

    fig, ax = plt.subplots(figsize=(7, 7))
    tri = Triangulation(VX, VY, EToV)
    pc = ax.tripcolor(tri, elem_patch_id, edgecolors="k", linewidth=0.6, alpha=0.85)
    fig.colorbar(pc, ax=ax, label="patch_id")

    cent = np.column_stack([VX[EToV].mean(axis=1), VY[EToV].mean(axis=1)])
    for k, (cx, cy) in enumerate(cent):
        ax.text(cx, cy, str(int(elem_patch_id[k])), ha="center", va="center", fontsize=7)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_table1_nodes_flat(
    VX: np.ndarray,
    VY: np.ndarray,
    EToV: np.ndarray,
    cache: SphereFlatGeometryCache,
    out: str | Path,
    title: str = "Table1 nodes on flattened mesh",
) -> Path:
    out = Path(out)

    fig, ax = plt.subplots(figsize=(7, 7))
    segs = _mesh_segments(VX, VY, EToV)
    ax.add_collection(LineCollection(segs, linewidths=0.5))

    sc = ax.scatter(
        cache.x_flat.ravel(),
        cache.y_flat.ravel(),
        c=cache.node_patch_id.ravel(),
        s=12,
    )
    fig.colorbar(sc, ax=ax, label="node patch_id")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.autoscale()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_sphere_patch_scatter(
    cache: SphereFlatGeometryCache,
    out: str | Path,
    title: str = "Mapped sphere nodes colored by patch_id",
) -> Path:
    out = Path(out)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        cache.X.ravel(),
        cache.Y.ravel(),
        cache.Z.ravel(),
        c=cache.node_patch_id.ravel(),
        s=10,
        depthshade=True,
    )
    fig.colorbar(sc, ax=ax, shrink=0.7, label="patch_id")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    
    max_range = np.nanmax([
        np.nanmax(cache.X) - np.nanmin(cache.X),
        np.nanmax(cache.Y) - np.nanmin(cache.Y),
        np.nanmax(cache.Z) - np.nanmin(cache.Z),
    ])
    mid_x = 0.5 * (np.nanmax(cache.X) + np.nanmin(cache.X))
    mid_y = 0.5 * (np.nanmax(cache.Y) + np.nanmin(cache.Y))
    mid_z = 0.5 * (np.nanmax(cache.Z) + np.nanmin(cache.Z))
    half = 0.5 * max_range
    ax.set_xlim(mid_x - half, mid_x + half)
    ax.set_ylim(mid_y - half, mid_y + half)
    ax.set_zlim(mid_z - half, mid_z + half)

    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_flat_scalar_nodes(
    cache: SphereFlatGeometryCache,
    values: np.ndarray,
    out: str | Path,
    title: str,
    colorbar_label: str,
    mask_bad: bool = False,
) -> Path:
    out = Path(out)

    values = np.asarray(values, dtype=float)
    if values.shape != cache.x_flat.shape:
        raise ValueError("values must have the same shape as cache.x_flat.")

    if mask_bad:
        values = np.where(cache.pole_mask, np.nan, values)

    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(
        cache.x_flat.ravel(),
        cache.y_flat.ravel(),
        c=values.ravel(),
        s=16,
    )
    fig.colorbar(sc, ax=ax, label=colorbar_label)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_pole_mask(
    cache: SphereFlatGeometryCache,
    out: str | Path,
    title: str = "Pole / bad-metric mask",
) -> Path:
    out = Path(out)

    fig, ax = plt.subplots(figsize=(7, 7))

    good = ~cache.pole_mask
    bad = cache.pole_mask

    ax.scatter(cache.x_flat[good], cache.y_flat[good], s=8, label="regular")
    ax.scatter(cache.x_flat[bad], cache.y_flat[bad], s=40, marker="x", label="pole/bad")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_flattened_velocity_quiver(
    cache: SphereFlatGeometryCache,
    u1: np.ndarray,
    u2: np.ndarray,
    out: str | Path,
    title: str = "Flattened velocity field",
    max_arrows: int = 800,
) -> Path:
    out = Path(out)

    x = cache.x_flat.ravel()
    y = cache.y_flat.ravel()
    a = np.asarray(u1, dtype=float).ravel()
    b = np.asarray(u2, dtype=float).ravel()

    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(a) & np.isfinite(b)

    idx = np.flatnonzero(good)
    if idx.size > max_arrows:
        idx = idx[:: max(1, idx.size // max_arrows)]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.quiver(x[idx], y[idx], a[idx], b[idx], angles="xy", scale_units="xy", scale=None)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_sphere_tangent_velocity_quiver(
    cache: SphereFlatGeometryCache,
    u_sph: np.ndarray,
    v_sph: np.ndarray,
    out: str | Path,
    title: str = "Sphere tangent velocity field",
    max_arrows: int = 800,
) -> Path:
    out = Path(out)

    TX, TY, TZ = sphere_tangent_xyz_velocity(
        lambda_=cache.lambda_,
        theta=cache.theta,
        u=u_sph,
        v=v_sph,
    )

    x = cache.X.ravel()
    y = cache.Y.ravel()
    z = cache.Z.ravel()
    a = TX.ravel()
    b = TY.ravel()
    c = TZ.ravel()

    good = (
        np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        & np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
    )

    idx = np.flatnonzero(good)
    if idx.size > max_arrows:
        idx = idx[:: max(1, idx.size // max_arrows)]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x[idx], y[idx], z[idx], s=5)
    ax.quiver(x[idx], y[idx], z[idx], a[idx], b[idx], c[idx], length=0.08, normalize=True)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def write_all_mapping_diagnostics(
    VX: np.ndarray,
    VY: np.ndarray,
    EToV: np.ndarray,
    elem_patch_id: np.ndarray,
    cache: SphereFlatGeometryCache,
    u1: np.ndarray,
    u2: np.ndarray,
    u_sph: np.ndarray,
    v_sph: np.ndarray,
    output_dir: str | Path,
    R: float = 1.0,
) -> list[Path]:
    output_dir = _ensure_dir(output_dir)
    paths: list[Path] = []

    paths.append(plot_flat_patch_mesh(
        VX, VY, EToV, elem_patch_id,
        output_dir / "01_flat_patch_mesh.png",
    ))

    paths.append(plot_table1_nodes_flat(
        VX, VY, EToV, cache,
        output_dir / "02_table1_nodes_flat.png",
    ))

    paths.append(plot_sphere_patch_scatter(
        cache,
        output_dir / "03_sphere_patch_scatter.png",
    ))

    radial = np.sqrt(cache.X**2 + cache.Y**2 + cache.Z**2)
    radial_err = np.abs(radial - R)

    paths.append(plot_flat_scalar_nodes(
        cache, radial_err,
        output_dir / "04_radial_error_flat.png",
        title="Radial error |sqrt(X^2+Y^2+Z^2)-R|",
        colorbar_label="radial error",
        mask_bad=False,
    ))

    paths.append(plot_flat_scalar_nodes(
        cache, cache.sqrtG,
        output_dir / "05_sqrtG_flat.png",
        title="sqrt(det(A^T A)) on flattened square",
        colorbar_label="sqrtG",
        mask_bad=True,
    ))

    paths.append(plot_flat_scalar_nodes(
        cache, cache.A_Ainv_err,
        output_dir / "06_A_Ainv_error_flat.png",
        title="||A Ainv - I||_inf",
        colorbar_label="A-Ainv error",
        mask_bad=True,
    ))

    paths.append(plot_pole_mask(
        cache,
        output_dir / "07_pole_mask.png",
    ))

    speed_flat = np.sqrt(u1**2 + u2**2)
    paths.append(plot_flat_scalar_nodes(
        cache, speed_flat,
        output_dir / "08_flattened_velocity_magnitude.png",
        title="Flattened velocity magnitude sqrt(u1^2+u2^2)",
        colorbar_label="|u_flat|",
        mask_bad=True,
    ))

    paths.append(plot_flattened_velocity_quiver(
        cache, u1, u2,
        output_dir / "09_flattened_velocity_quiver.png",
    ))

    paths.append(plot_sphere_tangent_velocity_quiver(
        cache, u_sph, v_sph,
        output_dir / "10_sphere_tangent_velocity_quiver.png",
    ))

    return paths
