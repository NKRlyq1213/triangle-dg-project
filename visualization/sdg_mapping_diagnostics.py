from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.tri import Triangulation

from geometry.sdg_sphere_mapping import sdg_detA_expected, sdg_sqrtG_expected
from geometry.sphere_flat_metrics import SphereFlatGeometryCache, per_patch_diagnostics
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


def plot_flat_patch_mesh(VX, VY, EToV, elem_patch_id, out):
    fig, ax = plt.subplots(figsize=(7, 7))
    tri = Triangulation(VX, VY, EToV)
    pc = ax.tripcolor(tri, elem_patch_id, edgecolors="k", linewidth=0.55, alpha=0.85)
    fig.colorbar(pc, ax=ax, label="patch_id")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("SDG flattened square mesh, colored by patch_id")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_table1_nodes_flat(VX, VY, EToV, cache: SphereFlatGeometryCache, out):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.add_collection(LineCollection(_mesh_segments(VX, VY, EToV), linewidths=0.45))
    sc = ax.scatter(cache.x_flat.ravel(), cache.y_flat.ravel(), c=cache.node_patch_id.ravel(), s=12)
    fig.colorbar(sc, ax=ax, label="node patch_id")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Table 1 nodes on SDG flattened mesh")
    ax.autoscale()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_sphere_patch_scatter(cache: SphereFlatGeometryCache, out):
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
    ax.set_title("SDG mapped sphere nodes, colored by patch_id")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_flat_scalar(cache: SphereFlatGeometryCache, values, out, title, label, mask_bad: bool = True):
    values = np.asarray(values, dtype=float)
    if values.shape != cache.x_flat.shape:
        raise ValueError("values must match cache.x_flat shape.")
    if mask_bad:
        values = np.where(cache.bad_mask, np.nan, values)

    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(cache.x_flat.ravel(), cache.y_flat.ravel(), c=values.ravel(), s=16)
    fig.colorbar(sc, ax=ax, label=label)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_mask(cache: SphereFlatGeometryCache, out):
    fig, ax = plt.subplots(figsize=(7, 7))
    good = ~cache.bad_mask
    bad = cache.bad_mask
    ax.scatter(cache.x_flat[good], cache.y_flat[good], s=8, label="regular")
    ax.scatter(cache.x_flat[bad], cache.y_flat[bad], s=45, marker="x", label="pole / bad")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("SDG pole / bad-node mask")
    ax.legend()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_per_patch_bar(cache: SphereFlatGeometryCache, R: float, key: str, out, title, ylabel):
    diag = per_patch_diagnostics(cache, R=R)
    pids = np.arange(1, 9)
    vals = np.array([diag[int(pid)][key] for pid in pids], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(pids, vals)
    ax.set_xticks(pids)
    ax.set_xlabel("patch_id")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_yscale("log")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_flattened_velocity_quiver(cache: SphereFlatGeometryCache, u1, u2, out, max_arrows: int = 900):
    x = cache.x_flat.ravel()
    y = cache.y_flat.ravel()
    a = np.asarray(u1, dtype=float).ravel()
    b = np.asarray(u2, dtype=float).ravel()
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(a) & np.isfinite(b)

    idx = np.flatnonzero(good)
    if idx.size > max_arrows:
        idx = idx[:: max(1, idx.size // max_arrows)]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.quiver(x[idx], y[idx], a[idx], b[idx], angles="xy")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("SDG flattened velocity quiver")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_sphere_tangent_velocity(cache: SphereFlatGeometryCache, u_sph, v_sph, out, max_arrows: int = 900):
    TX, TY, TZ = sphere_tangent_xyz_velocity(cache.lambda_, cache.theta, u_sph, v_sph)

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
    ax.set_title("SDG sphere tangent velocity")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_all_sdg_mapping_diagnostics(
    VX,
    VY,
    EToV,
    elem_patch_id,
    cache: SphereFlatGeometryCache,
    u1,
    u2,
    u_sph,
    v_sph,
    output_dir: str | Path,
    R: float = 1.0,
) -> list[Path]:
    output_dir = _ensure_dir(output_dir)
    paths: list[Path] = []

    def p(name: str) -> Path:
        return output_dir / name

    plot_flat_patch_mesh(VX, VY, EToV, elem_patch_id, p("01_sdg_flat_patch_mesh.png"))
    plot_table1_nodes_flat(VX, VY, EToV, cache, p("02_sdg_table1_nodes_flat.png"))
    plot_sphere_patch_scatter(cache, p("03_sdg_sphere_patch_scatter.png"))

    radial = np.sqrt(cache.X**2 + cache.Y**2 + cache.Z**2)
    plot_flat_scalar(
        cache,
        np.abs(radial - R),
        p("04_sdg_radial_error_flat.png"),
        "SDG radial error |sqrt(X^2+Y^2+Z^2)-R|",
        "radial error",
        mask_bad=False,
    )

    det_err = np.abs(cache.detA - sdg_detA_expected(R))
    sqrtG_err = np.abs(cache.sqrtG - sdg_sqrtG_expected(R))

    plot_flat_scalar(cache, cache.detA, p("05_sdg_detA_flat.png"), "SDG det(A)", "det(A)")
    plot_flat_scalar(cache, det_err, p("06_sdg_detA_error_flat.png"), "|det(A)-pi R^2|", "detA error")
    plot_flat_scalar(cache, cache.sqrtG, p("07_sdg_sqrtG_flat.png"), "SDG sqrt(G)", "sqrt(G)")
    plot_flat_scalar(cache, sqrtG_err, p("08_sdg_sqrtG_error_flat.png"), "|sqrt(G)-pi R^2|", "sqrtG error")
    plot_flat_scalar(cache, cache.A_Ainv_err, p("09_sdg_A_Ainv_error_flat.png"), "||A Ainv - I||inf", "A Ainv error")
    plot_flat_scalar(cache, cache.Ainv_numpy_err, p("10_sdg_Ainv_numpy_error_flat.png"), "||Ainv_SDG - inv(A)||inf", "Ainv check error")
    plot_mask(cache, p("11_sdg_pole_bad_mask.png"))

    plot_per_patch_bar(
        cache,
        R,
        "detA_error_max",
        p("12_sdg_per_patch_detA_error.png"),
        "Per-patch max |det(A)-pi R^2|",
        "max detA error",
    )
    plot_per_patch_bar(
        cache,
        R,
        "sqrtG_error_max",
        p("13_sdg_per_patch_sqrtG_error.png"),
        "Per-patch max |sqrt(G)-pi R^2|",
        "max sqrtG error",
    )
    plot_per_patch_bar(
        cache,
        R,
        "A_Ainv_err_max",
        p("14_sdg_per_patch_A_Ainv_error.png"),
        "Per-patch max ||A Ainv - I||inf",
        "max A Ainv error",
    )

    speed = np.sqrt(u1**2 + u2**2)
    plot_flat_scalar(
        cache,
        speed,
        p("15_sdg_flattened_velocity_magnitude.png"),
        "SDG flattened velocity magnitude",
        "|u_flat|",
    )
    plot_flattened_velocity_quiver(cache, u1, u2, p("16_sdg_flattened_velocity_quiver.png"))
    plot_sphere_tangent_velocity(cache, u_sph, v_sph, p("17_sdg_sphere_tangent_velocity.png"))

    return paths

