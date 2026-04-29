from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules and os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from geometry.sphere_manifold_metrics import ManifoldGeometryCache


def _ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _set_equal_3d_axes(ax, X, Y, Z) -> None:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Z = np.asarray(Z, dtype=float)

    max_range = np.nanmax(
        [
            np.nanmax(X) - np.nanmin(X),
            np.nanmax(Y) - np.nanmin(Y),
            np.nanmax(Z) - np.nanmin(Z),
        ]
    )
    half = 0.5 * max_range
    cx = 0.5 * (np.nanmax(X) + np.nanmin(X))
    cy = 0.5 * (np.nanmax(Y) + np.nanmin(Y))
    cz = 0.5 * (np.nanmax(Z) + np.nanmin(Z))
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_zlim(cz - half, cz + half)


def create_manifold_scalar_figure(
    geom: ManifoldGeometryCache,
    values: np.ndarray,
    *,
    title: str = "Manifold scalar field",
    label: str = "q",
):
    values = np.asarray(values, dtype=float)
    if values.shape != geom.X.shape:
        raise ValueError("values must match geometry nodal shape.")

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        geom.X.ravel(),
        geom.Y.ravel(),
        geom.Z.ravel(),
        c=values.ravel(),
        s=12,
    )
    fig.colorbar(sc, ax=ax, shrink=0.72, label=label)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    _set_equal_3d_axes(ax, geom.X, geom.Y, geom.Z)
    return fig, ax


def create_manifold_scalar_panel_figure(
    geom: ManifoldGeometryCache,
    panels: list[dict],
    *,
    suptitle: str | None = None,
):
    if not panels:
        raise ValueError("panels must not be empty.")

    ncols = len(panels)
    fig = plt.figure(figsize=(6.2 * ncols, 6.2))
    axes = []
    for idx, panel in enumerate(panels, start=1):
        values = np.asarray(panel["values"], dtype=float)
        if values.shape != geom.X.shape:
            raise ValueError("panel values must match geometry nodal shape.")
        ax = fig.add_subplot(1, ncols, idx, projection="3d")
        sc = ax.scatter(
            geom.X.ravel(),
            geom.Y.ravel(),
            geom.Z.ravel(),
            c=values.ravel(),
            s=12,
        )
        fig.colorbar(sc, ax=ax, shrink=0.72, label=str(panel.get("label", "q")))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(str(panel.get("title", f"Field {idx}")))
        _set_equal_3d_axes(ax, geom.X, geom.Y, geom.Z)
        axes.append(ax)

    if suptitle is not None:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig, axes


def plot_manifold_mesh(
    nodes_xyz: np.ndarray,
    EToV: np.ndarray,
    out: str | Path,
    title: str = "Spherical octahedron manifold mesh",
) -> Path:
    out = Path(out)
    nodes_xyz = np.asarray(nodes_xyz, dtype=float)
    EToV = np.asarray(EToV, dtype=int)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(
        nodes_xyz[:, 0],
        nodes_xyz[:, 1],
        nodes_xyz[:, 2],
        triangles=EToV,
        linewidth=0.4,
        edgecolor="k",
        alpha=0.72,
        color="#9ecae1",
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    _set_equal_3d_axes(ax, nodes_xyz[:, 0], nodes_xyz[:, 1], nodes_xyz[:, 2])
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_manifold_velocity_quiver(
    geom: ManifoldGeometryCache,
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
    out: str | Path,
    title: str = "3D solid-body tangent velocity",
    max_arrows: int = 900,
) -> Path:
    out = Path(out)

    x = geom.X.ravel()
    y = geom.Y.ravel()
    z = geom.Z.ravel()
    u = np.asarray(U, dtype=float).ravel()
    v = np.asarray(V, dtype=float).ravel()
    w = np.asarray(W, dtype=float).ravel()

    good = (
        np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        & np.isfinite(u) & np.isfinite(v) & np.isfinite(w)
    )
    idx = np.flatnonzero(good)
    if idx.size > max_arrows:
        idx = idx[:: max(1, idx.size // max_arrows)]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x[idx], y[idx], z[idx], s=5, alpha=0.65)
    ax.quiver(x[idx], y[idx], z[idx], u[idx], v[idx], w[idx], length=0.08, normalize=True)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    _set_equal_3d_axes(ax, x, y, z)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_manifold_divergence_nodes(
    geom: ManifoldGeometryCache,
    div: np.ndarray,
    out: str | Path,
    title: str = "Free-stream divergence error on 3D sphere",
    log_abs: bool = True,
) -> Path:
    out = Path(out)
    div = np.asarray(div, dtype=float)
    if div.shape != geom.X.shape:
        raise ValueError("div must match geometry nodal shape.")

    values = np.abs(div) if log_abs else div
    if log_abs:
        values = np.log10(np.maximum(values, 1.0e-16))
        label = "log10(|div_h|)"
    else:
        label = "div_h"

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        geom.X.ravel(),
        geom.Y.ravel(),
        geom.Z.ravel(),
        c=values.ravel(),
        s=12,
    )
    fig.colorbar(sc, ax=ax, shrink=0.72, label=label)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    _set_equal_3d_axes(ax, geom.X, geom.Y, geom.Z)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_manifold_element_mean_divergence(
    geom: ManifoldGeometryCache,
    div: np.ndarray,
    out: str | Path,
    title: str = "Element mean free-stream divergence error",
) -> Path:
    out = Path(out)
    div = np.asarray(div, dtype=float)
    if div.shape != geom.X.shape:
        raise ValueError("div must match geometry nodal shape.")

    elem_values = np.nanmean(div, axis=1)
    absmax = float(np.nanmax(np.abs(elem_values))) if elem_values.size else 1.0
    if absmax <= 0.0:
        absmax = 1.0

    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(vmin=-absmax, vmax=absmax)
    face_colors = cmap(norm(elem_values))

    vertices = geom.nodes_xyz[geom.EToV]
    collection = Poly3DCollection(
        vertices,
        facecolors=face_colors,
        edgecolors="k",
        linewidths=0.25,
        alpha=0.92,
    )

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(collection)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(mappable, ax=ax, shrink=0.72, label="mean div_h")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    _set_equal_3d_axes(ax, geom.nodes_xyz[:, 0], geom.nodes_xyz[:, 1], geom.nodes_xyz[:, 2])
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_manifold_convergence(
    results: list[dict],
    out: str | Path,
    title: str = "Manifold DG constant-field divergence h-convergence",
) -> Path:
    out = Path(out)
    if not results:
        raise ValueError("results must not be empty.")

    h = np.array([row["h"] for row in results], dtype=float)
    L2 = np.array([row["L2"] for row in results], dtype=float)
    Linf = np.array([row["Linf"] for row in results], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(h, L2, marker="o", label="L2")
    ax.loglog(h, Linf, marker="s", label="Linf")
    ax.invert_xaxis()
    ax.set_xlabel("h")
    ax.set_ylabel("error for q=1 RHS")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_manifold_lsrk_convergence(
    results: list[dict],
    out: str | Path,
    title: str = "Manifold DG LSRK sphere time stepping",
) -> Path:
    out = Path(out)
    if not results:
        raise ValueError("results must not be empty.")

    h = np.array([row["h"] for row in results], dtype=float)
    l2 = np.array([row["L2_error"] for row in results], dtype=float)
    linf = np.array([row["Linf_error"] for row in results], dtype=float)
    field_case = str(results[0].get("field_case", "field"))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(h, l2, marker="o", label="L2")
    ax.loglog(h, linf, marker="s", label="Linf")
    ax.invert_xaxis()
    ax.set_xlabel("h")
    ax.set_ylabel("error")
    ax.set_title(f"{title} ({field_case})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_manifold_l2_error_vs_time(
    histories: list[dict],
    out: str | Path,
    title: str = "Manifold DG L2 error vs time",
) -> Path:
    out = Path(out)
    if not histories:
        raise ValueError("histories must not be empty.")

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ymin = np.inf
    for history in histories:
        times = np.asarray(history["times"], dtype=float)
        l2 = np.asarray(history["l2"], dtype=float)
        l2_plot = np.maximum(l2, np.finfo(float).tiny)
        ymin = min(ymin, float(np.min(l2_plot)))
        ax.plot(
            times,
            l2_plot,
            linewidth=2.0,
            label=f"L2 error (n={int(history['mesh_level'])})",
        )

    field_case = str(histories[0].get("field_case", "field"))
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-14, top=1e1)
    ax.set_xlabel("time")
    ax.set_ylabel("L2 error (log scale)")
    ax.set_title(f"{title} ({field_case})")
    ax.grid(True, which="both", linestyle=":", linewidth=0.8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_manifold_mass_error_vs_time(
    histories: list[dict],
    out: str | Path,
    title: str = "Manifold DG mass drift vs time",
) -> Path:
    out = Path(out)
    if not histories:
        raise ValueError("histories must not be empty.")

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for history in histories:
        times = np.asarray(history["times"], dtype=float)
        mass_error = np.asarray(history["mass_error"], dtype=float)
        mass_error_plot = np.maximum(np.abs(mass_error), np.finfo(float).tiny)
        ax.plot(
            times,
            mass_error_plot,
            linewidth=2.0,
            label=f"|mass drift| (n={int(history['mesh_level'])})",
        )

    field_case = str(histories[0].get("field_case", "field"))
    ax.set_yscale("log")
    ax.set_xlabel("time")
    ax.set_ylabel("|mass error| (log scale)")
    ax.set_title(f"{title} ({field_case})")
    ax.grid(True, which="both", linestyle=":", linewidth=0.8)
    ax.set_ylim(bottom=1e-14, top=1e1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_manifold_scalar_nodes(
    geom: ManifoldGeometryCache,
    values: np.ndarray,
    out: str | Path,
    title: str = "Manifold scalar field",
    label: str = "q",
) -> Path:
    out = Path(out)
    fig, _ax = create_manifold_scalar_figure(
        geom,
        values,
        title=title,
        label=label,
    )
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def write_all_manifold_diagnostics(
    nodes_xyz: np.ndarray,
    EToV: np.ndarray,
    geom: ManifoldGeometryCache,
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray,
    div: np.ndarray,
    output_dir: str | Path,
) -> list[Path]:
    output_dir = _ensure_dir(output_dir)
    paths: list[Path] = []

    paths.append(plot_manifold_mesh(nodes_xyz, EToV, output_dir / "01_manifold_mesh.png"))
    paths.append(plot_manifold_velocity_quiver(geom, U, V, W, output_dir / "02_velocity_quiver.png"))
    paths.append(plot_manifold_divergence_nodes(geom, div, output_dir / "03_divergence_nodes.png"))
    paths.append(
        plot_manifold_element_mean_divergence(
            geom,
            div,
            output_dir / "04_divergence_element_mean.png",
        )
    )

    return paths
