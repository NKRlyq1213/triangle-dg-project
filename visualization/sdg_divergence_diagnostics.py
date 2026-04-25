from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.tri import Triangulation

from geometry.sphere_flat_metrics import SphereFlatGeometryCache


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


def plot_flat_nodes_scalar(
    cache: SphereFlatGeometryCache,
    values: np.ndarray,
    out: str | Path,
    title: str,
    label: str,
    mask_bad: bool = True,
    symmetric: bool = False,
) -> Path:
    out = Path(out)
    values = np.asarray(values, dtype=float)

    if values.shape != cache.x_flat.shape:
        raise ValueError("values must match cache.x_flat shape.")

    if mask_bad:
        values = np.where(cache.bad_mask, np.nan, values)

    fig, ax = plt.subplots(figsize=(7.5, 7.0))

    kwargs = {}
    if symmetric:
        finite = values[np.isfinite(values)]
        if finite.size:
            vmax = float(np.max(np.abs(finite)))
            if vmax > 0.0:
                kwargs["vmin"] = -vmax
                kwargs["vmax"] = vmax

    sc = ax.scatter(
        cache.x_flat.ravel(),
        cache.y_flat.ravel(),
        c=values.ravel(),
        s=18,
        **kwargs,
    )
    fig.colorbar(sc, ax=ax, label=label)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_flat_nodes_scalar_with_mesh(
    VX: np.ndarray,
    VY: np.ndarray,
    EToV: np.ndarray,
    cache: SphereFlatGeometryCache,
    values: np.ndarray,
    out: str | Path,
    title: str,
    label: str,
    mask_bad: bool = True,
    symmetric: bool = False,
) -> Path:
    out = Path(out)
    values = np.asarray(values, dtype=float)

    if values.shape != cache.x_flat.shape:
        raise ValueError("values must match cache.x_flat shape.")

    if mask_bad:
        values = np.where(cache.bad_mask, np.nan, values)

    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    ax.add_collection(LineCollection(_mesh_segments(VX, VY, EToV), linewidths=0.4, alpha=0.25))

    kwargs = {}
    if symmetric:
        finite = values[np.isfinite(values)]
        if finite.size:
            vmax = float(np.max(np.abs(finite)))
            if vmax > 0.0:
                kwargs["vmin"] = -vmax
                kwargs["vmax"] = vmax

    sc = ax.scatter(
        cache.x_flat.ravel(),
        cache.y_flat.ravel(),
        c=values.ravel(),
        s=18,
        **kwargs,
    )
    fig.colorbar(sc, ax=ax, label=label)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.autoscale()

    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_element_mean_scalar(
    VX: np.ndarray,
    VY: np.ndarray,
    EToV: np.ndarray,
    elem_values: np.ndarray,
    out: str | Path,
    title: str,
    label: str,
    symmetric: bool = False,
) -> Path:
    out = Path(out)
    elem_values = np.asarray(elem_values, dtype=float)

    if elem_values.shape != (EToV.shape[0],):
        raise ValueError("elem_values must have shape (K,).")

    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    tri = Triangulation(VX, VY, EToV)

    kwargs = {}
    if symmetric:
        finite = elem_values[np.isfinite(elem_values)]
        if finite.size:
            vmax = float(np.max(np.abs(finite)))
            if vmax > 0.0:
                kwargs["vmin"] = -vmax
                kwargs["vmax"] = vmax

    pc = ax.tripcolor(
        tri,
        facecolors=elem_values,
        edgecolors="k",
        linewidth=0.35,
        alpha=0.95,
        **kwargs,
    )
    fig.colorbar(pc, ax=ax, label=label)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def write_all_sdg_divergence_diagnostics(
    VX: np.ndarray,
    VY: np.ndarray,
    EToV: np.ndarray,
    cache: SphereFlatGeometryCache,
    q: np.ndarray,
    u1: np.ndarray,
    u2: np.ndarray,
    div: np.ndarray,
    output_dir: str | Path,
) -> list[Path]:
    output_dir = _ensure_dir(output_dir)
    paths: list[Path] = []

    def p(name: str) -> Path:
        return output_dir / name

    speed = np.sqrt(u1**2 + u2**2)
    rhs = -div

    div_elem_mean = np.nanmean(div, axis=1)
    rhs_elem_mean = np.nanmean(rhs, axis=1)

    paths.append(plot_flat_nodes_scalar_with_mesh(
        VX, VY, EToV, cache, q,
        p("01_q_field_nodes.png"),
        title="q field on flattened square nodes",
        label="q",
        symmetric=False,
    ))

    paths.append(plot_flat_nodes_scalar(
        cache, u1,
        p("02_u1_flattened_velocity.png"),
        title="flattened velocity component u1",
        label="u1",
        symmetric=True,
    ))

    paths.append(plot_flat_nodes_scalar(
        cache, u2,
        p("03_u2_flattened_velocity.png"),
        title="flattened velocity component u2",
        label="u2",
        symmetric=True,
    ))

    paths.append(plot_flat_nodes_scalar(
        cache, speed,
        p("04_flattened_velocity_magnitude.png"),
        title="flattened velocity magnitude",
        label="sqrt(u1^2+u2^2)",
        symmetric=False,
    ))

    paths.append(plot_flat_nodes_scalar(
        cache, div,
        p("05_volume_only_flattened_divergence_nodes.png"),
        title="volume-only div_flat(u q) at Table1 nodes",
        label="div_flat",
        symmetric=True,
    ))

    paths.append(plot_element_mean_scalar(
        VX, VY, EToV, div_elem_mean,
        p("06_volume_only_flattened_divergence_element_mean.png"),
        title="element mean of volume-only div_flat(u q)",
        label="mean div_flat",
        symmetric=True,
    ))

    paths.append(plot_flat_nodes_scalar(
        cache, rhs,
        p("07_volume_only_rhs_nodes.png"),
        title="volume-only RHS = -div_flat(u q)",
        label="rhs",
        symmetric=True,
    ))

    paths.append(plot_element_mean_scalar(
        VX, VY, EToV, rhs_elem_mean,
        p("08_volume_only_rhs_element_mean.png"),
        title="element mean of volume-only RHS",
        label="mean rhs",
        symmetric=True,
    ))

    paths.append(plot_free_stream_divergence_error_log(
        cache,
        div,
        p("09_free_stream_divergence_error_log.png"),
    ))

    return paths



def plot_free_stream_divergence_error_log(
    cache: SphereFlatGeometryCache,
    div: np.ndarray,
    out: str | Path,
    title: str = "Free-stream Divergence Error on 2D Flattened Domain",
    floor: float = 1.0e-10,
) -> Path:
    """
    Plot |div_flat(u)| using logarithmic color scale.

    This is the correct diagnostic for q=1 free-stream divergence error.
    """
    from matplotlib.colors import LogNorm

    out = Path(out)
    div = np.asarray(div, dtype=float)

    if div.shape != cache.x_flat.shape:
        raise ValueError("div must match cache.x_flat shape.")

    err = np.abs(div)
    err = np.where(cache.bad_mask, np.nan, err)
    err_plot = np.where(np.isfinite(err), np.maximum(err, floor), np.nan)

    fig, ax = plt.subplots(figsize=(7.5, 7.0))

    sc = ax.scatter(
        cache.x_flat.ravel(),
        cache.y_flat.ravel(),
        c=err_plot.ravel(),
        s=18,
        norm=LogNorm(vmin=floor, vmax=max(1.0, float(np.nanmax(err_plot)))),
    )

    fig.colorbar(sc, ax=ax, label="Divergence Error")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)

    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out
