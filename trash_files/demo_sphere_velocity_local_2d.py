from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from geometry.sphere_mesh import build_sphere_patch_mesh
from geometry.sphere_velocity import build_velocity_on_sphere_mesh
from geometry.sphere_patch_mesh import map_local_xy_to_patch_xy


def add_square_guides(ax, *, radius: float = 1.0) -> None:
    R = radius
    boundary = np.array(
        [
            [-R, -R],
            [ R, -R],
            [ R,  R],
            [-R,  R],
            [-R, -R],
        ],
        dtype=float,
    )

    ax.plot(boundary[:, 0], boundary[:, 1], linewidth=1.4)
    ax.axhline(0.0, linewidth=1.0, alpha=0.5)
    ax.axvline(0.0, linewidth=1.0, alpha=0.5)

    ax.set_xlim(-1.1 * R, 1.1 * R)
    ax.set_ylim(-1.1 * R, 1.1 * R)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def plot_local_contravariant_velocity(
    outdir: Path,
    *,
    nsub: int = 5,
    radius: float = 1.0,
    quad_table: str = "table1",
    quad_order: int = 4,
    u0: float = 1.0,
    alpha0: float = np.pi / 4.0,
    stride: int = 8,
    arrow_scale: float = 0.035,
) -> None:
    mesh = build_sphere_patch_mesh(
        nsub=nsub,
        radius=radius,
        quad_table=quad_table,
        quad_order=quad_order,
    )

    velocity = build_velocity_on_sphere_mesh(
        mesh,
        u0=u0,
        alpha0=alpha0,
    )

    fig, ax = plt.subplots(figsize=(9, 9))
    add_square_guides(ax, radius=radius)

    for patch_id in range(1, 9):
        mask = mesh.elem_to_patch == patch_id

        local_xy = mesh.volume_nodes_local[mask, :, :].reshape(-1, 2)
        u1 = velocity.u1[mask, :].reshape(-1)
        u2 = velocity.u2[mask, :].reshape(-1)

        xy_phys = map_local_xy_to_patch_xy(
            local_xy,
            patch_id,
            radius=radius,
        )

        idx = np.arange(0, xy_phys.shape[0], stride)

        ax.scatter(
            xy_phys[:, 0],
            xy_phys[:, 1],
            s=2.0,
            alpha=0.35,
        )

        ax.quiver(
            xy_phys[idx, 0],
            xy_phys[idx, 1],
            arrow_scale * u1[idx],
            arrow_scale * u2[idx],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.0022,
        )

        c = np.mean(xy_phys, axis=0)
        ax.text(c[0], c[1], f"T{patch_id}", fontsize=11, ha="center", va="center")

    ax.set_title(
        f"Local contravariant velocity field "
        f"(u1,u2), nsub={nsub}, alpha0={alpha0:.3f}"
    )

    outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outdir / "sphere_velocity_local_2d.png", dpi=240)
    plt.close(fig)


def main() -> None:
    outdir = Path("experiments_outputs/sphere_velocity")

    plot_local_contravariant_velocity(
        outdir,
        nsub=2,
        radius=1.0,
        quad_table="table1",
        quad_order=4,
        u0=1.0,
        alpha0=0,
        stride=8,
        arrow_scale=0.1,
    )

    print(f"[done] figures written to {outdir}")


if __name__ == "__main__":
    main()