from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from geometry.sphere_mesh import build_sphere_patch_mesh
from geometry.sphere_velocity import build_velocity_on_sphere_mesh


def add_xyz_axes(ax, *, radius: float = 1.0, scale: float = 1.35) -> None:
    L = scale * radius

    ax.quiver(0, 0, 0, L, 0, 0, arrow_length_ratio=0.08, linewidth=1.8)
    ax.text(L, 0, 0, "+X", fontsize=12)

    ax.quiver(0, 0, 0, 0, L, 0, arrow_length_ratio=0.08, linewidth=1.8)
    ax.text(0, L, 0, "+Y", fontsize=12)

    ax.quiver(0, 0, 0, 0, 0, L, arrow_length_ratio=0.08, linewidth=1.8)
    ax.text(0, 0, L, "+Z", fontsize=12)

    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_zlim(-L, L)


def spherical_basis_vectors(lambda_: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return e_lambda and e_theta in 3D Cartesian coordinates.

    e_lambda = (-sin lambda, cos lambda, 0)
    e_theta  = (-cos lambda sin theta,
                -sin lambda sin theta,
                 cos theta)
    """
    e_lambda = np.stack(
        [
            -np.sin(lambda_),
            np.cos(lambda_),
            np.zeros_like(lambda_),
        ],
        axis=-1,
    )

    e_theta = np.stack(
        [
            -np.cos(lambda_) * np.sin(theta),
            -np.sin(lambda_) * np.sin(theta),
            np.cos(theta),
        ],
        axis=-1,
    )

    return e_lambda, e_theta


def velocity_cartesian(
    lambda_: np.ndarray,
    theta: np.ndarray,
    u_sph: np.ndarray,
    v_sph: np.ndarray,
) -> np.ndarray:
    """
    Convert spherical components to 3D tangent Cartesian vectors.
    """
    e_lambda, e_theta = spherical_basis_vectors(lambda_, theta)

    return (
        u_sph[..., None] * e_lambda
        + v_sph[..., None] * e_theta
    )


def add_reference_sphere(ax, *, radius: float = 1.0) -> None:
    phi = np.linspace(0.0, 2.0 * np.pi, 72)
    theta = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 36)

    PHI, THETA = np.meshgrid(phi, theta)

    X = radius * np.cos(PHI) * np.cos(THETA)
    Y = radius * np.sin(PHI) * np.cos(THETA)
    Z = radius * np.sin(THETA)

    ax.plot_wireframe(X, Y, Z, linewidth=0.25, alpha=0.15)


def plot_sphere_velocity_3d(
    outdir: Path,
    *,
    nsub: int = 5,
    radius: float = 1.0,
    quad_table: str = "table1",
    quad_order: int = 4,
    u0: float = 1.0,
    alpha0: float = np.pi / 4.0,
    stride: int = 12,
    arrow_scale: float = 0.12,
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

    xyz = mesh.volume_xyz.reshape(-1, 3)
    lam = mesh.volume_lambda.reshape(-1)
    th = mesh.volume_theta.reshape(-1)
    u = velocity.u_sph.reshape(-1)
    v = velocity.v_sph.reshape(-1)

    Vxyz = velocity_cartesian(lam, th, u, v)

    # Downsample for readability.
    idx = np.arange(0, xyz.shape[0], stride)

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")

    add_reference_sphere(ax, radius=radius)

    ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        s=2.0,
        alpha=0.35,
    )

    ax.quiver(
        xyz[idx, 0],
        xyz[idx, 1],
        xyz[idx, 2],
        arrow_scale * Vxyz[idx, 0],
        arrow_scale * Vxyz[idx, 1],
        arrow_scale * Vxyz[idx, 2],
        length=1.0,
        normalize=False,
        linewidth=0.8,
    )

    add_xyz_axes(ax, radius=radius)
    ax.view_init(elev=25, azim=35)

    ax.set_title(
        f"Sphere velocity field "
        f"(nsub={nsub}, alpha0={alpha0:.3f})"
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect((1, 1, 1))

    outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outdir / "sphere_velocity_3d.png", dpi=240)
    plt.close(fig)


def main() -> None:
    outdir = Path("experiments_outputs/sphere_velocity")

    plot_sphere_velocity_3d(
        outdir,
        nsub=2,
        radius=1.0,
        quad_table="table1",
        quad_order=4,
        u0=1.0,
        alpha0=0,
        stride=12,
        arrow_scale=0.12,
    )

    print(f"[done] figures written to {outdir}")


if __name__ == "__main__":
    main()