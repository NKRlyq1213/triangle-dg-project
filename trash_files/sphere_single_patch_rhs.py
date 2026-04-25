from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from data.rule_registry import load_rule
from geometry.reference_triangle import reference_triangle_area
from geometry.sphere_patch_mesh import uniform_local_submesh
from geometry.affine_map import map_reference_nodes_to_all_elements
from geometry.metrics import affine_geometric_factors_from_mesh
from geometry.face_metrics import affine_face_geometry_from_mesh
from geometry.connectivity import build_face_connectivity

from operators.vandermonde2d import vandermonde2d, grad_vandermonde2d
from operators.differentiation import (
    differentiation_matrices_square,
    differentiation_matrices_weighted,
)
from operators.trace_policy import build_trace_policy
from operators.rhs_split_conservative_exact_trace import (
    rhs_split_conservative_exact_trace,
)

from geometry.sphere_mapping import local_xy_to_patch_angles
from geometry.sphere_velocity import (
    spherical_advection_velocity,
    contravariant_velocity_from_spherical,
)


@dataclass(frozen=True)
class SinglePatchRHSSetup:
    """
    Geometry/operator setup for a single sphere patch pulled back to local triangle.

    中文：
    - 這不是完整 8-patch solver。
    - 這只用來檢查單一 patch 上的 RHS 是否能保持 constant state。
    """

    patch_id: int
    nsub: int
    radius: float
    quad_table: str
    quad_order: int
    N: int

    rule: dict
    trace: dict
    VX: np.ndarray
    VY: np.ndarray
    EToV: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    Dr: np.ndarray
    Ds: np.ndarray
    geom: dict
    conn: dict
    face_geom: dict


def build_reference_diff_operators_from_rule(
    rule: dict,
    *,
    N: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build reference differentiation matrices Dr, Ds.

    Reuses repo Vandermonde and differentiation matrix builders.
    """
    rs = np.asarray(rule["rs"], dtype=float)
    ws = np.asarray(rule["ws"], dtype=float).reshape(-1)

    V = vandermonde2d(N, rs[:, 0], rs[:, 1])
    Vr, Vs = grad_vandermonde2d(N, rs[:, 0], rs[:, 1])

    if V.shape[0] == V.shape[1]:
        return differentiation_matrices_square(V, Vr, Vs)

    return differentiation_matrices_weighted(
        V,
        Vr,
        Vs,
        ws,
        area=reference_triangle_area(),
    )


def build_single_patch_local_mesh(
    *,
    nsub: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a local triangular mesh on:
        x >= 0, y >= 0, x + y <= 1

    Returns
    -------
    VX, VY:
        Vertex coordinates.
    EToV:
        Element-to-vertex connectivity.
    """
    points_local, EToV = uniform_local_submesh(nsub)

    VX = np.asarray(points_local[:, 0], dtype=float)
    VY = np.asarray(points_local[:, 1], dtype=float)
    EToV = np.asarray(EToV, dtype=int)

    return VX, VY, EToV


def build_single_patch_rhs_setup(
    *,
    patch_id: int,
    nsub: int,
    radius: float = 1.0,
    quad_table: str = "table1",
    quad_order: int = 4,
    N: int | None = None,
) -> SinglePatchRHSSetup:
    """
    Build all geometry/operator objects needed by existing planar RHS.

    Default:
        quad_table = "table1"
    """
    if N is None:
        N = quad_order

    if quad_table != "table1":
        raise ValueError("single-patch RHS smoke test is intended to use table1.")

    rule = load_rule(quad_table, quad_order)
    trace = build_trace_policy(rule, N=N)

    VX, VY, EToV = build_single_patch_local_mesh(nsub=nsub)

    X, Y = map_reference_nodes_to_all_elements(
        rule["rs"],
        VX,
        VY,
        EToV,
    )

    Dr, Ds = build_reference_diff_operators_from_rule(rule, N=N)

    geom = affine_geometric_factors_from_mesh(
        VX,
        VY,
        EToV,
        rule["rs"],
    )

    conn = build_face_connectivity(
        VX,
        VY,
        EToV,
        classify_boundary=None,
    )

    face_geom = affine_face_geometry_from_mesh(
        VX,
        VY,
        EToV,
        trace,
    )

    return SinglePatchRHSSetup(
        patch_id=patch_id,
        nsub=nsub,
        radius=radius,
        quad_table=quad_table,
        quad_order=quad_order,
        N=N,
        rule=rule,
        trace=trace,
        VX=VX,
        VY=VY,
        EToV=EToV,
        X=X,
        Y=Y,
        Dr=Dr,
        Ds=Ds,
        geom=geom,
        conn=conn,
        face_geom=face_geom,
    )


def q_exact_constant_one(x: np.ndarray, y: np.ndarray, t: float = 0.0) -> np.ndarray:
    """
    Exact constant solution q = 1.
    """
    del y, t
    return np.ones_like(x, dtype=float)


def make_local_contravariant_velocity(
    *,
    patch_id: int,
    radius: float = 1.0,
    u0: float = 1.0,
    alpha0: float = 0.0,
):
    """
    Return velocity(x,y,t) suitable for existing planar RHS.

    Input x,y are local patch coordinates:
        x >= 0, y >= 0, x + y <= 1

    Output:
        u1, u2
    where (u1,u2) are local contravariant velocity components.
    """

    def velocity(x: np.ndarray, y: np.ndarray, t: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        del t

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        lam, theta = local_xy_to_patch_angles(
            x,
            y,
            patch_id,
        )

        u_sph, v_sph = spherical_advection_velocity(
            lam,
            theta,
            u0=u0,
            alpha0=alpha0,
        )
        # print(f"velocity: max |u_sph| = {np.max(np.abs(u_sph)):.3e}, max |v_sph| = {np.max(np.abs(v_sph)):.3e}")
        u1, u2 = contravariant_velocity_from_spherical(
            u_sph,
            v_sph,
            x,
            y,
            patch_id,
            radius=radius,
        )

        return u1, u2

    return velocity


def compute_constant_state_single_patch_rhs(
    *,
    patch_id: int,
    nsub: int = 5,
    radius: float = 1.0,
    quad_table: str = "table1",
    quad_order: int = 4,
    N: int | None = None,
    u0: float = 1.0,
    alpha0: float = 0.0,
    tau: float = 0.0,
) -> tuple[np.ndarray, dict, SinglePatchRHSSetup]:
    """
    Compute RHS for q = 1 on one patch.

    We choose alpha0=0 by default because the resulting local contravariant
    velocity is simple and divergence-free under the current mapping.

    Returns
    -------
    rhs:
        Shape (K, Np)
    diagnostics:
        Existing RHS diagnostics.
    setup:
        SinglePatchRHSSetup.
    """
    setup = build_single_patch_rhs_setup(
        patch_id=patch_id,
        nsub=nsub,
        radius=radius,
        quad_table=quad_table,
        quad_order=quad_order,
        N=N,
    )

    q_elem = np.ones_like(setup.X)

    velocity = make_local_contravariant_velocity(
        patch_id=patch_id,
        radius=radius,
        u0=u0,
        alpha0=alpha0,
    )

    u_elem, v_elem = velocity(setup.X, setup.Y, t=0.0)

    surface_cache = {
        "x_face": setup.face_geom["x_face"],
        "y_face": setup.face_geom["y_face"],
        "nx": setup.face_geom["nx"],
        "ny": setup.face_geom["ny"],
        "is_boundary": setup.conn["is_boundary"],
    }

    rhs, diagnostics = rhs_split_conservative_exact_trace(
        q_elem=q_elem,
        u_elem=u_elem,
        v_elem=v_elem,
        Dr=setup.Dr,
        Ds=setup.Ds,
        geom=setup.geom,
        rule=setup.rule,
        trace=setup.trace,
        VX=setup.VX,
        VY=setup.VY,
        EToV=setup.EToV,
        q_exact=q_exact_constant_one,
        q_boundary=q_exact_constant_one,
        velocity=velocity,
        t=0.0,
        tau=tau,
        tau_interior=0.0,
        tau_qb=0.0,
        face_geom=setup.face_geom,
        physical_boundary_mode="exact_qb",
        use_numba=False,
        conn=setup.conn,
        surface_cache=surface_cache,
    )

    return rhs, diagnostics, setup