from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import math
import csv
import numpy as np

from data.table1_rules import load_table1_rule
from data.table2_rules import load_table2_rule

from geometry.reference_triangle import (
    reference_triangle_vertices,
    reference_triangle_area,
)
from geometry.mesh_structured import (
    structured_square_tri_mesh,
    triangle_signed_area,
)
from geometry.affine_map import (
    map_ref_to_phys,
    element_vertices,
)
from geometry.sampling import dense_barycentric_lattice

from operators.vandermonde2d import vandermonde2d
from operators.mass import mass_matrix_from_quadrature


@dataclass(frozen=True)
class FieldHConvergenceConfig:
    table_name: str = "table2"
    order: int = 4
    N: int = 4
    diagonal: str = "anti"
    mesh_levels: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256)
    eval_resolution: int = 12

    # Gaussian parameters
    x0: float = 0.5
    y0: float = 0.5
    sigma: float = 0.16

    verbose: bool = True


def gaussian_field(
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    y0: float,
    sigma: float,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * sigma ** 2))


def load_rule(table_name: str, order: int) -> dict:
    table_name = table_name.lower().strip()
    if table_name == "table1":
        return load_table1_rule(order)
    if table_name == "table2":
        return load_table2_rule(order)
    raise ValueError("table_name must be 'table1' or 'table2'.")


def build_fit_matrix(rule: dict, N: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the linear map:
        coeffs = Fit @ u_nodes

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (V_nodes, Fit)
        V_nodes shape = (Np, Nm)
        Fit     shape = (Nm, Np)
    """
    rs = np.asarray(rule["rs"], dtype=float)
    w = np.asarray(rule["ws"], dtype=float).reshape(-1)

    V_nodes = vandermonde2d(N, rs[:, 0], rs[:, 1])
    area_ref = reference_triangle_area()

    if V_nodes.shape[0] == V_nodes.shape[1]:
        # square nodal interpolation
        Fit = np.linalg.inv(V_nodes)
        return V_nodes, Fit

    # weighted projection
    M = mass_matrix_from_quadrature(V_nodes, w, area=area_ref)
    rhs = area_ref * (V_nodes.T * w[None, :])
    Fit = np.linalg.solve(M, rhs)
    return V_nodes, Fit


def build_evaluation_operators(
    rule: dict,
    N: int,
    rs_eval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build linear maps:
        u_h(nodes) = A_node  @ u_nodes
        u_h(dense) = A_dense @ u_nodes

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (A_node, A_dense)
    """
    rs_nodes = np.asarray(rule["rs"], dtype=float)
    V_nodes, Fit = build_fit_matrix(rule, N)

    V_eval = vandermonde2d(N, rs_eval[:, 0], rs_eval[:, 1])

    A_node = V_nodes @ Fit
    A_dense = V_eval @ Fit
    return A_node, A_dense


def compute_convergence_rate(errors: list[float]) -> list[float]:
    """
    rate_i = log2(E_i / E_{i+1})
    first entry is NaN
    """
    rates = [math.nan]
    for i in range(1, len(errors)):
        e_prev = errors[i - 1]
        e_curr = errors[i]
        if e_prev <= 0.0 or e_curr <= 0.0:
            rates.append(math.nan)
        else:
            rates.append(math.log(e_prev / e_curr, 2.0))
    return rates


def run_field_h_convergence(config: FieldHConvergenceConfig) -> list[dict]:
    """
    Run h-convergence study for Gaussian field reconstruction error.

    Computes both:
    1. nodal residual errors
    2. dense evaluation errors

    For each, report:
    - L2
    - Linf
    - observed rates
    """
    rule = load_rule(config.table_name, config.order)
    rs_nodes = np.asarray(rule["rs"], dtype=float)
    w_nodes = np.asarray(rule["ws"], dtype=float).reshape(-1)

    ref_vertices = reference_triangle_vertices()
    rs_eval = dense_barycentric_lattice(ref_vertices, resolution=config.eval_resolution)

    A_node, A_dense = build_evaluation_operators(rule, config.N, rs_eval)

    results: list[dict] = []

    for n in config.mesh_levels:
        t0 = perf_counter()

        VX, VY, EToV = structured_square_tri_mesh(
            nx=n,
            ny=n,
            diagonal=config.diagonal,
        )

        K = EToV.shape[0]
        Np = rs_nodes.shape[0]
        Nq = rs_eval.shape[0]

        node_l2_sq = 0.0
        node_linf = 0.0

        dense_l2_sq = 0.0
        dense_linf = 0.0

        for k in range(K):
            verts = element_vertices(VX, VY, EToV, k)

            # physical area of this element
            area_k = abs(triangle_signed_area(verts[0], verts[1], verts[2]))

            # nodal points on this element
            x_nodes, y_nodes = map_ref_to_phys(
                rs_nodes[:, 0],
                rs_nodes[:, 1],
                verts,
            )
            u_true_nodes = gaussian_field(
                x_nodes, y_nodes,
                x0=config.x0, y0=config.y0, sigma=config.sigma,
            )

            # reconstructed values back on nodal points
            u_rec_nodes = A_node @ u_true_nodes
            e_node = u_rec_nodes - u_true_nodes

            node_l2_sq += area_k * float(np.dot(w_nodes, e_node ** 2))
            node_linf = max(node_linf, float(np.max(np.abs(e_node))))

            # dense evaluation points on this element
            x_eval, y_eval = map_ref_to_phys(
                rs_eval[:, 0],
                rs_eval[:, 1],
                verts,
            )
            u_true_dense = gaussian_field(
                x_eval, y_eval,
                x0=config.x0, y0=config.y0, sigma=config.sigma,
            )

            u_rec_dense = A_dense @ u_true_nodes
            e_dense = u_rec_dense - u_true_dense

            # sampled L2-like error on dense points
            dense_l2_sq += area_k * float(np.mean(e_dense ** 2))
            dense_linf = max(dense_linf, float(np.max(np.abs(e_dense))))

        node_L2 = math.sqrt(node_l2_sq)
        dense_L2 = math.sqrt(dense_l2_sq)

        h = 1.0 / float(n)
        total_dof = K * Np
        elapsed = perf_counter() - t0

        row = {
            "nx": n,
            "ny": n,
            "K_tri": K,
            "h": h,
            "Np": Np,
            "Nq_dense": Nq,
            "total_dof": total_dof,
            "node_L2": node_L2,
            "node_Linf": node_linf,
            "dense_L2": dense_L2,
            "dense_Linf": dense_linf,
            "elapsed_sec": elapsed,
        }
        results.append(row)

        if config.verbose:
            print(
                f"[h-study] n={n:3d} | K={K:7d} | "
                f"node_L2={node_L2:.6e} | node_Linf={node_linf:.6e} | "
                f"dense_L2={dense_L2:.6e} | dense_Linf={dense_linf:.6e} | "
                f"time={elapsed:.2f}s"
            )

    # observed rates
    node_L2_list = [r["node_L2"] for r in results]
    node_Linf_list = [r["node_Linf"] for r in results]
    dense_L2_list = [r["dense_L2"] for r in results]
    dense_Linf_list = [r["dense_Linf"] for r in results]

    node_L2_rates = compute_convergence_rate(node_L2_list)
    node_Linf_rates = compute_convergence_rate(node_Linf_list)
    dense_L2_rates = compute_convergence_rate(dense_L2_list)
    dense_Linf_rates = compute_convergence_rate(dense_Linf_list)

    for i, r in enumerate(results):
        r["rate_node_L2"] = node_L2_rates[i]
        r["rate_node_Linf"] = node_Linf_rates[i]
        r["rate_dense_L2"] = dense_L2_rates[i]
        r["rate_dense_Linf"] = dense_Linf_rates[i]

    return results


def print_results_table(results: list[dict]) -> None:
    header = (
        f"{'n':>6s} {'K':>9s} {'h':>12s} "
        f"{'node_L2':>14s} {'rate':>8s} "
        f"{'node_Linf':>14s} {'rate':>8s} "
        f"{'dense_L2':>14s} {'rate':>8s} "
        f"{'dense_Linf':>14s} {'rate':>8s}"
    )
    print(header)
    print("-" * len(header))

    def fmt_rate(v):
        return "   -   " if not np.isfinite(v) else f"{v:8.3f}"

    for r in results:
        print(
            f"{r['nx']:6d} {r['K_tri']:9d} {r['h']:12.4e} "
            f"{r['node_L2']:14.6e} {fmt_rate(r['rate_node_L2'])} "
            f"{r['node_Linf']:14.6e} {fmt_rate(r['rate_node_Linf'])} "
            f"{r['dense_L2']:14.6e} {fmt_rate(r['rate_dense_L2'])} "
            f"{r['dense_Linf']:14.6e} {fmt_rate(r['rate_dense_Linf'])}"
        )


def save_results_csv(results: list[dict], filepath: str) -> None:
    if not results:
        raise ValueError("results is empty.")

    fieldnames = list(results[0].keys())
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)