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
from geometry.metrics import affine_metric_terms_from_vertices

from operators.vandermonde2d import vandermonde2d, grad_vandermonde2d
from operators.mass import mass_matrix_from_quadrature
from operators.differentiation import (
    differentiation_matrices_square,
    differentiation_matrices_weighted,
)
from operators.divergence_split import mapped_divergence_split_2d

from problems.coefficient_fields import coefficient_field_with_derivatives


@dataclass(frozen=True)
class DivHConvergenceConfig:
    table_name: str = "table2"
    order: int = 4
    N: int = 4
    diagonal: str = "anti"
    mesh_levels: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256)
    eval_resolution: int = 12

    # scalar field
    field_case: str = "gaussian"
    x0: float = 0.35
    y0: float = 0.55
    sigma: float = 0.16

    # coefficient field
    coeff_case: str = "constant_one"

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
    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * sigma ** 2)) #x**4 + y**4  


def gaussian_gradients(
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    y0: float,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = gaussian_field(x, y, x0=x0, y0=y0, sigma=sigma)
    vx = -((x - x0) / sigma**2) * v
    vy = -((y - y0) / sigma**2) * v
    return v, vx, vy


def exact_divergence_of_av_bv(
    x: np.ndarray,
    y: np.ndarray,
    coeff_case: str,
    x0: float,
    y0: float,
    sigma: float,
) -> np.ndarray:
    a, b, ax, ay, bx, by = coefficient_field_with_derivatives(coeff_case, x, y)
    v, vx, vy = gaussian_gradients(x, y, x0=x0, y0=y0, sigma=sigma)
    return ax * v + a * vx + by * v + b * vy


def load_rule(table_name: str, order: int) -> dict:
    table_name = table_name.lower().strip()
    if table_name == "table1":
        return load_table1_rule(order)
    if table_name == "table2":
        return load_table2_rule(order)
    raise ValueError("table_name must be 'table1' or 'table2'.")


def build_fit_matrix(rule: dict, N: int) -> tuple[np.ndarray, np.ndarray]:
    rs = np.asarray(rule["rs"], dtype=float)
    w = np.asarray(rule["ws"], dtype=float).reshape(-1)

    V_nodes = vandermonde2d(N, rs[:, 0], rs[:, 1])
    area_ref = reference_triangle_area()

    if V_nodes.shape[0] == V_nodes.shape[1]:
        Fit = np.linalg.inv(V_nodes)
        return V_nodes, Fit

    M = mass_matrix_from_quadrature(V_nodes, w, area=area_ref)
    rhs = area_ref * (V_nodes.T * w[None, :])
    Fit = np.linalg.solve(M, rhs)
    return V_nodes, Fit


def build_reference_diff_operators(rule: dict, N: int) -> tuple[np.ndarray, np.ndarray]:
    rs = np.asarray(rule["rs"], dtype=float)
    w = np.asarray(rule["ws"], dtype=float).reshape(-1)

    V = vandermonde2d(N, rs[:, 0], rs[:, 1])
    Vr, Vs = grad_vandermonde2d(N, rs[:, 0], rs[:, 1])

    if V.shape[0] == V.shape[1]:
        return differentiation_matrices_square(V, Vr, Vs)

    return differentiation_matrices_weighted(
        V, Vr, Vs, w, area=reference_triangle_area()
    )


def compute_convergence_rate(errors: list[float]) -> list[float]:
    rates = [math.nan]
    for i in range(1, len(errors)):
        e_prev = errors[i - 1]
        e_curr = errors[i]
        if e_prev <= 0.0 or e_curr <= 0.0:
            rates.append(math.nan)
        else:
            rates.append(math.log(e_prev / e_curr, 2.0))
    return rates


def run_div_h_convergence(config: DivHConvergenceConfig) -> list[dict]:
    rule = load_rule(config.table_name, config.order)
    rs_nodes = np.asarray(rule["rs"], dtype=float)
    w_nodes = np.asarray(rule["ws"], dtype=float).reshape(-1)

    Dr, Ds = build_reference_diff_operators(rule, config.N)

    ref_vertices = reference_triangle_vertices()
    rs_eval = dense_barycentric_lattice(ref_vertices, resolution=config.eval_resolution, boundary_mode = "no_top_vertex")

    _, Fit = build_fit_matrix(rule, config.N)

    V_eval = vandermonde2d(config.N, rs_eval[:, 0], rs_eval[:, 1])
    Vr_eval, Vs_eval = grad_vandermonde2d(config.N, rs_eval[:, 0], rs_eval[:, 1])

    A_eval = V_eval @ Fit

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
            area_k = abs(triangle_signed_area(verts[0], verts[1], verts[2]))

            gk = affine_metric_terms_from_vertices(verts)
            xr = gk["xr"]
            xs = gk["xs"]
            yr = gk["yr"]
            ys = gk["ys"]
            J = gk["J"]

            # ---------- nodal ----------
            x_nodes, y_nodes = map_ref_to_phys(rs_nodes[:, 0], rs_nodes[:, 1], verts)

            v_nodes = gaussian_field(
                x_nodes, y_nodes,
                x0=config.x0, y0=config.y0, sigma=config.sigma,
            )
            a_nodes, b_nodes, *_ = coefficient_field_with_derivatives(
                config.coeff_case, x_nodes, y_nodes
            )

            div_num_nodes = mapped_divergence_split_2d(
                v_nodes,
                a_nodes,
                b_nodes,
                Dr,
                Ds,
                xr=np.full_like(v_nodes, xr),
                xs=np.full_like(v_nodes, xs),
                yr=np.full_like(v_nodes, yr),
                ys=np.full_like(v_nodes, ys),
                J=np.full_like(v_nodes, J),
            )
            
            div_ex_nodes = exact_divergence_of_av_bv(
                x_nodes, y_nodes,
                coeff_case=config.coeff_case,
                x0=config.x0, y0=config.y0, sigma=config.sigma,
            )

            e_node = div_num_nodes - div_ex_nodes

            node_l2_sq += area_k * float(np.dot(w_nodes, e_node ** 2))
            node_linf = max(node_linf, float(np.max(np.abs(e_node))))

            # ---------- dense ----------
            coeffs_v = Fit @ v_nodes

            x_eval, y_eval = map_ref_to_phys(rs_eval[:, 0], rs_eval[:, 1], verts)

            v_dense_h = V_eval @ coeffs_v
            vr_dense_h = Vr_eval @ coeffs_v
            vs_dense_h = Vs_eval @ coeffs_v

            a_dense, b_dense, *_ = coefficient_field_with_derivatives(
                config.coeff_case, x_eval, y_eval
            )

            alpha_dense = ys * a_dense - xs * b_dense
            beta_dense = -yr * a_dense + xr * b_dense

            # For affine geometry, xr,xs,yr,ys,J are constant per element.
            # We still keep the general split-product form in (r,s):
            #
            # div_h^split = 1/J * 1/2 [ Dr(alpha v) + alpha Dr(v) + v Dr(alpha)
            #                           + Ds(beta v) + beta Ds(v) + v Ds(beta) ]
            #
            # On dense points we do not have Dr/Ds as nodal operators, so we use
            # the equivalent polynomial differentiation at dense points:
            #
            # Dr(alpha v_h) = (alpha v_h)_r
            # Ds(beta v_h)  = (beta  v_h)_s
            #
            # For general a,b, alpha and beta vary with x,y and thus with r,s.
            # We approximate alpha and beta pointwise exactly from the exact a,b
            # on dense points, and differentiate the polynomial v_h only.
            #
            # This yields a consistent dense diagnostic for the split divergence.
            #
            # Because alpha,beta are not represented as polynomial unknowns here,
            # we use the strong mapped divergence of the reconstructed flux:
            #
            #   F_h = (a * v_h, b * v_h)
            #   div_h = d_x(Fx_h) + d_y(Fy_h)
            #
            # with v_h differentiated through its polynomial representation.
            #
            Fx_h = a_dense * v_dense_h
            Fy_h = b_dense * v_dense_h

            # Exact gradients of a,b at dense points
            a_dense, b_dense, ax_dense, ay_dense, bx_dense, by_dense = coefficient_field_with_derivatives(
                config.coeff_case, x_eval, y_eval
            )

            # Physical derivatives of v_h from its reference derivatives
            rx = ys / J
            sx = -yr / J
            ry = -xs / J
            sy = xr / J

            vx_h = rx * vr_dense_h + sx * vs_dense_h
            vy_h = ry * vr_dense_h + sy * vs_dense_h

            div_num_dense = ax_dense * v_dense_h + a_dense * vx_h + by_dense * v_dense_h + b_dense * vy_h

            div_ex_dense = exact_divergence_of_av_bv(
                x_eval, y_eval,
                coeff_case=config.coeff_case,
                x0=config.x0, y0=config.y0, sigma=config.sigma,
            )

            e_dense = div_num_dense - div_ex_dense

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
                f"[div h-study] n={n:3d} | K={K:7d} | "
                f"node_L2={node_L2:.6e} | node_Linf={node_linf:.6e} | "
                f"dense_L2={dense_L2:.6e} | dense_Linf={dense_linf:.6e} | "
                f"time={elapsed:.2f}s"
            )

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