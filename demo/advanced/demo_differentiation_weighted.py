from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from data.table1_rules import load_table1_rule
from data.table2_rules import load_table2_rule
from experiments.output_paths import photo_output_dir
from geometry.reference_triangle import (
    reference_triangle_area,
    reference_triangle_vertices,
)
from geometry.sampling import dense_barycentric_lattice
from operators.vandermonde2d import vandermonde2d, grad_vandermonde2d
from operators.differentiation import differentiation_matrices_weighted
from operators.reconstruction import (
    fit_modal_coefficients_weighted,
    evaluate_modal_expansion,
)
from visualization.triangle_field import plot_triangle_field


def poly_u(r: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    degree-2 polynomial in the approximation space when N >= 2
    """
    return (
        1.0
        + 0.5 * r
        - 0.25 * s
        + 0.75 * r * s
        + 0.2 * r**2
        - 0.1 * s**2
        + 0.4 * s**4
        + 0.3 * r**2 * s**2
        - 0.2 * r**4
    )


def poly_ur_exact(r: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    ∂u/∂r
    """
    return 0.5 + 0.75 * s + 0.4 * r + 0.6 * r * s**2 - 0.8 * r**3


def poly_us_exact(r: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    ∂u/∂s
    """
    return -0.25 + 0.75 * r - 0.2 * s + 1.6 * s**3 + 0.6 * r**2 * s

def smooth_test_function(r: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    A smooth test function that is not a low-degree polynomial.
    This can be used to test the accuracy of the differentiation operators on more general functions.
    """
    r0 = -1/3
    s0 = -1/3
    sigma = np.sqrt(1/30)
    return np.exp(-((r - r0)**2 + (s - s0)**2) / (2.0 * sigma**2))

def smooth_test_function_r(r: np.ndarray, s: np.ndarray) -> np.ndarray:
    r0 = -1/3
    s0 = -1/3
    sigma = np.sqrt(1/30)
    return -((r - r0) / sigma**2) * smooth_test_function(r, s)

def smooth_test_function_s(r: np.ndarray, s: np.ndarray) -> np.ndarray:
    r0 = -1/3
    s0 = -1/3
    sigma = np.sqrt(1/30)
    return -((s - s0) / sigma**2) * smooth_test_function(r, s)

def report_error(name: str, num: np.ndarray, exact: np.ndarray) -> None:
    err = num - exact
    print(f"{name}:")
    print(f"  max abs error = {np.max(np.abs(err)):.3e}")
    print(f"  rms error     = {np.sqrt(np.mean(err**2)):.3e}")


def main() -> None:
    output_dir = photo_output_dir(__file__, "differentiation_weighted")
    #==================================================================
    #=====================table 1======================================
    #==================================================================
    # ------------------------------------------------------------
    # 1. setup
    # ------------------------------------------------------------
    table_order = 4
    N = 4
    resolution = 60

    area = reference_triangle_area()
    vertices = reference_triangle_vertices()

    rule = load_table1_rule(table_order)
    r = rule["rs"][:, 0]
    s = rule["rs"][:, 1]
    w = rule["ws"]

    # ------------------------------------------------------------
    # 2. build operators on sampling nodes
    # ------------------------------------------------------------
    V = vandermonde2d(N, r, s)
    Vr, Vs = grad_vandermonde2d(N, r, s)

    Dr, Ds = differentiation_matrices_weighted(
        V=V,
        Vr=Vr,
        Vs=Vs,
        weights=w,
        area=area,
    )

    # ------------------------------------------------------------
    # 3. nodal differentiation test
    # ------------------------------------------------------------
    u_nodes = poly_u(r, s)

    ur_nodes_num = Dr @ u_nodes
    us_nodes_num = Ds @ u_nodes

    ur_nodes_exact = poly_ur_exact(r, s)
    us_nodes_exact = poly_us_exact(r, s)

    print("=" * 60)
    print("NODAL DIFFERENTIATION TEST ON TABLE 1 RULE & POLTNOMIAL FUNCTION")
    print("=" * 60)
    report_error("d/dr on rule nodes", ur_nodes_num, ur_nodes_exact)
    report_error("d/ds on rule nodes", us_nodes_num, us_nodes_exact)

    # ------------------------------------------------------------
    # 4. dense-field derivative evaluation via modal coefficients
    # ------------------------------------------------------------
    '''
    coeffs = fit_modal_coefficients_weighted(
        u_nodes=u_nodes,
        V=V,
        weights=w,
        area=area,
    )

    from geometry.sampling import centroid_star_sampling
    ### AI: 避開邊界，確保評估點都在內部，這樣才能測試內部的微分準確度。
    rs_eval, theta_ids, rho = centroid_star_sampling(
        vertices=vertices,
        n_theta=120,
        n_r=50,
        include_endpoint=False,   # 關鍵：不要碰邊界
    )
    re = rs_eval[:, 0]
    se = rs_eval[:, 1]
    Vre, Vse = grad_vandermonde2d(N, re, se)

    ur_eval_num = evaluate_modal_expansion(Vre, coeffs)
    us_eval_num = evaluate_modal_expansion(Vse, coeffs)

    ur_eval_exact = poly_ur_exact(re, se)
    us_eval_exact = poly_us_exact(re, se)
    
    print()
    print("=" * 60)
    print("DENSE FIELD DIFFERENTIATION TEST ON TABLE 1 RULE & POLTNOMIAL FUNCTION")
    print("=" * 60)
    report_error("d/dr on dense interior points", ur_eval_num, ur_eval_exact)
    report_error("d/ds on dense interior points", us_eval_num, us_eval_exact)
    '''
    # ------------------------------------------------------------
    # 4. dense-field derivative evaluation via nodal derivatives
    #    then reconstruct to dense points (including vertices)
    # ------------------------------------------------------------
    from geometry.sampling import dense_barycentric_lattice

    # 先用已經建立好的 Dr, Ds 在 rule nodes 上求導數
    ur_nodes_num = Dr @ u_nodes
    us_nodes_num = Ds @ u_nodes

    # 把「導數場」當成新的 scalar field，各自求 modal coefficients
    coeffs_r = fit_modal_coefficients_weighted(
        u_nodes=ur_nodes_num,
        V=V,
        weights=w,
        area=area,
    )
    coeffs_s = fit_modal_coefficients_weighted(
        u_nodes=us_nodes_num,
        V=V,
        weights=w,
        area=area,
    )

    # 這裡可包含邊界與頂點
    # 注意：vertices 是 reference triangle 的三個點，所以這裡回傳的座標
    # 仍可直接當作 (r, s) 來用
    rs_eval = dense_barycentric_lattice(vertices=vertices, resolution=resolution)
    re = rs_eval[:, 0]
    se = rs_eval[:, 1]

    # 在 dense points 上只用普通 Vandermonde 評估，不再用 grad_vandermonde2d
    V_eval = vandermonde2d(N, re, se)

    ur_eval_num = evaluate_modal_expansion(V_eval, coeffs_r)
    us_eval_num = evaluate_modal_expansion(V_eval, coeffs_s)

    ur_eval_exact = poly_ur_exact(re, se)
    us_eval_exact = poly_us_exact(re, se)

    print()
    print("=" * 60)
    print("DENSE FIELD DIFFERENTIATION TEST ON TABLE 1 RULE & POLYNOMIAL FUNCTION")
    print("=" * 60)
    report_error("d/dr on dense points (including vertices)", ur_eval_num, ur_eval_exact)
    report_error("d/ds on dense points (including vertices)", us_eval_num, us_eval_exact)

    # 額外把三個頂點的數值列出來看
    vertex_indices = []
    for vx, vy in vertices:
        idx = np.where(
            (np.abs(re - vx) < 1e-14) &
            (np.abs(se - vy) < 1e-14)
        )[0]
        if idx.size > 0:
            vertex_indices.append(idx[0])

    print()
    print("=" * 60)
    print("VERTEX DERIVATIVE VALUES")
    print("=" * 60)
    for idx in vertex_indices:
        print(f"(r, s) = ({re[idx]: .1f}, {se[idx]: .1f})")
        print(f"  ur_num   = {ur_eval_num[idx]: .15e}")
        print(f"  ur_exact = {ur_eval_exact[idx]: .15e}")
        print(f"  us_num   = {us_eval_num[idx]: .15e}")
        print(f"  us_exact = {us_eval_exact[idx]: .15e}")
    # ------------------------------------------------------------
    # 5. visualize error fields
    # ------------------------------------------------------------
    err_r = ur_eval_num - ur_eval_exact
    err_s = us_eval_num - us_eval_exact

    fig1, ax1 = plot_triangle_field(
        rs_eval=rs_eval,
        u_eval=err_r,
        vertices=vertices,
        nodes=rule["rs"],
        title=f"d/dr poly error field (Table 1 order={table_order}, N={N})",
        levels=25,
        show_nodes=True,
    )
    fig1.savefig(output_dir / "differentiation_error_dr_table1_polynomial.png", dpi=200, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plot_triangle_field(
        rs_eval=rs_eval,
        u_eval=err_s,
        vertices=vertices,
        nodes=rule["rs"],
        title=f"d/ds poly error field (Table 1 order={table_order}, N={N})",
        levels=25,
        show_nodes=True,
    )
    fig2.savefig(output_dir / "differentiation_error_ds_table1_polynomial.png", dpi=200, bbox_inches="tight")
    plt.close(fig2)

    print()
    print("=" * 60)
    print("OUTPUT")
    print("=" * 60)
    print(f"Saved figures to: {output_dir}")
    print("  - differentiation_error_dr_table1_polynomial.png")
    print("  - differentiation_error_ds_table1_polynomial.png")

    err_r = ur_eval_num - ur_eval_exact
    idx = np.argmax(np.abs(err_r))

    print("worst point index:", idx)
    print("r, s =", re[idx], se[idx])
    print("numerical =", ur_eval_num[idx])
    print("exact     =", ur_eval_exact[idx])
    print("abs error =", abs(err_r[idx]))


    u_nodes = smooth_test_function(r, s)

    ur_nodes_num = Dr @ u_nodes
    us_nodes_num = Ds @ u_nodes

    ur_nodes_exact = smooth_test_function_r(r, s)
    us_nodes_exact = smooth_test_function_s(r, s)

    print("=" * 60)
    print("NODAL DIFFERENTIATION TEST ON TABLE 1 RULE & SMOOTH FUNCTION")
    print("=" * 60)
    report_error("d/dr on rule nodes", ur_nodes_num, ur_nodes_exact)
    report_error("d/ds on rule nodes", us_nodes_num, us_nodes_exact)

    # ------------------------------------------------------------
    # 4. dense-field derivative evaluation via modal coefficients
    # ------------------------------------------------------------
    '''
    coeffs = fit_modal_coefficients_weighted(
        u_nodes=u_nodes,
        V=V,
        weights=w,
        area=area,
    )

    from geometry.sampling import centroid_star_sampling

    rs_eval, theta_ids, rho = centroid_star_sampling(
        vertices=vertices,
        n_theta=120,
        n_r=50,
        include_endpoint=False,   # 關鍵：不要碰邊界
    )
    re = rs_eval[:, 0]
    se = rs_eval[:, 1]
    Vre, Vse = grad_vandermonde2d(N, re, se)

    ur_eval_num = evaluate_modal_expansion(Vre, coeffs)
    us_eval_num = evaluate_modal_expansion(Vse, coeffs)

    ur_eval_exact = smooth_test_function_r(re, se)
    us_eval_exact = smooth_test_function_s(re, se)

    print()
    print("=" * 60)
    print("DENSE FIELD DIFFERENTIATION TEST ON TABLE 1 RULE & SMOOTH FUNCTION")
    print("=" * 60)
    report_error("d/dr on dense interior points", ur_eval_num, ur_eval_exact)
    report_error("d/ds on dense interior points", us_eval_num, us_eval_exact)
    '''
    # ------------------------------------------------------------
    # 4. dense-field derivative evaluation via nodal derivatives
    #    then reconstruct to dense points (including vertices)
    # ------------------------------------------------------------
    from geometry.sampling import dense_barycentric_lattice

    # 先用已經建立好的 Dr, Ds 在 rule nodes 上求導數
    ur_nodes_num = Dr @ u_nodes
    us_nodes_num = Ds @ u_nodes

    # 把「導數場」當成新的 scalar field，各自求 modal coefficients
    coeffs_r = fit_modal_coefficients_weighted(
        u_nodes=ur_nodes_num,
        V=V,
        weights=w,
        area=area,
    )
    coeffs_s = fit_modal_coefficients_weighted(
        u_nodes=us_nodes_num,
        V=V,
        weights=w,
        area=area,
    )

    # 這裡可包含邊界與頂點
    # 注意：vertices 是 reference triangle 的三個點，所以這裡回傳的座標
    # 仍可直接當作 (r, s) 來用
    rs_eval = dense_barycentric_lattice(vertices=vertices, resolution=resolution)
    re = rs_eval[:, 0]
    se = rs_eval[:, 1]

    # 在 dense points 上只用普通 Vandermonde 評估，不再用 grad_vandermonde2d
    V_eval = vandermonde2d(N, re, se)

    ur_eval_num = evaluate_modal_expansion(V_eval, coeffs_r)
    us_eval_num = evaluate_modal_expansion(V_eval, coeffs_s)

    ur_eval_exact = smooth_test_function_r(re, se)
    us_eval_exact = smooth_test_function_s(re, se)

    print()
    print("=" * 60)
    print("DENSE FIELD DIFFERENTIATION TEST ON TABLE 1 RULE & SMOOTH FUNCTION")
    print("=" * 60)
    report_error("d/dr on dense points (including vertices)", ur_eval_num, ur_eval_exact)
    report_error("d/ds on dense points (including vertices)", us_eval_num, us_eval_exact)

    # 額外把三個頂點的數值列出來看
    vertex_indices = []
    for vx, vy in vertices:
        idx = np.where(
            (np.abs(re - vx) < 1e-14) &
            (np.abs(se - vy) < 1e-14)
        )[0]
        if idx.size > 0:
            vertex_indices.append(idx[0])

    print()
    print("=" * 60)
    print("VERTEX DERIVATIVE VALUES")
    print("=" * 60)
    for idx in vertex_indices:
        print(f"(r, s) = ({re[idx]: .1f}, {se[idx]: .1f})")
        print(f"  ur_num   = {ur_eval_num[idx]: .15e}")
        print(f"  ur_exact = {ur_eval_exact[idx]: .15e}")
        print(f"  us_num   = {us_eval_num[idx]: .15e}")
        print(f"  us_exact = {us_eval_exact[idx]: .15e}")
    # ------------------------------------------------------------
    # 5. visualize error fields
    # ------------------------------------------------------------
    err_r = ur_eval_num - ur_eval_exact
    err_s = us_eval_num - us_eval_exact

    fig1, ax1 = plot_triangle_field(
        rs_eval=rs_eval,
        u_eval=err_r,
        vertices=vertices,
        nodes=rule["rs"],
        title=f"d/dr smooth error field (Table 1 order={table_order}, N={N})",
        levels=25,
        show_nodes=True,
    )
    fig1.savefig(output_dir / "differentiation_error_dr_table1_smooth.png", dpi=200, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plot_triangle_field(
        rs_eval=rs_eval,
        u_eval=err_s,
        vertices=vertices,
        nodes=rule["rs"],
        title=f"d/ds smooth error field (Table 1 order={table_order}, N={N})",
        levels=25,
        show_nodes=True,
    )
    fig2.savefig(output_dir / "differentiation_error_ds_table1_smooth.png", dpi=200, bbox_inches="tight")
    plt.close(fig2)

    print()
    print("=" * 60)
    print("OUTPUT")
    print("=" * 60)
    print(f"Saved figures to: {output_dir}")
    print("  - differentiation_error_dr_table1_smooth.png")
    print("  - differentiation_error_ds_table1_smooth.png")

    err_r = ur_eval_num - ur_eval_exact
    idx = np.argmax(np.abs(err_r))

    print("worst point index:", idx)
    print("r, s =", re[idx], se[idx])
    print("numerical =", ur_eval_num[idx])
    print("exact     =", ur_eval_exact[idx])
    print("abs error =", abs(err_r[idx]))

    ur_eval_exact = smooth_test_function_r(re, se)
    us_eval_exact = smooth_test_function_s(re, se)

    fig1, ax1 = plot_triangle_field(
        rs_eval=rs_eval,
        u_eval=ur_eval_exact,
        vertices=vertices,
        nodes=rule["rs"],
        title=f"d/dr smooth exact field (Table 1 order={table_order}, N={N})",
        levels=25,
        show_nodes=True,
    )
    fig1.savefig(output_dir / "differentiation_exact_dr_table1_smooth.png", dpi=200, bbox_inches="tight")
    plt.close(fig1)
    fig2, ax2 = plot_triangle_field(
        rs_eval=rs_eval,
        u_eval=us_eval_exact,
        vertices=vertices,
        nodes=rule["rs"],
        title=f"d/ds smooth exact field (Table 1 order={table_order}, N={N})",
        levels=25,
        show_nodes=True,
    )
    fig2.savefig(output_dir / "differentiation_exact_ds_table1_smooth.png", dpi=200, bbox_inches="tight")
    plt.close(fig2)
    
    #==================================================================
    #=====================table 2======================================
    #==================================================================
    # ------------------------------------------------------------
    # 1. setup
    # ------------------------------------------------------------
    table_order = 4
    N = 4
    resolution = 60

    area = reference_triangle_area()
    vertices = reference_triangle_vertices()

    rule = load_table2_rule(table_order)
    r = rule["rs"][:, 0]
    s = rule["rs"][:, 1]
    w = rule["ws"]

    # ------------------------------------------------------------
    # 2. build operators on sampling nodes
    # ------------------------------------------------------------
    V = vandermonde2d(N, r, s)
    Vr, Vs = grad_vandermonde2d(N, r, s)

    Dr, Ds = differentiation_matrices_weighted(
        V=V,
        Vr=Vr,
        Vs=Vs,
        weights=w,
        area=area,
    )

    # ------------------------------------------------------------
    # 3. nodal differentiation test
    # ------------------------------------------------------------
    u_nodes = poly_u(r, s)

    ur_nodes_num = Dr @ u_nodes
    us_nodes_num = Ds @ u_nodes

    ur_nodes_exact = poly_ur_exact(r, s)
    us_nodes_exact = poly_us_exact(r, s)

    print("=" * 60)
    print("NODAL DIFFERENTIATION TEST")
    print("=" * 60)
    report_error("d/dr on rule nodes", ur_nodes_num, ur_nodes_exact)
    report_error("d/ds on rule nodes", us_nodes_num, us_nodes_exact)

    # ------------------------------------------------------------
    # 4. dense-field derivative evaluation via modal coefficients
    # ------------------------------------------------------------
    '''
    coeffs = fit_modal_coefficients_weighted(
        u_nodes=u_nodes,
        V=V,
        weights=w,
        area=area,
    )

    from geometry.sampling import centroid_star_sampling

    rs_eval, theta_ids, rho = centroid_star_sampling(
        vertices=vertices,
        n_theta=120,
        n_r=50,
        include_endpoint=False,   # 關鍵：不要碰邊界
    )
    re = rs_eval[:, 0]
    se = rs_eval[:, 1]
    Vre, Vse = grad_vandermonde2d(N, re, se)

    ur_eval_num = evaluate_modal_expansion(Vre, coeffs)
    us_eval_num = evaluate_modal_expansion(Vse, coeffs)

    ur_eval_exact = poly_ur_exact(re, se)
    us_eval_exact = poly_us_exact(re, se)

    print()
    print("=" * 60)
    print("DENSE FIELD DIFFERENTIATION TEST")
    print("=" * 60)
    report_error("d/dr on dense interior points", ur_eval_num, ur_eval_exact)
    report_error("d/ds on dense interior points", us_eval_num, us_eval_exact)
    '''
    # ------------------------------------------------------------
    # 4. dense-field derivative evaluation via nodal derivatives
    #    then reconstruct to dense points (including vertices)
    # ------------------------------------------------------------
    from geometry.sampling import dense_barycentric_lattice

    # 先用已經建立好的 Dr, Ds 在 rule nodes 上求導數
    ur_nodes_num = Dr @ u_nodes
    us_nodes_num = Ds @ u_nodes

    # 把「導數場」當成新的 scalar field，各自求 modal coefficients
    coeffs_r = fit_modal_coefficients_weighted(
        u_nodes=ur_nodes_num,
        V=V,
        weights=w,
        area=area,
    )
    coeffs_s = fit_modal_coefficients_weighted(
        u_nodes=us_nodes_num,
        V=V,
        weights=w,
        area=area,
    )

    # 這裡可包含邊界與頂點
    # 注意：vertices 是 reference triangle 的三個點，所以這裡回傳的座標
    # 仍可直接當作 (r, s) 來用
    rs_eval = dense_barycentric_lattice(vertices=vertices, resolution=resolution)
    re = rs_eval[:, 0]
    se = rs_eval[:, 1]

    # 在 dense points 上只用普通 Vandermonde 評估，不再用 grad_vandermonde2d
    V_eval = vandermonde2d(N, re, se)

    ur_eval_num = evaluate_modal_expansion(V_eval, coeffs_r)
    us_eval_num = evaluate_modal_expansion(V_eval, coeffs_s)

    ur_eval_exact = poly_ur_exact(re, se)
    us_eval_exact = poly_us_exact(re, se)

    print()
    print("=" * 60)
    print("DENSE FIELD DIFFERENTIATION TEST ON TABLE 2 RULE & POLYNOMIAL FUNCTION")
    print("=" * 60)
    report_error("d/dr on dense points (including vertices)", ur_eval_num, ur_eval_exact)
    report_error("d/ds on dense points (including vertices)", us_eval_num, us_eval_exact)

    # 額外把三個頂點的數值列出來看
    vertex_indices = []
    for vx, vy in vertices:
        idx = np.where(
            (np.abs(re - vx) < 1e-14) &
            (np.abs(se - vy) < 1e-14)
        )[0]
        if idx.size > 0:
            vertex_indices.append(idx[0])

    print()
    print("=" * 60)
    print("VERTEX DERIVATIVE VALUES")
    print("=" * 60)
    for idx in vertex_indices:
        print(f"(r, s) = ({re[idx]: .1f}, {se[idx]: .1f})")
        print(f"  ur_num   = {ur_eval_num[idx]: .15e}")
        print(f"  ur_exact = {ur_eval_exact[idx]: .15e}")
        print(f"  us_num   = {us_eval_num[idx]: .15e}")
        print(f"  us_exact = {us_eval_exact[idx]: .15e}")
    # ------------------------------------------------------------
    # 5. visualize error fields
    # ------------------------------------------------------------
    err_r = ur_eval_num - ur_eval_exact
    err_s = us_eval_num - us_eval_exact

    fig1, ax1 = plot_triangle_field(
        rs_eval=rs_eval,
        u_eval=err_r,
        vertices=vertices,
        nodes=rule["rs"],
        title=f"d/dr poly error field (Table2 order={table_order}, N={N})",
        levels=25,
        show_nodes=True,
    )
    fig1.savefig(output_dir / "differentiation_error_dr_table2_polynomial.png", dpi=200, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plot_triangle_field(
        rs_eval=rs_eval,
        u_eval=err_s,
        vertices=vertices,
        nodes=rule["rs"],
        title=f"d/ds poly error field (Table2 order={table_order}, N={N})",
        levels=25,
        show_nodes=True,
    )
    fig2.savefig(output_dir / "differentiation_error_ds_table2_polynomial.png", dpi=200, bbox_inches="tight")
    plt.close(fig2)

    print()
    print("=" * 60)
    print("OUTPUT")
    print("=" * 60)
    print(f"Saved figures to: {output_dir}")
    print("  - differentiation_error_dr_table2_polynomial.png")
    print("  - differentiation_error_ds_table2_polynomial.png")

    err_r = ur_eval_num - ur_eval_exact
    idx = np.argmax(np.abs(err_r))

    print("worst point index:", idx)
    print("r, s =", re[idx], se[idx])
    print("numerical =", ur_eval_num[idx])
    print("exact     =", ur_eval_exact[idx])
    print("abs error =", abs(err_r[idx]))

    u_nodes = smooth_test_function(r, s)

    ur_nodes_num = Dr @ u_nodes
    us_nodes_num = Ds @ u_nodes

    ur_nodes_exact = smooth_test_function_r(r, s)
    us_nodes_exact = smooth_test_function_s(r, s)

    print("=" * 60)
    print("NODAL DIFFERENTIATION TEST ON TABLE 2 RULE & SMOOTH FUNCTION")
    print("=" * 60)
    report_error("d/dr on rule nodes", ur_nodes_num, ur_nodes_exact)
    report_error("d/ds on rule nodes", us_nodes_num, us_nodes_exact)

    # ------------------------------------------------------------
    # 4. dense-field derivative evaluation via modal coefficients
    # ------------------------------------------------------------
    '''
    coeffs = fit_modal_coefficients_weighted(
        u_nodes=u_nodes,
        V=V,
        weights=w,
        area=area,
    )

    from geometry.sampling import centroid_star_sampling

    rs_eval, theta_ids, rho = centroid_star_sampling(
        vertices=vertices,
        n_theta=120,
        n_r=50,
        include_endpoint=False,   # 關鍵：不要碰邊界
    )
    re = rs_eval[:, 0]
    se = rs_eval[:, 1]
    Vre, Vse = grad_vandermonde2d(N, re, se)

    ur_eval_num = evaluate_modal_expansion(Vre, coeffs)
    us_eval_num = evaluate_modal_expansion(Vse, coeffs)

    ur_eval_exact = smooth_test_function_r(re, se)
    us_eval_exact = smooth_test_function_s(re, se)

    print()
    print("=" * 60)
    print("DENSE FIELD DIFFERENTIATION TEST ON TABLE 2 RULE & SMOOTH FUNCTION")
    print("=" * 60)
    report_error("d/dr on dense interior points", ur_eval_num, ur_eval_exact)
    report_error("d/ds on dense interior points", us_eval_num, us_eval_exact)
    '''
    # ------------------------------------------------------------
    # 4. dense-field derivative evaluation via nodal derivatives
    #    then reconstruct to dense points (including vertices)
    # ------------------------------------------------------------
    from geometry.sampling import dense_barycentric_lattice

    # 先用已經建立好的 Dr, Ds 在 rule nodes 上求導數
    ur_nodes_num = Dr @ u_nodes
    us_nodes_num = Ds @ u_nodes

    # 把「導數場」當成新的 scalar field，各自求 modal coefficients
    coeffs_r = fit_modal_coefficients_weighted(
        u_nodes=ur_nodes_num,
        V=V,
        weights=w,
        area=area,
    )
    coeffs_s = fit_modal_coefficients_weighted(
        u_nodes=us_nodes_num,
        V=V,
        weights=w,
        area=area,
    )

    # 這裡可包含邊界與頂點
    # 注意：vertices 是 reference triangle 的三個點，所以這裡回傳的座標
    # 仍可直接當作 (r, s) 來用
    rs_eval = dense_barycentric_lattice(vertices=vertices, resolution=resolution)
    re = rs_eval[:, 0]
    se = rs_eval[:, 1]

    # 在 dense points 上只用普通 Vandermonde 評估，不再用 grad_vandermonde2d
    V_eval = vandermonde2d(N, re, se)

    ur_eval_num = evaluate_modal_expansion(V_eval, coeffs_r)
    us_eval_num = evaluate_modal_expansion(V_eval, coeffs_s)

    ur_eval_exact = smooth_test_function_r(re, se)
    us_eval_exact = smooth_test_function_s(re, se)

    print()
    print("=" * 60)
    print("DENSE FIELD DIFFERENTIATION TEST ON TABLE 2 RULE & SMOOTH FUNCTION")
    print("=" * 60)
    report_error("d/dr on dense points (including vertices)", ur_eval_num, ur_eval_exact)
    report_error("d/ds on dense points (including vertices)", us_eval_num, us_eval_exact)

    # 額外把三個頂點的數值列出來看
    vertex_indices = []
    for vx, vy in vertices:
        idx = np.where(
            (np.abs(re - vx) < 1e-14) &
            (np.abs(se - vy) < 1e-14)
        )[0]
        if idx.size > 0:
            vertex_indices.append(idx[0])

    print()
    print("=" * 60)
    print("VERTEX DERIVATIVE VALUES")
    print("=" * 60)
    for idx in vertex_indices:
        print(f"(r, s) = ({re[idx]: .1f}, {se[idx]: .1f})")
        print(f"  ur_num   = {ur_eval_num[idx]: .15e}")
        print(f"  ur_exact = {ur_eval_exact[idx]: .15e}")
        print(f"  us_num   = {us_eval_num[idx]: .15e}")
        print(f"  us_exact = {us_eval_exact[idx]: .15e}")
    # ------------------------------------------------------------
    # 5. visualize error fields
    # ------------------------------------------------------------
    err_r = ur_eval_num - ur_eval_exact
    err_s = us_eval_num - us_eval_exact

    fig1, ax1 = plot_triangle_field(
        rs_eval=rs_eval,
        u_eval=err_r,
        vertices=vertices,
        nodes=rule["rs"],
        title=f"d/dr smooth error field (Table 2 order={table_order}, N={N})",
        levels=25,
        show_nodes=True,
    )
    fig1.savefig(output_dir / "differentiation_error_dr_table2_smooth.png", dpi=200, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plot_triangle_field(
        rs_eval=rs_eval,
        u_eval=err_s,
        vertices=vertices,
        nodes=rule["rs"],
        title=f"d/ds smooth error field (Table 2 order={table_order}, N={N})",
        levels=25,
        show_nodes=True,
    )
    fig2.savefig(output_dir / "differentiation_error_ds_table2_smooth.png", dpi=200, bbox_inches="tight")
    plt.close(fig2)

    print()
    print("=" * 60)
    print("OUTPUT")
    print("=" * 60)
    print(f"Saved figures to: {output_dir}")
    print("  - differentiation_error_dr_table2_smooth.png")
    print("  - differentiation_error_ds_table2_smooth.png")

    err_r = ur_eval_num - ur_eval_exact
    idx = np.argmax(np.abs(err_r))

    print("worst point index:", idx)
    print("r, s =", re[idx], se[idx])
    print("numerical =", ur_eval_num[idx])
    print("exact     =", ur_eval_exact[idx])
    print("abs error =", abs(err_r[idx]))

    ur_eval_exact = smooth_test_function_r(re, se)
    us_eval_exact = smooth_test_function_s(re, se)

    fig1, ax1 = plot_triangle_field(
        rs_eval=rs_eval,
        u_eval=ur_eval_exact,
        vertices=vertices,
        nodes=rule["rs"],
        title=f"d/dr smooth exact field (Table 2 order={table_order}, N={N})",
        levels=25,
        show_nodes=True,
    )
    fig1.savefig(output_dir / "differentiation_exact_dr_table2_smooth.png", dpi=200, bbox_inches="tight")
    plt.close(fig1)
    fig2, ax2 = plot_triangle_field(
        rs_eval=rs_eval,
        u_eval=us_eval_exact,
        vertices=vertices,
        nodes=rule["rs"],
        title=f"d/ds smooth exact field (Table 2 order={table_order}, N={N})",
        levels=25,
        show_nodes=True,
    )
    fig2.savefig(output_dir / "differentiation_exact_ds_table2_smooth.png", dpi=200, bbox_inches="tight")
    plt.close(fig2)

if __name__ == "__main__":
    main()
