from __future__ import annotations

from experiments.manifold_div_h_convergence import (
    ManifoldDivHConvergenceConfig,
    print_results_table,
    run_manifold_div_h_convergence,
)
from experiments.output_paths import photo_output_dir
from geometry.sphere_manifold_mesh import generate_spherical_octahedron_mesh
from geometry.sphere_manifold_metrics import build_manifold_geometry_cache
from operators.manifold_rhs import (
    build_manifold_table1_k4_reference_operators,
    manifold_rhs_constant_field,
)
from problems.sphere_advection import solid_body_velocity_xyz
from visualization.manifold_diagnostics import (
    plot_manifold_convergence,
    write_all_manifold_diagnostics,
)


def main() -> None:
    outdir = photo_output_dir(__file__, "manifold_divergence_diagnostics")
    ref_ops = build_manifold_table1_k4_reference_operators()

    n_div = 8
    nodes_xyz, EToV = generate_spherical_octahedron_mesh(n_div=n_div, R=1.0)
    geom = build_manifold_geometry_cache(
        nodes_xyz=nodes_xyz,
        EToV=EToV,
        rs_nodes=ref_ops.rs_nodes,
        Dr=ref_ops.Dr,
        Ds=ref_ops.Ds,
        R=1.0,
    )

    U, V, W = solid_body_velocity_xyz(geom.X, geom.Y, geom.Z, u0=1.0)
    diag = manifold_rhs_constant_field(geom, U, V, W, ref_ops=ref_ops)

    paths = write_all_manifold_diagnostics(
        nodes_xyz=nodes_xyz,
        EToV=EToV,
        geom=geom,
        U=U,
        V=V,
        W=W,
        div=diag["divergence"],
        output_dir=outdir,
    )

    results = run_manifold_div_h_convergence(
        ManifoldDivHConvergenceConfig(mesh_levels=(2, 4, 8, 16, 32), verbose=True)
    )
    conv_path = plot_manifold_convergence(results, outdir / "05_h_convergence.png")

    print()
    print_results_table(results)
    print()
    for path in [*paths, conv_path]:
        print("[OK] wrote " + str(path))


if __name__ == "__main__":
    main()
