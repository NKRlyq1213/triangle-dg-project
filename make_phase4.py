from pathlib import Path
import textwrap


FILES = {
    "geometry/sampling.py": textwrap.dedent(
        """
        from __future__ import annotations

        import numpy as np

        from .reference_triangle import reference_triangle_centroid
        from .barycentric import barycentric_to_cartesian


        def dense_barycentric_lattice(
            vertices: np.ndarray,
            resolution: int,
        ) -> np.ndarray:
            \"""
            Generate dense sampling points inside a triangle using a barycentric lattice.
            \"""
            if resolution < 1:
                raise ValueError("resolution must be >= 1")

            bary_points = []
            for i in range(resolution + 1):
                for j in range(resolution + 1 - i):
                    k = resolution - i - j
                    bary = np.array([i, j, k], dtype=float) / resolution
                    bary_points.append(bary)

            bary_points = np.array(bary_points, dtype=float)
            return barycentric_to_cartesian(bary_points, vertices)


        def centroid_star_sampling(
            vertices: np.ndarray,
            n_theta: int,
            n_r: int,
            include_endpoint: bool = True,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            \"""
            Generate centroid-centered star sampling points that fill the whole triangle.

            For each angle theta, compute the maximum admissible radius r_max(theta)
            before hitting the triangle boundary.

            Returns
            -------
            tuple[np.ndarray, np.ndarray, np.ndarray]
                (xy, theta_ids, radial_coordinates)
                - xy: sampled points, shape (n_points, 2)
                - theta_ids: angle index for each point, shape (n_points,)
                - radial_coordinates: normalized radial coordinate rho in [0,1],
                  shape (n_points,)
            \"""
            if n_theta < 3 or n_r < 2:
                raise ValueError("n_theta >= 3 and n_r >= 2 are required.")

            c = reference_triangle_centroid()
            cx, cy = float(c[0]), float(c[1])

            thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)

            pts = []
            theta_ids = []
            rhos = []

            for k, theta in enumerate(thetas):
                ct = float(np.cos(theta))
                st = float(np.sin(theta))

                candidates = []

                # x = 0
                if ct < -1e-14:
                    candidates.append((0.0 - cx) / ct)

                # y = 0
                if st < -1e-14:
                    candidates.append((0.0 - cy) / st)

                # x + y = 1
                denom = ct + st
                if denom > 1e-14:
                    candidates.append((1.0 - cx - cy) / denom)

                rmax = min(r for r in candidates if r > 0.0)

                rs = np.linspace(0.0, rmax, n_r, endpoint=include_endpoint)
                for j, r in enumerate(rs):
                    pts.append([cx + r * ct, cy + r * st])
                    theta_ids.append(k)
                    rho = 0.0 if rmax == 0.0 else r / rmax
                    rhos.append(rho)

            return (
                np.array(pts, dtype=float),
                np.array(theta_ids, dtype=int),
                np.array(rhos, dtype=float),
            )
        """
    ).strip() + "\n",

    "visualization/radial_plot.py": textwrap.dedent(
        """
        from __future__ import annotations

        import numpy as np
        import matplotlib.pyplot as plt


        def plot_radial_field(
            xy_eval: np.ndarray,
            u_eval: np.ndarray,
            vertices: np.ndarray,
            nodes: np.ndarray | None = None,
            title: str | None = None,
            ax=None,
        ):
            \"""
            Scatter-style visualization for centroid-based star sampling.
            \"""
            xy_eval = np.asarray(xy_eval, dtype=float)
            u_eval = np.asarray(u_eval, dtype=float).reshape(-1)

            if ax is None:
                fig, ax = plt.subplots(figsize=(6.5, 6))
            else:
                fig = ax.figure

            sc = ax.scatter(xy_eval[:, 0], xy_eval[:, 1], c=u_eval, s=12)
            fig.colorbar(sc, ax=ax)

            tri = np.vstack([vertices, vertices[0]])
            ax.plot(tri[:, 0], tri[:, 1], linewidth=1.5)

            if nodes is not None:
                nodes = np.asarray(nodes, dtype=float)
                ax.scatter(nodes[:, 0], nodes[:, 1], s=14, marker="o", alpha=0.8)

            ax.set_aspect("equal")
            ax.set_xlabel("xi")
            ax.set_ylabel("eta")
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.2)

            if title is not None:
                ax.set_title(title)

            return fig, ax


        def plot_radial_profile(
            rho: np.ndarray,
            u: np.ndarray,
            theta_ids: np.ndarray,
            n_curves_to_show: int = 8,
            title: str | None = None,
            ax=None,
        ):
            \"""
            Plot several radial profiles u(rho) for selected angle indices.
            \"""
            rho = np.asarray(rho, dtype=float)
            u = np.asarray(u, dtype=float).reshape(-1)
            theta_ids = np.asarray(theta_ids, dtype=int).reshape(-1)

            if ax is None:
                fig, ax = plt.subplots(figsize=(7, 4.5))
            else:
                fig = ax.figure

            unique_ids = np.unique(theta_ids)
            if unique_ids.size == 0:
                raise ValueError("theta_ids is empty.")

            step = max(1, unique_ids.size // n_curves_to_show)
            chosen = unique_ids[::step][:n_curves_to_show]

            for tid in chosen:
                mask = theta_ids == tid
                order = np.argsort(rho[mask])
                ax.plot(rho[mask][order], u[mask][order], label=f"theta_id={tid}")

            ax.set_xlabel("normalized radius rho")
            ax.set_ylabel("field value")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)

            if title is not None:
                ax.set_title(title)

            return fig, ax
        """
    ).strip() + "\n",

    "demo/demo_reconstruct_field_table1.py": textwrap.dedent(
        """
        from __future__ import annotations

        import matplotlib.pyplot as plt
        import numpy as np

        from data.table1_rules import load_table1_rule
        from geometry.reference_triangle import reference_triangle_vertices, reference_triangle_area
        from geometry.sampling import dense_barycentric_lattice
        from operators.vandermonde2d import vandermonde2d
        from operators.reconstruction import fit_modal_coefficients_weighted, evaluate_modal_expansion
        from problems.analytic_fields import ground_truth_function
        from visualization.triangle_field import plot_triangle_field


        def xy_to_rs(xy: np.ndarray) -> np.ndarray:
            xy = np.asarray(xy, dtype=float)
            rs = np.empty_like(xy)
            rs[:, 0] = 2.0 * xy[:, 0] - 1.0
            rs[:, 1] = 2.0 * xy[:, 1] - 1.0
            return rs


        def main() -> None:
            vertices = reference_triangle_vertices()
            area = reference_triangle_area()

            table_order = 4
            N = 4
            resolution = 50
            case_name = "smooth_bump"

            rule = load_table1_rule(table_order)

            xy_nodes = rule["xy"]
            rs_nodes = rule["rs"]
            w = rule["ws"]

            u_nodes = ground_truth_function(case_name, xy_nodes[:, 0], xy_nodes[:, 1])

            V = vandermonde2d(N, rs_nodes[:, 0], rs_nodes[:, 1])
            coeffs = fit_modal_coefficients_weighted(u_nodes, V, w, area=area)

            xy_eval = dense_barycentric_lattice(vertices, resolution=resolution)
            rs_eval = xy_to_rs(xy_eval)
            V_eval = vandermonde2d(N, rs_eval[:, 0], rs_eval[:, 1])
            u_eval = evaluate_modal_expansion(V_eval, coeffs)

            fig, ax = plot_triangle_field(
                xy_eval=xy_eval,
                u_eval=u_eval,
                vertices=vertices,
                nodes=xy_nodes,
                title=f"Reconstructed field: Table 1 order {table_order}, N={N}, case={case_name}",
                levels=25,
                show_nodes=True,
            )
            fig.savefig("reconstruct_field_table1_order4_N4.png", dpi=200, bbox_inches="tight")
            plt.close(fig)


        if __name__ == "__main__":
            main()
        """
    ).strip() + "\n",

    "demo/demo_error_field_table2.py": textwrap.dedent(
        """
        from __future__ import annotations

        import matplotlib.pyplot as plt
        import numpy as np

        from data.table2_rules import load_table2_rule
        from geometry.reference_triangle import reference_triangle_vertices, reference_triangle_area
        from geometry.sampling import dense_barycentric_lattice
        from operators.vandermonde2d import vandermonde2d
        from operators.reconstruction import fit_modal_coefficients_weighted, evaluate_modal_expansion
        from problems.analytic_fields import ground_truth_function
        from visualization.triangle_field import plot_triangle_field


        def xy_to_rs(xy: np.ndarray) -> np.ndarray:
            xy = np.asarray(xy, dtype=float)
            rs = np.empty_like(xy)
            rs[:, 0] = 2.0 * xy[:, 0] - 1.0
            rs[:, 1] = 2.0 * xy[:, 1] - 1.0
            return rs


        def main() -> None:
            vertices = reference_triangle_vertices()
            area = reference_triangle_area()

            table_order = 4
            N = 4
            resolution = 60
            case_name = "smooth_bump"

            rule = load_table2_rule(table_order)

            xy_nodes = rule["xy"]
            rs_nodes = rule["rs"]
            w = rule["ws"]

            u_nodes = ground_truth_function(case_name, xy_nodes[:, 0], xy_nodes[:, 1])

            V = vandermonde2d(N, rs_nodes[:, 0], rs_nodes[:, 1])
            coeffs = fit_modal_coefficients_weighted(u_nodes, V, w, area=area)

            xy_eval = dense_barycentric_lattice(vertices, resolution=resolution)
            rs_eval = xy_to_rs(xy_eval)
            V_eval = vandermonde2d(N, rs_eval[:, 0], rs_eval[:, 1])
            u_eval = evaluate_modal_expansion(V_eval, coeffs)
            u_exact = ground_truth_function(case_name, xy_eval[:, 0], xy_eval[:, 1])

            err = u_eval - u_exact

            fig, ax = plot_triangle_field(
                xy_eval=xy_eval,
                u_eval=err,
                vertices=vertices,
                nodes=xy_nodes,
                title=f"Error field: Table 2 order {table_order}, N={N}, case={case_name}",
                levels=25,
                show_nodes=True,
            )
            fig.savefig("error_field_table2_order4_N4.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

            print("max abs error =", np.max(np.abs(err)))
            print("l2-like rms error =", np.sqrt(np.mean(err**2)))


        if __name__ == "__main__":
            main()
        """
    ).strip() + "\n",

    "demo/demo_radial_sampling_table2.py": textwrap.dedent(
        """
        from __future__ import annotations

        import matplotlib.pyplot as plt
        import numpy as np

        from data.table2_rules import load_table2_rule
        from geometry.reference_triangle import reference_triangle_vertices, reference_triangle_area
        from geometry.sampling import centroid_star_sampling
        from operators.vandermonde2d import vandermonde2d
        from operators.reconstruction import fit_modal_coefficients_weighted, evaluate_modal_expansion
        from problems.analytic_fields import ground_truth_function
        from visualization.radial_plot import plot_radial_field, plot_radial_profile


        def xy_to_rs(xy: np.ndarray) -> np.ndarray:
            xy = np.asarray(xy, dtype=float)
            rs = np.empty_like(xy)
            rs[:, 0] = 2.0 * xy[:, 0] - 1.0
            rs[:, 1] = 2.0 * xy[:, 1] - 1.0
            return rs


        def main() -> None:
            vertices = reference_triangle_vertices()
            area = reference_triangle_area()

            table_order = 4
            N = 4
            case_name = "smooth_bump"

            rule = load_table2_rule(table_order)

            xy_nodes = rule["xy"]
            rs_nodes = rule["rs"]
            w = rule["ws"]

            u_nodes = ground_truth_function(case_name, xy_nodes[:, 0], xy_nodes[:, 1])

            V = vandermonde2d(N, rs_nodes[:, 0], rs_nodes[:, 1])
            coeffs = fit_modal_coefficients_weighted(u_nodes, V, w, area=area)

            xy_eval, theta_ids, rho = centroid_star_sampling(
                vertices=vertices,
                n_theta=120,
                n_r=50,
                include_endpoint=True,
            )
            rs_eval = xy_to_rs(xy_eval)
            V_eval = vandermonde2d(N, rs_eval[:, 0], rs_eval[:, 1])
            u_eval = evaluate_modal_expansion(V_eval, coeffs)

            fig1, ax1 = plot_radial_field(
                xy_eval=xy_eval,
                u_eval=u_eval,
                vertices=vertices,
                nodes=xy_nodes,
                title=f"Radial field: Table 2 order {table_order}, N={N}, case={case_name}",
            )
            fig1.savefig("radial_field_table2_order4_N4.png", dpi=200, bbox_inches="tight")
            plt.close(fig1)

            fig2, ax2 = plot_radial_profile(
                rho=rho,
                u=u_eval,
                theta_ids=theta_ids,
                n_curves_to_show=8,
                title=f"Radial profiles: Table 2 order {table_order}, N={N}, case={case_name}",
            )
            fig2.savefig("radial_profile_table2_order4_N4.png", dpi=200, bbox_inches="tight")
            plt.close(fig2)


        if __name__ == "__main__":
            main()
        """
    ).strip() + "\n",
}


def main() -> None:
    for rel_path, content in FILES.items():
        path = Path(rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"[OK] wrote {rel_path}")


if __name__ == "__main__":
    main()