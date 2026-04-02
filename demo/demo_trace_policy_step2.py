from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from data import load_table1_rule, load_table2_rule
from geometry import reference_triangle_vertices
from operators import (
    vandermonde2d,
    build_trace_policy,
    evaluate_embedded_face_values,
    evaluate_projected_face_values,
)


def _triangle_closed(vertices: np.ndarray) -> np.ndarray:
    return np.vstack([vertices, vertices[0]])


def _setup_axis(ax, vertices: np.ndarray, title: str) -> None:
    tri = _triangle_closed(vertices)
    ax.plot(tri[:, 0], tri[:, 1], color="black", linewidth=1.4)
    ax.set_aspect("equal")
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.12)
    ax.set_xlabel("r")
    ax.set_ylabel("s")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)


def _plot_face_ordered_points(ax, trace: dict, face_color: dict[int, str]) -> None:
    for face_id in (1, 2, 3):
        rs_face = trace["face_rs"][face_id]
        color = face_color[face_id]

        ax.scatter(
            rs_face[:, 0],
            rs_face[:, 1],
            s=55,
            color=color,
            label=f"face {face_id}",
            zorder=3,
        )

        for j, (rj, sj) in enumerate(rs_face):
            ax.text(
                rj,
                sj,
                str(j),
                fontsize=8,
                color="white",
                ha="center",
                va="center",
                bbox=dict(boxstyle="circle,pad=0.15", fc=color, ec="none", alpha=0.95),
                zorder=4,
            )


def main() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "photo"
    output_dir.mkdir(parents=True, exist_ok=True)

    vertices = reference_triangle_vertices()

    face_color = {
        1: "tab:blue",
        2: "tab:orange",
        3: "tab:green",
    }

    # -----------------------------
    # Table 1 : embedded trace
    # -----------------------------
    rule1 = load_table1_rule(4)
    trace1 = build_trace_policy(rule1)

    rs1 = rule1["rs"]

    # -----------------------------
    # Table 2 : projected trace
    # -----------------------------
    rule2 = load_table2_rule(4)
    trace2 = build_trace_policy(rule2, N=4, n_edge=5)

    rs2 = rule2["rs"]

    # -----------------------------
    # Polynomial exactness check
    # -----------------------------
    V1 = vandermonde2d(4, rs1[:, 0], rs1[:, 1])
    coeffs1 = np.linspace(0.3, 1.1, V1.shape[1])
    u1 = V1 @ coeffs1
    fvals1 = evaluate_embedded_face_values(u1, trace1)

    max_err1 = 0.0
    for face_id in (1, 2, 3):
        Vface = vandermonde2d(4, trace1["face_rs"][face_id][:, 0], trace1["face_rs"][face_id][:, 1])
        u_exact = Vface @ coeffs1
        max_err1 = max(max_err1, float(np.max(np.abs(fvals1[face_id] - u_exact))))

    V2 = vandermonde2d(4, rs2[:, 0], rs2[:, 1])
    coeffs2 = np.linspace(-0.6, 0.9, V2.shape[1])
    u2 = V2 @ coeffs2
    fvals2 = evaluate_projected_face_values(u2, trace2)

    max_err2 = 0.0
    for face_id in (1, 2, 3):
        Vface = vandermonde2d(4, trace2["face_rs"][face_id][:, 0], trace2["face_rs"][face_id][:, 1])
        u_exact = Vface @ coeffs2
        max_err2 = max(max_err2, float(np.max(np.abs(fvals2[face_id] - u_exact))))

    print("=== Step 2 trace-policy validation ===")
    print(f"Table 1 embedded face-value max error : {max_err1:.3e}")
    print(f"Table 2 projected face-value max error: {max_err2:.3e}")

    # -----------------------------
    # Visualization
    # -----------------------------
    fig, axs = plt.subplots(1, 2, figsize=(13, 6))

    # Table 1
    _setup_axis(axs[0], vertices, "Table 1: embedded face nodes")
    axs[0].scatter(rs1[:, 0], rs1[:, 1], s=24, color="lightgray", label="volume nodes", zorder=2)
    _plot_face_ordered_points(axs[0], trace1, face_color)
    axs[0].legend(loc="upper right", fontsize=8)

    # Table 2
    _setup_axis(axs[1], vertices, "Table 2: projected face nodes")
    axs[1].scatter(rs2[:, 0], rs2[:, 1], s=24, color="lightgray", label="volume nodes", zorder=2)
    _plot_face_ordered_points(axs[1], trace2, face_color)
    axs[1].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "demo_trace_policy_step2.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()