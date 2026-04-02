from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from data import load_table1_rule, load_table2_rule
from geometry import (
    structured_square_tri_mesh,
    validate_mesh_orientation,
    build_face_connectivity,
    map_reference_nodes_to_all_elements,
)
from operators import (
    build_trace_policy,
    pair_face_traces,
    interior_face_pair_mismatches,
)


def _global_poly(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (
        1.0
        + 2.0 * x
        - 0.5 * y
        + 0.75 * x * y
        + 0.30 * x**2
        - 0.20 * y**2
    )


def _draw_mesh(ax, VX: np.ndarray, VY: np.ndarray, EToV: np.ndarray) -> None:
    for vids in EToV:
        tri = np.array(
            [
                [VX[vids[0]], VY[vids[0]]],
                [VX[vids[1]], VY[vids[1]]],
                [VX[vids[2]], VY[vids[2]]],
                [VX[vids[0]], VY[vids[0]]],
            ],
            dtype=float,
        )
        ax.plot(tri[:, 0], tri[:, 1], linewidth=1.2, color="black")


def _highlight_selected_pair(ax, VX, VY, EToV, conn, k: int, f: int) -> None:
    _draw_mesh(ax, VX, VY, EToV)

    mids = conn["face_midpoints"]
    EToE = conn["EToE"]
    EToF = conn["EToF"]

    nbr = int(EToE[k, f - 1])
    nbr_f = int(EToF[k, f - 1])

    # mark the two elements
    for elem, color, label in [(k, "tab:blue", f"T{int(k+1)}"), (nbr, "tab:orange", f"T{int(nbr+1)}")]:
        vids = EToV[elem]
        tri = np.array(
            [
                [VX[vids[0]], VY[vids[0]]],
                [VX[vids[1]], VY[vids[1]]],
                [VX[vids[2]], VY[vids[2]]],
            ],
            dtype=float,
        )
        center = np.mean(tri, axis=0)
        ax.text(
            center[0],
            center[1],
            label,
            fontsize=11,
            ha="center",
            va="center",
            color=color,
            fontweight="bold",
        )

    # mark the paired face midpoints
    xm, ym = mids[k, f - 1]
    xn, yn = mids[nbr, nbr_f - 1]

    ax.scatter([xm], [ym], s=80, color="tab:blue", zorder=3)
    ax.scatter([xn], [yn], s=80, color="tab:orange", zorder=3)

    ax.text(xm, ym, f"f{f}", color="white", fontsize=9, ha="center", va="center")
    ax.text(xn, yn, f"f{nbr_f}", color="white", fontsize=9, ha="center", va="center")

    ax.set_aspect("equal")
    ax.set_title(f"Selected interior pair: (T{k+1}, f{f}) ↔ (T{nbr+1}, f{nbr_f})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.2)


def _plot_trace_pair(ax, paired: dict, k: int, f: int, title: str) -> None:
    t = np.asarray(paired["face_t"][f], dtype=float)
    uM = np.asarray(paired["uM"][k, f - 1, :], dtype=float)
    uP = np.asarray(paired["uP"][k, f - 1, :], dtype=float)

    ax.plot(t, uM, "o-", label="uM (local trace)", linewidth=1.8)
    ax.plot(t, uP, "s--", label="uP (neighbor-aligned)", linewidth=1.8)

    diff = np.max(np.abs(uM - uP))
    ax.set_title(f"{title}\nmax |uM-uP| = {diff:.3e}")
    ax.set_xlabel("local face parameter t")
    ax.set_ylabel("trace value")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

def _plot_global_exchange_mismatch(ax, VX, VY, EToV, conn, paired, title: str) -> None:
    _draw_mesh(ax, VX, VY, EToV)

    mids = conn["face_midpoints"]
    items = interior_face_pair_mismatches(paired)

    # boundary face 先畫成灰色
    for k, f in conn["boundary_faces"]:
        xm, ym = mids[k, f - 1]
        ax.scatter([xm], [ym], s=28, color="lightgray", marker="x", zorder=2)

    # interior face mismatch
    errs = np.array([item["max_abs_mismatch"] for item in items], dtype=float)

    # 若全部都非常小，避免 color scale 壞掉
    vmin = min(np.min(errs), 1e-16)
    vmax = max(np.max(errs), 1e-16)

    # 如果誤差跨很多數量級，用 log10 顯示比較直觀
    log_errs = np.log10(np.maximum(errs, 1e-16))

    xs = []
    ys = []
    for item in items:
        k = item["k"]
        f = item["f"]
        xm, ym = mids[k, f - 1]
        xs.append(xm)
        ys.append(ym)

    sc = ax.scatter(
        xs,
        ys,
        c=log_errs,
        s=70,
        cmap="viridis",
        edgecolors="black",
        linewidths=0.4,
        zorder=3,
    )

    # 每條 interior face 旁邊標一下 pair id 與誤差
    for item in items:
        k = item["k"]
        f = item["f"]
        nbr = item["nbr"]
        nbr_f = item["nbr_f"]
        err = item["max_abs_mismatch"]

        xm, ym = mids[k, f - 1]
        ax.text(
            xm,
            ym,
            f"({k+1},f{f})\n↔({nbr+1},f{nbr_f})\n{err:.1e}",
            fontsize=6.8,
            ha="center",
            va="bottom",
            color="black",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75),
            zorder=4,
        )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(r"$\log_{10}(\max |u_M-u_P|)$")

    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.2)

def main() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "photo"
    output_dir.mkdir(parents=True, exist_ok=True)

    VX, VY, EToV = structured_square_tri_mesh(
        nx=2,
        ny=2,
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        diagonal="anti",
    )
    validate_mesh_orientation(VX, VY, EToV)
    conn = build_face_connectivity(VX, VY, EToV, classify_boundary="box")

    # We keep using the same representative pair:
    # (T2, f3) <-> (T3, f2) in 1-based human notation
    # internal 0-based element ids:
    k_sel = 1   # T2
    f_sel = 3

    # -----------------------------
    # Table 1
    # -----------------------------
    rule1 = load_table1_rule(4)
    trace1 = build_trace_policy(rule1)
    X1, Y1 = map_reference_nodes_to_all_elements(rule1["rs"], VX, VY, EToV)
    u_elem1 = _global_poly(X1, Y1)
    paired1 = pair_face_traces(u_elem1, conn, trace1)
    mis1 = interior_face_pair_mismatches(paired1)

    # -----------------------------
    # Table 2
    # -----------------------------
    rule2 = load_table2_rule(4)
    trace2 = build_trace_policy(rule2, N=4, n_edge=5)
    X2, Y2 = map_reference_nodes_to_all_elements(rule2["rs"], VX, VY, EToV)
    u_elem2 = _global_poly(X2, Y2)
    paired2 = pair_face_traces(u_elem2, conn, trace2)
    mis2 = interior_face_pair_mismatches(paired2)

    print("=== Step 3 exchange diagnostics ===")
    print("Table 1 pair mismatches:")
    for item in mis1:
        print(
            f"  (T{item['k']+1}, f{item['f']}) ↔ "
            f"(T{item['nbr']+1}, f{item['nbr_f']}) : "
            f"{item['max_abs_mismatch']:.3e}"
        )

    print("\nTable 2 pair mismatches:")
    for item in mis2:
        print(
            f"  (T{item['k']+1}, f{item['f']}) ↔ "
            f"(T{item['nbr']+1}, f{item['nbr_f']}) : "
            f"{item['max_abs_mismatch']:.3e}"
        )

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    _plot_global_exchange_mismatch(
        axs[0],
        VX, VY, EToV,
        conn,
        paired1,
        "Global exchange mismatch: Table 1",
    )

    _plot_global_exchange_mismatch(
        axs[1],
        VX, VY, EToV,
        conn,
        paired2,
        "Global exchange mismatch: Table 2",
    )

    fig.tight_layout()
    fig.savefig(output_dir / "demo_exchange_step3_global.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()