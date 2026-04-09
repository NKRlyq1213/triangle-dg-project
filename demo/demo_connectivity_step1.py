from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from geometry import (
    structured_square_tri_mesh,
    validate_mesh_orientation,
    build_face_connectivity,
    validate_face_connectivity,
)


def element_centroids(VX: np.ndarray, VY: np.ndarray, EToV: np.ndarray) -> np.ndarray:
    VX = np.asarray(VX, dtype=float).reshape(-1)
    VY = np.asarray(VY, dtype=float).reshape(-1)
    EToV = np.asarray(EToV, dtype=int)

    centers = np.zeros((EToV.shape[0], 2), dtype=float)
    for k, vids in enumerate(EToV):
        centers[k, 0] = np.mean(VX[vids])
        centers[k, 1] = np.mean(VY[vids])
    return centers


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


def _plot_element_ids(ax, VX, VY, EToV) -> None:
    _draw_mesh(ax, VX, VY, EToV)
    centers = element_centroids(VX, VY, EToV)
    for k, (xc, yc) in enumerate(centers):
        ax.text(xc, yc, f"T{k+1}", ha="center", va="center", fontsize=11, fontweight="bold")

    ax.set_aspect("equal")
    ax.set_title("Element IDs")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.2)


def _plot_face_ids_and_orientations(
    ax,
    VX,
    VY,
    EToV,
    conn,
    face_filter: int | None = None,
) -> None:
    _draw_mesh(ax, VX, VY, EToV)

    face_vids = conn["face_vertex_ids"]
    mids = conn["face_midpoints"]

    # 每個 local face 固定一個顏色
    face_colors = {
        1: "tab:blue",
        2: "tab:orange",
        3: "tab:green",
    }

    for k in range(EToV.shape[0]):
        for jf in range(3):
            f = jf + 1

            if face_filter is not None and f != face_filter:
                continue

            color = face_colors[f]

            va, vb = face_vids[k, jf]
            x0, y0 = VX[va], VY[va]
            x1, y1 = VX[vb], VY[vb]
            xm, ym = mids[k, jf]

            dx = x1 - x0
            dy = y1 - y0

            # 短箭頭，置中於 face midpoint
            scale = 0.28
            ax.arrow(
                xm - 0.5 * scale * dx,
                ym - 0.5 * scale * dy,
                scale * dx,
                scale * dy,
                head_width=0.02,
                head_length=0.03,
                length_includes_head=True,
                color=color,
                alpha=0.9,
            )

            ax.text(
                xm,
                ym,
                f"f{f}",
                color="white",
                fontsize=9,
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    fc=color,
                    ec="none",
                    alpha=0.9,
                ),
            )

    # 簡單圖例
    for f, color in face_colors.items():
        ax.plot([], [], color=color, linewidth=2, label=f"face {f}")

    ax.legend(loc="upper right", fontsize=8)

    ax.set_aspect("equal")
    if face_filter is None:
        ax.set_title("Local face IDs and orientations")
    else:
        ax.set_title(f"Local face orientation: only f{face_filter}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.2)


def _plot_pairs_and_boundary(ax, VX, VY, EToV, conn) -> None:
    _draw_mesh(ax, VX, VY, EToV)

    EToE = conn["EToE"]
    EToF = conn["EToF"]
    is_boundary = conn["is_boundary"]
    mids = conn["face_midpoints"]

    drawn_pairs = set()

    # boundary groups for coloring
    boundary_groups = conn["boundary_groups"]
    group_color = {
        "left": "tab:green",
        "right": "tab:orange",
        "bottom": "tab:purple",
        "top": "tab:brown",
        "boundary_default": "tab:gray",
    }

    # interior pairs
    for k in range(EToV.shape[0]):
        for jf in range(3):
            f = jf + 1
            if is_boundary[k, jf]:
                continue

            nbr = int(EToE[k, jf])
            nbr_f = int(EToF[k, jf])

            pair_key = tuple(sorted(((k, f), (nbr, nbr_f))))
            if pair_key in drawn_pairs:
                continue
            drawn_pairs.add(pair_key)

            xm, ym = mids[k, jf]
            ax.scatter([xm], [ym], s=40, color="tab:red", zorder=3)
            ax.text(
                xm,
                ym,
                f"({k+1},f{f})↔({nbr+1},f{nbr_f})",
                fontsize=7.5,
                ha="center",
                va="bottom",
                color="tab:red",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8),
            )

    # boundary faces
    for gname, items in boundary_groups.items():
        if gname not in group_color:
            continue
        color = group_color[gname]
        for k, f in items:
            xm, ym = mids[k, f - 1]
            ax.scatter([xm], [ym], s=55, color=color, zorder=3)
            ax.text(
                xm,
                ym,
                gname,
                fontsize=7,
                ha="center",
                va="top",
                color=color,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.75),
            )

    ax.set_aspect("equal")
    ax.set_title("Interior pairs and boundary classification")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.2)


def _expected_truth_2x2() -> dict[str, np.ndarray]:
    """
    Truth tables for nx=ny=2, diagonal='anti'.

    Element ids are 0-based internally:
        T1 -> 0, ..., T8 -> 7

    Face ids stored in EToF are 1-based:
        f1, f2, f3
    """
    expected_EToE = np.array(
        [
            [1, -1, -1],   # T1
            [4,  0,  2],   # T2
            [3,  1, -1],   # T3
            [6,  2, -1],   # T4
            [5, -1,  1],   # T5
            [-1, 4,  6],   # T6
            [7,  5,  3],   # T7
            [-1, 6, -1],   # T8
        ],
        dtype=int,
    )

    expected_EToF = np.array(
        [
            [2, -1, -1],   # T1
            [3,  1,  2],   # T2
            [2,  3, -1],   # T3
            [3,  1, -1],   # T4
            [2, -1,  1],   # T5
            [-1, 1,  2],   # T6
            [2,  3,  1],   # T7
            [-1, 1, -1],   # T8
        ],
        dtype=int,
    )

    expected_is_boundary = np.array(
        [
            [False, True,  True ],
            [False, False, False],
            [False, False, True ],
            [False, False, True ],
            [False, True,  False],
            [True,  False, False],
            [False, False, False],
            [True,  False, True ],
        ],
        dtype=bool,
    )

    expected_face_flip = np.array(
        [
            [True,  False, False],
            [True,  True,  True ],
            [True,  True,  False],
            [True,  True,  False],
            [True,  False, True ],
            [False, True,  True ],
            [True,  True,  True ],
            [False, True,  False],
        ],
        dtype=bool,
    )

    return {
        "EToE": expected_EToE,
        "EToF": expected_EToF,
        "is_boundary": expected_is_boundary,
        "face_flip": expected_face_flip,
    }


def main() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "photo"
    output_dir.mkdir(parents=True, exist_ok=True)

    VX, VY, EToV = structured_square_tri_mesh(
        nx=3,
        ny=3,
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        diagonal="anti",
    )
    validate_mesh_orientation(VX, VY, EToV)

    conn = build_face_connectivity(VX, VY, EToV, classify_boundary="box", tol=1e-12)
    summary = validate_face_connectivity(EToV, conn)

    expected = _expected_truth_2x2()

    ok_EToE = np.array_equal(conn["EToE"], expected["EToE"])
    ok_EToF = np.array_equal(conn["EToF"], expected["EToF"])
    ok_is_boundary = np.array_equal(conn["is_boundary"], expected["is_boundary"])
    ok_face_flip = np.array_equal(conn["face_flip"], expected["face_flip"])

    print("=== Connectivity summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\n=== Truth-table checks (2x2, anti diagonal) ===")
    print(f"EToE correct       : {ok_EToE}")
    print(f"EToF correct       : {ok_EToF}")
    print(f"is_boundary correct: {ok_is_boundary}")
    print(f"face_flip correct  : {ok_face_flip}")

    print("\nEToE =")
    print(conn["EToE"])
    print("\nEToF =")
    print(conn["EToF"])
    print("\nis_boundary =")
    print(conn["is_boundary"].astype(int))
    print("\nface_flip =")
    print(conn["face_flip"].astype(int))

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    _plot_element_ids(axs[0], VX, VY, EToV)
    _plot_face_ids_and_orientations(axs[1], VX, VY, EToV, conn)
    _plot_pairs_and_boundary(axs[2], VX, VY, EToV, conn)
    fig.tight_layout()
    fig.savefig(output_dir / "demo_connectivity_step1_2x2.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()