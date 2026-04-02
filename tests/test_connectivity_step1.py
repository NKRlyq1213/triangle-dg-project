from __future__ import annotations

import numpy as np

from geometry import (
    structured_square_tri_mesh,
    validate_mesh_orientation,
    build_face_connectivity,
    validate_face_connectivity,
)


def test_connectivity_step1_truth_tables_2x2_anti() -> None:
    VX, VY, EToV = structured_square_tri_mesh(
        nx=2,
        ny=2,
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        diagonal="anti",
    )
    validate_mesh_orientation(VX, VY, EToV)

    conn = build_face_connectivity(VX, VY, EToV, classify_boundary="box", tol=1e-12)
    summary = validate_face_connectivity(EToV, conn)

    expected_EToE = np.array(
        [
            [1, -1, -1],
            [4,  0,  2],
            [3,  1, -1],
            [6,  2, -1],
            [5, -1,  1],
            [-1, 4,  6],
            [7,  5,  3],
            [-1, 6, -1],
        ],
        dtype=int,
    )

    expected_EToF = np.array(
        [
            [2, -1, -1],
            [3,  1,  2],
            [2,  3, -1],
            [3,  1, -1],
            [2, -1,  1],
            [-1, 1,  2],
            [2,  3,  1],
            [-1, 1, -1],
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

    assert np.array_equal(conn["EToE"], expected_EToE)
    assert np.array_equal(conn["EToF"], expected_EToF)
    assert np.array_equal(conn["is_boundary"], expected_is_boundary)
    assert np.array_equal(conn["face_flip"], expected_face_flip)

    assert summary["n_elements"] == 8
    assert summary["n_total_local_faces"] == 24
    assert summary["n_boundary_faces"] == 8
    assert summary["n_unique_interior_faces"] == 8