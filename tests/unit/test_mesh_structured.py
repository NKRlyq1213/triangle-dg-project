import numpy as np

from geometry.mesh_structured import (
    structured_square_tri_mesh,
    element_areas_and_orientations,
    validate_mesh_orientation,
)


def test_structured_square_tri_mesh_counts():
    VX, VY, EToV = structured_square_tri_mesh(nx=2, ny=2)

    assert VX.shape == (9,)
    assert VY.shape == (9,)
    assert EToV.shape == (8, 3)


def test_structured_square_tri_mesh_orientation():
    VX, VY, EToV = structured_square_tri_mesh(nx=2, ny=2)

    signed_areas, is_ccw = element_areas_and_orientations(VX, VY, EToV)

    assert np.all(is_ccw)
    assert np.all(signed_areas > 0.0)


def test_structured_square_tri_mesh_area_values():
    VX, VY, EToV = structured_square_tri_mesh(nx=2, ny=2)

    signed_areas, _ = element_areas_and_orientations(VX, VY, EToV)

    # [0,1]^2 area = 1, split into 8 equal triangles
    expected = 1.0 / 8.0
    assert np.allclose(signed_areas, expected)




def test_validate_mesh_orientation_passes():
    VX, VY, EToV = structured_square_tri_mesh(nx=3, ny=4)
    validate_mesh_orientation(VX, VY, EToV)