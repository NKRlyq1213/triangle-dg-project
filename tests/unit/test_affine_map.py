import numpy as np

from geometry.reference_triangle import reference_triangle_vertices
from geometry.mesh_structured import structured_square_tri_mesh
from geometry.affine_map import (
    reference_shape_functions,
    map_ref_to_phys,
    map_ref_to_phys_points,
    element_vertices,
    map_reference_nodes_to_element,
    map_reference_nodes_to_all_elements,
)


def test_reference_shape_functions_sum_to_one():
    rs = np.array([
        [-1.0, -1.0],
        [ 1.0, -1.0],
        [-1.0,  1.0],
        [-0.2, -0.3],
        [ 0.0, -0.5],
    ])
    phi = reference_shape_functions(rs[:, 0], rs[:, 1])
    assert np.allclose(np.sum(phi, axis=1), 1.0)


def test_reference_vertices_map_to_physical_vertices():
    verts = np.array([
        [0.2, 0.1],
        [1.1, 0.0],
        [0.3, 0.9],
    ], dtype=float)

    ref_verts = reference_triangle_vertices()
    xy = map_ref_to_phys_points(ref_verts, verts)

    assert np.allclose(xy, verts)


def test_element_vertices_extraction():
    VX, VY, EToV = structured_square_tri_mesh(nx=2, ny=2)
    verts = element_vertices(VX, VY, EToV, elem_id=0)

    expected = np.array([
        [0.0, 0.0],
        [0.5, 0.0],
        [0.0, 0.5],
    ])
    assert np.allclose(verts, expected)


def test_map_reference_nodes_to_element_shape():
    VX, VY, EToV = structured_square_tri_mesh(nx=2, ny=2)
    rs_nodes = np.array([
        [-1.0, -1.0],
        [ 1.0, -1.0],
        [-1.0,  1.0],
        [ 0.0, -0.5],
    ])
    xy = map_reference_nodes_to_element(rs_nodes, VX, VY, EToV, elem_id=0)

    assert xy.shape == (4, 2)


def test_map_reference_nodes_to_all_elements_shape():
    VX, VY, EToV = structured_square_tri_mesh(nx=2, ny=2)
    rs_nodes = np.array([
        [-1.0, -1.0],
        [ 1.0, -1.0],
        [-1.0,  1.0],
    ])

    X, Y = map_reference_nodes_to_all_elements(rs_nodes, VX, VY, EToV)

    assert X.shape == (8, 3)
    assert Y.shape == (8, 3)

if __name__ == "__main__":
    test_reference_shape_functions_sum_to_one()
    test_reference_vertices_map_to_physical_vertices()
    test_element_vertices_extraction()
    test_map_reference_nodes_to_element_shape()
    test_map_reference_nodes_to_all_elements_shape()