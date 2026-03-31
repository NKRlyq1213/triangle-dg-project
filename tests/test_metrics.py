import numpy as np

from data.table1_rules import load_table1_rule
from data.table2_rules import load_table2_rule
from geometry.mesh_structured import structured_square_tri_mesh
from geometry.affine_map import map_reference_nodes_to_all_elements
from geometry.metrics import affine_geometric_factors_from_mesh


def _run_metric_check(rule: dict):
    rs = rule["rs"]

    VX, VY, EToV = structured_square_tri_mesh(nx=2, ny=2, diagonal="anti")
    X, Y = map_reference_nodes_to_all_elements(rs, VX, VY, EToV)
    g = affine_geometric_factors_from_mesh(VX, VY, EToV, rs)

    K = EToV.shape[0]
    Np = rs.shape[0]

    assert X.shape == (K, Np)
    assert Y.shape == (K, Np)

    for key in ["xr", "xs", "yr", "ys", "J", "rx", "sx", "ry", "sy"]:
        assert g[key].shape == (K, Np)

    assert np.all(g["J"] > 0.0)

    # [0,1]^2 cut into 8 triangles => each triangle area = 1/8
    # reference triangle area = 2
    # affine area relation: area_phys = J * area_ref
    # hence J = (1/8)/2 = 1/16
    assert np.allclose(g["J"], 1.0 / 16.0)

    # For element 0 under diagonal="anti":
    # vertices = (0,0), (0.5,0), (0.0 ,0.5)
    # exact affine metrics:
    # xr = 0.25, xs = 0.25, yr = 0.0, ys = 0.25
    assert np.allclose(g["xr"][0], 0.25)
    assert np.allclose(g["xs"][0], 0.0)
    assert np.allclose(g["yr"][0], 0.0)
    assert np.allclose(g["ys"][0], 0.25)
    assert np.allclose(g["rx"][0], 4.0)
    assert np.allclose(g["sx"][0], 0.0)
    assert np.allclose(g["ry"][0], 0.0)
    assert np.allclose(g["sy"][0], 4.0)


def test_metrics_table1_order4():
    rule = load_table1_rule(4)
    _run_metric_check(rule)


def test_metrics_table2_order4():
    rule = load_table2_rule(4)
    _run_metric_check(rule)


if __name__ == "__main__":
    test_metrics_table1_order4()
    test_metrics_table2_order4()
    print("test_metrics: all checks passed.")