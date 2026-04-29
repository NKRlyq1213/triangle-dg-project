from __future__ import annotations

import numpy as np
import pytest

from geometry.connectivity import validate_face_connectivity
from geometry.sphere_manifold_mesh import generate_spherical_octahedron_mesh
from geometry.sphere_manifold_metrics import build_manifold_geometry_cache
from operators.manifold_rhs import (
    build_manifold_exchange_cache,
    build_manifold_face_connectivity,
    build_manifold_table1_k4_reference_operators,
    build_manifold_vmaps,
    manifold_rhs_exchange,
    manifold_surface_term,
    manifold_surface_term_from_exchange,
    pair_manifold_face_traces,
)
from problems.sphere_advection import gaussian_bell_xyz, solid_body_velocity_xyz
from experiments.manifold_lsrk_convergence import (
    ManifoldLSRKConvergenceConfig,
    run_manifold_lsrk_convergence,
)


def _fixture(n_div: int = 2):
    ref_ops = build_manifold_table1_k4_reference_operators()
    nodes_xyz, EToV = generate_spherical_octahedron_mesh(n_div=n_div, R=1.0)
    geom = build_manifold_geometry_cache(
        nodes_xyz=nodes_xyz,
        EToV=EToV,
        rs_nodes=ref_ops.rs_nodes,
        Dr=ref_ops.Dr,
        Ds=ref_ops.Ds,
        R=1.0,
    )
    U, V, W = solid_body_velocity_xyz(geom.X, geom.Y, geom.Z)
    cache = build_manifold_exchange_cache(
        EToV=EToV,
        ref_ops=ref_ops,
        geom=geom,
        U=U,
        V=V,
        W=W,
        use_numba=False,
    )
    return ref_ops, nodes_xyz, EToV, geom, U, V, W, cache


def test_manifold_exchange_is_closed_and_pairs_neighbor_traces():
    ref_ops, _, EToV, geom, *_rest, cache = _fixture()

    summary = validate_face_connectivity(EToV, cache.conn)
    assert summary["n_boundary_faces"] == 0

    q = np.arange(geom.X.size, dtype=float).reshape(geom.X.shape)
    paired = pair_manifold_face_traces(q, cache.conn, cache.trace, use_numba=False)

    vmapM, vmapP, _ = build_manifold_vmaps(
        EToV,
        ref_ops.face_node_ids,
        Np=geom.X.shape[1],
    )
    qM_vmap = q.reshape(-1)[vmapM]
    qP_vmap = q.reshape(-1)[vmapP]
    qM_exchange = paired["uM"].transpose(1, 2, 0).reshape(qM_vmap.shape)
    qP_exchange = paired["uP"].transpose(1, 2, 0).reshape(qP_vmap.shape)

    assert np.array_equal(qM_exchange, qM_vmap)
    assert np.array_equal(qP_exchange, qP_vmap)


def test_manifold_face_extraction_is_one_hot():
    ref_ops = build_manifold_table1_k4_reference_operators()

    assert np.all(np.sum(ref_ops.face_extraction, axis=1) == 1.0)
    for face_id in (1, 2, 3):
        rows = np.arange((face_id - 1) * 5, face_id * 5)
        ids = ref_ops.face_node_ids[face_id]
        assert np.array_equal(np.argmax(ref_ops.face_extraction[rows], axis=1), ids)


def test_exchange_surface_term_matches_vmap_reference_for_random_q():
    ref_ops, _, EToV, geom, U, V, W, cache = _fixture()
    q = np.random.default_rng(2026).normal(size=geom.X.shape)

    vmapM, vmapP, _ = build_manifold_vmaps(
        EToV,
        ref_ops.face_node_ids,
        Np=geom.X.shape[1],
    )
    surface_vmap = manifold_surface_term(q, geom, U, V, W, ref_ops, vmapM, vmapP)
    surface_exchange = manifold_surface_term_from_exchange(
        q,
        geom,
        (U, V, W),
        ref_ops,
        cache,
        use_numba=False,
    )

    assert np.allclose(surface_exchange, surface_vmap, atol=0.0, rtol=0.0)


def test_constant_rhs_exchange_matches_free_stream_diagnostic():
    ref_ops, _, _, geom, U, V, W, cache = _fixture()
    q = np.ones_like(geom.X)

    rhs = manifold_rhs_exchange(
        q,
        geom,
        (U, V, W),
        cache,
        ref_ops=ref_ops,
        use_numba=False,
    )
    surface = manifold_surface_term_from_exchange(
        q,
        geom,
        (U, V, W),
        ref_ops,
        cache,
        use_numba=False,
    )

    assert np.max(np.abs(surface)) < 1.0e-14
    assert np.all(np.isfinite(rhs))


def test_short_manifold_lsrk_errors_decrease_under_refinement():
    config = ManifoldLSRKConvergenceConfig(
        mesh_levels=(2, 4, 8),
        tf=0.03,
        cfl=0.1,
        field_case="gaussian",
        use_numba=False,
        verbose=False,
    )
    rows = run_manifold_lsrk_convergence(config)
    gaussian_l2 = [row["L2_error"] for row in rows]
    gaussian_linf = [row["Linf_error"] for row in rows]

    assert all(np.isfinite(v) for v in gaussian_l2)
    assert all(np.isfinite(v) for v in gaussian_linf)
    assert gaussian_l2[1] < gaussian_l2[0]
    assert gaussian_l2[2] < gaussian_l2[1]
    assert gaussian_linf[1] < gaussian_linf[0]
    assert gaussian_linf[2] < gaussian_linf[1]


def test_gaussian_initial_field_changes_with_center_xyz():
    X = np.array([1.0, 0.0, 0.0])
    Y = np.array([0.0, 1.0, 0.0])
    Z = np.array([0.0, 0.0, 1.0])

    q_x = gaussian_bell_xyz(X, Y, Z, R=1.0, center_xyz=(1.0, 0.0, 0.0))
    q_y = gaussian_bell_xyz(X, Y, Z, R=1.0, center_xyz=(0.0, 1.0, 0.0))

    assert not np.allclose(q_x, q_y)
    assert q_x[0] > q_y[0]
    assert q_y[1] > q_x[1]


def test_initial_preset_updates_resolved_center():
    config = ManifoldLSRKConvergenceConfig(
        mesh_levels=(2,),
        tf=0.01,
        cfl=0.1,
        field_case="gaussian",
        center_xyz=(1.0, 0.0, 0.0),
        initial_preset="north_pole",
        use_numba=False,
        verbose=False,
    )
    row = run_manifold_lsrk_convergence(config)[0]

    assert row["initial_preset"] == "north_pole"
    assert row["center_x"] == pytest.approx(0.0)
    assert row["center_y"] == pytest.approx(0.0)
    assert row["center_z"] == pytest.approx(1.0)


def test_constant_field_drift_decreases_under_refinement():
    config = ManifoldLSRKConvergenceConfig(
        mesh_levels=(2, 4, 8),
        tf=0.03,
        cfl=0.1,
        field_case="constant",
        constant_value=1.0,
        use_numba=False,
        verbose=False,
    )
    rows = run_manifold_lsrk_convergence(config)
    drift_l2 = [row["L2_error"] for row in rows]
    drift_linf = [row["Linf_error"] for row in rows]

    assert all(np.isfinite(v) for v in drift_l2)
    assert all(np.isfinite(v) for v in drift_linf)
    assert drift_l2[1] < drift_l2[0]
    assert drift_l2[2] < drift_l2[1]
    assert drift_linf[1] < drift_linf[0]
    assert drift_linf[2] < drift_linf[1]
    assert all(abs(float(row["const_L2_drift"]) - float(row["L2_error"])) < 1.0e-15 for row in rows)
    assert all(abs(float(row["const_Linf_drift"]) - float(row["Linf_error"])) < 1.0e-15 for row in rows)


def test_lax_friedrichs_uses_smaller_dt_than_upwind():
    upwind_config = ManifoldLSRKConvergenceConfig(
        mesh_levels=(2,),
        tf=0.01,
        cfl=0.1,
        field_case="gaussian",
        flux_type="upwind",
        alpha_lf=1.0,
        use_numba=False,
        verbose=False,
    )
    lf_config = ManifoldLSRKConvergenceConfig(
        mesh_levels=(2,),
        tf=0.01,
        cfl=0.1,
        field_case="gaussian",
        flux_type="lax_friedrichs",
        alpha_lf=1.5,
        use_numba=False,
        verbose=False,
    )

    upwind_row = run_manifold_lsrk_convergence(upwind_config)[0]
    lf_row = run_manifold_lsrk_convergence(lf_config)[0]

    assert upwind_row["flux_type"] == "upwind"
    assert lf_row["flux_type"] == "lax_friedrichs"
    assert lf_row["dt"] < upwind_row["dt"]
    assert np.isfinite(lf_row["L2_error"])
    assert np.isfinite(lf_row["Linf_error"])


def test_recorded_history_is_monotone_and_matches_final_time():
    config = ManifoldLSRKConvergenceConfig(
        mesh_levels=(2,),
        tf=0.03,
        cfl=0.1,
        field_case="gaussian",
        record_history=True,
        use_numba=False,
        verbose=False,
    )
    row = run_manifold_lsrk_convergence(config)[0]
    history = row["history"]

    times = np.asarray(history["times"], dtype=float)
    l2 = np.asarray(history["l2"], dtype=float)
    linf = np.asarray(history["linf"], dtype=float)
    mass = np.asarray(history["mass"], dtype=float)
    mass_error = np.asarray(history["mass_error"], dtype=float)
    mass_rel_error = np.asarray(history["mass_rel_error"], dtype=float)

    assert times.ndim == 1
    assert np.all(np.diff(times) >= 0.0)
    assert times[0] == pytest.approx(0.0)
    assert l2[0] == pytest.approx(0.0)
    assert linf[0] == pytest.approx(0.0)
    assert mass_error[0] == pytest.approx(0.0)
    assert np.all(np.isfinite(l2))
    assert np.all(np.isfinite(linf))
    assert np.all(np.isfinite(mass))
    assert np.all(np.isfinite(mass_error))
    assert np.all(np.isfinite(mass_rel_error) | np.isnan(mass_rel_error))
    assert np.array_equal(np.asarray(history["step_ids"], dtype=int), np.arange(times.size, dtype=int))
    assert times[-1] == pytest.approx(float(row["tf"]))
    assert float(history["tf_used"]) == pytest.approx(float(row["tf"]))
    assert float(history["mass0"]) == pytest.approx(float(mass[0]))
    assert float(history["mass_error"][-1]) == pytest.approx(float(row["mass_error"]))
    assert bool(history["reached_tf"]) is True


def test_snapshot_capture_records_requested_times():
    config = ManifoldLSRKConvergenceConfig(
        mesh_levels=(2,),
        tf=0.03,
        cfl=0.1,
        field_case="constant",
        record_history=True,
        snapshot_times=(0.0, 0.02, 0.03),
        use_numba=False,
        verbose=False,
    )
    row = run_manifold_lsrk_convergence(config)[0]
    snapshots = row["snapshots"]

    assert len(snapshots) == 3
    assert snapshots[0]["time_requested"] == pytest.approx(0.0)
    assert snapshots[0]["time_actual"] == pytest.approx(0.0)
    assert snapshots[-1]["time_requested"] == pytest.approx(0.03)
    assert snapshots[-1]["time_actual"] == pytest.approx(float(row["tf"]))
    assert all(np.asarray(s["q"]).shape == row["artifacts"]["geom"].X.shape for s in snapshots)
    assert all(np.asarray(s["error"]).shape == row["artifacts"]["geom"].X.shape for s in snapshots)


def test_step_snapshots_store_every_accepted_step():
    config = ManifoldLSRKConvergenceConfig(
        mesh_levels=(2,),
        tf=0.03,
        cfl=0.1,
        field_case="gaussian",
        record_history=True,
        record_step_snapshots=True,
        use_numba=False,
        verbose=False,
    )
    row = run_manifold_lsrk_convergence(config)[0]
    history = row["history"]
    step_snapshots = row["step_snapshots"]

    assert len(step_snapshots) == len(history["times"])
    assert [int(s["step_index"]) for s in step_snapshots] == list(range(len(step_snapshots)))
    assert step_snapshots[0]["time_actual"] == pytest.approx(0.0)
    assert step_snapshots[-1]["time_actual"] == pytest.approx(float(row["tf"]))
    assert all(np.asarray(s["q"]).shape == row["artifacts"]["geom"].X.shape for s in step_snapshots)
    assert all(np.asarray(s["q_ref"]).shape == row["artifacts"]["geom"].X.shape for s in step_snapshots)
