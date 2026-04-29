from __future__ import annotations

import csv
import contextlib
import io
from pathlib import Path
from unittest.mock import patch

from experiments.manifold_lsrk_convergence import save_time_history_csv
from cli.run_manifold_lsrk_convergence import _field_slug
from visualization.manifold_diagnostics import (
    plot_manifold_l2_error_vs_time,
    plot_manifold_mass_error_vs_time,
)


def _make_result_with_history(*, mesh_level: int, field_case: str = "gaussian") -> dict:
    return {
        "n_div": int(mesh_level),
        "history": {
            "mesh_level": int(mesh_level),
            "h": 1.0 / float(mesh_level),
            "field_case": str(field_case),
            "flux_type": "upwind",
            "alpha_lf": 1.0,
            "initial_preset": "custom",
            "center_x": 1.0,
            "center_y": 0.0,
            "center_z": 0.0,
            "mass0": 2.0,
            "mass": [2.0, 2.0 + 1.0e-4 / mesh_level, 2.0 + 2.0e-4 / mesh_level],
            "mass_error": [0.0, 1.0e-4 / mesh_level, 2.0e-4 / mesh_level],
            "mass_rel_error": [0.0, 5.0e-5 / mesh_level, 1.0e-4 / mesh_level],
            "error_reference": "exact_gaussian" if field_case == "gaussian" else "constant_drift",
            "step_ids": [0, 1, 2],
            "times": [0.0, 0.1, 0.2],
            "l2": [0.0, 1.0e-3 / mesh_level, 5.0e-4 / mesh_level],
            "linf": [0.0, 2.0e-3 / mesh_level, 1.0e-3 / mesh_level],
            "max_abs_q": [1.0, 1.1, 1.2],
            "reached_tf": True,
            "tf_used": 0.2,
            "nsteps": 2,
        },
    }


def test_save_time_history_csv_writes_expected_rows() -> None:
    results = [_make_result_with_history(mesh_level=2), _make_result_with_history(mesh_level=4)]
    out_path = Path("ignored_history.csv")
    buffer = io.StringIO()

    with patch("pathlib.Path.open", return_value=contextlib.nullcontext(buffer)):
        save_time_history_csv(results, out_path)

    buffer.seek(0)
    rows = list(csv.DictReader(io.StringIO(buffer.getvalue())))

    assert len(rows) == 6
    assert rows[0]["mesh_level"] == "2"
    assert rows[0]["field_case"] == "gaussian"
    assert rows[0]["flux_type"] == "upwind"
    assert rows[0]["alpha_lf"] == "1.0"
    assert rows[0]["initial_preset"] == "custom"
    assert rows[0]["center_x"] == "1.0"
    assert rows[0]["mass0"] == "2.0"
    assert rows[0]["mass_error"] == "0.0"
    assert rows[0]["step_index"] == "0"
    assert rows[0]["time"] == "0.0"
    assert rows[-1]["mesh_level"] == "4"
    assert rows[-1]["step_index"] == "2"
    assert rows[-1]["tf_used"] == "0.2"


def test_plot_manifold_l2_error_vs_time_writes_png() -> None:
    histories = [
        _make_result_with_history(mesh_level=2)["history"],
        _make_result_with_history(mesh_level=4)["history"],
    ]
    out_path = Path("ignored_plot.png")

    with patch("matplotlib.figure.Figure.savefig") as mock_savefig:
        plot_manifold_l2_error_vs_time(histories, out_path)

    mock_savefig.assert_called_once()
    args, kwargs = mock_savefig.call_args
    assert args[0] == out_path
    assert kwargs["dpi"] == 220


def test_plot_manifold_mass_error_vs_time_writes_png() -> None:
    histories = [
        _make_result_with_history(mesh_level=2)["history"],
        _make_result_with_history(mesh_level=4)["history"],
    ]
    out_path = Path("ignored_mass_plot.png")

    with patch("matplotlib.figure.Figure.savefig") as mock_savefig:
        plot_manifold_mass_error_vs_time(histories, out_path)

    mock_savefig.assert_called_once()
    args, kwargs = mock_savefig.call_args
    assert args[0] == out_path
    assert kwargs["dpi"] == 220


def test_field_slug_includes_gaussian_center_location() -> None:
    assert _field_slug("gaussian", 1.0, "custom", (1.0, 0.0, 0.0), 1.0) == "gaussian_cx1_cy0_cz0"
    assert _field_slug("gaussian", 1.0, "north_pole", (1.0, 0.0, 0.0), 1.0) == "gaussian_north_pole"
    assert _field_slug("constant", 2.5, "custom", (1.0, 0.0, 0.0), 1.0) == "constant_v2p5"
