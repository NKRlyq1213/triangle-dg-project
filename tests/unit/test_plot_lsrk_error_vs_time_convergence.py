from __future__ import annotations

import csv
import math

import pytest

from cli import plot_lsrk_error_vs_time as plot_cli


def _make_curve(
    *,
    mesh_level: int,
    reached_tf: bool,
    l2_final: float,
    linf_final: float,
    tf_used: float = 1.0,
    tau_interior: float = 0.0,
    tau_qb: float = 0.0,
) -> dict:
    return {
        "mesh_level": int(mesh_level),
        "reached_tf": bool(reached_tf),
        "tf_used": float(tf_used),
        "tau_interior": float(tau_interior),
        "tau_qb": float(tau_qb),
        "l2": [float(l2_final)],
        "linf": [float(linf_final)],
    }


def test_build_convergence_summary_ok_sorts_by_mesh_and_computes_rates() -> None:
    # Feed unsorted levels to verify the summary always reports coarse-to-fine order.
    curves = [
        _make_curve(mesh_level=16, reached_tf=True, l2_final=(1.0 / 16.0) ** 2, linf_final=2.0 * (1.0 / 16.0) ** 2),
        _make_curve(mesh_level=4, reached_tf=True, l2_final=(1.0 / 4.0) ** 2, linf_final=2.0 * (1.0 / 4.0) ** 2),
        _make_curve(mesh_level=8, reached_tf=True, l2_final=(1.0 / 8.0) ** 2, linf_final=2.0 * (1.0 / 8.0) ** 2),
    ]

    rows, status = plot_cli._build_convergence_summary(curves)

    assert status == "ok"
    assert [int(r["mesh_level"]) for r in rows] == [4, 8, 16]
    assert [float(r["h"]) for r in rows] == pytest.approx([0.25, 0.125, 0.0625])

    assert math.isnan(float(rows[0]["rate_L2"]))
    assert float(rows[1]["rate_L2"]) == pytest.approx(2.0)
    assert float(rows[2]["rate_L2"]) == pytest.approx(2.0)

    assert math.isnan(float(rows[0]["rate_Linf"]))
    assert float(rows[1]["rate_Linf"]) == pytest.approx(2.0)
    assert float(rows[2]["rate_Linf"]) == pytest.approx(2.0)


def test_build_convergence_summary_returns_unavailable_when_any_mesh_stops_early() -> None:
    curves = [
        _make_curve(mesh_level=4, reached_tf=True, l2_final=1.0e-2, linf_final=2.0e-2),
        _make_curve(mesh_level=8, reached_tf=False, l2_final=2.0e-3, linf_final=4.0e-3),
    ]

    rows, status = plot_cli._build_convergence_summary(curves)

    assert status == "strict_final_time_not_met"
    assert all(math.isnan(float(r["rate_L2"])) for r in rows)
    assert all(math.isnan(float(r["rate_Linf"])) for r in rows)


def test_build_convergence_summary_requires_at_least_two_mesh_levels() -> None:
    rows, status = plot_cli._build_convergence_summary(
        [_make_curve(mesh_level=8, reached_tf=True, l2_final=1.0e-3, linf_final=2.0e-3)]
    )

    assert status == "insufficient_mesh_levels"
    assert len(rows) == 1
    assert math.isnan(float(rows[0]["rate_L2"]))
    assert math.isnan(float(rows[0]["rate_Linf"]))


def test_build_convergence_annotation_text_includes_last_and_average_rates() -> None:
    curves = [
        _make_curve(mesh_level=4, reached_tf=True, l2_final=(1.0 / 4.0) ** 2, linf_final=2.0 * (1.0 / 4.0) ** 2),
        _make_curve(mesh_level=8, reached_tf=True, l2_final=(1.0 / 8.0) ** 2, linf_final=2.0 * (1.0 / 8.0) ** 2),
        _make_curve(mesh_level=16, reached_tf=True, l2_final=(1.0 / 16.0) ** 2, linf_final=2.0 * (1.0 / 16.0) ** 2),
    ]
    rows, status = plot_cli._build_convergence_summary(curves)

    text = plot_cli._build_convergence_annotation_text(rows, rate_status=status)

    assert "p_L2(last)=2.000" in text
    assert "p_L2(avg)=2.000" in text
    assert "p_Linf(last)=2.000" in text
    assert "p_Linf(avg)=2.000" in text


def test_build_convergence_annotation_text_reports_unavailable_status() -> None:
    rows = [
        {
            "mesh_level": 4,
            "h": 0.25,
            "reached_tf": False,
            "tf_used": 0.8,
            "tau_interior": 0.0,
            "tau_qb": 0.0,
            "L2_error_final": 1.0,
            "rate_L2": float("nan"),
            "Linf_error_final": 1.0,
            "rate_Linf": float("nan"),
        }
    ]

    text = plot_cli._build_convergence_annotation_text(rows, rate_status="strict_final_time_not_met")

    assert "unavailable" in text
    assert "strict_final_time_not_met" in text


def test_save_csv_convergence_summary_writes_rate_status(tmp_path) -> None:
    rows = [
        {
            "mesh_level": 4,
            "h": 0.25,
            "reached_tf": True,
            "tf_used": 1.0,
            "tau_interior": 0.0,
            "tau_qb": 0.0,
            "L2_error_final": 1.0e-2,
            "rate_L2": float("nan"),
            "Linf_error_final": 2.0e-2,
            "rate_Linf": float("nan"),
        }
    ]
    out_path = tmp_path / "convergence_summary.csv"

    plot_cli._save_csv_convergence_summary(out_path, rows, rate_status="insufficient_mesh_levels")

    with out_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        loaded = list(reader)

    assert len(loaded) == 1
    assert loaded[0]["mesh_level"] == "4"
    assert loaded[0]["h"] == "0.25"
    assert loaded[0]["reached_tf"] == "True"
    assert loaded[0]["rate_status"] == "insufficient_mesh_levels"
