from __future__ import annotations

import math
import sys

import pytest

from cli import run_lsrk_h_convergence as cli


def _make_row(
    *,
    nx: int,
    h: float,
    reached_tf: bool,
    l2_error: float,
    linf_error: float,
    l2_error_at_stop: float,
    linf_error_at_stop: float,
    elapsed_sec: float,
) -> dict:
    return {
        "nx": int(nx),
        "h": float(h),
        "reached_tf": bool(reached_tf),
        "L2_error": float(l2_error),
        "Linf_error": float(linf_error),
        "L2_error_at_stop": float(l2_error_at_stop),
        "Linf_error_at_stop": float(linf_error_at_stop),
        "elapsed_sec": float(elapsed_sec),
    }


def test_parse_time_cli_defaults_to_false(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_lsrk_h_convergence"])
    args = cli._parse_args()
    assert bool(args.time_cli) is False


def test_parse_time_cli_flag_can_enable(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_lsrk_h_convergence", "--time-cli"])
    args = cli._parse_args()
    assert bool(args.time_cli) is True


def test_build_time_scan_summary_rows_ok_computes_rates() -> None:
    tf_map = {
        0.2: [
            _make_row(
                nx=4,
                h=0.25,
                reached_tf=True,
                l2_error=(0.25**4),
                linf_error=(0.25**4) * 2.0,
                l2_error_at_stop=(0.25**4),
                linf_error_at_stop=(0.25**4) * 2.0,
                elapsed_sec=0.3,
            ),
            _make_row(
                nx=8,
                h=0.125,
                reached_tf=True,
                l2_error=(0.125**4),
                linf_error=(0.125**4) * 2.0,
                l2_error_at_stop=(0.125**4),
                linf_error_at_stop=(0.125**4) * 2.0,
                elapsed_sec=0.6,
            ),
        ]
    }

    rows = cli._build_time_scan_summary_rows(tf_map)

    assert len(rows) == 1
    row = rows[0]
    assert row["rate_status"] == "ok"
    assert row["reached_tf_all"] is True
    assert row["finest_n"] == 8
    assert row["total_elapsed_sec"] == pytest.approx(0.9)
    assert row["p_L2_last"] == pytest.approx(4.0)
    assert row["p_Linf_last"] == pytest.approx(4.0)


def test_build_time_scan_summary_rows_strict_final_time_gating() -> None:
    tf_map = {
        0.2: [
            _make_row(
                nx=4,
                h=0.25,
                reached_tf=True,
                l2_error=1.0e-3,
                linf_error=2.0e-3,
                l2_error_at_stop=1.0e-3,
                linf_error_at_stop=2.0e-3,
                elapsed_sec=0.2,
            ),
            _make_row(
                nx=8,
                h=0.125,
                reached_tf=False,
                l2_error=math.nan,
                linf_error=math.nan,
                l2_error_at_stop=8.0e-5,
                linf_error_at_stop=1.6e-4,
                elapsed_sec=0.4,
            ),
        ]
    }

    rows = cli._build_time_scan_summary_rows(tf_map)

    assert len(rows) == 1
    row = rows[0]
    assert row["rate_status"] == "strict_final_time_not_met"
    assert row["reached_tf_all"] is False
    assert math.isnan(float(row["p_L2_last"]))
    assert math.isnan(float(row["p_Linf_last"]))
    assert row["L2_error_finest"] == pytest.approx(8.0e-5)
    assert row["Linf_error_finest"] == pytest.approx(1.6e-4)


def test_flatten_results_by_tf_adds_tf_scan_and_sorts() -> None:
    tf_map = {
        1.0: [
            _make_row(
                nx=8,
                h=0.125,
                reached_tf=True,
                l2_error=1.0e-4,
                linf_error=2.0e-4,
                l2_error_at_stop=1.0e-4,
                linf_error_at_stop=2.0e-4,
                elapsed_sec=1.0,
            )
        ],
        0.5: [
            _make_row(
                nx=4,
                h=0.25,
                reached_tf=True,
                l2_error=1.0e-3,
                linf_error=2.0e-3,
                l2_error_at_stop=1.0e-3,
                linf_error_at_stop=2.0e-3,
                elapsed_sec=0.5,
            )
        ],
    }

    rows = cli._flatten_results_by_tf(tf_map)

    assert len(rows) == 2
    assert float(rows[0]["tf_scan"]) == pytest.approx(0.5)
    assert int(rows[0]["nx"]) == 4
    assert float(rows[1]["tf_scan"]) == pytest.approx(1.0)
    assert int(rows[1]["nx"]) == 8
