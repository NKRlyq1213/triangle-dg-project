from __future__ import annotations

import sys

from cli import run_lsrk_h_convergence


def test_run_lsrk_use_numba_flag_defaults_to_true(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_lsrk_h_convergence"])
    args = run_lsrk_h_convergence._parse_args()
    assert bool(args.use_numba) is True


def test_run_lsrk_use_numba_flag_can_disable_numba(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_lsrk_h_convergence", "--no-use-numba"])
    args = run_lsrk_h_convergence._parse_args()
    assert bool(args.use_numba) is False
