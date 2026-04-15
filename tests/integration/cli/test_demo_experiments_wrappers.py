from __future__ import annotations

import importlib
import runpy
import sys

import pytest


WRAPPER_PAIRS = (
    ("demo.experiments.run_field_h_convergence", "cli.run_field_h_convergence"),
    ("demo.experiments.run_div_h_convergence", "cli.run_div_h_convergence"),
    ("demo.experiments.run_rhs_exchange_benchmark", "cli.run_rhs_exchange_benchmark"),
    ("demo.experiments.run_lsrk_h_convergence", "cli.run_lsrk_h_convergence"),
    ("demo.experiments.plot_lsrk_error_vs_time", "cli.plot_lsrk_error_vs_time"),
)


@pytest.mark.parametrize(("wrapper_module", "cli_module"), WRAPPER_PAIRS)
def test_wrapper_exports_same_main_symbol(wrapper_module: str, cli_module: str) -> None:
    wrapper = importlib.import_module(wrapper_module)
    cli = importlib.import_module(cli_module)
    assert wrapper.main is cli.main


@pytest.mark.parametrize(("wrapper_module", "cli_module"), WRAPPER_PAIRS)
def test_wrapper_main_exec_delegates_to_cli(wrapper_module: str, cli_module: str, monkeypatch) -> None:
    cli = importlib.import_module(cli_module)
    calls = {"count": 0}

    def fake_main() -> None:
        calls["count"] += 1

    monkeypatch.setattr(cli, "main", fake_main)
    sys.modules.pop(wrapper_module, None)
    runpy.run_module(wrapper_module, run_name="__main__")

    assert calls["count"] == 1
