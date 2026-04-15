from __future__ import annotations

from operators import (
    upwind_flux_and_penalty,
    upwind_flux_and_penalty_exact_trace,
    upwind_flux_and_penalty_exchange,
    volume_term_split_conservative,
    volume_term_split_conservative_exact_trace,
    volume_term_split_conservative_exchange,
)


def test_volume_term_exports_are_explicit_and_compatible() -> None:
    assert volume_term_split_conservative is volume_term_split_conservative_exchange
    assert volume_term_split_conservative_exact_trace is not volume_term_split_conservative_exchange


def test_upwind_flux_exports_are_explicit_and_compatible() -> None:
    assert upwind_flux_and_penalty is upwind_flux_and_penalty_exchange
    assert upwind_flux_and_penalty_exact_trace is not upwind_flux_and_penalty_exchange
