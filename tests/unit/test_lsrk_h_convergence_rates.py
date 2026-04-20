from __future__ import annotations

import math

import numpy as np
import pytest

from experiments.lsrk_h_convergence import compute_convergence_rate


def test_compute_convergence_rate_matches_log2_when_h_halves() -> None:
    errors = [1.0, 1.0 / 16.0, 1.0 / 256.0]
    h_values = [1.0, 0.5, 0.25]

    rates = compute_convergence_rate(errors, h_values)

    assert math.isnan(rates[0])
    assert rates[1] == pytest.approx(4.0)
    assert rates[2] == pytest.approx(4.0)


def test_compute_convergence_rate_uses_general_h_ratio() -> None:
    errors = [1.0, 1.0 / 9.0]
    h_values = [1.0, 1.0 / 3.0]

    rates = compute_convergence_rate(errors, h_values)
    expected = math.log(errors[0] / errors[1]) / math.log(h_values[0] / h_values[1])

    assert math.isnan(rates[0])
    assert rates[1] == pytest.approx(expected)


def test_compute_convergence_rate_returns_nan_on_invalid_inputs() -> None:
    errors = [1.0, np.nan, -1.0, 1.0, 1.0, 1.0]
    h_values = [1.0, 0.5, 0.25, 0.0, 0.125, 0.125]

    rates = compute_convergence_rate(errors, h_values)

    assert math.isnan(rates[0])
    assert math.isnan(rates[1])  # e_curr is nan
    assert math.isnan(rates[2])  # e_curr <= 0
    assert math.isnan(rates[3])  # h_curr <= 0
    assert math.isnan(rates[4])  # h_prev <= 0
    assert math.isnan(rates[5])  # h_prev == h_curr


def test_compute_convergence_rate_requires_same_length() -> None:
    with pytest.raises(ValueError, match="same length"):
        compute_convergence_rate([1.0, 0.5], [1.0])
