from __future__ import annotations

from .analytic_fields import ground_truth_function


def initial_condition(case_name: str, x, y, **params):
    """
    Wrapper for initial data / analytic test fields.
    """
    return ground_truth_function(case_name, x, y, **params)
