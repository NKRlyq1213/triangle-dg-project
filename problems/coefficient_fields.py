from __future__ import annotations

import numpy as np


def coefficient_field_with_derivatives(
    case_name: str,
    x,
    y,
    **params,
):
    """
    Coefficient fields (a, b) and their spatial derivatives.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (a, b, ax, ay, bx, by)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    case = case_name.lower().strip()

    if case == "constant_one":
        a = np.ones_like(x)
        b = np.ones_like(x)
        ax = np.zeros_like(x)
        ay = np.zeros_like(x)
        bx = np.zeros_like(x)
        by = np.zeros_like(x)
        return a, b, ax, ay, bx, by

    if case == "linear":
        a = 1.0 + 0.3 * x - 0.2 * y
        b = -0.4 + 0.1 * x + 0.25 * y
        ax = np.full_like(x, 0.3)
        ay = np.full_like(x, -0.2)
        bx = np.full_like(x, 0.1)
        by = np.full_like(x, 0.25)
        return a, b, ax, ay, bx, by

    if case == "trig":
        a = 1.0 + 0.2 * np.sin(np.pi * x) * np.cos(np.pi * y)
        b = 0.8 + 0.15 * np.cos(np.pi * x) * np.sin(np.pi * y)

        ax = 0.2 * np.pi * np.cos(np.pi * x) * np.cos(np.pi * y)
        ay = -0.2 * np.pi * np.sin(np.pi * x) * np.sin(np.pi * y)

        bx = -0.15 * np.pi * np.sin(np.pi * x) * np.sin(np.pi * y)
        by = 0.15 * np.pi * np.cos(np.pi * x) * np.cos(np.pi * y)
        return a, b, ax, ay, bx, by

    raise ValueError(
        "Unknown case_name. Available: 'constant_one', 'linear', 'trig'."
    )