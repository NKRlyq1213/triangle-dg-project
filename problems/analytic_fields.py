from __future__ import annotations

import numpy as np


def ground_truth_function(case_name: str, x, y, **params) -> np.ndarray:
    """
    Analytic test fields defined on the reference triangle in (xi, eta).

    Available cases
    ---------------
    - "poly2"
    - "smooth_bump"
    - "trig"
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    case = case_name.lower().strip()

    if case == "poly2":
        # Quadratic polynomial in (xi, eta)
        return (
            1.0
            + 0.8 * x
            - 0.4 * y
            + 0.6 * x * y
            + 0.3 * x**2
        )

    if case == "smooth_bump":
        x0 = params.get("x0", 0.35)
        y0 = params.get("y0", 0.30)
        sigma = params.get("sigma", 0.12)
        return np.exp(-((x - x0)**2 + (y - y0)**2) / (2.0 * sigma**2))

    if case == "trig":
        return np.sin(np.pi * x) * np.cos(0.5 * np.pi * y)

    raise ValueError(f"Unknown case_name: {case_name}")
