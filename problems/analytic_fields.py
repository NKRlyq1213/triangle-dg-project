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
            - 0.2 * y**2
            + 0.1 * x**2 * y
            - 0.05 * x * y**2
            + 0.02 * x**2 * y**2
        )

    if case == "smooth_bump":
        x0 = params.get("x0", 1/3)
        y0 = params.get("y0", 1/3)
        sigma = params.get("sigma", np.sqrt(1/30))
        return np.exp(-((x - x0)**2 + (y - y0)**2) / (2.0 * sigma**2))

    if case == "trig":
        return np.sin(np.pi * x) * np.cos(0.5 * np.pi * y)

    raise ValueError(f"Unknown case_name: {case_name}")
