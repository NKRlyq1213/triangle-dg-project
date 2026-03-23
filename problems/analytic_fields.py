from __future__ import annotations

import numpy as np


def ground_truth_function(case_name: str, r, s, **params) -> np.ndarray:
    """
    Analytic test fields defined on the reference triangle in (r, s).

    Available cases
    ---------------
    - "poly2"
    - "smooth_bump"
    - "trig"
    """
    r = np.asarray(r, dtype=float)
    s = np.asarray(s, dtype=float)
    
    case = case_name.lower().strip()

    if case == "poly2":
        # Quadratic polynomial in (xi, eta)
        return (
            1.0
            + 0.8 * r
            - 0.4 * s
            + 0.6 * r * s
            + 0.3 * r**2
            - 0.2 * s**2
            + 0.1 * r**2 * s
            - 0.05 * r * s**2
            + 0.02 * r**2 * s**2
            - 0.05 * r**3 * s**2
        )

    if case == "smooth_bump":
        r0 = params.get("r0", -1/3)
        s0 = params.get("s0", -1/3)
        sigma = params.get("sigma", np.sqrt(1/30))
        return np.exp(-((r - r0)**2 + (s - s0)**2) / (2.0 * sigma**2))

    if case == "trig":
        return np.sin(np.pi * r) * np.cos(0.5 * np.pi * s)

    raise ValueError(f"Unknown case_name: {case_name}")
