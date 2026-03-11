from __future__ import annotations


def num_modes_2d(N: int) -> int:
    """
    Number of 2D polynomial modes on a triangle up to total degree N.
    """
    if N < 0:
        raise ValueError("N must be >= 0")
    return (N + 1) * (N + 2) // 2


def mode_indices_2d(N: int) -> list[tuple[int, int]]:
    """
    Return the list of mode index pairs (i, j) such that i + j <= N.

    Ordering:
        (0,0),
        (0,1), (1,0),
        (0,2), (1,1), (2,0), ...
    """
    if N < 0:
        raise ValueError("N must be >= 0")

    out: list[tuple[int, int]] = []
    for total_deg in range(N + 1):
        for i in range(total_deg + 1):
            j = total_deg - i
            out.append((i, j))
    return out
