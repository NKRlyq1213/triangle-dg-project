from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SpherePatch:
    """
    Metadata for one of the eight triangular patches.

    中文說明：
    - patch_id: 1,...,8
    - sx, sy, sz: 對應 octahedron / sphere face 的符號
    - upper: True 表示上半球，False 表示下半球
    - sector: longitude sector index, 0,1,2,3
    """

    patch_id: int
    sx: int
    sy: int
    sz: int
    upper: bool
    sector: int


_PATCHES: dict[int, SpherePatch] = {
    1: SpherePatch(1, +1, +1, +1, True,  0),
    2: SpherePatch(2, +1, +1, -1, False, 0),
    3: SpherePatch(3, -1, +1, +1, True,  1),
    4: SpherePatch(4, -1, +1, -1, False, 1),
    5: SpherePatch(5, -1, -1, +1, True,  2),
    6: SpherePatch(6, -1, -1, -1, False, 2),
    7: SpherePatch(7, +1, -1, +1, True,  3),
    8: SpherePatch(8, +1, -1, -1, False, 3),
}


def get_patch(patch_id: int) -> SpherePatch:
    """Return metadata for patch_id in {1,...,8}."""
    if patch_id not in _PATCHES:
        raise ValueError("patch_id must be one of 1,2,...,8.")
    return _PATCHES[patch_id]


def all_patch_ids() -> tuple[int, ...]:
    """Return all patch ids."""
    return tuple(_PATCHES.keys())


def patch_signs(patch_id: int) -> tuple[int, int, int]:
    """Return (sx, sy, sz) for the target face F_i."""
    p = get_patch(patch_id)
    return p.sx, p.sy, p.sz


def assert_patch_local_xy(x, y, *, tol: float = 1e-12) -> None:
    """
    Check local triangle coordinates.

    Expected patch-local coordinates:
        x >= 0, y >= 0, x + y <= 1

    注意：
    這裡使用 normalized R=1 local coordinates。
    若你使用 physical radius R，請先除以 R。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if np.any(x < -tol):
        raise ValueError("patch-local x must be >= 0.")
    if np.any(y < -tol):
        raise ValueError("patch-local y must be >= 0.")
    if np.any(x + y > 1.0 + tol):
        raise ValueError("patch-local x + y must be <= 1.")


def local_xy_from_reference_rs(rs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert repo reference triangle nodes (r,s) to local triangle coordinates (x,y).

    Repo reference triangle:
        v1 = (-1,-1)
        v2 = ( 1,-1)
        v3 = (-1, 1)

    Local triangle:
        x >= 0, y >= 0, x + y <= 1

    Affine relation:
        x = (r + 1)/2
        y = (s + 1)/2

    Parameters
    ----------
    rs:
        Shape (Np, 2).

    Returns
    -------
    x, y:
        Each shape (Np,).
    """
    rs = np.asarray(rs, dtype=float)
    if rs.ndim != 2 or rs.shape[1] != 2:
        raise ValueError("rs must have shape (Np, 2).")

    r = rs[:, 0]
    s = rs[:, 1]
    x = 0.5 * (r + 1.0)
    y = 0.5 * (s + 1.0)

    assert_patch_local_xy(x, y)
    return x, y