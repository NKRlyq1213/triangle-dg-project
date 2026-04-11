from __future__ import annotations

import numpy as np
import pytest

from operators.divergence_split import (
    mapped_divergence_split_2d,
    build_mapped_divergence_split_cache_2d,
)


def test_mapped_divergence_split_numba_matches_numpy() -> None:
    pytest.importorskip("numba")

    rng = np.random.default_rng(20260411)
    K = 1000
    Np = 22

    v = rng.normal(size=(K, Np))
    a = rng.normal(size=(K, Np))
    b = rng.normal(size=(K, Np))

    xr = rng.normal(size=(K, Np))
    xs = rng.normal(size=(K, Np))
    yr = rng.normal(size=(K, Np))
    ys = rng.normal(size=(K, Np))
    J = np.abs(rng.normal(size=(K, Np))) + 0.5

    Dr = rng.normal(size=(Np, Np))
    Ds = rng.normal(size=(Np, Np))

    out_numpy = mapped_divergence_split_2d(
        v=v,
        a=a,
        b=b,
        Dr=Dr,
        Ds=Ds,
        xr=xr,
        xs=xs,
        yr=yr,
        ys=ys,
        J=J,
        use_numba=False,
    )

    out_numba = mapped_divergence_split_2d(
        v=v,
        a=a,
        b=b,
        Dr=Dr,
        Ds=Ds,
        xr=xr,
        xs=xs,
        yr=yr,
        ys=ys,
        J=J,
        use_numba=True,
    )

    assert np.allclose(out_numpy, out_numba, atol=1e-11, rtol=1e-11)


def test_mapped_divergence_split_cache_matches_baseline() -> None:
    rng = np.random.default_rng(20260411)
    K = 400
    Np = 22

    v = rng.normal(size=(K, Np))
    a = rng.normal(size=(K, Np))
    b = rng.normal(size=(K, Np))

    xr = rng.normal(size=(K, Np))
    xs = rng.normal(size=(K, Np))
    yr = rng.normal(size=(K, Np))
    ys = rng.normal(size=(K, Np))
    J = np.abs(rng.normal(size=(K, Np))) + 0.5

    Dr = rng.normal(size=(Np, Np))
    Ds = rng.normal(size=(Np, Np))

    split_cache = build_mapped_divergence_split_cache_2d(
        a=a,
        b=b,
        Dr=Dr,
        Ds=Ds,
        xr=xr,
        xs=xs,
        yr=yr,
        ys=ys,
        J=J,
    )

    out_baseline = mapped_divergence_split_2d(
        v=v,
        a=a,
        b=b,
        Dr=Dr,
        Ds=Ds,
        xr=xr,
        xs=xs,
        yr=yr,
        ys=ys,
        J=J,
        use_numba=False,
    )

    out_cached = mapped_divergence_split_2d(
        v=v,
        a=a,
        b=b,
        Dr=Dr,
        Ds=Ds,
        xr=xr,
        xs=xs,
        yr=yr,
        ys=ys,
        J=J,
        use_numba=False,
        split_cache=split_cache,
    )

    assert np.allclose(out_baseline, out_cached, atol=1e-11, rtol=1e-11)
