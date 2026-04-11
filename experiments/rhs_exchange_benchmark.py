from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import csv
import importlib.util
import numpy as np

from data.table1_rules import load_table1_rule
from geometry.reference_triangle import reference_triangle_area
from geometry.mesh_structured import structured_square_tri_mesh
from geometry.affine_map import map_reference_nodes_to_all_elements
from geometry.metrics import affine_geometric_factors_from_mesh
from geometry.face_metrics import affine_face_geometry_from_mesh
from geometry.connectivity import build_face_connectivity

from operators.vandermonde2d import vandermonde2d, grad_vandermonde2d
from operators.differentiation import (
    differentiation_matrices_square,
    differentiation_matrices_weighted,
)
from operators.trace_policy import build_trace_policy
from operators.rhs_split_conservative_exchange import rhs_split_conservative_exchange


@dataclass(frozen=True)
class RHSExchangeBenchmarkConfig:
    order: int = 4
    N: int = 4
    diagonal: str = "anti"
    mesh_levels: tuple[int, ...] = (8, 16, 32)
    repeats: int = 80
    warmup: int = 8
    t_eval: float = 0.1
    tau: float = 0.0
    backends: tuple[str, ...] = ("numpy", "auto", "numba")
    modes: tuple[str, ...] = ("full", "perf")
    verbose: bool = True


class _CaseData(dict):
    pass


def _numba_installed() -> bool:
    return importlib.util.find_spec("numba") is not None


def _resolve_requested_use_numba(backend: str) -> bool | None:
    name = str(backend).lower().strip()
    if name == "numpy":
        return False
    if name == "numba":
        return True
    if name == "auto":
        return None
    raise ValueError("backend must be 'numpy', 'auto', or 'numba'.")


def _resolve_mode(mode: str) -> tuple[bool, bool]:
    mode = str(mode).lower().strip()
    if mode == "full":
        return True, True
    if mode == "perf":
        return False, False
    raise ValueError("mode must be 'full' or 'perf'.")


def _build_reference_diff_operators(rule: dict, N: int) -> tuple[np.ndarray, np.ndarray]:
    rs = np.asarray(rule["rs"], dtype=float)
    ws = np.asarray(rule["ws"], dtype=float).reshape(-1)

    V = vandermonde2d(N, rs[:, 0], rs[:, 1])
    Vr, Vs = grad_vandermonde2d(N, rs[:, 0], rs[:, 1])

    if V.shape[0] == V.shape[1]:
        return differentiation_matrices_square(V, Vr, Vs)

    return differentiation_matrices_weighted(
        V, Vr, Vs, ws, area=reference_triangle_area()
    )


def _q_exact_sinx(x: np.ndarray, y: np.ndarray, t: float = 0.0) -> np.ndarray:
    return np.sin(x - t)


def _velocity_one_one(x: np.ndarray, y: np.ndarray, t: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    return np.ones_like(x), np.ones_like(y)


def _build_case_data(config: RHSExchangeBenchmarkConfig, n: int) -> _CaseData:
    rule = load_table1_rule(config.order)
    trace = build_trace_policy(rule)

    Dr, Ds = _build_reference_diff_operators(rule, config.N)

    VX, VY, EToV = structured_square_tri_mesh(nx=n, ny=n, diagonal=config.diagonal)
    conn = build_face_connectivity(VX, VY, EToV, classify_boundary="box")

    X, Y = map_reference_nodes_to_all_elements(rule["rs"], VX, VY, EToV)

    q_elem = _q_exact_sinx(X, Y, t=0.0)
    u_elem = np.ones_like(q_elem)
    v_elem = np.ones_like(q_elem)

    geom = affine_geometric_factors_from_mesh(VX, VY, EToV, rule["rs"])
    face_geom = affine_face_geometry_from_mesh(VX, VY, EToV, trace)

    return _CaseData(
        rule=rule,
        trace=trace,
        Dr=Dr,
        Ds=Ds,
        conn=conn,
        geom=geom,
        face_geom=face_geom,
        q_elem=q_elem,
        u_elem=u_elem,
        v_elem=v_elem,
        K=int(EToV.shape[0]),
        Np=int(q_elem.shape[1]),
    )


def _bench_one(
    case: _CaseData,
    config: RHSExchangeBenchmarkConfig,
    use_numba: bool | None,
    compute_mismatches: bool,
    return_diagnostics: bool,
) -> tuple[float, float]:
    def call_rhs() -> np.ndarray:
        rhs, _ = rhs_split_conservative_exchange(
            q_elem=case["q_elem"],
            u_elem=case["u_elem"],
            v_elem=case["v_elem"],
            Dr=case["Dr"],
            Ds=case["Ds"],
            geom=case["geom"],
            rule=case["rule"],
            trace=case["trace"],
            conn=case["conn"],
            face_geom=case["face_geom"],
            q_boundary=_q_exact_sinx,
            velocity=_velocity_one_one,
            t=config.t_eval,
            tau=config.tau,
            compute_mismatches=compute_mismatches,
            return_diagnostics=return_diagnostics,
            use_numba=use_numba,
        )
        return rhs

    for _ in range(config.warmup):
        call_rhs()

    checksum = 0.0
    t0 = perf_counter()
    for _ in range(config.repeats):
        rhs = call_rhs()
        checksum += float(np.sum(rhs))
    elapsed = perf_counter() - t0

    return elapsed, checksum


def run_rhs_exchange_benchmark(config: RHSExchangeBenchmarkConfig) -> list[dict]:
    results: list[dict] = []
    baseline_ms_by_mesh: dict[int, float] = {}
    has_numba = _numba_installed()

    for n in config.mesh_levels:
        case = _build_case_data(config, n)

        if config.verbose:
            print(f"[setup] n={n} -> K={case['K']}, Np={case['Np']}")

        mesh_rows: list[dict] = []

        for backend in config.backends:
            requested_use_numba = _resolve_requested_use_numba(backend)
            effective_use_numba = bool(has_numba) and (
                requested_use_numba is None or requested_use_numba
            )

            for mode in config.modes:
                compute_mismatches, return_diagnostics = _resolve_mode(mode)

                elapsed, checksum = _bench_one(
                    case=case,
                    config=config,
                    use_numba=requested_use_numba,
                    compute_mismatches=compute_mismatches,
                    return_diagnostics=return_diagnostics,
                )

                per_call_ms = 1000.0 * elapsed / float(config.repeats)

                row = {
                    "nx": n,
                    "ny": n,
                    "K_tri": case["K"],
                    "Np": case["Np"],
                    "repeats": int(config.repeats),
                    "warmup": int(config.warmup),
                    "backend": str(backend),
                    "mode": str(mode),
                    "requested_use_numba": str(requested_use_numba),
                    "effective_use_numba": bool(effective_use_numba),
                    "numba_installed": bool(has_numba),
                    "compute_mismatches": bool(compute_mismatches),
                    "return_diagnostics": bool(return_diagnostics),
                    "elapsed_sec": float(elapsed),
                    "per_call_ms": float(per_call_ms),
                    "checksum": float(checksum),
                }
                mesh_rows.append(row)

                if config.verbose:
                    print(
                        f"  [bench] backend={backend:>5s} mode={mode:>4s} "
                        f"per_call={per_call_ms:9.4f} ms "
                        f"(numba_effective={effective_use_numba})"
                    )

        baseline_candidates = [
            r for r in mesh_rows if r["backend"] == "numpy" and r["mode"] == "full"
        ]
        if len(baseline_candidates) != 1:
            raise RuntimeError("Expected exactly one baseline row: backend=numpy, mode=full")

        baseline_ms = float(baseline_candidates[0]["per_call_ms"])
        baseline_ms_by_mesh[n] = baseline_ms

        for row in mesh_rows:
            row["speedup_vs_numpy_full"] = baseline_ms / float(row["per_call_ms"])

        results.extend(mesh_rows)

    return results


def print_results_table(results: list[dict]) -> None:
    header = (
        f"{'n':>5s} {'K':>8s} {'backend':>8s} {'mode':>6s} "
        f"{'numba':>7s} {'per_call_ms':>12s} {'speedup':>9s}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{int(r['nx']):5d} "
            f"{int(r['K_tri']):8d} "
            f"{str(r['backend']):>8s} "
            f"{str(r['mode']):>6s} "
            f"{str(r['effective_use_numba']):>7s} "
            f"{float(r['per_call_ms']):12.4f} "
            f"{float(r['speedup_vs_numpy_full']):9.3f}"
        )


def save_results_csv(results: list[dict], csv_path: str) -> None:
    if not results:
        raise ValueError("results is empty")

    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
