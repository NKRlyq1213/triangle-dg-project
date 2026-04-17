from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from time import perf_counter
import math
import csv
import numpy as np
from typing import Callable

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
from operators.mass import mass_matrix_from_quadrature
from operators.trace_policy import build_trace_policy
from operators.rhs_split_conservative_exchange import (
    rhs_split_conservative_exchange,
    build_surface_exchange_cache,
    build_volume_split_cache,
    resolve_effective_taus,
)
from operators.rhs_split_conservative_exact_trace import rhs_split_conservative_exact_trace

from time_integration.lsrk54 import integrate_lsrk54
from time_integration.lsrk54 import RK4A, RK4B, RK4C
from time_integration.CFL import mesh_min_altitude, cfl_dt_from_h, vmax_from_uv


@dataclass(frozen=True)
class LSRKHConvergenceConfig:
    table_name: str = "table1"
    order: int = 4
    N: int = 4
    diagonal: str = "anti"
    mesh_levels: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256)

    # Requested fixed setup from user:
    # - q(x,y,t)=sin(2*pi*(x-t))
    # - CFL=1
    # - final times tf=2*pi and tf=20*pi
    cfl: float = 1.0
    tf_values: tuple[float, ...] = (np.pi*2, np.pi*20)

    tau: float = 0.0
    tau_interior: float | None = None
    tau_qb: float | None = None
    use_numba: bool | None = True
    surface_inverse_mass_mode: str = "diagonal"
    surface_backend: str = "face-major"
    interior_trace_mode: str = "exchange"
    test_function_mode: str = "sin2pi_x"
    physical_boundary_mode: str = "exact_qb"
    face_order_mode: str = "triangle"
    use_surface_cache: bool = True
    use_rk_stage_boundary_correction: bool = False
    q_boundary_correction: Callable | None = None
    q_boundary_correction_mode: str = "all"
    verbose: bool = True


@dataclass(frozen=True)
class TransportTestFunctionSpec:
    mode: str
    phase_x: float
    phase_y: float
    phase_t: float
    vel_x: float
    vel_y: float


_TEST_FUNCTION_SPECS = {
    "sin2pi_x": TransportTestFunctionSpec(
        mode="sin2pi_x",
        phase_x=1.0,
        phase_y=0.0,
        phase_t=1.0,
        vel_x=1.0,
        vel_y=0.0,
    ),
    "sin2pi_y": TransportTestFunctionSpec(
        mode="sin2pi_y",
        phase_x=0.0,
        phase_y=1.0,
        phase_t=1.0,
        vel_x=0.0,
        vel_y=1.0,
    ),
    "sin2pi_xy": TransportTestFunctionSpec(
        mode="sin2pi_xy",
        phase_x=1.0,
        phase_y=1.0,
        phase_t=2.0,
        vel_x=1.0,
        vel_y=1.0,
    ),
}


def _resolve_test_function_spec(mode: str) -> TransportTestFunctionSpec:
    key = str(mode).strip().lower()
    aliases = {
        "x": "sin2pi_x",
        "y": "sin2pi_y",
        "xy": "sin2pi_xy",
    }
    key = aliases.get(key, key)
    spec = _TEST_FUNCTION_SPECS.get(key)
    if spec is None:
        raise ValueError(
            "test_function_mode must be one of: 'sin2pi_x', 'sin2pi_y', 'sin2pi_xy'."
        )
    return spec


def _make_q_exact(spec: TransportTestFunctionSpec) -> Callable:
    k = 2.0 * np.pi

    def q_exact(x: np.ndarray, y: np.ndarray, t: float = 0.0) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        phase = k * (spec.phase_x * x + spec.phase_y * y - spec.phase_t * float(t))
        return np.sin(phase)

    return q_exact


def _make_velocity(spec: TransportTestFunctionSpec) -> Callable:
    def velocity(x: np.ndarray, y: np.ndarray, t: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        del t
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return spec.vel_x * np.ones_like(x), spec.vel_y * np.ones_like(y)

    return velocity


_DEFAULT_TEST_PROFILE = _TEST_FUNCTION_SPECS["sin2pi_x"]
_DEFAULT_Q_EXACT = _make_q_exact(_DEFAULT_TEST_PROFILE)
_DEFAULT_VELOCITY = _make_velocity(_DEFAULT_TEST_PROFILE)


class RKStageBoundaryCorrection:
    """
    Notebook-style boundary state evolution synchronized with LSRK stage calls.

    This tracks [g, g_t, g_tt] using the same low-storage RK coefficients and
    returns delta_qB = g_stage - g_exact(t_stage).

    The effective step size is inferred from consecutive stage-call times,
    so the correction remains consistent on the final short step.
    """

    def __init__(self, dt: float, profile: TransportTestFunctionSpec):
        dt = float(dt)
        if dt <= 0.0:
            raise ValueError("dt must be positive for RKStageBoundaryCorrection.")

        self._dt_default = dt
        self._profile = profile
        self._k = 2.0 * np.pi
        self._initialized = False
        self._stage = 0
        self._last_t = -np.inf
        self._prev_t = 0.0
        self._prev_stage = 0
        self._have_prev_call = False

        self._g = None
        self._g_t = None
        self._g_tt = None
        self._Kg = None
        self._Kgt = None
        self._Kgtt = None

    def _phase(self, x_face: np.ndarray, y_face: np.ndarray, t: float) -> np.ndarray:
        return self._k * (
            self._profile.phase_x * x_face
            + self._profile.phase_y * y_face
            - self._profile.phase_t * float(t)
        )

    def _reset(
        self,
        x_face: np.ndarray,
        y_face: np.ndarray,
        t: float,
        expected_shape: tuple[int, int, int],
    ) -> None:
        phase = self._phase(x_face, y_face, t)
        lam = self._k * self._profile.phase_t

        self._g = np.sin(phase)
        self._g_t = -lam * np.cos(phase)
        self._g_tt = -(lam**2) * np.sin(phase)

        self._Kg = np.zeros(expected_shape, dtype=float)
        self._Kgt = np.zeros(expected_shape, dtype=float)
        self._Kgtt = np.zeros(expected_shape, dtype=float)

        self._stage = 0
        self._last_t = float(t)
        self._prev_t = float(t)
        self._prev_stage = 0
        self._have_prev_call = False
        self._initialized = True

    def _infer_step_dt(
        self,
        *,
        t_prev: float,
        t_curr: float,
        stage_prev: int,
        stage_curr: int,
    ) -> float:
        c_prev = float(RK4C[stage_prev])
        c_curr = float(RK4C[stage_curr])

        if stage_curr > stage_prev:
            frac = c_curr - c_prev
        else:
            frac = (1.0 - c_prev) + c_curr

        if frac <= 0.0:
            return float(self._dt_default)

        dt_step = (float(t_curr) - float(t_prev)) / frac
        if (not np.isfinite(dt_step)) or dt_step <= 0.0:
            return float(self._dt_default)
        return float(dt_step)

    def _advance_one_stage(
        self,
        x_face: np.ndarray,
        y_face: np.ndarray,
        *,
        t_stage: float,
        stage: int,
        dt_step: float,
    ) -> None:
        phase = self._phase(x_face, y_face, t_stage)
        lam = self._k * self._profile.phase_t
        g_ttt = (lam**3) * np.cos(phase)

        self._Kg *= RK4A[stage]
        self._Kgt *= RK4A[stage]
        self._Kgtt *= RK4A[stage]

        self._Kg += dt_step * self._g_t
        self._Kgt += dt_step * self._g_tt
        self._Kgtt += dt_step * g_ttt

        self._g += RK4B[stage] * self._Kg
        self._g_t += RK4B[stage] * self._Kgt
        self._g_tt += RK4B[stage] * self._Kgtt

    def __call__(
        self,
        x_face: np.ndarray,
        y_face: np.ndarray,
        t: float,
        qM: np.ndarray,
        ndotV: np.ndarray,
        is_boundary: np.ndarray,
        q_boundary_exact: np.ndarray,
    ) -> np.ndarray:
        del qM, ndotV, is_boundary

        t = float(t)
        x_face = np.asarray(x_face, dtype=float)
        y_face = np.asarray(y_face, dtype=float)
        q_boundary_exact = np.asarray(q_boundary_exact, dtype=float)

        if x_face.shape != q_boundary_exact.shape:
            raise ValueError("x_face and q_boundary_exact must share shape (K, 3, Nfp).")
        if y_face.shape != q_boundary_exact.shape:
            raise ValueError("y_face and q_boundary_exact must share shape (K, 3, Nfp).")

        if (
            (not self._initialized)
            or (t < self._last_t - 1e-14)
            or (self._g is None)
            or (self._g.shape != q_boundary_exact.shape)
        ):
            self._reset(
                x_face=x_face,
                y_face=y_face,
                t=t,
                expected_shape=q_boundary_exact.shape,
            )

        if self._have_prev_call:
            prev_stage = int(self._prev_stage)
            curr_stage = int(self._stage)

            if curr_stage != ((prev_stage + 1) % 5):
                self._reset(
                    x_face=x_face,
                    y_face=y_face,
                    t=t,
                    expected_shape=q_boundary_exact.shape,
                )
            else:
                dt_step = self._infer_step_dt(
                    t_prev=float(self._prev_t),
                    t_curr=t,
                    stage_prev=prev_stage,
                    stage_curr=curr_stage,
                )
                self._advance_one_stage(
                    x_face=x_face,
                    y_face=y_face,
                    t_stage=float(self._prev_t),
                    stage=prev_stage,
                    dt_step=dt_step,
                )

        delta = self._g - q_boundary_exact

        self._prev_t = t
        self._prev_stage = int(self._stage)
        self._have_prev_call = True

        self._stage = (self._stage + 1) % 5
        self._last_t = t

        return delta


def build_rk_stage_boundary_correction(
    dt: float,
    profile: TransportTestFunctionSpec,
) -> Callable:
    return RKStageBoundaryCorrection(dt=dt, profile=profile)


def build_reference_diff_operators_from_rule(rule: dict, N: int) -> tuple[np.ndarray, np.ndarray]:
    rs = np.asarray(rule["rs"], dtype=float)
    ws = np.asarray(rule["ws"], dtype=float).reshape(-1)

    V = vandermonde2d(N, rs[:, 0], rs[:, 1])
    Vr, Vs = grad_vandermonde2d(N, rs[:, 0], rs[:, 1])

    if V.shape[0] == V.shape[1]:
        return differentiation_matrices_square(V, Vr, Vs)

    return differentiation_matrices_weighted(
        V, Vr, Vs, ws, area=reference_triangle_area()
    )


def build_polynomial_l2_projector_from_rule(rule: dict, N: int) -> np.ndarray:
    rs = np.asarray(rule["rs"], dtype=float)
    ws = np.asarray(rule["ws"], dtype=float).reshape(-1)

    V = vandermonde2d(N, rs[:, 0], rs[:, 1])
    area = reference_triangle_area()
    M = mass_matrix_from_quadrature(V, ws, area=area)
    rhs = area * (V.T * ws[None, :])
    proj_modal = np.linalg.solve(M, rhs)
    return V @ proj_modal


def build_projected_inverse_mass_from_rule(rule: dict, N: int) -> np.ndarray:
    rs = np.asarray(rule["rs"], dtype=float)
    ws = np.asarray(rule["ws"], dtype=float).reshape(-1)
    if np.any(ws <= 0.0):
        raise ValueError("rule['ws'] must be strictly positive.")

    V = vandermonde2d(N, rs[:, 0], rs[:, 1])
    area = reference_triangle_area()
    M = mass_matrix_from_quadrature(V, ws, area=area)

    rhs = area * V.T
    projected_inverse_mass = V @ np.linalg.inv(M) @ rhs
    if projected_inverse_mass.shape != (ws.size, ws.size):
        raise ValueError("Projected inverse-mass size must be (Np, Np).")

    return projected_inverse_mass


def q_exact_sinx(x: np.ndarray, y: np.ndarray, t: float = 0.0) -> np.ndarray:
    return _DEFAULT_Q_EXACT(x, y, t)


def velocity_one_one(x: np.ndarray, y: np.ndarray, t: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    return _DEFAULT_VELOCITY(x, y, t)


def weighted_l2_error(err: np.ndarray, rule: dict, face_geom: dict) -> float:
    ws = np.asarray(rule["ws"], dtype=float).reshape(-1)
    area = np.asarray(face_geom["area"], dtype=float).reshape(-1)

    val = 0.0
    for k in range(err.shape[0]):
        val += area[k] * np.dot(ws, err[k] ** 2)
    return float(np.sqrt(val))


def compute_convergence_rate(errors: list[float]) -> list[float]:
    rates = [math.nan]
    for i in range(1, len(errors)):
        e_prev = float(errors[i - 1])
        e_curr = float(errors[i])
        if (not np.isfinite(e_prev)) or (not np.isfinite(e_curr)):
            rates.append(math.nan)
        elif e_prev <= 0.0 or e_curr <= 0.0:
            rates.append(math.nan)
        else:
            rates.append(math.log(e_prev / e_curr, 2.0))
    return rates


def _normalize_qb_mode(qb_mode: str) -> str:
    mode = str(qb_mode).strip().lower()
    if mode not in ("off", "on", "compare"):
        raise ValueError("qb_mode must be one of: 'off', 'on', 'compare'.")
    return mode


def _validate_config(config: LSRKHConvergenceConfig) -> None:
    if str(config.table_name).lower().strip() != "table1":
        raise ValueError("LSRK h-convergence currently supports table_name='table1' only.")
    if config.cfl <= 0.0:
        raise ValueError("cfl must be positive.")
    if len(config.mesh_levels) == 0:
        raise ValueError("mesh_levels must be non-empty.")
    if any(n <= 0 for n in config.mesh_levels):
        raise ValueError("All mesh levels must be positive integers.")
    if len(config.tf_values) == 0:
        raise ValueError("tf_values must be non-empty.")
    if any(tf <= 0.0 for tf in config.tf_values):
        raise ValueError("All tf values must be positive.")
    resolve_effective_taus(
        tau=config.tau,
        tau_interior=config.tau_interior,
        tau_qb=config.tau_qb,
    )
    surface_inverse_mass_mode = str(config.surface_inverse_mass_mode).strip().lower()
    if surface_inverse_mass_mode not in ("diagonal", "projected"):
        raise ValueError("surface_inverse_mass_mode must be one of: 'diagonal', 'projected'.")
    interior_trace_mode = str(config.interior_trace_mode).strip().lower()
    if interior_trace_mode not in ("exchange", "exact_trace"):
        raise ValueError("interior_trace_mode must be one of: 'exchange', 'exact_trace'.")
    test_function_mode = str(config.test_function_mode).strip().lower()
    if test_function_mode not in ("sin2pi_x", "sin2pi_y", "sin2pi_xy", "x", "y", "xy"):
        raise ValueError(
            "test_function_mode must be one of: 'sin2pi_x', 'sin2pi_y', 'sin2pi_xy'."
        )
    physical_boundary_mode = str(config.physical_boundary_mode).strip().lower()
    if physical_boundary_mode not in ("exact_qb", "opposite_boundary", "periodic_vmap"):
        raise ValueError(
            "physical_boundary_mode must be one of: 'exact_qb', 'opposite_boundary', 'periodic_vmap'."
        )
    face_order_mode = str(config.face_order_mode).strip().lower()
    if face_order_mode not in ("triangle", "simplex", "simplex_strict"):
        raise ValueError("face_order_mode must be one of: 'triangle', 'simplex', 'simplex_strict'.")
    if interior_trace_mode != "exchange" and face_order_mode != "triangle":
        raise ValueError(
            "face_order_mode='simplex' and 'simplex_strict' currently support interior_trace_mode='exchange' only."
        )
    if face_order_mode == "simplex_strict" and surface_inverse_mass_mode != "projected":
        raise ValueError(
            "face_order_mode='simplex_strict' requires surface_inverse_mass_mode='projected'."
        )
    if interior_trace_mode == "exact_trace" and surface_inverse_mass_mode != "diagonal":
        raise ValueError(
            "surface_inverse_mass_mode='projected' is not supported with interior_trace_mode='exact_trace'."
        )
    q_boundary_correction_mode = str(config.q_boundary_correction_mode).strip().lower()
    if q_boundary_correction_mode not in ("inflow", "boundary", "all"):
        raise ValueError("q_boundary_correction_mode must be one of: 'inflow', 'boundary', 'all'.")
    if config.q_boundary_correction is not None and (not callable(config.q_boundary_correction)):
        raise ValueError("q_boundary_correction must be callable or None.")
    if config.use_rk_stage_boundary_correction and config.q_boundary_correction is not None:
        raise ValueError(
            "Use either use_rk_stage_boundary_correction=True or q_boundary_correction callable, not both."
        )
    exact_source_exists = (interior_trace_mode == "exact_trace") or (physical_boundary_mode == "exact_qb")
    if (config.use_rk_stage_boundary_correction or config.q_boundary_correction is not None) and (
        not exact_source_exists
    ):
        raise ValueError(
            "q_boundary_correction requires at least one exact source: "
            "interior_trace_mode='exact_trace' or physical_boundary_mode='exact_qb'."
        )


def _prepare_study_context(config: LSRKHConvergenceConfig) -> dict:
    surface_inverse_mass_mode = str(config.surface_inverse_mass_mode).strip().lower()
    interior_trace_mode = str(config.interior_trace_mode).strip().lower()
    physical_boundary_mode = str(config.physical_boundary_mode).strip().lower()
    face_order_mode = str(config.face_order_mode).strip().lower()
    q_boundary_correction_mode = str(config.q_boundary_correction_mode).strip().lower()
    tau_interior_eff, tau_qb_eff = resolve_effective_taus(
        tau=config.tau,
        tau_interior=config.tau_interior,
        tau_qb=config.tau_qb,
    )

    test_function_spec = _resolve_test_function_spec(config.test_function_mode)
    q_exact = _make_q_exact(test_function_spec)
    velocity = _make_velocity(test_function_spec)

    rule = load_table1_rule(config.order)
    trace = build_trace_policy(rule)
    Dr, Ds = build_reference_diff_operators_from_rule(rule, config.N)

    surface_inverse_mass_t = None
    if surface_inverse_mass_mode == "projected":
        surface_inverse_mass = build_projected_inverse_mass_from_rule(rule, config.N)
        surface_inverse_mass_t = np.ascontiguousarray(surface_inverse_mass.T, dtype=float)

    return {
        "surface_inverse_mass_mode": surface_inverse_mass_mode,
        "interior_trace_mode": interior_trace_mode,
        "physical_boundary_mode": physical_boundary_mode,
        "face_order_mode": face_order_mode,
        "q_boundary_correction_mode": q_boundary_correction_mode,
        "tau_interior": float(tau_interior_eff),
        "tau_qb": float(tau_qb_eff),
        "test_function_spec": test_function_spec,
        "q_exact": q_exact,
        "velocity": velocity,
        "rule": rule,
        "trace": trace,
        "Dr": Dr,
        "Ds": Ds,
        "surface_inverse_mass_t": surface_inverse_mass_t,
    }


def _prepare_level_state(
    config: LSRKHConvergenceConfig,
    context: dict,
    n: int,
) -> dict:
    rule = context["rule"]
    trace = context["trace"]
    velocity = context["velocity"]
    interior_trace_mode = context["interior_trace_mode"]
    physical_boundary_mode = context["physical_boundary_mode"]
    face_order_mode = context["face_order_mode"]

    VX, VY, EToV = structured_square_tri_mesh(
        nx=n,
        ny=n,
        diagonal=config.diagonal,
    )

    conn = build_face_connectivity(VX, VY, EToV, classify_boundary="box")

    X, Y = map_reference_nodes_to_all_elements(rule["rs"], VX, VY, EToV)
    q0 = context["q_exact"](X, Y, t=0.0)
    u_elem, v_elem = velocity(X, Y, t=0.0)

    geom = affine_geometric_factors_from_mesh(VX, VY, EToV, rule["rs"])
    face_geom = affine_face_geometry_from_mesh(VX, VY, EToV, trace)

    volume_split_cache = build_volume_split_cache(
        u_elem=u_elem,
        v_elem=v_elem,
        Dr=context["Dr"],
        Ds=context["Ds"],
        geom=geom,
    )

    surface_cache = None
    if config.use_surface_cache or interior_trace_mode == "exact_trace":
        periodic_nodes = physical_boundary_mode == "periodic_vmap"
        surface_cache = build_surface_exchange_cache(
            rule=rule,
            trace=trace,
            conn=conn,
            face_geom=face_geom,
            face_order_mode=face_order_mode,
            X_nodes=X if periodic_nodes else None,
            Y_nodes=Y if periodic_nodes else None,
        )

    u_face, v_face = velocity(
        np.asarray(face_geom["x_face"], dtype=float),
        np.asarray(face_geom["y_face"], dtype=float),
        t=0.0,
    )
    ndotV_precomputed = np.ascontiguousarray(
        np.asarray(face_geom["nx"], dtype=float) * np.asarray(u_face, dtype=float)
        + np.asarray(face_geom["ny"], dtype=float) * np.asarray(v_face, dtype=float),
        dtype=float,
    )
    ndotV_flat_precomputed = np.ascontiguousarray(
        ndotV_precomputed.reshape(-1, int(trace["nfp"])),
        dtype=float,
    )

    hmin = mesh_min_altitude(VX, VY, EToV)
    vmax = vmax_from_uv(u_elem, v_elem)
    dt_nominal = cfl_dt_from_h(cfl=config.cfl, h=hmin, N=config.N, vmax=vmax)

    return {
        "n": int(n),
        "VX": VX,
        "VY": VY,
        "EToV": EToV,
        "conn": conn,
        "X": X,
        "Y": Y,
        "q0": q0,
        "u_elem": u_elem,
        "v_elem": v_elem,
        "geom": geom,
        "face_geom": face_geom,
        "volume_split_cache": volume_split_cache,
        "surface_cache": surface_cache,
        "ndotV_precomputed": ndotV_precomputed,
        "ndotV_flat_precomputed": ndotV_flat_precomputed,
        "hmin": float(hmin),
        "dt_nominal": float(dt_nominal),
    }


def _resolve_level_q_boundary_correction(
    config: LSRKHConvergenceConfig,
    context: dict,
    *,
    dt_nominal: float,
) -> tuple[Callable | None, str]:
    interior_trace_mode = context["interior_trace_mode"]
    physical_boundary_mode = context["physical_boundary_mode"]
    test_function_spec = context["test_function_spec"]

    q_boundary_correction = config.q_boundary_correction
    q_boundary_correction_kind = "none"

    if config.use_rk_stage_boundary_correction:
        q_boundary_correction = build_rk_stage_boundary_correction(
            dt=dt_nominal,
            profile=test_function_spec,
        )
        q_boundary_correction_kind = "rk_stage"
    elif q_boundary_correction is not None:
        q_boundary_correction_kind = "custom"

    if interior_trace_mode == "exact_trace":
        if q_boundary_correction_kind == "rk_stage":
            q_boundary_correction_kind = "rk_stage_exact_trace"
        elif q_boundary_correction_kind == "custom":
            q_boundary_correction_kind = "custom_exact_trace"
    elif physical_boundary_mode == "exact_qb":
        if q_boundary_correction_kind == "rk_stage":
            q_boundary_correction_kind = "rk_stage_exact_qb"
        elif q_boundary_correction_kind == "custom":
            q_boundary_correction_kind = "custom_exact_qb"

    return q_boundary_correction, q_boundary_correction_kind


def _build_rhs_function(
    config: LSRKHConvergenceConfig,
    context: dict,
    level_state: dict,
    q_boundary_correction: Callable | None,
) -> Callable[[float, np.ndarray], np.ndarray]:
    interior_trace_mode = context["interior_trace_mode"]

    def rhs(t: float, q: np.ndarray) -> np.ndarray:
        if interior_trace_mode == "exchange":
            total_rhs, _ = rhs_split_conservative_exchange(
                q_elem=q,
                u_elem=level_state["u_elem"],
                v_elem=level_state["v_elem"],
                Dr=context["Dr"],
                Ds=context["Ds"],
                geom=level_state["geom"],
                rule=context["rule"],
                trace=context["trace"],
                conn=level_state["conn"],
                face_geom=level_state["face_geom"],
                q_boundary=context["q_exact"],
                velocity=context["velocity"],
                t=t,
                tau=config.tau,
                tau_interior=context["tau_interior"],
                tau_qb=context["tau_qb"],
                compute_mismatches=False,
                return_diagnostics=False,
                use_numba=config.use_numba,
                surface_backend=config.surface_backend,
                surface_cache=level_state["surface_cache"],
                ndotV_precomputed=level_state["ndotV_precomputed"],
                ndotV_flat_precomputed=level_state["ndotV_flat_precomputed"],
                volume_split_cache=level_state["volume_split_cache"],
                surface_inverse_mass_T=context["surface_inverse_mass_t"],
                physical_boundary_mode=context["physical_boundary_mode"],
                face_order_mode=context["face_order_mode"],
                q_boundary_correction=q_boundary_correction,
                q_boundary_correction_mode=context["q_boundary_correction_mode"],
                X_nodes=level_state["X"],
                Y_nodes=level_state["Y"],
            )
            return total_rhs

        total_rhs, _ = rhs_split_conservative_exact_trace(
            q_elem=q,
            u_elem=level_state["u_elem"],
            v_elem=level_state["v_elem"],
            Dr=context["Dr"],
            Ds=context["Ds"],
            geom=level_state["geom"],
            rule=context["rule"],
            trace=context["trace"],
            VX=level_state["VX"],
            VY=level_state["VY"],
            EToV=level_state["EToV"],
            q_exact=context["q_exact"],
            q_boundary=context["q_exact"],
            velocity=context["velocity"],
            t=t,
            tau=config.tau,
            tau_interior=context["tau_interior"],
            tau_qb=context["tau_qb"],
            face_geom=level_state["face_geom"],
            physical_boundary_mode=context["physical_boundary_mode"],
            q_boundary_correction=q_boundary_correction,
            q_boundary_correction_mode=context["q_boundary_correction_mode"],
            use_numba=config.use_numba,
            conn=level_state["conn"],
            surface_cache=level_state["surface_cache"],
        )
        return total_rhs

    return rhs


def _run_lsrk_h_convergence_for_tf(
    config: LSRKHConvergenceConfig,
    *,
    tf: float,
    context: dict,
) -> list[dict]:
    results: list[dict] = []

    for n in config.mesh_levels:
        t0 = perf_counter()
        level_state = _prepare_level_state(config, context, int(n))

        q_boundary_correction, q_boundary_correction_kind = _resolve_level_q_boundary_correction(
            config,
            context,
            dt_nominal=level_state["dt_nominal"],
        )

        rhs = _build_rhs_function(
            config,
            context,
            level_state,
            q_boundary_correction,
        )

        qf, tf_used, nsteps = integrate_lsrk54(
            rhs=rhs,
            q0=level_state["q0"],
            t0=0.0,
            tf=float(tf),
            dt=level_state["dt_nominal"],
        )

        tf_target = float(tf)
        reached_tf = bool(np.isclose(tf_used, tf_target, atol=1e-12, rtol=1e-12))

        q_exact_at_stop = context["q_exact"](level_state["X"], level_state["Y"], t=tf_used)
        err_at_stop = qf - q_exact_at_stop
        L2_error_at_stop = weighted_l2_error(err_at_stop, context["rule"], level_state["face_geom"])
        Linf_error_at_stop = float(np.max(np.abs(err_at_stop)))

        if reached_tf:
            q_exact_final = context["q_exact"](level_state["X"], level_state["Y"], t=tf_target)
            err = qf - q_exact_final
            L2_error = weighted_l2_error(err, context["rule"], level_state["face_geom"])
            Linf_error = float(np.max(np.abs(err)))
        else:
            L2_error = math.nan
            Linf_error = math.nan

        h = 1.0 / float(level_state["n"])
        K = int(level_state["EToV"].shape[0])
        Np = int(level_state["q0"].shape[1])
        total_dof = K * Np
        elapsed = perf_counter() - t0

        row = {
            "nx": int(level_state["n"]),
            "ny": int(level_state["n"]),
            "K_tri": K,
            "h": h,
            "hmin": float(level_state["hmin"]),
            "Np": Np,
            "total_dof": int(total_dof),
            "diagonal": str(config.diagonal),
            "tf_target": tf_target,
            "tf": float(tf_used),
            "reached_tf": reached_tf,
            "cfl": float(config.cfl),
            "dt_nominal": float(level_state["dt_nominal"]),
            "nsteps": int(nsteps),
            "tau": float(config.tau),
            "tau_interior": float(context["tau_interior"]),
            "tau_qb": float(context["tau_qb"]),
            "L2_error": float(L2_error),
            "Linf_error": float(Linf_error),
            "L2_error_at_stop": float(L2_error_at_stop),
            "Linf_error_at_stop": float(Linf_error_at_stop),
            "elapsed_sec": float(elapsed),
            "surface_inverse_mass_mode": context["surface_inverse_mass_mode"],
            "interior_trace_mode": context["interior_trace_mode"],
            "test_function_mode": context["test_function_spec"].mode,
            "physical_boundary_mode": context["physical_boundary_mode"],
            "face_order_mode": context["face_order_mode"],
            "q_boundary_correction_kind": q_boundary_correction_kind,
            "q_boundary_correction_mode": context["q_boundary_correction_mode"],
        }
        results.append(row)

        if config.verbose:
            status = "ok" if reached_tf else "stopped_early"
            l2_display = L2_error if reached_tf else L2_error_at_stop
            linf_display = Linf_error if reached_tf else Linf_error_at_stop
            print(
                f"[lsrk h-study] tf={tf:>5.1f} | n={int(level_state['n']):>3d} | K={K:>7d} | "
                f"L2={l2_display:.6e} | Linf={linf_display:.6e} | "
                f"steps={nsteps:>7d} | dt={float(level_state['dt_nominal']):.3e} | time={elapsed:.2f}s"
            )

    L2_list = [r["L2_error"] for r in results]
    Linf_list = [r["Linf_error"] for r in results]
    L2_rates = compute_convergence_rate(L2_list)
    Linf_rates = compute_convergence_rate(Linf_list)

    for i, row in enumerate(results):
        row["rate_L2"] = L2_rates[i]
        row["rate_Linf"] = Linf_rates[i]

    return results


def _run_lsrk_h_convergence_for_config(
    config: LSRKHConvergenceConfig,
) -> dict[float, list[dict]]:
    _validate_config(config)
    context = _prepare_study_context(config)

    out: dict[float, list[dict]] = {}
    for tf in config.tf_values:
        out[float(tf)] = _run_lsrk_h_convergence_for_tf(
            config,
            tf=float(tf),
            context=context,
        )
    return out


def run_lsrk_study(
    config: LSRKHConvergenceConfig,
    *,
    qb_mode: str = "off",
) -> dict[str, dict[float, list[dict]]]:
    """
    Unified LSRK h-convergence entrypoint.

    qb_mode:
    - "off": baseline run only
    - "on": RK-stage exact-source correction run only
    - "compare": run both baseline and RK-stage exact-source correction
    """
    _validate_config(config)
    mode = _normalize_qb_mode(qb_mode)

    if mode == "compare":
        if config.q_boundary_correction is not None:
            raise ValueError(
                "run_lsrk_study with qb_mode='compare' requires q_boundary_correction=None."
            )
        variants = (
            (
                "baseline",
                replace(
                    config,
                    use_rk_stage_boundary_correction=False,
                    q_boundary_correction=None,
                ),
            ),
            (
                "rk_stage_correction",
                replace(
                    config,
                    use_rk_stage_boundary_correction=True,
                    q_boundary_correction=None,
                ),
            ),
        )
    elif mode == "on":
        if config.q_boundary_correction is not None:
            raise ValueError(
                "run_lsrk_study with qb_mode='on' requires q_boundary_correction=None."
            )
        variants = (
            (
                "rk_stage_correction",
                replace(
                    config,
                    use_rk_stage_boundary_correction=True,
                    q_boundary_correction=None,
                ),
            ),
        )
    else:
        variants = (
            (
                "baseline",
                replace(
                    config,
                    use_rk_stage_boundary_correction=False,
                    q_boundary_correction=None,
                ),
            ),
        )

    out: dict[str, dict[float, list[dict]]] = {}
    for label, cfg in variants:
        out[label] = _run_lsrk_h_convergence_for_config(cfg)
    return out


def print_results_table(results: list[dict], title: str | None = None) -> None:
    if title is not None:
        print(title)

    header = (
        f"{'n':>6s} {'K':>9s} {'h':>12s} "
        f"{'dt':>12s} {'steps':>8s} "
        f"{'L2_error':>14s} {'rate':>8s} "
        f"{'Linf_error':>14s} {'rate':>8s} "
        f"{'time(s)':>10s}"
    )
    print(header)
    print("-" * len(header))

    def fmt_rate(v: float) -> str:
        return "   -   "+ " " if not np.isfinite(v) else f"{v:8.3f}"

    for r in results:
        status = "ok" if bool(r.get("reached_tf", True)) else "stopped_early"
        print(
            f"{r['nx']:6d} {r['K_tri']:9d} {r['h']:12.4e} "
            f"{r['dt_nominal']:12.4e} {r['nsteps']:8d} "
            f"{r['L2_error']:14.6e} {fmt_rate(r['rate_L2'])} "
            f"{r['Linf_error']:14.6e} {fmt_rate(r['rate_Linf'])} "
            f"{r['elapsed_sec']:10.2f}"
        )


def save_results_csv(results: list[dict], filepath: str) -> None:
    if not results:
        raise ValueError("results is empty.")

    fieldnames = list(results[0].keys())
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
