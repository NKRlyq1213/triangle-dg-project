from .vandermonde2d import vandermonde2d, grad_vandermonde2d
from .mass import mass_matrix_from_quadrature
from .differentiation import (
    differentiation_matrices_square,
    differentiation_matrices_weighted,
)
from .reconstruction import (
    fit_modal_coefficients_square,
    fit_modal_coefficients_weighted,
    evaluate_modal_expansion,
    PolynomialReconstruction,
)
from .boundary import (
    edge_nodes_rs,
    edge_vandermonde2d,
    volume_to_edge_operator,
    evaluate_on_edge,
    evaluate_on_all_edges,
)
from .split_form import split_advective_operator_2d
from .divergence_split import (
    mapped_divergence_split_2d,
    mapped_divergence_conservative_2d,
)
from .trace_policy import (
    face_parameter_t_from_rs,
    build_trace_policy,
    evaluate_embedded_face_values,
    evaluate_projected_face_values,
)
from .exchange import (
    evaluate_all_face_values,
    unique_interior_face_pairs,
    pair_face_traces,
    interior_face_pair_mismatches,
)
from .sdg_flattened_divergence import (
    build_table1_reference_diff_operators,
    divergence_stats_by_patch,
    sdg_flattened_cartesian_divergence,
)
from .manifold_rhs import (
    ManifoldExchangeCache,
    ManifoldReferenceOperators,
    build_manifold_exchange_cache,
    build_manifold_face_connectivity,
    build_manifold_table1_k4_reference_operators,
    build_manifold_vmaps,
    constant_field_divergence_error,
    manifold_contravariant_velocity,
    manifold_rhs,
    manifold_rhs_exchange,
    manifold_rhs_constant_field,
    manifold_surface_term,
    manifold_surface_term_from_exchange,
    manifold_volume_divergence,
    pair_manifold_face_traces,
)

from .rhs_split_conservative_exact_trace import (
    volume_term_split_conservative as volume_term_split_conservative_exact_trace,
    surface_term_from_exact_trace,
    upwind_flux_and_penalty as upwind_flux_and_penalty_exact_trace,
)

from .rhs_split_conservative_exchange import (
    volume_term_split_conservative as volume_term_split_conservative_exchange,
    upwind_flux_and_penalty as upwind_flux_and_penalty_exchange,
    upwind_penalty_simplified,
    fill_exterior_state,
    fill_boundary_exterior_state_upwind,
    surface_term_from_exchange,
)

# Backward-compatible aliases keep existing import behavior on exchange backend.
volume_term_split_conservative = volume_term_split_conservative_exchange
upwind_flux_and_penalty = upwind_flux_and_penalty_exchange

__all__ = [
    "vandermonde2d",
    "grad_vandermonde2d",
    "mass_matrix_from_quadrature",
    "differentiation_matrices_square",
    "differentiation_matrices_weighted",
    "fit_modal_coefficients_square",
    "fit_modal_coefficients_weighted",
    "evaluate_modal_expansion",
    "PolynomialReconstruction",
    "edge_nodes_rs",
    "edge_vandermonde2d",
    "volume_to_edge_operator",
    "evaluate_on_edge",
    "evaluate_on_all_edges",
    "split_advective_operator_2d",
    "mapped_divergence_split_2d",
    "mapped_divergence_conservative_2d",
    "face_parameter_t_from_rs",
    "build_trace_policy",
    "evaluate_embedded_face_values",
    "evaluate_projected_face_values",
    "evaluate_all_face_values",
    "unique_interior_face_pairs",
    "pair_face_traces",
    "interior_face_pair_mismatches",
    "build_table1_reference_diff_operators",
    "divergence_stats_by_patch",
    "sdg_flattened_cartesian_divergence",
    "ManifoldExchangeCache",
    "ManifoldReferenceOperators",
    "build_manifold_exchange_cache",
    "build_manifold_face_connectivity",
    "build_manifold_table1_k4_reference_operators",
    "build_manifold_vmaps",
    "constant_field_divergence_error",
    "manifold_contravariant_velocity",
    "manifold_rhs",
    "manifold_rhs_exchange",
    "manifold_rhs_constant_field",
    "manifold_surface_term",
    "manifold_surface_term_from_exchange",
    "manifold_volume_divergence",
    "pair_manifold_face_traces",
    "volume_term_split_conservative",
    "volume_term_split_conservative_exact_trace",
    "volume_term_split_conservative_exchange",
    "surface_term_from_exact_trace",
    "upwind_flux_and_penalty",
    "upwind_flux_and_penalty_exact_trace",
    "upwind_flux_and_penalty_exchange",
    "upwind_penalty_simplified",
    "fill_exterior_state",
    "fill_boundary_exterior_state_upwind",
    "surface_term_from_exchange",
]
