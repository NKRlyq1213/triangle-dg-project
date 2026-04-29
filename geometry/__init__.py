from __future__ import annotations

from .connectivity import build_face_connectivity, validate_face_connectivity
from .mesh_structured import structured_square_tri_mesh, validate_mesh_orientation
from .reference_triangle import (
    reference_triangle_area,
    reference_triangle_centroid,
    reference_triangle_vertices,
)
from .sphere_flat_mesh import mesh_summary, sphere_flat_square_mesh
from .sphere_flat_metrics import (
    SphereFlatGeometryCache,
    build_sphere_flat_geometry_cache,
    geometry_diagnostics,
    per_patch_diagnostics,
)
from .sphere_manifold_mesh import (
    generate_spherical_octahedron_mesh,
    spherical_mesh_hmin,
)
from .sphere_manifold_metrics import (
    ManifoldGeometryCache,
    build_manifold_geometry_cache,
    map_reference_nodes_to_sphere,
)
from .sphere_square_patches import (
    A_Ainv_error,
    A_matrix_from_xy_patch,
    Ainv_from_xy_patch,
    lambda_theta_from_xy_patch,
    metric_sqrtG_from_A,
    patch_id_from_xy,
    sphere_xyz_from_lambda_theta,
    sphere_xyz_from_xy_patch,
)
from .sdg_sphere_mapping import (
    SDGMappingResult,
    sdg_A_Ainv_error,
    sdg_A_from_lambda_theta_patch,
    sdg_Ainv_from_A_explicit,
    sdg_Ainv_numpy_error,
    sdg_Ainv_stable_from_lambda_theta_patch,
    sdg_Ainv_T1_stable,
    sdg_Ainv_with_T1_stable_patch,
    sdg_detA_expected,
    sdg_lambda_theta_from_xy_patch,
    sdg_mapping_from_xy_patch,
    sdg_sphere_xyz_from_lambda_theta,
    sdg_sqrtG_expected,
    sdg_sqrtG_from_A,
)
from .sdg_seam_connectivity import (
    SDGSeamPair,
    build_sdg_sphere_face_connectivity,
    map_face_samples_to_sphere,
    sample_face_points,
    seam_pair_xyz_errors,
    validate_sdg_sphere_connectivity,
)

__all__ = [
    "build_face_connectivity",
    "validate_face_connectivity",
    "structured_square_tri_mesh",
    "validate_mesh_orientation",
    "reference_triangle_area",
    "reference_triangle_centroid",
    "reference_triangle_vertices",
    "mesh_summary",
    "sphere_flat_square_mesh",
    "SphereFlatGeometryCache",
    "build_sphere_flat_geometry_cache",
    "geometry_diagnostics",
    "per_patch_diagnostics",
    "generate_spherical_octahedron_mesh",
    "spherical_mesh_hmin",
    "ManifoldGeometryCache",
    "build_manifold_geometry_cache",
    "map_reference_nodes_to_sphere",
    "A_Ainv_error",
    "A_matrix_from_xy_patch",
    "Ainv_from_xy_patch",
    "lambda_theta_from_xy_patch",
    "metric_sqrtG_from_A",
    "patch_id_from_xy",
    "sphere_xyz_from_lambda_theta",
    "sphere_xyz_from_xy_patch",
    "SDGMappingResult",
    "sdg_A_Ainv_error",
    "sdg_A_from_lambda_theta_patch",
    "sdg_Ainv_from_A_explicit",
    "sdg_Ainv_numpy_error",
    "sdg_Ainv_stable_from_lambda_theta_patch",
    "sdg_Ainv_T1_stable",
    "sdg_Ainv_with_T1_stable_patch",
    "sdg_detA_expected",
    "sdg_lambda_theta_from_xy_patch",
    "sdg_mapping_from_xy_patch",
    "sdg_sphere_xyz_from_lambda_theta",
    "sdg_sqrtG_expected",
    "sdg_sqrtG_from_A",
    "SDGSeamPair",
    "build_sdg_sphere_face_connectivity",
    "map_face_samples_to_sphere",
    "sample_face_points",
    "seam_pair_xyz_errors",
    "validate_sdg_sphere_connectivity",
]
