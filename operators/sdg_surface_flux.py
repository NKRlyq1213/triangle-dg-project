from __future__ import annotations

from dataclasses import dataclass
import numpy as np


_LOCAL_FACE_VERTICES = (
    (1, 2),  # face 0: edge v2 -> v3, opposite v1
    (2, 0),  # face 1: edge v3 -> v1, opposite v2
    (0, 1),  # face 2: edge v1 -> v2, opposite v3
)


@dataclass(frozen=True)
class ReferenceFaceData:
    face_node_ids: tuple[np.ndarray, np.ndarray, np.ndarray]
    face_wratio: tuple[np.ndarray, np.ndarray, np.ndarray]
    nfp: int


@dataclass(frozen=True)
class SurfaceConnectivity:
    neighbor_elem: np.ndarray       # shape (K, 3), -1 for boundary
    neighbor_face: np.ndarray       # shape (K, 3), -1 for boundary
    neighbor_node_ids: np.ndarray   # shape (K, 3, nfp), -1 for boundary
    is_boundary: np.ndarray         # shape (K, 3)
    face_length: np.ndarray         # shape (K, 3)
    face_normal_x: np.ndarray       # shape (K, 3)
    face_normal_y: np.ndarray       # shape (K, 3)
    area_flat: np.ndarray           # shape (K,)
    max_match_error: float
    n_internal_faces: int
    n_boundary_faces: int


def build_reference_face_data(
    rule: dict,
    *,
    tol: float = 1.0e-12,
) -> ReferenceFaceData:
    r"""
    Extract Table1 edge nodes for the three reference triangle faces.

    Reference triangle:
        v1 = (-1, -1)
        v2 = ( 1, -1)
        v3 = (-1,  1)

    Face convention:
        face 0: v2-v3, r+s = 0
        face 1: v3-v1, r = -1
        face 2: v1-v2, s = -1

    Uses:
        wratio_i = edge_weight_i / volume_weight_i

    This matches diagonal-mass lifting:

        M_i^{-1} * int_face p phi_i ds
        ~ length * we_i / (area * ws_i) * p_i
    """
    rs = np.asarray(rule["rs"], dtype=float)
    ws = np.asarray(rule["ws"], dtype=float).reshape(-1)
    we = np.asarray(rule["we"], dtype=float).reshape(-1)

    if rs.ndim != 2 or rs.shape[1] != 2:
        raise ValueError("rule['rs'] must have shape (Np, 2).")
    if ws.shape != (rs.shape[0],) or we.shape != (rs.shape[0],):
        raise ValueError("rule weights do not match rs.")

    r = rs[:, 0]
    s = rs[:, 1]

    masks = [
        np.abs(r + s) <= tol,
        np.abs(r + 1.0) <= tol,
        np.abs(s + 1.0) <= tol,
    ]

    # Sort each face along its oriented local edge.
    sort_keys = [
        0.5 * (s + 1.0),       # face 0: v2 -> v3
        0.5 * (1.0 - s),       # face 1: v3 -> v1
        0.5 * (r + 1.0),       # face 2: v1 -> v2
    ]

    face_node_ids = []
    face_wratio = []

    for f in range(3):
        ids = np.nonzero(masks[f])[0]
        if ids.size == 0:
            raise ValueError(f"No edge nodes found on reference face {f}.")

        ids = ids[np.argsort(sort_keys[f][ids])]

        if np.any(~np.isfinite(we[ids])):
            raise ValueError(
                f"Reference face {f} contains nodes without edge weights."
            )

        wratio = we[ids] / ws[ids]
        face_node_ids.append(ids.astype(int))
        face_wratio.append(wratio.astype(float))

    nfp = int(face_node_ids[0].size)
    if any(ids.size != nfp for ids in face_node_ids):
        raise ValueError("All faces must have the same number of face nodes.")

    return ReferenceFaceData(
        face_node_ids=tuple(face_node_ids),
        face_wratio=tuple(face_wratio),
        nfp=nfp,
    )


def _triangle_area(vertices: np.ndarray) -> float:
    v0, v1, v2 = vertices
    return 0.5 * (
        (v1[0] - v0[0]) * (v2[1] - v0[1])
        - (v1[1] - v0[1]) * (v2[0] - v0[0])
    )


def _map_reference_nodes_to_element(
    rs: np.ndarray,
    vertices: np.ndarray,
) -> np.ndarray:
    r = np.asarray(rs[:, 0], dtype=float)
    s = np.asarray(rs[:, 1], dtype=float)

    v0 = vertices[0]
    v1 = vertices[1]
    v2 = vertices[2]

    # Reference triangle:
    # x = v0 + 0.5(r+1)(v1-v0) + 0.5(s+1)(v2-v0)
    return (
        v0[None, :]
        + 0.5 * (r[:, None] + 1.0) * (v1 - v0)[None, :]
        + 0.5 * (s[:, None] + 1.0) * (v2 - v0)[None, :]
    )


def _face_geometry_for_element(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Return length, outward normal x, outward normal y for 3 local faces.

    Vertices must be CCW. For an oriented CCW boundary edge a -> b,
    outward normal is:

        n = (dy, -dx) / length.
    """
    area = _triangle_area(vertices)
    if area <= 0.0:
        raise ValueError("Element vertices must be CCW.")

    length = np.zeros(3, dtype=float)
    nx = np.zeros(3, dtype=float)
    ny = np.zeros(3, dtype=float)

    for f, (ia, ib) in enumerate(_LOCAL_FACE_VERTICES):
        a = vertices[ia]
        b = vertices[ib]
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        L = float(np.sqrt(dx * dx + dy * dy))
        if L <= 0.0:
            raise ValueError("Degenerate face with zero length.")

        length[f] = L
        nx[f] = dy / L
        ny[f] = -dx / L

    return length, nx, ny


def build_surface_connectivity(
    VX: np.ndarray,
    VY: np.ndarray,
    EToV: np.ndarray,
    rs_nodes: np.ndarray,
    ref_face: ReferenceFaceData,
    *,
    match_tol: float = 1.0e-10,
) -> SurfaceConnectivity:
    r"""
    Build element face connectivity and face-node matching.

    Interior faces are paired through shared global vertex pairs.
    Boundary faces remain boundary.

    For each interior face, neighbor_node_ids[k,f,i] gives the matching node
    index on the neighbor element.
    """
    VX = np.asarray(VX, dtype=float).reshape(-1)
    VY = np.asarray(VY, dtype=float).reshape(-1)
    EToV = np.asarray(EToV, dtype=int)
    rs_nodes = np.asarray(rs_nodes, dtype=float)

    if EToV.ndim != 2 or EToV.shape[1] != 3:
        raise ValueError("EToV must have shape (K, 3).")
    if rs_nodes.ndim != 2 or rs_nodes.shape[1] != 2:
        raise ValueError("rs_nodes must have shape (Np, 2).")

    K = EToV.shape[0]
    nfp = ref_face.nfp

    neighbor_elem = -np.ones((K, 3), dtype=int)
    neighbor_face = -np.ones((K, 3), dtype=int)
    neighbor_node_ids = -np.ones((K, 3, nfp), dtype=int)
    is_boundary = np.ones((K, 3), dtype=bool)

    face_length = np.zeros((K, 3), dtype=float)
    face_normal_x = np.zeros((K, 3), dtype=float)
    face_normal_y = np.zeros((K, 3), dtype=float)
    area_flat = np.zeros(K, dtype=float)

    # Face map from undirected vertex pair -> list[(k,f)].
    face_map: dict[tuple[int, int], list[tuple[int, int]]] = {}

    for k in range(K):
        vids = EToV[k]
        vertices = np.column_stack([VX[vids], VY[vids]])

        area = _triangle_area(vertices)
        if area <= 0.0:
            raise ValueError(f"Element {k} is not CCW or is degenerate.")

        area_flat[k] = area

        L, nx, ny = _face_geometry_for_element(vertices)
        face_length[k, :] = L
        face_normal_x[k, :] = nx
        face_normal_y[k, :] = ny

        local_face_vertex_ids = [
            (vids[1], vids[2]),
            (vids[2], vids[0]),
            (vids[0], vids[1]),
        ]

        for f, (a, b) in enumerate(local_face_vertex_ids):
            key = tuple(sorted((int(a), int(b))))
            face_map.setdefault(key, []).append((k, f))

    # Precompute physical coordinates of each element's reference nodes.
    node_xy = np.zeros((K, rs_nodes.shape[0], 2), dtype=float)
    for k in range(K):
        vids = EToV[k]
        vertices = np.column_stack([VX[vids], VY[vids]])
        node_xy[k, :, :] = _map_reference_nodes_to_element(rs_nodes, vertices)

    max_match_error = 0.0
    n_internal_faces = 0
    n_boundary_faces = 0

    for key, faces in face_map.items():
        if len(faces) == 1:
            n_boundary_faces += 1
            continue

        if len(faces) != 2:
            raise ValueError(f"Non-manifold face detected for vertex pair {key}: {faces}")

        (ka, fa), (kb, fb) = faces

        neighbor_elem[ka, fa] = kb
        neighbor_face[ka, fa] = fb
        neighbor_elem[kb, fb] = ka
        neighbor_face[kb, fb] = fa

        is_boundary[ka, fa] = False
        is_boundary[kb, fb] = False

        ids_a = ref_face.face_node_ids[fa]
        ids_b = ref_face.face_node_ids[fb]

        xy_a = node_xy[ka, ids_a, :]
        xy_b = node_xy[kb, ids_b, :]

        # Match owner nodes on a to closest nodes on b.
        for i in range(nfp):
            d = np.linalg.norm(xy_b - xy_a[i, :][None, :], axis=1)
            j = int(np.argmin(d))
            max_match_error = max(max_match_error, float(d[j]))
            if d[j] > match_tol:
                raise ValueError(
                    f"Face-node match error too large: {d[j]} on ({ka},{fa}) -> ({kb},{fb})"
                )
            neighbor_node_ids[ka, fa, i] = ids_b[j]

        # Match owner nodes on b to closest nodes on a.
        for i in range(nfp):
            d = np.linalg.norm(xy_a - xy_b[i, :][None, :], axis=1)
            j = int(np.argmin(d))
            max_match_error = max(max_match_error, float(d[j]))
            if d[j] > match_tol:
                raise ValueError(
                    f"Face-node match error too large: {d[j]} on ({kb},{fb}) -> ({ka},{fa})"
                )
            neighbor_node_ids[kb, fb, i] = ids_a[j]

        n_internal_faces += 1

    return SurfaceConnectivity(
        neighbor_elem=neighbor_elem,
        neighbor_face=neighbor_face,
        neighbor_node_ids=neighbor_node_ids,
        is_boundary=is_boundary,
        face_length=face_length,
        face_normal_x=face_normal_x,
        face_normal_y=face_normal_y,
        area_flat=area_flat,
        max_match_error=float(max_match_error),
        n_internal_faces=int(n_internal_faces),
        n_boundary_faces=int(n_boundary_faces),
    )


def _penalty_upwind_family(
    ndotF: np.ndarray,
    qM: np.ndarray,
    qP: np.ndarray,
    *,
    tau: float = 0.0,
) -> np.ndarray:
    r"""
    Penalty p = f - f* for upwind-family flux.

    f = ndotF qM

    f* = 1/2(ndotF qM + ndotF qP)
         + 1/2(1 - tau)|ndotF|(qM - qP)

    Therefore:

        p = 1/2[ndotF - (1 - tau)|ndotF|](qM - qP)

    tau = 0 gives pure upwind:
        p = min(ndotF, 0)(qM - qP)
    """
    ndotF = np.asarray(ndotF, dtype=float)
    qM = np.asarray(qM, dtype=float)
    qP = np.asarray(qP, dtype=float)

    if not (ndotF.shape == qM.shape == qP.shape):
        raise ValueError("ndotF, qM, qP must have the same shape.")

    tau = float(tau)
    if tau == 0.0:
        return np.minimum(ndotF, 0.0) * (qM - qP)

    return 0.5 * (ndotF - (1.0 - tau) * np.abs(ndotF)) * (qM - qP)


def sdg_surface_penalty_rhs(
    q: np.ndarray,
    Fx_area: np.ndarray,
    Fy_area: np.ndarray,
    ref_face: ReferenceFaceData,
    conn: SurfaceConnectivity,
    *,
    J_area: float,
    tau: float = 0.0,
    boundary_mode: str = "same_state",
    q_boundary_value: float = 1.0,
) -> np.ndarray:
    r"""
    Surface penalty contribution to q_t for equal-area conservative SDG.

    Continuous pulled-back conservative equation:

        d_t(J q) + div(F_area q) = 0

    The strong-form DG correction is:

        q_t += (1/J) M^{-1} int_face (f - f*) phi ds

    Here diagonal mass lifting is used:

        contribution_i =
            (1/J) * (length / area_flat) * (edge_weight_i / volume_weight_i) * p_i

    Boundary modes
    --------------
    same_state:
        qP = qM on boundary. Good for constant-state preservation.
    constant:
        qP = q_boundary_value on boundary.
    """
    q = np.asarray(q, dtype=float)
    Fx_area = np.asarray(Fx_area, dtype=float)
    Fy_area = np.asarray(Fy_area, dtype=float)

    if not (q.shape == Fx_area.shape == Fy_area.shape):
        raise ValueError("q, Fx_area, Fy_area must have the same shape.")
    if q.ndim != 2:
        raise ValueError("q must have shape (K, Np).")

    K, Np = q.shape
    if conn.neighbor_elem.shape != (K, 3):
        raise ValueError("connectivity K does not match q.")
    if J_area == 0.0:
        raise ValueError("J_area must be nonzero.")

    mode = str(boundary_mode).strip().lower()
    if mode not in ("same_state", "constant"):
        raise ValueError("boundary_mode must be 'same_state' or 'constant'.")

    rhs_surface = np.zeros_like(q, dtype=float)

    for k in range(K):
        for f in range(3):
            ids = ref_face.face_node_ids[f]
            wratio = ref_face.face_wratio[f]

            qM = q[k, ids]
            FxM = Fx_area[k, ids]
            FyM = Fy_area[k, ids]

            ndotF = (
                conn.face_normal_x[k, f] * FxM
                + conn.face_normal_y[k, f] * FyM
            )

            if conn.is_boundary[k, f]:
                if mode == "same_state":
                    qP = qM
                else:
                    qP = np.full_like(qM, float(q_boundary_value))
            else:
                kp = conn.neighbor_elem[k, f]
                idsP = conn.neighbor_node_ids[k, f, :]
                qP = q[kp, idsP]

            p = _penalty_upwind_family(
                ndotF,
                qM,
                qP,
                tau=tau,
            )

            scale = (
                conn.face_length[k, f]
                / conn.area_flat[k]
                / float(J_area)
            )

            rhs_surface[k, ids] += scale * wratio * p

    return rhs_surface


def surface_connectivity_summary(conn: SurfaceConnectivity) -> dict[str, float | int]:
    return {
        "n_internal_faces": int(conn.n_internal_faces),
        "n_boundary_faces": int(conn.n_boundary_faces),
        "max_match_error": float(conn.max_match_error),
        "min_area_flat": float(np.min(conn.area_flat)),
        "max_area_flat": float(np.max(conn.area_flat)),
        "min_face_length": float(np.min(conn.face_length)),
        "max_face_length": float(np.max(conn.face_length)),
    }
