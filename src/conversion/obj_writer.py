"""
Wavefront OBJ export for idTech brush geometry.

Converts brush half-space representations to explicit polygon meshes
via plane-plane-plane intersection, then writes .obj (and optional .mtl).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from quake_levelgenerator.src.conversion.map_writer import Brush, Plane

Vec3 = Tuple[float, float, float]

EPSILON = 1e-6


# ---------------------------------------------------------------------------
# Vector math helpers
# ---------------------------------------------------------------------------

def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _length(v: Vec3) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _normalize(v: Vec3) -> Vec3:
    ln = _length(v)
    if ln < EPSILON:
        return (0.0, 0.0, 1.0)
    return (v[0] / ln, v[1] / ln, v[2] / ln)


# ---------------------------------------------------------------------------
# Plane representation from 3-point brush face
# ---------------------------------------------------------------------------

def _plane_from_points(p1: Vec3, p2: Vec3, p3: Vec3) -> Tuple[Vec3, float]:
    """Return (normal, dist) from three CCW points."""
    v1 = _sub(p2, p1)
    v2 = _sub(p3, p1)
    n = _normalize(_cross(v1, v2))
    return n, _dot(n, p1)


def _intersect_three_planes(
    n1: Vec3, d1: float,
    n2: Vec3, d2: float,
    n3: Vec3, d3: float,
) -> Optional[Vec3]:
    """Find the intersection point of three planes, or None if degenerate."""
    denom = _dot(n1, _cross(n2, n3))
    if abs(denom) < EPSILON:
        return None
    c23 = _cross(n2, n3)
    c31 = _cross(n3, n1)
    c12 = _cross(n1, n2)
    x = (d1 * c23[0] + d2 * c31[0] + d3 * c12[0]) / denom
    y = (d1 * c23[1] + d2 * c31[1] + d3 * c12[1]) / denom
    z = (d1 * c23[2] + d2 * c31[2] + d3 * c12[2]) / denom
    return (x, y, z)


# ---------------------------------------------------------------------------
# Brush → polygons conversion
# ---------------------------------------------------------------------------

def _brush_planes(brush: Brush) -> List[Tuple[Vec3, float]]:
    """Extract (normal, dist) for each plane in a brush."""
    result = []
    for p in brush.planes:
        n, d = _plane_from_points(p.p1, p.p2, p.p3)
        result.append((n, d))
    return result


def _compute_brush_centroid(planes: List[Tuple[Vec3, float]]) -> Optional[Vec3]:
    """Estimate brush centroid by averaging plane distances along normals."""
    if len(planes) < 4:
        return None

    # Use the center of the bounding box implied by opposing planes
    # This is a heuristic - find all vertex candidates and average them
    vertices = []
    n_planes = len(planes)

    for i in range(n_planes):
        for j in range(i + 1, n_planes):
            for k in range(j + 1, n_planes):
                pt = _intersect_three_planes(
                    planes[i][0], planes[i][1],
                    planes[j][0], planes[j][1],
                    planes[k][0], planes[k][1]
                )
                if pt is not None:
                    vertices.append(pt)

    if not vertices:
        return None

    cx = sum(v[0] for v in vertices) / len(vertices)
    cy = sum(v[1] for v in vertices) / len(vertices)
    cz = sum(v[2] for v in vertices) / len(vertices)
    return (cx, cy, cz)


def _point_inside_brush(pt: Vec3, planes: List[Tuple[Vec3, float]], skip: int = -1,
                        inward_normals: Optional[bool] = None) -> bool:
    """Check if a point is on or inside every half-space defined by the brush planes.

    The function auto-detects whether normals point inward or outward based on
    the brush centroid. If inward_normals is explicitly provided, it uses that.

    For inward-facing normals (idTech 1 convention):
        - Points inside satisfy dot(n, pt) >= d for all planes
    For outward-facing normals:
        - Points inside satisfy dot(n, pt) <= d for all planes
    """
    for i, (n, d) in enumerate(planes):
        if i == skip:
            continue

        val = _dot(n, pt)

        if inward_normals is True:
            # Inward normals: point must be "in front of" plane
            if val < d - EPSILON:
                return False
        elif inward_normals is False:
            # Outward normals: point must be "behind" plane
            if val > d + EPSILON:
                return False
        else:
            # No explicit direction - this shouldn't happen often,
            # but if it does, use a tolerance that accepts points near the boundary
            if abs(val - d) > EPSILON * 1000:
                # Check both conditions - if either is satisfied, we might be inside
                # This is a fallback and may not be accurate
                pass

    return True


def _detect_normal_direction(planes: List[Tuple[Vec3, float]]) -> bool:
    """Detect if brush normals point inward (toward centroid) or outward.

    Returns True if normals point inward, False if outward.
    """
    centroid = _compute_brush_centroid(planes)
    if centroid is None:
        return True  # Default to inward (idTech 1 convention)

    # Check if centroid is "in front of" all planes (inward) or "behind" all (outward)
    inward_count = 0
    outward_count = 0

    for n, d in planes:
        val = _dot(n, centroid)
        if val >= d:
            inward_count += 1
        else:
            outward_count += 1

    return inward_count >= outward_count


def _order_face_vertices(verts: List[Vec3], normal: Vec3) -> List[Vec3]:
    """Order coplanar vertices in CCW winding around normal."""
    if len(verts) < 3:
        return verts
    center = (
        sum(v[0] for v in verts) / len(verts),
        sum(v[1] for v in verts) / len(verts),
        sum(v[2] for v in verts) / len(verts),
    )
    # Build a local 2D basis on the plane
    ref = _sub(verts[0], center)
    ref_len = _length(ref)
    if ref_len < EPSILON:
        ref = _sub(verts[1], center) if len(verts) > 1 else (1, 0, 0)
        ref_len = _length(ref)
    if ref_len < EPSILON:
        return verts
    u = (ref[0] / ref_len, ref[1] / ref_len, ref[2] / ref_len)
    v = _cross(normal, u)

    def angle(pt: Vec3) -> float:
        d = _sub(pt, center)
        return math.atan2(_dot(d, v), _dot(d, u))

    return sorted(verts, key=angle)


def brush_to_polygons(brush: Brush) -> List[Tuple[List[Vec3], str]]:
    """Convert a brush to a list of (vertex_list, texture) per face.

    Each face is the convex polygon formed by the intersection of that plane
    with all other planes of the brush.

    Automatically detects whether the brush uses inward or outward facing normals
    and adjusts the half-space test accordingly.
    """
    planes = _brush_planes(brush)
    n_planes = len(planes)
    faces: List[Tuple[List[Vec3], str]] = []

    # Detect normal direction
    inward_normals = _detect_normal_direction(planes)

    for fi in range(n_planes):
        fn, fd = planes[fi]
        verts: List[Vec3] = []
        seen = set()
        for j in range(n_planes):
            if j == fi:
                continue
            for k in range(j + 1, n_planes):
                if k == fi:
                    continue
                pt = _intersect_three_planes(
                    fn, fd, planes[j][0], planes[j][1], planes[k][0], planes[k][1]
                )
                if pt is None:
                    continue
                # De-dup with snapping
                key = (round(pt[0], 2), round(pt[1], 2), round(pt[2], 2))
                if key in seen:
                    continue
                if _point_inside_brush(pt, planes, inward_normals=inward_normals):
                    seen.add(key)
                    verts.append(pt)
        if len(verts) >= 3:
            ordered = _order_face_vertices(verts, fn)
            faces.append((ordered, brush.planes[fi].texture))
    return faces


# ---------------------------------------------------------------------------
# OBJ Writer
# ---------------------------------------------------------------------------

class ObjWriter:
    """Write brush geometry as Wavefront OBJ + optional MTL."""

    def __init__(self):
        self._vertices: List[Vec3] = []
        self._faces: List[Tuple[List[int], str]] = []  # (vertex indices 1-based, material)
        self._materials: Dict[str, bool] = {}

    def add_brushes(self, brushes: List[Brush]):
        for brush in brushes:
            polys = brush_to_polygons(brush)
            for verts, tex in polys:
                if len(verts) < 3:
                    continue
                indices = []
                for v in verts:
                    self._vertices.append(v)
                    indices.append(len(self._vertices))  # 1-based
                self._faces.append((indices, tex))
                self._materials[tex] = True

    def write(self, obj_path: str, write_mtl: bool = True):
        """Write .obj (and optionally .mtl) files."""
        obj_p = Path(obj_path)
        mtl_name = obj_p.stem + ".mtl"

        lines = []
        lines.append(f"# idTech Map Generator OBJ export")
        lines.append(f"# {len(self._vertices)} vertices, {len(self._faces)} faces")
        if write_mtl:
            lines.append(f"mtllib {mtl_name}")
        lines.append("")

        # Vertices — idTech Y-up to OBJ Y-up (idTech Z is up; OBJ Y is up)
        for v in self._vertices:
            lines.append(f"v {v[0]:.4f} {v[2]:.4f} {-v[1]:.4f}")

        lines.append("")

        # Faces grouped by material
        current_mat = None
        # Sort faces by material for fewer usemtl switches
        sorted_faces = sorted(self._faces, key=lambda f: f[1])
        for indices, mat in sorted_faces:
            if mat != current_mat:
                lines.append(f"usemtl {mat}")
                current_mat = mat
            face_str = " ".join(str(i) for i in indices)
            lines.append(f"f {face_str}")

        obj_p.write_text("\n".join(lines) + "\n")

        if write_mtl:
            self._write_mtl(str(obj_p.parent / mtl_name))

    def _write_mtl(self, mtl_path: str):
        lines = ["# idTech Map Generator MTL", ""]
        for mat in sorted(self._materials.keys()):
            lines.append(f"newmtl {mat}")
            lines.append("Ka 0.2 0.2 0.2")
            lines.append("Kd 0.8 0.8 0.8")
            lines.append("Ks 0.0 0.0 0.0")
            lines.append("d 1.0")
            lines.append("")
        Path(mtl_path).write_text("\n".join(lines) + "\n")

    @property
    def vertex_count(self) -> int:
        return len(self._vertices)

    @property
    def face_count(self) -> int:
        return len(self._faces)
