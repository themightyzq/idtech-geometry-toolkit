"""
Mesh builder for converting idTech brushes to renderable geometry.

Uses the existing brush_to_polygons() function from obj_writer.py and
converts the output to vertex/index buffers for OpenGL rendering.
"""

from __future__ import annotations

import hashlib
import math
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum

from quake_levelgenerator.src.conversion.map_writer import Brush, Plane
from quake_levelgenerator.src.conversion.obj_writer import brush_to_polygons

Vec3 = Tuple[float, float, float]
Vec2 = Tuple[float, float]

# idTech texture scale: 64 world units = 1 texture tile
TEXTURE_SCALE = 64.0


class SurfaceType(Enum):
    """Surface type classification based on face normal or texture name."""
    FLOOR = "floor"           # Z-up normal (nz > 0.7)
    CEILING = "ceiling"       # Z-down normal (nz < -0.7)
    WALL = "wall"             # Mostly horizontal normal
    STRUCTURAL = "structural"  # Pillars, arches, stairs
    TRIM = "trim"             # Decorative trim


def _get_texture_to_surface_map() -> Dict[str, SurfaceType]:
    """Get mapping of texture names to surface types from TEXTURE_SETTINGS.

    Returns:
        Dict mapping texture name -> SurfaceType
    """
    try:
        from quake_levelgenerator.src.generators.textures import TEXTURE_SETTINGS
        return {
            TEXTURE_SETTINGS.get_texture("floor"): SurfaceType.FLOOR,
            TEXTURE_SETTINGS.get_texture("ceiling"): SurfaceType.CEILING,
            TEXTURE_SETTINGS.get_texture("wall"): SurfaceType.WALL,
            TEXTURE_SETTINGS.get_texture("structural"): SurfaceType.STRUCTURAL,
            TEXTURE_SETTINGS.get_texture("trim"): SurfaceType.TRIM,
        }
    except Exception:
        # Fallback if TEXTURE_SETTINGS not available
        return {}


DEBUG_CLASSIFICATION = False  # Enable to trace surface classification

def classify_surface(normal: Vec3, texture_name: Optional[str] = None) -> SurfaceType:
    """Classify a surface based on normal direction first, then texture name.

    For floors and ceilings, always use normal-based classification to ensure
    correct surface type even when primitives use the same texture for all surfaces.
    For walls and other surfaces, try texture matching first.

    Args:
        normal: Normalized face normal (nx, ny, nz)
        texture_name: Optional texture name from the face

    Returns:
        SurfaceType enum value
    """
    nz = normal[2]

    # ALWAYS classify floors and ceilings by normal direction
    # This ensures correct classification even when primitives use _box() for everything
    if nz > 0.7:
        if DEBUG_CLASSIFICATION:
            import sys
            print(f"[CLASSIFY] nz={nz:.2f} -> FLOOR (by normal)", file=sys.stderr)
            sys.stderr.flush()
        return SurfaceType.FLOOR
    elif nz < -0.7:
        if DEBUG_CLASSIFICATION:
            import sys
            print(f"[CLASSIFY] nz={nz:.2f} -> CEILING (by normal)", file=sys.stderr)
            sys.stderr.flush()
        return SurfaceType.CEILING

    # For non-floor/ceiling surfaces, try texture matching for structural/trim
    if texture_name:
        tex_map = _get_texture_to_surface_map()
        if DEBUG_CLASSIFICATION:
            import sys
            print(f"[CLASSIFY] texture='{texture_name}', checking tex_map", file=sys.stderr)
            sys.stderr.flush()
        if texture_name in tex_map:
            surface_type = tex_map[texture_name]
            # Only use texture match for structural/trim (not floor/ceiling/wall)
            if surface_type in (SurfaceType.STRUCTURAL, SurfaceType.TRIM):
                if DEBUG_CLASSIFICATION:
                    print(f"[CLASSIFY] -> {surface_type} (matched by texture)", file=sys.stderr)
                    sys.stderr.flush()
                return surface_type

    # Default to WALL for vertical surfaces
    if DEBUG_CLASSIFICATION:
        import sys
        print(f"[CLASSIFY] nz={nz:.2f} -> WALL (default)", file=sys.stderr)
        sys.stderr.flush()
    return SurfaceType.WALL


@dataclass
class RenderMesh:
    """Renderable mesh data for OpenGL."""
    # Vertex data: position (3) + normal (3) + uv (2) + color (3) = 11 floats per vertex
    vertices: np.ndarray  # Shape: (N, 11), dtype=float32
    # Triangle indices
    indices: np.ndarray   # Shape: (M, 3), dtype=uint32
    # Bounding box
    bounds_min: Tuple[float, float, float]
    bounds_max: Tuple[float, float, float]

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def triangle_count(self) -> int:
        return len(self.indices)

    @property
    def is_empty(self) -> bool:
        return len(self.vertices) == 0


def _texture_to_color(texture_name: str) -> Tuple[float, float, float]:
    """Generate a deterministic color from texture name using hash."""
    # Hash the texture name
    h = hashlib.md5(texture_name.encode()).hexdigest()

    # Use hash bytes for RGB
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0

    # Boost saturation and keep colors visible
    # Adjust to mid-range values to ensure visibility
    r = 0.3 + r * 0.5
    g = 0.3 + g * 0.5
    b = 0.3 + b * 0.5

    return (r, g, b)


def _compute_normal(v0: Vec3, v1: Vec3, v2: Vec3) -> Vec3:
    """Compute face normal from three vertices (CCW winding)."""
    # Edge vectors
    e1 = (v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2])
    e2 = (v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2])

    # Cross product
    nx = e1[1] * e2[2] - e1[2] * e2[1]
    ny = e1[2] * e2[0] - e1[0] * e2[2]
    nz = e1[0] * e2[1] - e1[1] * e2[0]

    # Normalize
    length = (nx * nx + ny * ny + nz * nz) ** 0.5
    if length > 1e-6:
        nx /= length
        ny /= length
        nz /= length
    else:
        nx, ny, nz = 0.0, 0.0, 1.0

    return (nx, ny, nz)


def _compute_uv(vertex: Vec3, normal: Vec3, plane: Optional[Plane] = None) -> Vec2:
    """Compute UV coordinates using idTech planar projection.

    Projects the vertex onto a 2D plane based on the dominant axis of the normal,
    then applies rotation, scale, and offset from the plane parameters.

    Args:
        vertex: 3D vertex position (x, y, z)
        normal: Face normal vector (nx, ny, nz)
        plane: Optional Plane object with texture parameters

    Returns:
        UV coordinates (u, v)
    """
    x, y, z = vertex
    nx, ny, nz = normal

    # Determine projection axis based on dominant normal component
    abs_nx, abs_ny, abs_nz = abs(nx), abs(ny), abs(nz)

    if abs_nz >= abs_nx and abs_nz >= abs_ny:
        # Z-dominant (floor/ceiling): project to XY plane
        u, v = x, y
    elif abs_nx >= abs_ny:
        # X-dominant (E/W wall): project to YZ plane
        u, v = y, z
    else:
        # Y-dominant (N/S wall): project to XZ plane
        u, v = x, z

    # Get texture parameters from plane (or use defaults)
    x_offset = 0.0
    y_offset = 0.0
    rotation = 0.0
    x_scale = 1.0
    y_scale = 1.0

    if plane is not None:
        x_offset = plane.x_offset
        y_offset = plane.y_offset
        rotation = plane.rotation
        x_scale = plane.x_scale if plane.x_scale != 0 else 1.0
        y_scale = plane.y_scale if plane.y_scale != 0 else 1.0

    # Apply rotation (convert degrees to radians)
    if rotation != 0.0:
        rad = math.radians(rotation)
        cos_r = math.cos(rad)
        sin_r = math.sin(rad)
        u_new = u * cos_r - v * sin_r
        v_new = u * sin_r + v * cos_r
        u, v = u_new, v_new

    # Apply scale and offset
    # idTech convention: 64 world units = 1 texture tile
    u = u / (x_scale * TEXTURE_SCALE) + x_offset / TEXTURE_SCALE
    v = v / (y_scale * TEXTURE_SCALE) + y_offset / TEXTURE_SCALE

    return (u, v)


class MeshBuilder:
    """Converts idTech brushes to renderable mesh data."""

    def __init__(self):
        self._vertices: List[List[float]] = []
        self._indices: List[List[int]] = []
        self._bounds_min: Optional[List[float]] = None
        self._bounds_max: Optional[List[float]] = None

    def clear(self):
        """Clear all mesh data."""
        self._vertices.clear()
        self._indices.clear()
        self._bounds_min = None
        self._bounds_max = None

    def add_brushes(self, brushes: List[Brush]):
        """Convert brushes to mesh data and accumulate."""
        for brush in brushes:
            self._add_brush(brush)

    def _add_brush(self, brush: Brush):
        """Convert a single brush to triangulated mesh data."""
        polygons = brush_to_polygons(brush)

        # Build a lookup from texture name to plane for UV parameters
        # Note: Multiple planes may have the same texture, but for UV we need
        # the specific plane that generated each face. We match by texture name
        # as an approximation since brush_to_polygons returns (vertices, texture).
        texture_to_plane = {}
        for plane in brush.planes:
            # Store plane by texture name (last one wins if duplicates)
            texture_to_plane[plane.texture] = plane

        for face_idx, (vertices, texture) in enumerate(polygons):
            if len(vertices) < 3:
                continue

            color = _texture_to_color(texture)

            # Triangulate the polygon (fan triangulation)
            # Works correctly for convex polygons from brush faces
            first_idx = len(self._vertices)

            # Compute face normal from first triangle
            normal = _compute_normal(vertices[0], vertices[1], vertices[2])

            # Get plane for UV computation (may be None if not found)
            plane = texture_to_plane.get(texture)

            # Add vertices
            for v in vertices:
                # Update bounds
                self._update_bounds(v)

                # Compute UV coordinates
                uv = _compute_uv(v, normal, plane)

                # Position + Normal + UV + Color (11 floats)
                self._vertices.append([
                    v[0], v[1], v[2],                 # Position (3)
                    normal[0], normal[1], normal[2],  # Normal (3)
                    uv[0], uv[1],                     # UV (2)
                    color[0], color[1], color[2]      # Color (3)
                ])

            # Add triangles (fan from first vertex)
            for i in range(1, len(vertices) - 1):
                self._indices.append([
                    first_idx,
                    first_idx + i,
                    first_idx + i + 1
                ])

    def _update_bounds(self, v: Vec3):
        """Update bounding box with new vertex."""
        if self._bounds_min is None:
            self._bounds_min = [v[0], v[1], v[2]]
            self._bounds_max = [v[0], v[1], v[2]]
        else:
            self._bounds_min[0] = min(self._bounds_min[0], v[0])
            self._bounds_min[1] = min(self._bounds_min[1], v[1])
            self._bounds_min[2] = min(self._bounds_min[2], v[2])
            self._bounds_max[0] = max(self._bounds_max[0], v[0])
            self._bounds_max[1] = max(self._bounds_max[1], v[1])
            self._bounds_max[2] = max(self._bounds_max[2], v[2])

    def build(self) -> RenderMesh:
        """Build the final renderable mesh."""
        if not self._vertices:
            return RenderMesh(
                vertices=np.array([], dtype=np.float32),
                indices=np.array([], dtype=np.uint32),
                bounds_min=(0.0, 0.0, 0.0),
                bounds_max=(0.0, 0.0, 0.0)
            )

        vertices = np.array(self._vertices, dtype=np.float32)
        indices = np.array(self._indices, dtype=np.uint32)

        bounds_min = tuple(self._bounds_min) if self._bounds_min else (0.0, 0.0, 0.0)
        bounds_max = tuple(self._bounds_max) if self._bounds_max else (0.0, 0.0, 0.0)

        return RenderMesh(
            vertices=vertices,
            indices=indices,
            bounds_min=bounds_min,
            bounds_max=bounds_max
        )


def build_mesh_from_brushes(brushes: List[Brush]) -> RenderMesh:
    """Convenience function to build mesh from brushes in one call."""
    builder = MeshBuilder()
    builder.add_brushes(brushes)
    return builder.build()


def build_wireframe_mesh(brushes: List[Brush]) -> Tuple[np.ndarray, np.ndarray]:
    """Build wireframe mesh (edges only) from brushes.

    Returns:
        Tuple of (vertices, indices) for line rendering.
        vertices: Shape (N, 3), dtype=float32
        indices: Shape (M, 2), dtype=uint32 (line segments)
    """
    vertices: List[List[float]] = []
    indices: List[List[int]] = []
    vertex_map = {}  # (x, y, z) -> index

    def get_or_add_vertex(v: Vec3) -> int:
        key = (round(v[0], 2), round(v[1], 2), round(v[2], 2))
        if key in vertex_map:
            return vertex_map[key]
        idx = len(vertices)
        vertices.append([v[0], v[1], v[2]])
        vertex_map[key] = idx
        return idx

    for brush in brushes:
        polygons = brush_to_polygons(brush)
        for verts, _ in polygons:
            if len(verts) < 2:
                continue
            # Add edges around polygon
            for i in range(len(verts)):
                v0 = verts[i]
                v1 = verts[(i + 1) % len(verts)]
                i0 = get_or_add_vertex(v0)
                i1 = get_or_add_vertex(v1)
                # Avoid duplicate edges (order-independent)
                edge_key = (min(i0, i1), max(i0, i1))
                indices.append([i0, i1])

    if not vertices:
        return np.array([], dtype=np.float32), np.array([], dtype=np.uint32)

    return (
        np.array(vertices, dtype=np.float32),
        np.array(indices, dtype=np.uint32)
    )


@dataclass
class SurfaceMeshes:
    """Meshes grouped by surface type for per-surface texturing."""
    floor: RenderMesh
    ceiling: RenderMesh
    wall: RenderMesh
    structural: RenderMesh
    trim: RenderMesh
    # Combined bounds for camera fitting
    bounds_min: Tuple[float, float, float]
    bounds_max: Tuple[float, float, float]

    @property
    def is_empty(self) -> bool:
        return (self.floor.is_empty and self.ceiling.is_empty and
                self.wall.is_empty and self.structural.is_empty and self.trim.is_empty)

    @property
    def total_triangles(self) -> int:
        return (self.floor.triangle_count + self.ceiling.triangle_count +
                self.wall.triangle_count + self.structural.triangle_count +
                self.trim.triangle_count)


def build_surface_meshes(brushes: List[Brush]) -> SurfaceMeshes:
    """Build separate meshes for each surface type.

    This enables per-surface texturing where each surface type can have
    a different texture applied. Classification is done first by texture name
    (matching against TEXTURE_SETTINGS), then falling back to normal direction.

    Args:
        brushes: List of idTech brushes to convert

    Returns:
        SurfaceMeshes with separate meshes for floor, ceiling, wall, structural, trim
    """
    # Separate vertex/index lists for each surface type
    surface_data: Dict[SurfaceType, Tuple[List[List[float]], List[List[int]]]] = {
        SurfaceType.FLOOR: ([], []),
        SurfaceType.CEILING: ([], []),
        SurfaceType.WALL: ([], []),
        SurfaceType.STRUCTURAL: ([], []),
        SurfaceType.TRIM: ([], []),
    }

    # Overall bounds
    bounds_min: Optional[List[float]] = None
    bounds_max: Optional[List[float]] = None

    def update_bounds(v: Vec3):
        nonlocal bounds_min, bounds_max
        if bounds_min is None:
            bounds_min = [v[0], v[1], v[2]]
            bounds_max = [v[0], v[1], v[2]]
        else:
            bounds_min[0] = min(bounds_min[0], v[0])
            bounds_min[1] = min(bounds_min[1], v[1])
            bounds_min[2] = min(bounds_min[2], v[2])
            bounds_max[0] = max(bounds_max[0], v[0])
            bounds_max[1] = max(bounds_max[1], v[1])
            bounds_max[2] = max(bounds_max[2], v[2])

    for brush in brushes:
        polygons = brush_to_polygons(brush)

        # Build texture to plane lookup
        texture_to_plane = {}
        for plane in brush.planes:
            texture_to_plane[plane.texture] = plane

        for vertices, texture in polygons:
            if len(vertices) < 3:
                continue

            color = _texture_to_color(texture)
            normal = _compute_normal(vertices[0], vertices[1], vertices[2])
            plane = texture_to_plane.get(texture)

            # Classify surface by texture name first, then by normal
            surface_type = classify_surface(normal, texture)

            # Get target lists for this surface type
            target_verts, target_indices = surface_data[surface_type]

            first_idx = len(target_verts)

            # Add vertices
            for v in vertices:
                update_bounds(v)
                uv = _compute_uv(v, normal, plane)
                target_verts.append([
                    v[0], v[1], v[2],
                    normal[0], normal[1], normal[2],
                    uv[0], uv[1],
                    color[0], color[1], color[2]
                ])

            # Add triangles (fan from first vertex)
            for i in range(1, len(vertices) - 1):
                target_indices.append([first_idx, first_idx + i, first_idx + i + 1])

    # Helper to build RenderMesh from lists
    def make_mesh(verts: List, indices: List, bmin, bmax) -> RenderMesh:
        if not verts:
            return RenderMesh(
                vertices=np.array([], dtype=np.float32),
                indices=np.array([], dtype=np.uint32),
                bounds_min=(0.0, 0.0, 0.0),
                bounds_max=(0.0, 0.0, 0.0)
            )
        return RenderMesh(
            vertices=np.array(verts, dtype=np.float32),
            indices=np.array(indices, dtype=np.uint32),
            bounds_min=tuple(bmin) if bmin else (0.0, 0.0, 0.0),
            bounds_max=tuple(bmax) if bmax else (0.0, 0.0, 0.0)
        )

    final_min = tuple(bounds_min) if bounds_min else (0.0, 0.0, 0.0)
    final_max = tuple(bounds_max) if bounds_max else (0.0, 0.0, 0.0)

    # Build meshes for each surface type
    floor_verts, floor_indices = surface_data[SurfaceType.FLOOR]
    ceiling_verts, ceiling_indices = surface_data[SurfaceType.CEILING]
    wall_verts, wall_indices = surface_data[SurfaceType.WALL]
    structural_verts, structural_indices = surface_data[SurfaceType.STRUCTURAL]
    trim_verts, trim_indices = surface_data[SurfaceType.TRIM]

    return SurfaceMeshes(
        floor=make_mesh(floor_verts, floor_indices, bounds_min, bounds_max),
        ceiling=make_mesh(ceiling_verts, ceiling_indices, bounds_min, bounds_max),
        wall=make_mesh(wall_verts, wall_indices, bounds_min, bounds_max),
        structural=make_mesh(structural_verts, structural_indices, bounds_min, bounds_max),
        trim=make_mesh(trim_verts, trim_indices, bounds_min, bounds_max),
        bounds_min=final_min,
        bounds_max=final_max
    )
