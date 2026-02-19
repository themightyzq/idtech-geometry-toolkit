"""
MAP file format writers for idTech 1 and idTech 4.

Each writer serialises entities + brushes into the target format's text
representation.  The writers operate on PlaneGeometry objects from
plane_math.py.
"""

from __future__ import annotations
import io
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional

from quake_levelgenerator.src.conversion.plane_math import PlaneGeometry, Vec3


# ---------------------------------------------------------------
# Texture Path Normalization
# ---------------------------------------------------------------

def normalize_texture_path(texture_path: str, for_format: str = "idtech4") -> str:
    """Normalize texture paths for game-relative format.

    Converts absolute filesystem paths to game-relative paths suitable
    for idTech engines. This handles:
    - Stripping file extensions (idTech 4 uses material names without extensions)
    - Converting absolute paths to relative by detecting common base directories
    - Ensuring forward slashes for path separators

    Common base directory patterns detected:
    - textures/      -> textures/...
    - materials/     -> textures/materials/...
    - /textures/     -> textures/...
    - /base/textures -> textures/...

    Args:
        texture_path: The texture path to normalize (may be absolute or relative)
        for_format: Target format ("idtech1" or "idtech4")

    Returns:
        Normalized game-relative texture path

    Examples:
        >>> normalize_texture_path("/home/user/game/textures/brick.tga", "idtech4")
        "textures/brick"
        >>> normalize_texture_path("textures/materials/stone_d.png", "idtech4")
        "textures/materials/stone_d"
        >>> normalize_texture_path("BRICK1_5", "idtech1")
        "BRICK1_5"
    """
    if not texture_path:
        return texture_path

    # Normalize path separators to forward slashes
    normalized = texture_path.replace("\\", "/")

    # Strip file extension for idTech 4 (material references)
    if for_format == "idtech4":
        # Remove common texture extensions
        extensions = [".tga", ".png", ".jpg", ".jpeg", ".dds", ".bmp"]
        for ext in extensions:
            if normalized.lower().endswith(ext):
                normalized = normalized[:-len(ext)]
                break

    # If it's an absolute path, convert to relative
    if os.path.isabs(texture_path) or normalized.startswith("/"):
        # Look for common texture directory markers
        markers = ["/textures/", "/materials/", "/base/"]
        for marker in markers:
            idx = normalized.lower().find(marker)
            if idx != -1:
                # Start from the marker or just after if it starts with /
                if marker.startswith("/"):
                    normalized = normalized[idx + 1:]  # Skip leading /
                else:
                    normalized = normalized[idx:]
                break
        else:
            # No marker found - just use the filename
            normalized = os.path.basename(normalized)

    # Ensure forward slashes
    normalized = normalized.replace("\\", "/")

    # Remove any leading slashes
    normalized = normalized.lstrip("/")

    return normalized


class MapFormatWriter(ABC):
    """Abstract base for MAP format writers."""

    @abstractmethod
    def format_name(self) -> str: ...

    @abstractmethod
    def write_plane(self, plane: PlaneGeometry) -> str:
        """Return a single-line string representing one brush face."""
        ...

    def write_brush(self, planes: List[PlaneGeometry], brush_index: int = 0) -> str:
        lines = ["{\n"]
        for p in planes:
            lines.append(f"  {self.write_plane(p)}\n")
        lines.append("}\n")
        return "".join(lines)

    def write_entity(
        self,
        classname: str,
        properties: Dict[str, str],
        brushes: List[List[PlaneGeometry]],
        entity_index: int = 0,
    ) -> str:
        lines = ["{\n"]
        lines.append(f'  "classname" "{classname}"\n')
        for k, v in properties.items():
            if k == "classname":
                continue
            lines.append(f'  "{k}" "{v}"\n')
        for i, brush_planes in enumerate(brushes):
            lines.append(self.write_brush(brush_planes, brush_index=i))
        lines.append("}\n")
        return "".join(lines)


class IdTech1Writer(MapFormatWriter):
    """idTech 1 (Quake / Half-Life) 3-point plane format.

    Format per face:
        ( x1 y1 z1 ) ( x2 y2 z2 ) ( x3 y3 z3 ) TEXTURE offX offY rot scX scY
    """

    def format_name(self) -> str:
        return "idtech1"

    def write_plane(self, plane: PlaneGeometry) -> str:
        p1, p2, p3 = plane.to_three_points()

        def _fmt(v: Vec3) -> str:
            return f"( {_rint(v[0])} {_rint(v[1])} {_rint(v[2])} )"

        # Normalize texture path for idTech 1 (strips absolute paths to filename)
        texture = normalize_texture_path(plane.texture, "idtech1")

        return (
            f"{_fmt(p1)} {_fmt(p2)} {_fmt(p3)} "
            f"{texture} "
            f"{int(plane.offset_x)} {int(plane.offset_y)} "
            f"{int(plane.rotation)} "
            f"{plane.scale_x:.6g} {plane.scale_y:.6g}"
        )


class IdTech4Writer(MapFormatWriter):
    """idTech 4 (Doom 3) brushDef3 format.

    Format per face:
        ( nx ny nz dist ) ( ( sx sy ox ) ( tx ty oy ) ) "material/path"
    """

    def format_name(self) -> str:
        return "idtech4"

    def write_plane(self, plane: PlaneGeometry) -> str:
        n, d = plane.to_normal_distance()
        # Doom 3 brushDef3: normals point outward (away from solid).
        # Our internal representation has normals pointing inward (idTech 1
        # convention), so negate the normal to flip the direction.
        #
        # brushDef3 plane equation: n·p + d = 0  (i.e., n·p = -d)
        # Our internal equation: n·p = dist
        #
        # When we negate the normal (n → -n), the internal equation becomes:
        #   (-n)·p = -dist, but brushDef3 uses (-n)·p + d = 0
        #   So: -dist = -d → d = dist
        #
        # Therefore: negate normal, keep distance as-is.
        nx, ny, nz = -n[0] + 0.0, -n[1] + 0.0, -n[2] + 0.0  # +0.0 clears negative zero
        # Snap distance to integer - idTech coordinates should be integers.
        # This eliminates floating-point artifacts like 7.83774e-15 → 0.
        dd = _snap_float(d + 0.0)

        # Normalize texture path for idTech 4 (strips extensions, converts to relative)
        texture = normalize_texture_path(plane.texture, "idtech4")

        return (
            f"( {nx:.6g} {ny:.6g} {nz:.6g} {dd:.6g} ) "
            f"( ( {plane.scale_x:.6g} 0 {plane.offset_x:.6g} ) "
            f"( 0 {plane.scale_y:.6g} {plane.offset_y:.6g} ) ) "
            f'"{texture}"'
        )

    def write_entity(
        self,
        classname: str,
        properties: Dict[str, str],
        brushes: List[List[PlaneGeometry]],
        entity_index: int = 0,
    ) -> str:
        lines = [f"// entity {entity_index}\n", "{\n"]
        lines.append(f'  "classname" "{classname}"\n')
        for k, v in properties.items():
            if k == "classname":
                continue
            lines.append(f'  "{k}" "{v}"\n')
        for i, brush_planes in enumerate(brushes):
            lines.append(self.write_brush(brush_planes, brush_index=i))
        lines.append("}\n")
        return "".join(lines)

    def write_brush(self, planes: List[PlaneGeometry], brush_index: int = 0) -> str:
        lines = [f"  // brush {brush_index}\n", "  {\n", "    brushDef3\n", "    {\n"]
        for p in planes:
            lines.append(f"      {self.write_plane(p)}\n")
        lines.append("    }\n")
        lines.append("  }\n")
        return "".join(lines)


# ---------------------------------------------------------------
# Factory
# ---------------------------------------------------------------

_WRITERS = {
    "idtech1": IdTech1Writer,
    "idtech4": IdTech4Writer,
}


def get_writer(format_name: str) -> MapFormatWriter:
    """Return a writer instance for the given format name.

    Raises ValueError for unknown formats.
    """
    cls = _WRITERS.get(format_name.lower())
    if cls is None:
        raise ValueError(f"Unknown export format '{format_name}'. Available: {list(_WRITERS)}")
    return cls()


# ---------------------------------------------------------------
# Helper
# ---------------------------------------------------------------

def _rint(v: float) -> int:
    """Round to nearest integer for MAP coordinate output."""
    return int(round(v))


def _snap_float(v: float, epsilon: float = 1e-9) -> float:
    """Snap a floating-point value to the nearest integer if close enough.

    Eliminates floating-point artifacts like 7.83774e-15 → 0.
    Values more than epsilon away from an integer are preserved as-is.

    This provides a safety net at the output layer for any coordinates
    that bypassed earlier snapping (e.g., from BSP generator).
    """
    rounded = round(v)
    if abs(v - rounded) < epsilon:
        return float(rounded)
    return v
