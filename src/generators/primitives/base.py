"""
Base classes for geometric primitives.

Each primitive defines a parameter schema and generates a list of Brush
objects that can be written directly to a .map file.
"""

from __future__ import annotations
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

from quake_levelgenerator.src.conversion.map_writer import Brush, Plane

if TYPE_CHECKING:
    from .portal_system import PortalTag, PortalDirection


Vec3 = Tuple[float, float, float]


@dataclass
class PrimitiveParameters:
    """Common parameters shared by all primitives."""
    origin: Vec3 = (0.0, 0.0, 0.0)
    rotation: float = 0.0       # Y-axis rotation in degrees
    scale: float = 1.0
    texture: str = "CRATE1_5"


class GeometricPrimitive(ABC):
    """Abstract base for all geometry primitives."""

    def __init__(self):
        self.params = PrimitiveParameters()
        self._brush_id = 0
        # Surface-specific textures (empty = use self.params.texture)
        self._texture_wall: str = ""
        self._texture_floor: str = ""
        self._texture_ceiling: str = ""
        self._texture_trim: str = ""
        self._texture_structural: str = ""
        # Portal tags generated during last generate() call
        # Tags mark exact portal positions for alignment validation
        self._generated_tags: List['PortalTag'] = []

    # ------------------------------------------------------------------
    # Surface texture properties
    # ------------------------------------------------------------------

    @property
    def texture_wall(self) -> str:
        """Texture for wall surfaces."""
        return self._texture_wall or self.params.texture

    @property
    def texture_floor(self) -> str:
        """Texture for floor surfaces."""
        return self._texture_floor or self.params.texture

    @property
    def texture_ceiling(self) -> str:
        """Texture for ceiling surfaces."""
        return self._texture_ceiling or self.params.texture

    @property
    def texture_trim(self) -> str:
        """Texture for trim/decorative surfaces."""
        return self._texture_trim or self.params.texture

    @property
    def texture_structural(self) -> str:
        """Texture for structural elements (pillars, stairs, arches)."""
        return self._texture_structural or self.params.texture

    # ------------------------------------------------------------------
    # Portal Tag System
    # ------------------------------------------------------------------

    def get_generated_tags(self) -> List['PortalTag']:
        """Return tags generated during last generate() call.

        Tags are created by _register_portal_tag() during generate().
        The layout generator collects these tags to validate alignment.

        Returns:
            List of PortalTag objects in local coordinates
        """
        return self._generated_tags

    def _reset_tags(self) -> None:
        """Reset the tag list. Call at start of generate()."""
        self._generated_tags = []

    def _register_portal_tag(
        self,
        portal_id: str,
        center_x: float,
        center_y: float,
        center_z: float,
        direction: 'PortalDirection',
        width: float = None,
        height: float = None,
        tag_type: str = "portal"
    ) -> None:
        """Register a portal tag at the actual portal position.

        Call this during generate() at each portal position.
        The layout generator will transform these to world space.

        Args:
            portal_id: ID of the portal (entrance, exit, front, back, etc.)
            center_x: X coordinate of portal center (local coords)
            center_y: Y coordinate of portal center (local coords)
            center_z: Z coordinate of portal floor (local coords)
            direction: Which way the portal faces
            width: Portal width (default: PORTAL_WIDTH)
            height: Portal height (default: PORTAL_HEIGHT)
            tag_type: Type of tag ("portal" or "stair_connection")
        """
        # Import here to avoid circular imports
        from .portal_system import PortalTag, PORTAL_WIDTH, PORTAL_HEIGHT

        tag = PortalTag(
            primitive_id="",  # Set by layout generator
            portal_id=portal_id,
            center_x=center_x,
            center_y=center_y,
            center_z=center_z,
            direction=direction,
            width=width if width is not None else PORTAL_WIDTH,
            height=height if height is not None else PORTAL_HEIGHT,
            tag_type=tag_type,
        )
        self._generated_tags.append(tag)

    # ------------------------------------------------------------------
    # Surface-specific brush helpers
    # ------------------------------------------------------------------

    def _floor_box(self, x1: float, y1: float, z1: float,
                   x2: float, y2: float, z2: float) -> 'Brush':
        """Create a floor brush with floor texture."""
        return self._box(x1, y1, z1, x2, y2, z2, texture=self.texture_floor)

    def _ceiling_box(self, x1: float, y1: float, z1: float,
                     x2: float, y2: float, z2: float) -> 'Brush':
        """Create a ceiling brush with ceiling texture."""
        return self._box(x1, y1, z1, x2, y2, z2, texture=self.texture_ceiling)

    def _wall_box(self, x1: float, y1: float, z1: float,
                  x2: float, y2: float, z2: float) -> 'Brush':
        """Create a wall brush with wall texture."""
        return self._box(x1, y1, z1, x2, y2, z2, texture=self.texture_wall)

    def _structural_box(self, x1: float, y1: float, z1: float,
                        x2: float, y2: float, z2: float) -> 'Brush':
        """Create a structural brush with structural texture."""
        return self._box(x1, y1, z1, x2, y2, z2, texture=self.texture_structural)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def generate(self) -> List[Brush]:
        """Generate brushes for this primitive."""
        ...

    @classmethod
    @abstractmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return a dict describing each parameter.

        Each key is a parameter name; value is a dict with:
            type: "float" | "int" | "bool" | "choice"
            default: default value
            min / max: optional numeric range
            choices: list of strings (only for type=="choice")
            label: human-readable label
        """
        ...

    @classmethod
    @abstractmethod
    def get_display_name(cls) -> str: ...

    @classmethod
    @abstractmethod
    def get_category(cls) -> str: ...

    # ------------------------------------------------------------------
    # Parameter application
    # ------------------------------------------------------------------

    def apply_params(self, params: Dict[str, Any]):
        """Apply a dict of parameter values (from the GUI or pipeline)."""
        schema = self.get_parameter_schema()
        for key, value in params.items():
            if key in ("origin", "rotation", "scale", "texture"):
                setattr(self.params, key, value)
            elif key in ("texture_wall", "texture_floor", "texture_ceiling",
                         "texture_trim", "texture_structural"):
                # Set surface-specific textures
                setattr(self, f"_{key}", value)
            elif key in ("portal_target_x", "portal_target_y"):
                # Portal target coordinates for grid-aligned polygonal rooms
                # Stored directly on instance for access by _generate_polygonal_walls
                setattr(self, f"_{key}", value)
            elif key in schema:
                setattr(self, key, value)

    # ------------------------------------------------------------------
    # Brush helpers
    # ------------------------------------------------------------------

    # idTech coordinate snapping tolerance - values within this of an integer
    # are snapped to that integer (eliminates floating point artifacts)
    SNAP_EPSILON = 1e-6

    def _next_id(self) -> int:
        self._brush_id += 1
        return self._brush_id

    def _snap_coord(self, v: float) -> float:
        """Snap a coordinate to the nearest integer.

        idTech engines require integer coordinates for reliable BSP compilation.
        This eliminates floating-point artifacts from trigonometric calculations
        (e.g., 7.83774e-15 → 0, 110.851 → 111).

        All primitive geometry flows through _box() and _wedge(), so snapping
        here ensures ALL generated coordinates are clean integers.
        """
        rounded = round(v)
        # If very close to an integer, snap exactly (avoids 0.999999 → 1)
        if abs(v - rounded) < self.SNAP_EPSILON:
            return float(rounded)
        # Otherwise round to nearest integer
        return float(rounded)

    def _box(self, x1: float, y1: float, z1: float,
             x2: float, y2: float, z2: float,
             texture: str = "") -> Brush:
        """Create an axis-aligned box brush.

        Uses idTech 1 winding convention (inward-facing normals) for
        consistency with brush_generator. The format writer handles
        conversion to idTech 4 outward-facing normals.

        All coordinates are snapped to integers for idTech compatibility.
        Coordinates are normalized so x1 <= x2, y1 <= y2, z1 <= z2 to ensure
        correct plane winding regardless of input order.
        """
        tex = texture or self.params.texture
        # Snap all coordinates to integers for clean idTech geometry
        x1, y1, z1 = self._snap_coord(x1), self._snap_coord(y1), self._snap_coord(z1)
        x2, y2, z2 = self._snap_coord(x2), self._snap_coord(y2), self._snap_coord(z2)

        # Normalize coordinates: ensure min <= max on each axis
        # This fixes inverted brushes that can occur from calculations like
        # stair_cx + stairwell_r > ox + r in Tower's floor cutout logic
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        if z1 > z2:
            z1, z2 = z2, z1

        # idTech 1 convention: normals point inward (toward brush center)
        # This matches brush_generator._create_box_brush() winding order
        planes = [
            Plane((x1, y1, z1), (x1, y2, z1), (x1, y1, z2), tex),  # Left (X=x1)
            Plane((x2, y1, z1), (x2, y1, z2), (x2, y2, z1), tex),  # Right (X=x2)
            Plane((x1, y1, z1), (x1, y1, z2), (x2, y1, z1), tex),  # Front (Y=y1)
            Plane((x1, y2, z1), (x2, y2, z1), (x1, y2, z2), tex),  # Back (Y=y2)
            Plane((x1, y1, z1), (x2, y1, z1), (x1, y2, z1), tex),  # Bottom (Z=z1)
            Plane((x1, y1, z2), (x1, y2, z2), (x2, y1, z2), tex),  # Top (Z=z2)
        ]
        return Brush(planes=planes, brush_id=self._next_id())

    def _wedge(self, x1: float, y1: float, z1: float,
               x2: float, y2: float, z2: float,
               ramp_axis: str = "y", texture: str = "") -> Brush:
        """Create a wedge / ramp brush (triangular cross-section).

        ramp_axis: which horizontal axis ramps upward ("x" or "y").
        The ramp goes from z1 at the low end to z2 at the high end.

        All coordinates are snapped to integers for idTech compatibility.
        """
        tex = texture or self.params.texture
        # Snap all coordinates to integers for clean idTech geometry
        x1, y1, z1 = self._snap_coord(x1), self._snap_coord(y1), self._snap_coord(z1)
        x2, y2, z2 = self._snap_coord(x2), self._snap_coord(y2), self._snap_coord(z2)
        if ramp_axis == "y":
            # Y-axis ramp: slopes from front (y1, z1) to back (y2, z2)
            planes = [
                # Left side (-X): triangle
                Plane((x1, y1, z1), (x1, y2, z2), (x1, y2, z1), tex),
                # Right side (+X): triangle
                Plane((x2, y1, z1), (x2, y2, z1), (x2, y2, z2), tex),
                # Front (-Y): small vertical face at low end
                Plane((x1, y1, z1), (x2, y1, z1), (x2, y1, z2), tex),
                # Back (+Y): full vertical face at high end
                Plane((x2, y2, z1), (x1, y2, z1), (x1, y2, z2), tex),
                # Bottom (-Z): flat horizontal surface
                Plane((x1, y1, z1), (x1, y2, z1), (x2, y1, z1), tex),
                # Slope: ramp surface from front-low to back-high
                Plane((x1, y1, z1), (x2, y1, z1), (x1, y2, z2), tex),
            ]
        else:
            # X-axis ramp: slopes from left (x1, z1) to right (x2, z2)
            planes = [
                # Left side (-X): small vertical face at low end
                Plane((x1, y1, z1), (x1, y2, z1), (x1, y1, z2), tex),
                # Right side (+X): full vertical face at high end
                Plane((x2, y2, z1), (x2, y1, z1), (x2, y1, z2), tex),
                # Front (-Y): triangle
                Plane((x1, y1, z1), (x2, y1, z2), (x2, y1, z1), tex),
                # Back (+Y): triangle
                Plane((x1, y2, z1), (x2, y2, z1), (x2, y2, z2), tex),
                # Bottom (-Z): flat horizontal surface
                Plane((x1, y1, z1), (x2, y1, z1), (x1, y2, z1), tex),
                # Slope: ramp surface from left-low to right-high
                Plane((x1, y1, z1), (x1, y2, z1), (x2, y1, z2), tex),
            ]
        return Brush(planes=planes, brush_id=self._next_id())

    def _radial_segment(
        self,
        cx: float, cy: float, z1: float, z2: float,
        inner_r: float, outer_r: float,
        angle1: float, angle2: float,
        texture: str = ""
    ) -> Brush:
        """Create a radial/pie-slice brush segment for arches and spiral stairs.

        Creates a wedge-shaped brush that spans from angle1 to angle2 around
        center (cx, cy), between inner_r and outer_r radii, from z1 to z2.

        This is essential for creating proper curved geometry like arches,
        spiral stairs, and cylindrical structures.

        Args:
            cx, cy: Center point of the radial geometry
            z1, z2: Bottom and top Z coordinates
            inner_r: Inner radius (0 for solid pie slices)
            outer_r: Outer radius
            angle1, angle2: Start and end angles in radians (0 = +X axis)
            texture: Optional texture override
        """
        tex = texture or self.params.texture

        # Calculate the four corner points
        cos1, sin1 = math.cos(angle1), math.sin(angle1)
        cos2, sin2 = math.cos(angle2), math.sin(angle2)

        # Inner corners
        ix1 = self._snap_coord(cx + cos1 * inner_r)
        iy1 = self._snap_coord(cy + sin1 * inner_r)
        ix2 = self._snap_coord(cx + cos2 * inner_r)
        iy2 = self._snap_coord(cy + sin2 * inner_r)

        # Outer corners
        ox1 = self._snap_coord(cx + cos1 * outer_r)
        oy1 = self._snap_coord(cy + sin1 * outer_r)
        ox2 = self._snap_coord(cx + cos2 * outer_r)
        oy2 = self._snap_coord(cy + sin2 * outer_r)

        z1 = self._snap_coord(z1)
        z2 = self._snap_coord(z2)

        # Build planes for the 6-sided brush
        # Bottom face
        planes = [
            Plane((ix1, iy1, z1), (ox1, oy1, z1), (ix2, iy2, z1), tex),
            # Top face
            Plane((ix1, iy1, z2), (ix2, iy2, z2), (ox1, oy1, z2), tex),
            # Inner face (if inner_r > 0)
            Plane((ix1, iy1, z1), (ix2, iy2, z1), (ix1, iy1, z2), tex),
            # Outer face
            Plane((ox1, oy1, z1), (ox1, oy1, z2), (ox2, oy2, z1), tex),
            # Side face 1 (at angle1)
            Plane((ix1, iy1, z1), (ix1, iy1, z2), (ox1, oy1, z1), tex),
            # Side face 2 (at angle2)
            Plane((ix2, iy2, z1), (ox2, oy2, z1), (ix2, iy2, z2), tex),
        ]
        return Brush(planes=planes, brush_id=self._next_id())

    def _arch_voussoir(
        self,
        cx: float, cy: float, base_z: float,
        inner_r: float, outer_r: float,
        angle1: float, angle2: float,
        depth: float,
        texture: str = ""
    ) -> Brush:
        """Create a voussoir (arch stone) brush for proper arch construction.

        Creates a wedge-shaped brush that forms part of an arch curve.
        The brush spans from angle1 to angle2 around center (cx, base_z),
        with the arch extending in the Y direction by depth.

        This creates brushes in the X-Z plane (vertical arch) extending
        along Y (depth), which is the standard arch orientation.

        Args:
            cx: X center of the arch
            cy: Y position (front of arch)
            base_z: Z position of arch center (spring line)
            inner_r: Inner radius of arch opening
            outer_r: Outer radius (inner_r + thickness)
            angle1, angle2: Start and end angles in radians (0 = right, pi/2 = up)
            depth: How far the arch extends in Y
            texture: Optional texture override
        """
        tex = texture or self.params.texture

        # Calculate corner points in X-Z plane
        cos1, sin1 = math.cos(angle1), math.sin(angle1)
        cos2, sin2 = math.cos(angle2), math.sin(angle2)

        # Inner corners (opening side)
        ix1 = self._snap_coord(cx + cos1 * inner_r)
        iz1 = self._snap_coord(base_z + sin1 * inner_r)
        ix2 = self._snap_coord(cx + cos2 * inner_r)
        iz2 = self._snap_coord(base_z + sin2 * inner_r)

        # Outer corners
        oox1 = self._snap_coord(cx + cos1 * outer_r)
        oz1 = self._snap_coord(base_z + sin1 * outer_r)
        oox2 = self._snap_coord(cx + cos2 * outer_r)
        oz2 = self._snap_coord(base_z + sin2 * outer_r)

        y1 = self._snap_coord(cy)
        y2 = self._snap_coord(cy + depth)

        # 6 faces: front, back, inner curve, outer curve, side1, side2
        planes = [
            # Front face (Y = y1)
            Plane((ix1, y1, iz1), (ix2, y1, iz2), (oox1, y1, oz1), tex),
            # Back face (Y = y2)
            Plane((ix1, y2, iz1), (oox1, y2, oz1), (ix2, y2, iz2), tex),
            # Inner face (curved opening)
            Plane((ix1, y1, iz1), (ix1, y2, iz1), (ix2, y1, iz2), tex),
            # Outer face
            Plane((oox1, y1, oz1), (oox2, y1, oz2), (oox1, y2, oz1), tex),
            # Side 1 (radial at angle1)
            Plane((ix1, y1, iz1), (oox1, y1, oz1), (ix1, y2, iz1), tex),
            # Side 2 (radial at angle2)
            Plane((ix2, y1, iz2), (ix2, y2, iz2), (oox2, y1, oz2), tex),
        ]
        return Brush(planes=planes, brush_id=self._next_id())

    # ------------------------------------------------------------------
    # Polygonal shell generation for configurable room shapes
    # ------------------------------------------------------------------

    def _generate_polygonal_walls(
        self,
        cx: float, cy: float,
        z1: float, z2: float,
        radius: float,
        sides: int,
        wall_thickness: float,
        portal_segment: int = -1,
        portal_width: float = 96,
        portal_height: float = 88,
        texture: str = "",
        portal_target_x: float = None,
        portal_target_y: float = None,
        portal_segments: List[Dict[str, Any]] = None
    ) -> List[Brush]:
        """Generate N-sided polygonal walls around a center point.

        Creates wall segments as trapezoidal prisms forming an N-sided polygon.
        Each wall is a convex brush spanning from the outer radius to inner radius.

        Args:
            cx, cy: Center point of the polygon
            z1, z2: Bottom and top Z coordinates
            radius: Outer radius of the polygon
            sides: Number of sides (3-16)
            wall_thickness: Thickness of walls
            portal_segment: DEPRECATED - Segment index for single portal (-1 for no portal).
                           Use portal_segments for multiple portals.
            portal_width: Width of portal opening (used with portal_segment)
            portal_height: Height of portal opening (used with portal_segment)
            texture: Optional texture override
            portal_target_x: Optional target X position for portal center (grid-aligned).
                           If provided, portal is placed at this X instead of segment midpoint.
            portal_target_y: Optional target Y position for portal center (grid-aligned).
                           If provided, portal is placed at this Y instead of segment midpoint.
            portal_segments: List of portal definitions for multi-portal support.
                           Each dict contains: {'segment': int, 'width': float, 'height': float,
                           'target_x': float (optional), 'target_y': float (optional)}
                           If provided, overrides portal_segment parameter.

        Returns:
            List of wall brushes forming the polygonal shell
        """
        tex = texture or self.params.texture
        brushes: List[Brush] = []
        angle_step = 2 * math.pi / sides

        # Build portal lookup map from portal_segments if provided
        # Otherwise fall back to legacy single-portal support
        portal_map: Dict[int, Dict[str, Any]] = {}
        if portal_segments:
            for p in portal_segments:
                seg = p.get('segment', -1)
                if seg >= 0:
                    portal_map[seg] = p
        elif portal_segment >= 0:
            # Legacy single-portal mode
            portal_map[portal_segment] = {
                'segment': portal_segment,
                'width': portal_width,
                'height': portal_height,
                'target_x': portal_target_x,
                'target_y': portal_target_y
            }

        for i in range(sides):
            angle1 = i * angle_step - math.pi / 2  # Start from -Y (SOUTH)
            angle2 = (i + 1) * angle_step - math.pi / 2

            # Outer corners of this wall segment
            x1_out = self._snap_coord(cx + radius * math.cos(angle1))
            y1_out = self._snap_coord(cy + radius * math.sin(angle1))
            x2_out = self._snap_coord(cx + radius * math.cos(angle2))
            y2_out = self._snap_coord(cy + radius * math.sin(angle2))

            # Inner corners (radius - wall_thickness)
            inner_r = radius - wall_thickness
            x1_in = self._snap_coord(cx + inner_r * math.cos(angle1))
            y1_in = self._snap_coord(cy + inner_r * math.sin(angle1))
            x2_in = self._snap_coord(cx + inner_r * math.cos(angle2))
            y2_in = self._snap_coord(cy + inner_r * math.sin(angle2))

            z1_snap = self._snap_coord(z1)
            z2_snap = self._snap_coord(z2)

            if i in portal_map:
                # Generate portal opening on this segment
                p_info = portal_map[i]
                brushes.extend(self._generate_portal_on_segment(
                    x1_out, y1_out, x2_out, y2_out,
                    x1_in, y1_in, x2_in, y2_in,
                    z1_snap, z2_snap,
                    p_info.get('width', portal_width),
                    p_info.get('height', portal_height),
                    tex,
                    target_x=p_info.get('target_x'),
                    target_y=p_info.get('target_y')
                ))
            else:
                # Create solid wall brush as 8-vertex prism
                brushes.append(self._wall_segment_brush(
                    x1_out, y1_out, x2_out, y2_out,
                    x1_in, y1_in, x2_in, y2_in,
                    z1_snap, z2_snap, tex
                ))

        return brushes

    def _wall_segment_brush(
        self,
        x1_out: float, y1_out: float, x2_out: float, y2_out: float,
        x1_in: float, y1_in: float, x2_in: float, y2_in: float,
        z1: float, z2: float,
        texture: str
    ) -> Brush:
        """Create a wall segment brush as a convex 6-sided prism.

        The wall is a trapezoidal prism with:
        - Outer face: from (x1_out, y1_out) to (x2_out, y2_out)
        - Inner face: from (x1_in, y1_in) to (x2_in, y2_in)
        - Top and bottom horizontal faces
        - Two end faces at the corners
        """
        # For short wall segments where inner and outer nearly coincide,
        # use a simple box approximation
        dx = x2_out - x1_out
        dy = y2_out - y1_out

        # Calculate bounding box for this segment
        min_x = min(x1_out, x2_out, x1_in, x2_in)
        max_x = max(x1_out, x2_out, x1_in, x2_in)
        min_y = min(y1_out, y2_out, y1_in, y2_in)
        max_y = max(y1_out, y2_out, y1_in, y2_in)

        # Use box for axis-aligned or very short segments
        if abs(dx) < 2 or abs(dy) < 2:
            return self._box(min_x, min_y, z1, max_x, max_y, z2, texture)

        # For angled segments, create proper trapezoidal prism
        # 6 planes: outer, inner, left end, right end, bottom, top
        planes = [
            # Bottom face (Z=z1)
            Plane(
                (x1_out, y1_out, z1),
                (x2_out, y2_out, z1),
                (x1_in, y1_in, z1),
                texture
            ),
            # Top face (Z=z2)
            Plane(
                (x1_out, y1_out, z2),
                (x1_in, y1_in, z2),
                (x2_out, y2_out, z2),
                texture
            ),
            # Outer face
            Plane(
                (x1_out, y1_out, z1),
                (x1_out, y1_out, z2),
                (x2_out, y2_out, z1),
                texture
            ),
            # Inner face
            Plane(
                (x1_in, y1_in, z1),
                (x2_in, y2_in, z1),
                (x1_in, y1_in, z2),
                texture
            ),
            # Left end face (at corner 1)
            Plane(
                (x1_out, y1_out, z1),
                (x1_in, y1_in, z1),
                (x1_out, y1_out, z2),
                texture
            ),
            # Right end face (at corner 2)
            Plane(
                (x2_out, y2_out, z1),
                (x2_out, y2_out, z2),
                (x2_in, y2_in, z1),
                texture
            ),
        ]
        return Brush(planes=planes, brush_id=self._next_id())

    def _generate_portal_on_segment(
        self,
        x1_out: float, y1_out: float, x2_out: float, y2_out: float,
        x1_in: float, y1_in: float, x2_in: float, y2_in: float,
        z1: float, z2: float,
        portal_width: float, portal_height: float,
        texture: str,
        target_x: float = None,
        target_y: float = None
    ) -> List[Brush]:
        """Generate a portal opening on an angled wall segment.

        Creates 3 brushes: left piece, right piece, and lintel above portal.

        For simplicity with angled walls, we use bounding boxes for the portal pieces.
        This ensures convex brushes and proper sealing.

        Args:
            x1_out, y1_out: First outer corner of segment
            x2_out, y2_out: Second outer corner of segment
            x1_in, y1_in: First inner corner of segment
            x2_in, y2_in: Second inner corner of segment
            z1, z2: Bottom and top Z coordinates
            portal_width: Width of portal opening
            portal_height: Height of portal opening
            texture: Texture for wall brushes
            target_x: Optional target X position for portal center (grid-aligned).
                     If provided with target_y, overrides segment midpoint calculation.
            target_y: Optional target Y position for portal center (grid-aligned).
                     If provided with target_x, overrides segment midpoint calculation.
        """
        brushes: List[Brush] = []

        # Calculate segment direction and length
        seg_dx = x2_out - x1_out
        seg_dy = y2_out - y1_out
        seg_len = math.sqrt(seg_dx * seg_dx + seg_dy * seg_dy)

        # Calculate portal center position as fraction along segment
        # Use target position if provided (for grid-aligned polygonal rooms)
        if target_x is not None and target_y is not None and seg_len > 0:
            # Project target onto segment to find center fraction
            # Vector from segment start to target
            to_target_x = target_x - x1_out
            to_target_y = target_y - y1_out
            # Dot product to find projection along segment
            dot = (to_target_x * seg_dx + to_target_y * seg_dy) / (seg_len * seg_len)
            center_frac = max(0.1, min(0.9, dot))  # Clamp to valid range
        else:
            center_frac = 0.5  # Default to segment midpoint

        # Portal width as fraction of segment
        portal_fraction = portal_width / seg_len if seg_len > 0 else 0.5
        half_portal = portal_fraction / 2

        # Calculate portal edge points along segment using CENTER fraction
        # Left side of portal
        left_frac = center_frac - half_portal
        p_left_x_out = self._snap_coord(x1_out + seg_dx * left_frac)
        p_left_y_out = self._snap_coord(y1_out + seg_dy * left_frac)
        p_left_x_in = self._snap_coord(x1_in + (x2_in - x1_in) * left_frac)
        p_left_y_in = self._snap_coord(y1_in + (y2_in - y1_in) * left_frac)

        # Right side of portal
        right_frac = center_frac + half_portal
        p_right_x_out = self._snap_coord(x1_out + seg_dx * right_frac)
        p_right_y_out = self._snap_coord(y1_out + seg_dy * right_frac)
        p_right_x_in = self._snap_coord(x1_in + (x2_in - x1_in) * right_frac)
        p_right_y_in = self._snap_coord(y1_in + (y2_in - y1_in) * right_frac)

        portal_top = self._snap_coord(z1 + portal_height)

        # Left piece (from segment start to portal left edge)
        if left_frac > 0.05:
            brushes.append(self._wall_segment_brush(
                x1_out, y1_out, p_left_x_out, p_left_y_out,
                x1_in, y1_in, p_left_x_in, p_left_y_in,
                z1, z2, texture
            ))

        # Right piece (from portal right edge to segment end)
        if right_frac < 0.95:
            brushes.append(self._wall_segment_brush(
                p_right_x_out, p_right_y_out, x2_out, y2_out,
                p_right_x_in, p_right_y_in, x2_in, y2_in,
                z1, z2, texture
            ))

        # Lintel above portal (full segment width at portal location)
        if portal_top < z2:
            brushes.append(self._wall_segment_brush(
                p_left_x_out, p_left_y_out, p_right_x_out, p_right_y_out,
                p_left_x_in, p_left_y_in, p_right_x_in, p_right_y_in,
                portal_top, z2, texture
            ))

        return brushes

    def _generate_polygonal_floor(
        self,
        cx: float, cy: float,
        z: float,
        radius: float,
        sides: int,
        thickness: float,
        is_ceiling: bool = False,
        texture: str = "",
        vestibule_clip_zone: Dict[str, Any] = None,
        vestibule_clip_zones: List[Dict[str, Any]] = None
    ) -> List[Brush]:
        """Generate N-sided polygonal floor or ceiling as a bounding box.

        Creates floor/ceiling as box brush(es) covering the polygon's bounding
        rectangle. When vestibule clip zones are provided, splits the floor/ceiling
        into multiple brushes that avoid the vestibule zones to prevent overlapping
        geometry.

        Uses standard 6-plane box geometry for TrenchBroom compatibility.
        Previous triangular prism approach (5-plane brushes) was not rendered
        correctly by TrenchBroom.

        Args:
            cx, cy: Center point of the polygon
            z: Z coordinate (bottom of floor, top of ceiling)
            radius: Radius of the polygon (extends slightly beyond for sealing)
            sides: Number of sides (3-16)
            thickness: Thickness of floor/ceiling slab
            is_ceiling: If True, slab extends upward from z, else downward
            texture: Optional texture override
            vestibule_clip_zone: Optional single dict (for backward compat) with keys:
                - 'direction': 'SOUTH', 'NORTH', 'EAST', or 'WEST'
                - 'min_x', 'max_x', 'min_y', 'max_y': clip zone bounds
            vestibule_clip_zones: Optional list of clip zone dicts (for multi-portal rooms)
                When provided, floor/ceiling brushes avoid ALL zones.

        Returns:
            List of floor/ceiling box brushes
        """
        tex = texture or self.params.texture
        angle_step = 2 * math.pi / sides

        # Extend radius slightly for overlap with walls
        ext_radius = radius + thickness

        if is_ceiling:
            z1 = self._snap_coord(z)
            z2 = self._snap_coord(z + thickness)
        else:
            z1 = self._snap_coord(z - thickness)
            z2 = self._snap_coord(z)

        # Compute bounding box of the polygon vertices
        min_x = float('inf')
        max_x = float('-inf')
        min_y = float('inf')
        max_y = float('-inf')

        for i in range(sides):
            angle = i * angle_step - math.pi / 2
            x = cx + ext_radius * math.cos(angle)
            y = cy + ext_radius * math.sin(angle)
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

        # Snap to grid
        min_x = self._snap_coord(min_x)
        max_x = self._snap_coord(max_x)
        min_y = self._snap_coord(min_y)
        max_y = self._snap_coord(max_y)

        # Normalize clip zones to a list
        clip_zones = []
        if vestibule_clip_zones:
            clip_zones = [z for z in vestibule_clip_zones if z is not None]
        elif vestibule_clip_zone is not None:
            clip_zones = [vestibule_clip_zone]

        # If no clip zones, return single box brush
        if not clip_zones:
            return [self._box(min_x, min_y, z1, max_x, max_y, z2, tex)]

        # Use the multi-zone clipping approach
        return self._generate_clipped_floor_brushes(
            min_x, min_y, max_x, max_y, z1, z2, clip_zones, tex
        )

    def _generate_clipped_floor_brushes(
        self,
        floor_min_x: float, floor_min_y: float,
        floor_max_x: float, floor_max_y: float,
        z1: float, z2: float,
        clip_zones: List[Dict[str, Any]],
        texture: str
    ) -> List[Brush]:
        """Generate floor brushes with multiple clip zones carved out.

        Uses a cell decomposition approach: the floor bounding box is divided
        into a grid based on the clip zone boundaries, then cells that do NOT
        overlap any clip zone are converted to brushes.

        Args:
            floor_min_x, floor_min_y: Floor bounding box minimum
            floor_max_x, floor_max_y: Floor bounding box maximum
            z1, z2: Z coordinates of floor slab
            clip_zones: List of clip zone dicts with 'min_x', 'max_x', 'min_y', 'max_y'
            texture: Texture for floor brushes

        Returns:
            List of floor box brushes that avoid all clip zones
        """
        min_dim = 8  # Minimum brush dimension

        # Collect all X and Y boundaries from floor and clip zones
        x_boundaries = {floor_min_x, floor_max_x}
        y_boundaries = {floor_min_y, floor_max_y}

        for zone in clip_zones:
            x_boundaries.add(self._snap_coord(zone.get('min_x', floor_min_x)))
            x_boundaries.add(self._snap_coord(zone.get('max_x', floor_max_x)))
            y_boundaries.add(self._snap_coord(zone.get('min_y', floor_min_y)))
            y_boundaries.add(self._snap_coord(zone.get('max_y', floor_max_y)))

        # Sort boundaries
        x_sorted = sorted(x_boundaries)
        y_sorted = sorted(y_boundaries)

        brushes: List[Brush] = []

        # Generate brushes for each cell that doesn't overlap any clip zone
        for i in range(len(x_sorted) - 1):
            for j in range(len(y_sorted) - 1):
                cell_min_x = x_sorted[i]
                cell_max_x = x_sorted[i + 1]
                cell_min_y = y_sorted[j]
                cell_max_y = y_sorted[j + 1]

                # Skip cells that are too small
                if cell_max_x - cell_min_x < min_dim or cell_max_y - cell_min_y < min_dim:
                    continue

                # Check if cell overlaps any clip zone
                overlaps_clip = False
                for zone in clip_zones:
                    zone_min_x = self._snap_coord(zone.get('min_x', floor_min_x))
                    zone_max_x = self._snap_coord(zone.get('max_x', floor_max_x))
                    zone_min_y = self._snap_coord(zone.get('min_y', floor_min_y))
                    zone_max_y = self._snap_coord(zone.get('max_y', floor_max_y))

                    # Check overlap (cell center inside zone OR cell fully inside zone)
                    cell_cx = (cell_min_x + cell_max_x) / 2
                    cell_cy = (cell_min_y + cell_max_y) / 2
                    if (zone_min_x <= cell_cx <= zone_max_x and
                        zone_min_y <= cell_cy <= zone_max_y):
                        overlaps_clip = True
                        break

                if not overlaps_clip:
                    brushes.append(self._box(
                        cell_min_x, cell_min_y, z1,
                        cell_max_x, cell_max_y, z2,
                        texture
                    ))

        # Fallback: if no brushes generated, create the full floor
        if not brushes:
            brushes.append(self._box(floor_min_x, floor_min_y, z1,
                                     floor_max_x, floor_max_y, z2, texture))

        return brushes

    def _find_portal_segment(self, sides: int, direction: str) -> int:
        """Find wall segment index closest to the specified direction.

        Args:
            sides: Number of polygon sides
            direction: Cardinal direction (NORTH, SOUTH, EAST, WEST)

        Returns:
            Segment index (0 to sides-1) facing closest to the specified direction
        """
        # Target angles for each direction (in degrees, 0 = +X/EAST)
        target_angles = {
            'SOUTH': 270,  # -Y
            'NORTH': 90,   # +Y
            'EAST': 0,     # +X
            'WEST': 180,   # -X
        }
        target = target_angles.get(direction.upper(), 270)

        angle_step = 360.0 / sides
        best_segment = 0
        best_diff = 360

        for i in range(sides):
            # Segment center angle (midpoint of the segment arc)
            # Segments start from -90° (SOUTH) to match _generate_polygonal_walls()
            # which uses: angle = i * angle_step - math.pi / 2
            segment_angle = (i * angle_step - 90) % 360
            diff = abs(segment_angle - target)
            if diff > 180:
                diff = 360 - diff
            if diff < best_diff:
                best_diff = diff
                best_segment = i

        return best_segment

    def _get_polygon_min_radius(self, sides: int) -> float:
        """Get minimum radius for a given number of sides to avoid degenerate geometry.

        Higher side counts need larger radii to ensure vertex separation after
        integer snapping.
        """
        # Minimum radius scales with side count
        # At least 8 units per vertex separation
        return max(64, sides * 8)

    def _get_polygon_segment_midpoint(
        self,
        cx: float, cy: float,
        radius: float,
        sides: int,
        segment_index: int
    ) -> Tuple[float, float]:
        """Calculate the midpoint of a polygon wall segment (chord midpoint).

        This is the correct position for portal placement on polygonal walls.
        The chord midpoint is the average of the two corner vertex positions,
        NOT a point on the circumscribed circle at the midpoint angle.

        CRITICAL: Portal geometry is generated at the segment chord midpoint
        (see _generate_polygonal_walls lines 660-661). Portal tags MUST use
        the same calculation for alignment to work.

        Args:
            cx, cy: Center of polygon
            radius: Radius of polygon (to outer wall surface)
            sides: Number of polygon sides
            segment_index: Which segment (0 to sides-1), starting from SOUTH

        Returns:
            (x, y) tuple of the segment chord midpoint
        """
        angle_step = 2 * math.pi / sides
        # Segments start from -90 degrees (SOUTH) to match wall generation
        angle1 = segment_index * angle_step - math.pi / 2
        angle2 = (segment_index + 1) * angle_step - math.pi / 2

        # Chord midpoint is average of corner positions
        # NOT: radius * cos((angle1 + angle2) / 2) which gives a different point
        mid_x = cx + radius * (math.cos(angle1) + math.cos(angle2)) / 2
        mid_y = cy + radius * (math.sin(angle1) + math.sin(angle2)) / 2

        return mid_x, mid_y

    def _set_portal_target_for_polygonal(
        self,
        cx: float, cy: float,
        radius: float,
        sides: int,
        portal_segment: int
    ) -> None:
        """Set portal target coordinates for grid-aligned portal placement.

        For polygonal rooms, the layout system expects portals at grid-aligned
        positions (room center X = cx). This method sets the target so that
        _generate_portal_on_segment projects the portal to be centered at X=cx
        on the wall segment.

        IMPORTANT: Portal tags must also use (cx, ...) for the center_x to
        ensure geometry and tags align. Don't use chord midpoint - use room
        center coordinates.

        Args:
            cx, cy: Center of polygon (room origin X = cx for grid alignment)
            radius: Radius of polygon
            sides: Number of polygon sides
            portal_segment: Which segment has the portal (0 = SOUTH)
        """
        if portal_segment < 0:
            # No portal, clear targets
            self._portal_target_x = None
            self._portal_target_y = None
            return

        # Use room center X for grid alignment
        # The Y coordinate uses the segment's approximate position
        angle_step = 2 * math.pi / sides
        segment_center_angle = (portal_segment + 0.5) * angle_step - math.pi / 2

        self._portal_target_x = cx
        self._portal_target_y = cy + radius * math.sin(segment_center_angle)

    def _generate_portal_vestibule(
        self,
        cx: float, cy: float,
        oz: float,
        polygon_inner_edge: float,
        footprint_edge: float,
        portal_width: float,
        portal_height: float,
        wall_thickness: float,
        direction: str,
        has_portal: bool = True,
        room_height: float = 128.0,
        texture: str = ""
    ) -> List[Brush]:
        """Generate rectangular vestibule between polygon and footprint edge.

        The vestibule approach solves the polygonal portal alignment problem:
        - Polygon walls stop short of the footprint boundary
        - A rectangular vestibule bridges the gap with axis-aligned walls
        - The portal is cut in the axis-aligned vestibule wall at the footprint edge
        - This ensures clean grid-aligned connections to halls

        Creates:
        - Floor extending from polygon to footprint edge (sealed with t overlap)
        - Ceiling same extent
        - Two side walls (perpendicular to direction)
        - Back wall at footprint edge (with portal if has_portal)
        - NO wall at polygon side (open to polygon interior)

        Args:
            cx, cy: Center point of the room (used for portal centering)
            oz: Floor Z coordinate
            polygon_inner_edge: Where the polygon wall interior edge stops
                               (e.g., cy - inner_radius for SOUTH)
            footprint_edge: Where the footprint boundary is (hall connects here)
            portal_width: Width of portal opening
            portal_height: Height of portal opening
            wall_thickness: Wall thickness (t) for sealed geometry
            direction: Cardinal direction ('SOUTH', 'NORTH', 'EAST', 'WEST')
            has_portal: Whether to cut a portal in the back wall
            room_height: Full room height (for walls above portal)
            texture: Optional texture override

        Returns:
            List of brushes forming the sealed vestibule corridor
        """
        brushes: List[Brush] = []
        tex = texture or getattr(self, 'texture_wall', self.params.texture)
        floor_tex = getattr(self, 'texture_floor', tex)
        ceiling_tex = getattr(self, 'texture_ceiling', tex)
        t = wall_thickness
        hw = portal_width / 2
        ph = portal_height
        h = room_height

        direction = direction.upper()

        if direction == 'SOUTH':
            # Vestibule extends from polygon (north) to footprint edge (south)
            # Portal faces SOUTH (player exits room going south)
            vest_min_x = self._snap_coord(cx - hw - t)
            vest_max_x = self._snap_coord(cx + hw + t)
            vest_min_y = self._snap_coord(footprint_edge)  # South edge (footprint boundary)
            vest_max_y = self._snap_coord(polygon_inner_edge)  # North edge (polygon interior)

            # Floor - extends outward at footprint edge, extends INTO polygon zone
            # to seal with angled polygon walls (overlapping brushes OK in idTech)
            brushes.append(self._box(
                vest_min_x - t, vest_min_y - t, oz - t,
                vest_max_x + t, vest_max_y + t, oz,  # Extend into polygon zone
                floor_tex
            ))

            # Ceiling - same pattern as floor
            brushes.append(self._box(
                vest_min_x - t, vest_min_y - t, oz + h,
                vest_max_x + t, vest_max_y + t, oz + h + t,  # Extend into polygon zone
                ceiling_tex
            ))

            # West wall - extend INTO polygon wall zone to seal junction
            # idTech handles overlapping brushes correctly
            brushes.append(self._box(
                vest_min_x - t, vest_min_y - t, oz,
                vest_min_x, vest_max_y + t, oz + h,  # Extend into polygon zone
                tex
            ))

            # East wall - same pattern
            brushes.append(self._box(
                vest_max_x, vest_min_y - t, oz,
                vest_max_x + t, vest_max_y + t, oz + h,  # Extend into polygon zone
                tex
            ))

            # South wall with portal (at footprint edge)
            if has_portal:
                # 3-piece wall: left, right, lintel
                portal_left = self._snap_coord(cx - hw)
                portal_right = self._snap_coord(cx + hw)

                # Left piece - wall thickness extends inward from footprint edge
                brushes.append(self._box(
                    vest_min_x - t, vest_min_y, oz,
                    portal_left, vest_min_y + t, oz + h,
                    tex
                ))
                # Right piece
                brushes.append(self._box(
                    portal_right, vest_min_y, oz,
                    vest_max_x + t, vest_min_y + t, oz + h,
                    tex
                ))
                # Lintel above portal
                if ph < h:
                    brushes.append(self._box(
                        portal_left, vest_min_y, oz + ph,
                        portal_right, vest_min_y + t, oz + h,
                        tex
                    ))
            else:
                # Solid wall
                brushes.append(self._box(
                    vest_min_x - t, vest_min_y, oz,
                    vest_max_x + t, vest_min_y + t, oz + h,
                    tex
                ))

            # NO north wall - open to polygon interior

        elif direction == 'NORTH':
            # Vestibule extends from polygon (south) to footprint edge (north)
            vest_min_x = self._snap_coord(cx - hw - t)
            vest_max_x = self._snap_coord(cx + hw + t)
            vest_min_y = self._snap_coord(polygon_inner_edge)  # South edge (polygon interior)
            vest_max_y = self._snap_coord(footprint_edge)  # North edge (footprint boundary)

            # Floor - extend INTO polygon zone on south side to seal junction
            brushes.append(self._box(
                vest_min_x - t, vest_min_y - t, oz - t,  # Extend into polygon zone
                vest_max_x + t, vest_max_y + t, oz,
                floor_tex
            ))

            # Ceiling
            brushes.append(self._box(
                vest_min_x - t, vest_min_y - t, oz + h,  # Extend into polygon zone
                vest_max_x + t, vest_max_y + t, oz + h + t,
                ceiling_tex
            ))

            # West wall - extend INTO polygon wall zone to seal junction
            brushes.append(self._box(
                vest_min_x - t, vest_min_y - t, oz,  # Extend into polygon zone
                vest_min_x, vest_max_y + t, oz + h,
                tex
            ))

            # East wall
            brushes.append(self._box(
                vest_max_x, vest_min_y - t, oz,  # Extend into polygon zone
                vest_max_x + t, vest_max_y + t, oz + h,
                tex
            ))

            # North wall with portal (at footprint edge)
            if has_portal:
                portal_left = self._snap_coord(cx - hw)
                portal_right = self._snap_coord(cx + hw)

                # Left piece - wall extends inward from footprint
                brushes.append(self._box(
                    vest_min_x - t, vest_max_y - t, oz,
                    portal_left, vest_max_y, oz + h,
                    tex
                ))
                # Right piece
                brushes.append(self._box(
                    portal_right, vest_max_y - t, oz,
                    vest_max_x + t, vest_max_y, oz + h,
                    tex
                ))
                # Lintel
                if ph < h:
                    brushes.append(self._box(
                        portal_left, vest_max_y - t, oz + ph,
                        portal_right, vest_max_y, oz + h,
                        tex
                    ))
            else:
                brushes.append(self._box(
                    vest_min_x - t, vest_max_y - t, oz,
                    vest_max_x + t, vest_max_y, oz + h,
                    tex
                ))

        elif direction == 'WEST':
            # Vestibule extends from polygon (east) to footprint edge (west)
            vest_min_x = self._snap_coord(footprint_edge)  # West edge (footprint boundary)
            vest_max_x = self._snap_coord(polygon_inner_edge)  # East edge (polygon interior)
            vest_min_y = self._snap_coord(cy - hw - t)
            vest_max_y = self._snap_coord(cy + hw + t)

            # Floor - extend INTO polygon zone on east side
            brushes.append(self._box(
                vest_min_x - t, vest_min_y - t, oz - t,
                vest_max_x + t, vest_max_y + t, oz,  # Extend into polygon zone
                floor_tex
            ))

            # Ceiling - extend INTO polygon zone
            brushes.append(self._box(
                vest_min_x - t, vest_min_y - t, oz + h,
                vest_max_x + t, vest_max_y + t, oz + h + t,  # Extend into polygon zone
                ceiling_tex
            ))

            # South wall - extend INTO polygon zone
            brushes.append(self._box(
                vest_min_x - t, vest_min_y - t, oz,
                vest_max_x + t, vest_min_y, oz + h,  # Extend into polygon zone
                tex
            ))

            # North wall - extend INTO polygon zone
            brushes.append(self._box(
                vest_min_x - t, vest_max_y, oz,
                vest_max_x + t, vest_max_y + t, oz + h,  # Extend into polygon zone
                tex
            ))

            # West wall with portal - extends inward from footprint
            if has_portal:
                portal_bottom = self._snap_coord(cy - hw)
                portal_top = self._snap_coord(cy + hw)

                # Bottom piece
                brushes.append(self._box(
                    vest_min_x, vest_min_y - t, oz,
                    vest_min_x + t, portal_bottom, oz + h,
                    tex
                ))
                # Top piece
                brushes.append(self._box(
                    vest_min_x, portal_top, oz,
                    vest_min_x + t, vest_max_y + t, oz + h,
                    tex
                ))
                # Lintel (above portal height)
                if ph < h:
                    brushes.append(self._box(
                        vest_min_x, portal_bottom, oz + ph,
                        vest_min_x + t, portal_top, oz + h,
                        tex
                    ))
            else:
                brushes.append(self._box(
                    vest_min_x, vest_min_y - t, oz,
                    vest_min_x + t, vest_max_y + t, oz + h,
                    tex
                ))

        else:  # EAST
            # Vestibule extends from polygon (west) to footprint edge (east)
            vest_min_x = self._snap_coord(polygon_inner_edge)  # West edge (polygon interior)
            vest_max_x = self._snap_coord(footprint_edge)  # East edge (footprint boundary)
            vest_min_y = self._snap_coord(cy - hw - t)
            vest_max_y = self._snap_coord(cy + hw + t)

            # Floor - extend INTO polygon zone on west side
            brushes.append(self._box(
                vest_min_x - t, vest_min_y - t, oz - t,  # Extend into polygon zone
                vest_max_x + t, vest_max_y + t, oz,
                floor_tex
            ))

            # Ceiling - extend INTO polygon zone
            brushes.append(self._box(
                vest_min_x - t, vest_min_y - t, oz + h,  # Extend into polygon zone
                vest_max_x + t, vest_max_y + t, oz + h + t,
                ceiling_tex
            ))

            # South wall - extend INTO polygon zone
            brushes.append(self._box(
                vest_min_x - t, vest_min_y - t, oz,  # Extend into polygon zone
                vest_max_x + t, vest_min_y, oz + h,
                tex
            ))

            # North wall - extend INTO polygon zone
            brushes.append(self._box(
                vest_min_x - t, vest_max_y, oz,  # Extend into polygon zone
                vest_max_x + t, vest_max_y + t, oz + h,
                tex
            ))

            # East wall with portal - extends inward from footprint
            if has_portal:
                portal_bottom = self._snap_coord(cy - hw)
                portal_top = self._snap_coord(cy + hw)

                # Bottom piece
                brushes.append(self._box(
                    vest_max_x - t, vest_min_y - t, oz,
                    vest_max_x, portal_bottom, oz + h,
                    tex
                ))
                # Top piece
                brushes.append(self._box(
                    vest_max_x - t, portal_top, oz,
                    vest_max_x, vest_max_y + t, oz + h,
                    tex
                ))
                # Lintel
                if ph < h:
                    brushes.append(self._box(
                        vest_max_x - t, portal_bottom, oz + ph,
                        vest_max_x, portal_top, oz + h,
                        tex
                    ))
            else:
                brushes.append(self._box(
                    vest_max_x - t, vest_min_y - t, oz,
                    vest_max_x, vest_max_y + t, oz + h,
                    tex
                ))

        return brushes

    def _get_vestibule_params(
        self,
        cx: float, cy: float,
        radius: float,
        sides: int,
        portal_segment: int,
        wall_thickness: float,
        room_origin: Tuple[float, float, float] = None,
        room_length: float = None,
        room_width: float = None
    ) -> Tuple[float, float, str]:
        """Calculate vestibule parameters from polygon geometry.

        Returns the polygon inner edge position, footprint edge position,
        and direction string needed for _generate_portal_vestibule().

        Args:
            cx, cy: Center of polygon
            radius: Outer radius of polygon
            sides: Number of polygon sides
            portal_segment: Which segment has the portal
            wall_thickness: Wall thickness
            room_origin: Optional (ox, oy, oz) tuple for accurate footprint edge calc
            room_length: Optional room length (Y extent) for accurate footprint edge
            room_width: Optional room width (X extent) for accurate footprint edge

        Returns:
            (polygon_inner_edge, footprint_edge, direction) tuple
        """
        t = wall_thickness

        # Determine portal direction from segment angle
        angle_step = 2 * math.pi / sides
        segment_center_angle = (portal_segment + 0.5) * angle_step - math.pi / 2
        angle_deg = (math.degrees(segment_center_angle) + 360) % 360

        # Calculate inner radius (where polygon walls end on inside)
        inner_radius = radius - t

        # Compute footprint edge based on room origin if provided
        # Otherwise fall back to polygon-relative calculation
        if room_origin is not None:
            ox, oy, oz = room_origin
            # Use actual room boundaries for footprint edge
            # The vestibule should extend to the room's edge (with wall overlap)
            if 225 <= angle_deg <= 315:
                # SOUTH facing - footprint edge at room's south cell boundary
                polygon_inner_edge = cy - inner_radius
                footprint_edge = oy  # Exactly at cell boundary (hall provides overlap)
                direction = 'SOUTH'
            elif 45 <= angle_deg <= 135:
                # NORTH facing - footprint edge at room's north cell boundary
                polygon_inner_edge = cy + inner_radius
                # North edge = oy + room_length
                if room_length is not None:
                    footprint_edge = oy + room_length  # Exactly at cell boundary
                else:
                    # Fallback: derive from polygon center (cy = oy + length/2)
                    footprint_edge = 2 * cy - oy
                direction = 'NORTH'
            elif 135 < angle_deg < 225:
                # WEST facing - footprint edge at room's west cell boundary
                polygon_inner_edge = cx - inner_radius
                # West edge = ox - half_width
                if room_width is not None:
                    footprint_edge = ox - room_width / 2  # Exactly at cell boundary
                else:
                    footprint_edge = cx - radius - t  # Fallback
                direction = 'WEST'
            else:
                # EAST facing - footprint edge at room's east cell boundary
                polygon_inner_edge = cx + inner_radius
                if room_width is not None:
                    footprint_edge = ox + room_width / 2  # Exactly at cell boundary
                else:
                    footprint_edge = cx + radius + t  # Fallback
                direction = 'EAST'
        else:
            # Legacy fallback: polygon-relative calculation
            # This may not align perfectly with footprint boundaries
            footprint_margin = radius + t * 2

            if 225 <= angle_deg <= 315:
                # SOUTH facing
                polygon_inner_edge = cy - inner_radius
                footprint_edge = cy - footprint_margin
                direction = 'SOUTH'
            elif 45 <= angle_deg <= 135:
                # NORTH facing
                polygon_inner_edge = cy + inner_radius
                footprint_edge = cy + footprint_margin
                direction = 'NORTH'
            elif 135 < angle_deg < 225:
                # WEST facing
                polygon_inner_edge = cx - inner_radius
                footprint_edge = cx - footprint_margin
                direction = 'WEST'
            else:
                # EAST facing
                polygon_inner_edge = cx + inner_radius
                footprint_edge = cx + footprint_margin
                direction = 'EAST'

        return polygon_inner_edge, footprint_edge, direction

    def _compute_vestibule_clip_zone(
        self,
        cx: float, cy: float,
        radius: float,
        sides: int,
        portal_segment: int,
        portal_width: float,
        wall_thickness: float,
        room_origin: Tuple[float, float, float] = None,
        room_length: float = None,
        room_width: float = None
    ) -> Dict[str, Any]:
        """Compute the vestibule clip zone for floor/ceiling generation.

        When a polygonal room has a vestibule connector, this method calculates
        the rectangular zone that the vestibule occupies, so the polygon's
        floor/ceiling generation can exclude this zone (avoiding overlapping
        brushes).

        Args:
            cx, cy: Center of polygon
            radius: Outer radius of polygon
            sides: Number of polygon sides
            portal_segment: Which segment has the portal (-1 if no portal)
            portal_width: Width of portal opening
            wall_thickness: Wall thickness (t)
            room_origin: Optional (ox, oy, oz) for footprint edge calculation
            room_length: Optional room length (Y extent)
            room_width: Optional room width (X extent)

        Returns:
            Dict with keys: 'direction', 'min_x', 'max_x', 'min_y', 'max_y'
            Returns None if no portal segment or vestibule would be too small.
        """
        if portal_segment < 0:
            return None

        t = wall_thickness
        hw = portal_width / 2

        # Get vestibule parameters
        polygon_inner_edge, footprint_edge, direction = self._get_vestibule_params(
            cx, cy, radius, sides, portal_segment, t,
            room_origin=room_origin,
            room_length=room_length,
            room_width=room_width
        )

        # Check if vestibule would have sufficient depth
        min_depth = 8.0
        if direction in ('SOUTH', 'WEST'):
            if polygon_inner_edge <= footprint_edge + min_depth:
                return None
        else:  # NORTH, EAST
            if polygon_inner_edge >= footprint_edge - min_depth:
                return None

        # Calculate clip zone bounds based on direction
        # The clip zone MUST match the actual vestibule floor/ceiling footprint exactly
        # Vestibule floor uses: vest_min_x - t to vest_max_x + t (where vest = cx ± (hw + t))
        # So the vestibule floor X extent is: cx - hw - 2t to cx + hw + 2t
        if direction == 'SOUTH':
            # Vestibule extends from footprint_edge (south) to polygon_inner_edge (north)
            # Floor X width includes extra t for side wall overlap on each side
            clip_min_x = self._snap_coord(cx - hw - t - t)  # vest_min_x - t
            clip_max_x = self._snap_coord(cx + hw + t + t)  # vest_max_x + t
            clip_min_y = self._snap_coord(footprint_edge - t)
            clip_max_y = self._snap_coord(polygon_inner_edge)

        elif direction == 'NORTH':
            # Vestibule extends from polygon_inner_edge (south) to footprint_edge (north)
            clip_min_x = self._snap_coord(cx - hw - t - t)  # vest_min_x - t
            clip_max_x = self._snap_coord(cx + hw + t + t)  # vest_max_x + t
            clip_min_y = self._snap_coord(polygon_inner_edge)
            clip_max_y = self._snap_coord(footprint_edge + t)

        elif direction == 'WEST':
            # Vestibule extends from footprint_edge (west) to polygon_inner_edge (east)
            clip_min_x = self._snap_coord(footprint_edge - t)
            clip_max_x = self._snap_coord(polygon_inner_edge)
            clip_min_y = self._snap_coord(cy - hw - t - t)  # vest_min_y - t
            clip_max_y = self._snap_coord(cy + hw + t + t)  # vest_max_y + t

        else:  # EAST
            # Vestibule extends from polygon_inner_edge (west) to footprint_edge (east)
            clip_min_x = self._snap_coord(polygon_inner_edge)
            clip_max_x = self._snap_coord(footprint_edge + t)
            clip_min_y = self._snap_coord(cy - hw - t - t)  # vest_min_y - t
            clip_max_y = self._snap_coord(cy + hw + t + t)  # vest_max_y + t

        return {
            'direction': direction,
            'min_x': clip_min_x,
            'max_x': clip_max_x,
            'min_y': clip_min_y,
            'max_y': clip_max_y
        }

    def _generate_polygonal_portal_connector(
        self,
        cx: float, cy: float,
        oz: float,
        radius: float,
        sides: int,
        portal_segment: int,
        portal_width: float,
        portal_height: float,
        wall_thickness: float,
        texture: str = "",
        room_height: float = None,
        room_origin: Tuple[float, float, float] = None,
        room_length: float = None,
        room_width: float = None
    ) -> List[Brush]:
        """Generate vestibule corridor from polygon interior to footprint edge.

        The vestibule approach:
        - Creates a rectangular corridor with axis-aligned walls
        - Polygon walls don't need to reach the footprint boundary
        - Portal is cut in the axis-aligned vestibule wall
        - Ensures clean grid-aligned connections to halls

        Args:
            cx, cy: Center point of polygon (room center)
            oz: Floor Z coordinate
            radius: Outer radius of polygon
            sides: Number of polygon sides
            portal_segment: Which segment has the portal
            portal_width: Width of portal opening
            portal_height: Height of portal opening
            wall_thickness: Wall thickness for vestibule walls
            texture: Texture for vestibule brushes
            room_height: Full room height (defaults to portal_height + 40)
            room_origin: Optional (ox, oy, oz) for accurate footprint edge calculation
            room_length: Optional room length (Y extent) for accurate footprint edge
            room_width: Optional room width (X extent) for accurate footprint edge

        Returns:
            List of brushes forming the vestibule corridor
        """
        if portal_segment < 0:
            return []

        # Get vestibule parameters with room origin for accurate footprint alignment
        polygon_inner_edge, footprint_edge, direction = self._get_vestibule_params(
            cx, cy, radius, sides, portal_segment, wall_thickness,
            room_origin=room_origin,
            room_length=room_length,
            room_width=room_width
        )

        # Check if vestibule would have positive depth
        # If the polygon already extends to/past the footprint boundary, skip vestibule
        # This can happen with large polygons (high shell_sides) in small footprints
        min_depth = 8.0  # Minimum vestibule depth to be meaningful
        if direction in ('SOUTH', 'WEST'):
            # For SOUTH/WEST: polygon_inner_edge should be > footprint_edge + min_depth
            if polygon_inner_edge <= footprint_edge + min_depth:
                return []
        else:  # NORTH, EAST
            # For NORTH/EAST: polygon_inner_edge should be < footprint_edge - min_depth
            if polygon_inner_edge >= footprint_edge - min_depth:
                return []

        # Default room height if not specified
        if room_height is None:
            room_height = portal_height + 40

        return self._generate_portal_vestibule(
            cx=cx, cy=cy,
            oz=oz,
            polygon_inner_edge=polygon_inner_edge,
            footprint_edge=footprint_edge,
            portal_width=portal_width,
            portal_height=portal_height,
            wall_thickness=wall_thickness,
            direction=direction,
            has_portal=True,
            room_height=room_height,
            texture=texture
        )

    # ------------------------------------------------------------------
    # Polygonal solid generation for pillars and columns
    # ------------------------------------------------------------------

    def _calculate_taper_radius(
        self,
        base_radius: float,
        t: float,
        taper: float,
        curve: str = "linear"
    ) -> float:
        """Calculate the radius at height t with taper curve.

        Args:
            base_radius: Radius at the bottom of the pillar
            t: Normalized height (0.0 = bottom, 1.0 = top)
            taper: Taper amount (0.0 = no taper, 0.8 = 80% narrower at top)
            curve: Taper curve type: "linear", "concave", "convex", "entasis"

        Returns:
            Radius at the specified height
        """
        if curve == "linear":
            factor = 1.0 - taper * t
        elif curve == "concave":
            # Slower taper at bottom, faster at top (quadratic)
            factor = 1.0 - taper * (t * t)
        elif curve == "convex":
            # Faster taper at bottom, slower at top (square root)
            factor = 1.0 - taper * math.sqrt(t)
        elif curve == "entasis":
            # Greek column bulge - outward curve at middle (15% bulge for visibility)
            bulge = 0.15 * math.sin(t * math.pi)
            factor = 1.0 - taper * t + bulge
        else:
            factor = 1.0 - taper * t

        return base_radius * max(0.1, factor)

    def _generate_polygonal_solid(
        self,
        cx: float, cy: float, z1: float, z2: float,
        radius: float,
        sides: int,
        texture: str = ""
    ) -> List[Brush]:
        """Generate an N-sided polygonal solid prism.

        Creates a solid pillar/column with the specified number of sides.

        Args:
            cx, cy: Center point of the polygon
            z1, z2: Bottom and top Z coordinates
            radius: Radius of the polygon (to vertex)
            sides: Number of sides (3-16)
            texture: Optional texture override

        Returns:
            List of brushes forming the solid
        """
        tex = texture or self.params.texture
        brushes: List[Brush] = []

        if sides == 4:
            # Square: use simple box for efficiency
            hw = radius * 0.707  # cos(45°) for inscribed square
            hw = self._snap_coord(hw)
            brushes.append(self._box(cx - hw, cy - hw, z1, cx + hw, cy + hw, z2, tex))
        elif sides == 3:
            # Triangle: use proper 5-face prism
            brushes.append(self._generate_triangular_prism(cx, cy, z1, z2, radius, tex))
        elif sides <= 6:
            # Pentagon/Hexagon: use 2 overlapping boxes
            hw = radius * 0.707
            hw2 = radius * 0.5
            brushes.append(self._box(cx - hw, cy - hw2, z1, cx + hw, cy + hw2, z2, tex))
            brushes.append(self._box(cx - hw2, cy - hw, z1, cx + hw2, cy + hw, z2, tex))
        elif sides <= 8:
            # 7-8 sides: use 2 overlapping boxes for octagonal appearance
            hw = radius * 0.707
            hw2 = radius * 0.6
            brushes.append(self._box(cx - hw, cy - hw2, z1, cx + hw, cy + hw2, z2, tex))
            brushes.append(self._box(cx - hw2, cy - hw, z1, cx + hw2, cy + hw, z2, tex))
        else:
            # High side count (9+): use 3 overlapping boxes for rounder appearance
            # This avoids degenerate radial segments with inner_r=0
            # Box 1: axis-aligned
            hw = radius * 0.707
            brushes.append(self._box(cx - hw, cy - hw, z1, cx + hw, cy + hw, z2, tex))
            # Box 2: rotated ~30° (approximated with offset dimensions)
            hw2 = radius * 0.6
            hw3 = radius * 0.35
            brushes.append(self._box(cx - hw2, cy - hw3, z1, cx + hw2, cy + hw3, z2, tex))
            brushes.append(self._box(cx - hw3, cy - hw2, z1, cx + hw3, cy + hw2, z2, tex))

        return brushes

    def _generate_triangular_prism(
        self,
        cx: float, cy: float, z1: float, z2: float,
        radius: float,
        texture: str
    ) -> Brush:
        """Generate a triangular prism (3-sided pillar).

        Creates a 5-face brush: 3 vertical sides + top + bottom.
        """
        # Calculate triangle vertices (pointing north)
        angle_offset = -math.pi / 2  # Start from top
        angles = [angle_offset + i * 2 * math.pi / 3 for i in range(3)]

        x1 = self._snap_coord(cx + radius * math.cos(angles[0]))
        y1 = self._snap_coord(cy + radius * math.sin(angles[0]))
        x2 = self._snap_coord(cx + radius * math.cos(angles[1]))
        y2 = self._snap_coord(cy + radius * math.sin(angles[1]))
        x3 = self._snap_coord(cx + radius * math.cos(angles[2]))
        y3 = self._snap_coord(cy + radius * math.sin(angles[2]))

        z1 = self._snap_coord(z1)
        z2 = self._snap_coord(z2)

        # 5 faces: bottom, top, 3 sides
        planes = [
            # Bottom face
            Plane((x1, y1, z1), (x2, y2, z1), (x3, y3, z1), texture),
            # Top face
            Plane((x1, y1, z2), (x3, y3, z2), (x2, y2, z2), texture),
            # Side 1: v1 to v2
            Plane((x1, y1, z1), (x1, y1, z2), (x2, y2, z1), texture),
            # Side 2: v2 to v3
            Plane((x2, y2, z1), (x2, y2, z2), (x3, y3, z1), texture),
            # Side 3: v3 to v1
            Plane((x3, y3, z1), (x3, y3, z2), (x1, y1, z1), texture),
        ]
        return Brush(planes=planes, brush_id=self._next_id())

    def _generate_tapered_polygonal_solid(
        self,
        cx: float, cy: float, z1: float, z2: float,
        base_radius: float,
        sides: int,
        taper: float = 0.0,
        taper_curve: str = "linear",
        taper_segments: int = 2,
        texture: str = ""
    ) -> List[Brush]:
        """Generate a tapered N-sided polygonal solid.

        Creates a pillar that narrows from bottom to top using multiple
        stacked sections with decreasing radii.

        Args:
            cx, cy: Center point of the polygon
            z1, z2: Bottom and top Z coordinates
            base_radius: Radius at the bottom
            sides: Number of sides (3-16)
            taper: Taper amount (0.0 to 0.8)
            taper_curve: Curve type: "linear", "concave", "convex", "entasis"
            taper_segments: Number of segments for curve approximation (1-8)
            texture: Optional texture override

        Returns:
            List of brushes forming the tapered solid
        """
        if taper < 0.01:
            # No taper - use simple solid
            return self._generate_polygonal_solid(cx, cy, z1, z2, base_radius, sides, texture)

        tex = texture or self.params.texture
        brushes: List[Brush] = []

        # Clamp segments
        taper_segments = max(1, min(8, taper_segments))
        height = z2 - z1
        segment_height = height / taper_segments

        for seg in range(taper_segments):
            seg_z1 = z1 + seg * segment_height
            seg_z2 = z1 + (seg + 1) * segment_height

            # Calculate radius at bottom and top of this segment
            t_bottom = seg / taper_segments
            t_top = (seg + 1) / taper_segments

            r_bottom = self._calculate_taper_radius(base_radius, t_bottom, taper, taper_curve)
            r_top = self._calculate_taper_radius(base_radius, t_top, taper, taper_curve)

            # Use average radius for this segment (approximation)
            seg_radius = (r_bottom + r_top) / 2

            brushes.extend(self._generate_polygonal_solid(
                cx, cy, seg_z1, seg_z2, seg_radius, sides, tex
            ))

        return brushes

    def _generate_room_pillar(
        self,
        cx: float, cy: float, z1: float, z2: float,
        width: float,
        style: str = "square",
        capital: bool = False,
        capital_height: float = 16.0,
        capital_ratio: float = 1.3,
        texture: str = ""
    ) -> List[Brush]:
        """Generate a pillar at position using room's pillar settings.

        This is a convenience method for room modules to generate pillars
        with configurable style parameters without instantiating the full
        Pillar primitive.

        Args:
            cx, cy: Center position of the pillar
            z1, z2: Bottom and top of pillar (before capital)
            width: Pillar width (diameter for round styles)
            style: Style name: "square", "hexagonal", "octagonal", "round"
            capital: Add a capital block at top
            capital_height: Height of capital block
            capital_ratio: Width multiplier for capital
            texture: Optional texture override

        Returns:
            List of brushes forming the pillar
        """
        tex = texture or self.params.texture
        brushes: List[Brush] = []

        # Convert style to sides
        style_sides = {
            "square": 4,
            "hexagonal": 6,
            "octagonal": 8,
            "round": 12
        }
        sides = style_sides.get(style.lower(), 4)

        # Calculate radius from width
        hw = width / 2

        # Adjust body top if capital
        body_top = z2
        if capital:
            body_top = z2 - capital_height

        # Generate shaft
        brushes.extend(self._generate_polygonal_solid(
            cx, cy, z1, body_top,
            hw, sides, tex
        ))

        # Generate capital
        if capital:
            capital_radius = hw * capital_ratio
            brushes.extend(self._generate_polygonal_solid(
                cx, cy, body_top, z2,
                capital_radius, sides, tex
            ))

        return brushes
