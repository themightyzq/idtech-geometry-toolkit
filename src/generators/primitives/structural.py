"""
Structural geometry primitives: staircases, arches, pillars.

These are OPEN ELEMENTS - they are meant to be placed INSIDE rooms or between
sealed spaces. They do NOT need to be sealed themselves.

See CLAUDE.md for the distinction between sealed (room) and open (structural) primitives.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Tuple

from quake_levelgenerator.src.conversion.map_writer import Brush, Plane
from .base import GeometricPrimitive, Vec3


class StraightStaircase(GeometricPrimitive):
    """A straight staircase made of box steps."""

    width: float = 128.0
    length: float = 256.0
    height: float = 128.0
    step_height: float = 16.0
    railing: bool = False

    @classmethod
    def get_display_name(cls) -> str:
        return "Straight Staircase"

    @classmethod
    def get_category(cls) -> str:
        return "Structural"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "width": {
                "type": "float", "default": 128.0, "min": 64, "max": 512, "label": "Width",
                "description": "Total width of the staircase (player is 32 units wide)"
            },
            "length": {
                "type": "float", "default": 256.0, "min": 64, "max": 1024, "label": "Length",
                "description": "Horizontal run of the staircase from bottom to top"
            },
            "height": {
                "type": "float", "default": 128.0, "min": 16, "max": 512, "label": "Height",
                "description": "Total vertical rise of the staircase"
            },
            "step_height": {
                "type": "float", "default": 12.0, "min": 4, "max": 16, "label": "Step Height",
                "description": "Height of each step (max 18 for player traversal, 16 comfortable)"
            },
            "railing": {
                "type": "bool", "default": False, "label": "Add Railings",
                "description": "Add protective railings on both sides of the staircase"
            },
        }

    def generate(self) -> List[Brush]:
        ox, oy, oz = self.params.origin
        num_steps = max(1, int(self.height / self.step_height))
        step_depth = self.length / num_steps
        brushes: List[Brush] = []

        for i in range(num_steps):
            sx = ox - self.width / 2
            sy = oy + i * step_depth
            sz = oz + i * self.step_height
            brushes.append(self._structural_box(
                sx, sy, sz,
                sx + self.width, sy + step_depth, sz + self.step_height,
            ))

        if self.railing:
            rail_w = 8.0
            for side in (-1, 1):
                rx = ox + side * (self.width / 2)
                for i in range(num_steps):
                    sy = oy + i * step_depth
                    sz = oz + i * self.step_height
                    if side == -1:
                        brushes.append(self._structural_box(
                            rx - rail_w, sy, sz,
                            rx, sy + step_depth, sz + self.step_height + 48,
                        ))
                    else:
                        brushes.append(self._structural_box(
                            rx, sy, sz,
                            rx + rail_w, sy + step_depth, sz + self.step_height + 48,
                        ))
        return brushes


class Arch(GeometricPrimitive):
    """A semicircular arch with voussoir (wedge-shaped) stone segments.

    Creates architecturally correct round arches with:
    - Wedge-shaped voussoirs that follow the curve
    - Vertical legs (jambs) supporting the arch
    - Configurable segment count for smoother/blockier curves
    - Optional flat top for integration into walls/doorways
    """

    width: float = 128.0
    arch_height: float = 64.0   # Height of the arch curve (semicircle radius)
    leg_height: float = 64.0    # Height of jambs below the arch
    depth: float = 32.0
    segments: int = 8
    thickness: float = 16.0
    flat_top: bool = False      # If True, adds lintel above for wall integration

    @classmethod
    def get_display_name(cls) -> str:
        return "Arch"

    @classmethod
    def get_category(cls) -> str:
        return "Structural"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "width": {
                "type": "float", "default": 128.0, "min": 64, "max": 512, "label": "Opening Width",
                "description": "Width of the passable opening between the arch legs"
            },
            "arch_height": {
                "type": "float", "default": 64.0, "min": 32, "max": 256, "label": "Arch Height",
                "description": "Height of the semicircular curve (radius of the arch)"
            },
            "leg_height": {
                "type": "float", "default": 64.0, "min": 0, "max": 256, "label": "Leg Height",
                "description": "Height of vertical jambs below the arch curve (0 for keyhole arches)"
            },
            "depth": {
                "type": "float", "default": 32.0, "min": 8, "max": 128, "label": "Depth",
                "description": "Thickness of the arch from front to back"
            },
            "segments": {
                "type": "int", "default": 8, "min": 2, "max": 24, "label": "Segments",
                "description": "Number of voussoir (wedge) segments in the arch curve (6-8 recommended)"
            },
            "thickness": {
                "type": "float", "default": 16.0, "min": 4, "max": 64, "label": "Thickness",
                "description": "Radial thickness of the arch voussoirs"
            },
            "flat_top": {
                "type": "bool", "default": False, "label": "Flat Top (for walls)",
                "description": "Add lintel and spandrel fills for integration into rectangular walls"
            },
        }

    def generate(self) -> List[Brush]:
        ox, oy, oz = self.params.origin
        brushes: List[Brush] = []
        hw = self.width / 2  # Half-width of the opening

        # Arch geometry:
        # - arch_height controls the vertical rise of the semicircular curve
        # - leg_height controls how tall the jambs are below the arch
        # - For a true semicircle, arch_height should equal width/2 (inner_r)
        # - Smaller arch_height creates a flatter segmental arch
        inner_r = min(self.arch_height, hw)  # Arch can't be wider than opening
        outer_r = inner_r + self.thickness

        # Spring line is at top of jambs
        spring_z = oz + self.leg_height

        # Jambs (vertical legs) - extend from ground to spring line
        # Left jamb
        brushes.append(self._structural_box(
            ox - hw - self.thickness, oy - self.depth / 2, oz,
            ox - hw, oy + self.depth / 2, spring_z,
        ))
        # Right jamb
        brushes.append(self._structural_box(
            ox + hw, oy - self.depth / 2, oz,
            ox + hw + self.thickness, oy + self.depth / 2, spring_z,
        ))

        # Generate arch curve as voussoir segments
        # Angles from 0 (right horizontal) to pi (left horizontal)
        for i in range(self.segments):
            angle1 = math.pi * i / self.segments
            angle2 = math.pi * (i + 1) / self.segments

            brushes.append(self._arch_voussoir(
                ox, oy - self.depth / 2, spring_z,
                inner_r, outer_r,
                angle1, angle2,
                self.depth
            ))

        # Flat top: fill spandrel areas between arch curve and rectangular frame
        if self.flat_top:
            crown_z = spring_z + inner_r  # Top of semicircle
            lintel_height = self.thickness * 2  # Lintel above the crown
            total_top_z = crown_z + lintel_height

            # Top lintel spanning full width above the arch
            brushes.append(self._structural_box(
                ox - hw - self.thickness, oy - self.depth / 2, crown_z,
                ox + hw + self.thickness, oy + self.depth / 2, total_top_z,
            ))

            # Jamb extensions: fill the gap between jamb top and first voussoir
            # The voussoirs have straight edges, so there's a triangular gap
            # between spring_z and where the voussoir outer edge curves inward
            brushes.extend(self._generate_jamb_extensions(
                ox, oy, spring_z, inner_r, outer_r, hw
            ))

            # Left spandrel: fills corner between jamb top and lintel
            brushes.extend(self._generate_spandrel(
                ox, oy, spring_z, inner_r, outer_r, hw, crown_z, left_side=True
            ))

            # Right spandrel
            brushes.extend(self._generate_spandrel(
                ox, oy, spring_z, inner_r, outer_r, hw, crown_z, left_side=False
            ))

        return brushes

    def _generate_jamb_extensions(
        self,
        ox: float, oy: float, spring_z: float,
        inner_r: float, outer_r: float, hw: float
    ) -> List[Brush]:
        """Generate jamb extension brushes to fill gaps at the spring line.

        The voussoirs have straight edges (not curved), creating a triangular
        gap between the jamb top and where the voussoir outer edge curves away
        from the frame edge. This method fills that gap with a single brush
        per side for clean, predictable geometry.

        NOTE: Only generates fills for high segment counts (8+) where the
        first voussoir is a small wedge near the spring line. For low segment
        counts, the voussoirs themselves cover this area adequately.
        """
        brushes: List[Brush] = []

        # Only generate jamb extensions for segment counts >= 8
        # For lower counts, the voussoirs are large enough that there's no
        # meaningful gap to fill (the "gap" would overlap the arch opening)
        if self.segments < 8:
            return brushes

        # First voussoir outer edge endpoints
        angle1 = math.pi / self.segments
        v_x2 = outer_r * math.cos(angle1)
        v_z2 = spring_z + outer_r * math.sin(angle1)

        # Frame edge position
        frame_x = hw + self.thickness

        # If outer_r <= hw + thickness, the voussoir outer edge starts at or
        # inside the frame edge, creating a triangular gap.
        if outer_r <= frame_x + 0.1:
            # Use a SINGLE rectangular fill per side that covers the entire
            # gap area. This is simpler and more stable than multiple strips.
            # The fill extends from the jamb top (spring_z) to v_z2, and from
            # the voussoir x position to the frame edge.
            #
            # For clean geometry, we use the minimum voussoir X (at v_z2)
            # which ensures the fill doesn't overlap the voussoir.
            fill_width = frame_x - v_x2
            fill_height = v_z2 - spring_z

            # Only create fills if they're reasonably small (not spanning half the arch)
            max_reasonable_width = hw / 2  # Don't fill more than quarter of arch width
            if fill_width >= 2.0 and fill_height >= 2.0 and fill_width <= max_reasonable_width:
                # Right side fill
                brushes.append(self._structural_box(
                    ox + v_x2, oy - self.depth / 2, spring_z,
                    ox + frame_x, oy + self.depth / 2, v_z2,
                ))
                # Left side fill (mirrored)
                brushes.append(self._structural_box(
                    ox - frame_x, oy - self.depth / 2, spring_z,
                    ox - v_x2, oy + self.depth / 2, v_z2,
                ))

        return brushes

    def _generate_spandrel(
        self,
        ox: float, oy: float, spring_z: float,
        inner_r: float, outer_r: float, hw: float, crown_z: float,
        left_side: bool
    ) -> List[Brush]:
        """Generate spandrel fill for one side of a flat-top arch.

        Spandrels are the roughly triangular areas between the arch curve
        and the rectangular frame.

        Uses INVERSE scaling: fewer strips for more segments (since the
        voussoirs already approximate the curve well at high segment counts).
        """
        brushes: List[Brush] = []

        # Minimum brush dimensions to avoid degenerate geometry
        MIN_WIDTH = 2.0

        # Build the voussoir OUTER edge profile (where spandrel meets voussoir).
        edge_points: List[Tuple[float, float]] = []  # (x, z) points along voussoir outer edge

        for i in range(self.segments + 1):
            angle = math.pi * i / self.segments
            x = outer_r * math.cos(angle)
            h = outer_r * math.sin(angle)
            z = spring_z + h
            edge_points.append((x, z))

        # INVERSE scaling: more segments = fewer fill strips needed
        # Low segments (2-4): need more strips to approximate missing curve detail
        # High segments (12+): voussoirs already approximate curve, need fewer fills
        if self.segments <= 4:
            num_strips = 6
        elif self.segments <= 8:
            num_strips = 4
        else:
            # High segment counts: use just 3 strips for clean geometry
            num_strips = 3

        total_height = crown_z - spring_z

        for i in range(num_strips):
            strip_z1 = spring_z + total_height * i / num_strips
            strip_z2 = spring_z + total_height * (i + 1) / num_strips

            if strip_z1 >= crown_z:
                continue

            # Find the arch edge x-coordinate at the TOP of the strip
            # This gives us the edge closest to center for full coverage
            arch_x_top = self._find_arch_edge_x(edge_points, min(strip_z2, crown_z - 1), left_side)
            arch_x_bot = self._find_arch_edge_x(edge_points, strip_z1, left_side)

            # Use the edge position closest to center (smallest absolute value)
            if arch_x_top is not None and arch_x_bot is not None:
                if left_side:
                    arch_x = max(arch_x_top, arch_x_bot)  # Less negative = closer to center
                else:
                    arch_x = min(arch_x_top, arch_x_bot)  # Less positive = closer to center
            elif arch_x_top is not None:
                arch_x = arch_x_top
            elif arch_x_bot is not None:
                arch_x = arch_x_bot
            else:
                continue

            if left_side:
                outer_x = ox - hw - self.thickness
                fill_to_x = ox + arch_x
                fill_to_x = min(fill_to_x, ox)  # Don't cross centerline

                if fill_to_x - outer_x >= MIN_WIDTH:
                    brushes.append(self._structural_box(
                        outer_x, oy - self.depth / 2, strip_z1,
                        fill_to_x, oy + self.depth / 2, strip_z2,
                    ))
            else:
                outer_x = ox + hw + self.thickness
                fill_from_x = ox + arch_x
                fill_from_x = max(fill_from_x, ox)  # Don't cross centerline

                if outer_x - fill_from_x >= MIN_WIDTH:
                    brushes.append(self._structural_box(
                        fill_from_x, oy - self.depth / 2, strip_z1,
                        outer_x, oy + self.depth / 2, strip_z2,
                    ))

        return brushes

    def _find_arch_edge_x(
        self,
        edge_points: List[Tuple[float, float]],
        z: float,
        left_side: bool
    ) -> float | None:
        """Find the x-coordinate of the arch edge at height z.

        Interpolates along the straight voussoir segment edges.
        Returns the x value on the appropriate side (negative for left, positive for right).
        """
        # Find which segment contains this z level
        for i in range(len(edge_points) - 1):
            x1, z1 = edge_points[i]
            x2, z2 = edge_points[i + 1]

            # Check if z is within this segment's z range
            z_min, z_max = min(z1, z2), max(z1, z2)
            if z_min <= z <= z_max:
                # Interpolate x along this segment
                if abs(z2 - z1) < 0.001:
                    # Horizontal segment
                    interp_x = (x1 + x2) / 2
                else:
                    t = (z - z1) / (z2 - z1)
                    interp_x = x1 + t * (x2 - x1)

                # Return the appropriate side
                if left_side and interp_x < 0:
                    return interp_x
                elif not left_side and interp_x > 0:
                    return interp_x

        return None


class Pillar(GeometricPrimitive):
    """A pillar with configurable polygonal cross-section and segment styles.

    User-friendly pillar generator with segment customization for creating
    visually interesting pillars.

    Shaft Sides:
        - 3: Triangular
        - 4: Square (default)
        - 6: Hexagonal
        - 8: Octagonal (max - higher counts use same geometry)

    Segment Styles (when segments > 1):
        - uniform: All segments same width
        - alternating: Wide/narrow pattern for banded look
        - bulging: Middle segments wider (barrel shape)
        - necked: Middle segments narrower (hourglass shape)

    Ruin Styles:
        - broken_top: Irregular jagged top (2-4 brushes)
        - partial: Clean horizontal cut (1 section)
        - tilted: Leaning pillar (box + wedge, 2 brushes)

    Capital and base always match shaft sides for visual consistency.
    """

    # Height clamping constants (per CLAUDE.md §2: NEVER create degenerate geometry)
    MIN_SHAFT_HEIGHT = 16.0      # Minimum visible shaft height
    MIN_ORNAMENT_HEIGHT = 8.0    # Minimum height for capital/base to be generated
    MAX_BASE_RATIO = 0.4         # Base can use at most 40% of pillar height
    MAX_CAPITAL_RATIO = 0.3      # Capital can use at most 30% of pillar height

    width: float = 32.0
    pillar_height: float = 128.0

    # Shaft parameters
    shaft_sides: int = 4            # 3-8 sides (capped for performance)
    shaft_segments: int = 1         # 1-6 segments for customization
    segment_style: str = "uniform"  # uniform, alternating, bulging, necked

    # Capital parameters
    capital: bool = False
    capital_height: float = 16.0
    capital_width_ratio: float = 1.3

    # Base parameters
    base_plinth: bool = False
    base_height: float = 16.0
    base_width_ratio: float = 1.5

    # Ruin parameters
    ruined: bool = False
    ruin_style: str = "broken_top"   # broken_top, partial, tilted
    ruin_amount: float = 0.3         # 0.1-0.9
    random_seed: int = 0             # For reproducible ruins

    @classmethod
    def get_display_name(cls) -> str:
        return "Pillar"

    @classmethod
    def get_category(cls) -> str:
        return "Structural"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "width": {
                "type": "float", "default": 32.0, "min": 16, "max": 128, "label": "Width",
                "description": "Base width of the pillar shaft"
            },
            "pillar_height": {
                "type": "float", "default": 128.0, "min": 32, "max": 512, "label": "Height",
                "description": "Total height from floor to top of capital (if any)"
            },

            # Shaft
            "shaft_sides": {
                "type": "int", "default": 4, "min": 3, "max": 8, "label": "Shaft Sides",
                "description": "Number of sides (4=square, 6=hex, 8=octagon)"
            },
            "shaft_segments": {
                "type": "int", "default": 1, "min": 1, "max": 6, "label": "Segments",
                "description": "Number of shaft segments (1=simple, 2-6=allows segment styling)"
            },
            "segment_style": {
                "type": "choice", "default": "uniform",
                "choices": ["uniform", "alternating", "bulging", "necked"], "label": "Segment Style",
                "description": "uniform=same width, alternating=banded, bulging=barrel, necked=hourglass"
            },

            # Capital
            "capital": {
                "type": "bool", "default": False, "label": "Add Capital",
                "description": "Add decorative capital on top of the pillar shaft"
            },
            "capital_height": {
                "type": "float", "default": 16.0, "min": 8, "max": 64, "label": "Capital Height",
                "description": "Vertical height of the capital ornament"
            },
            "capital_width_ratio": {
                "type": "float", "default": 1.3, "min": 1.0, "max": 2.0, "label": "Capital Width",
                "description": "Capital width as ratio of shaft width (1.3 = 30% wider)"
            },

            # Base
            "base_plinth": {
                "type": "bool", "default": False, "label": "Add Base",
                "description": "Add decorative base/plinth at the bottom of the pillar"
            },
            "base_height": {
                "type": "float", "default": 16.0, "min": 8, "max": 64, "label": "Base Height",
                "description": "Vertical height of the base plinth"
            },
            "base_width_ratio": {
                "type": "float", "default": 1.5, "min": 1.0, "max": 2.5, "label": "Base Width",
                "description": "Base width as ratio of shaft width (1.5 = 50% wider)"
            },

            # Ruin
            "ruined": {
                "type": "bool", "default": False, "label": "Ruined",
                "description": "Enable ruin mode (disables capital, creates damaged appearance)"
            },
            "ruin_style": {
                "type": "choice", "default": "broken_top",
                "choices": ["broken_top", "partial", "tilted"], "label": "Ruin Style",
                "description": "broken_top = jagged chunks, partial = clean break, tilted = leaning pillar"
            },
            "ruin_amount": {
                "type": "float", "default": 0.3, "min": 0.1, "max": 0.9, "label": "Ruin Amount",
                "description": "Severity of damage (0.1 = minor chips, 0.9 = mostly destroyed)"
            },
            "random_seed": {
                "type": "int", "default": 0, "min": 0, "max": 999999, "label": "Random Seed",
                "description": "Seed for reproducible ruin patterns (same seed = same damage)"
            },
        }

    def _get_segment_width_ratio(self, segment_index: int, total_segments: int, style: str) -> float:
        """Calculate width multiplier for a segment based on style.

        Args:
            segment_index: Which segment (0 = bottom, total_segments-1 = top)
            total_segments: Total number of segments
            style: Segment style (uniform, alternating, bulging, necked)

        Returns:
            Width multiplier (1.0 = normal, >1.0 = wider, <1.0 = narrower)
        """
        if total_segments <= 1 or style == "uniform":
            return 1.0

        # Normalized position (0.0 = bottom, 1.0 = top)
        t = segment_index / (total_segments - 1) if total_segments > 1 else 0.5

        if style == "alternating":
            # Alternating wide/narrow pattern (20% variation)
            return 1.1 if segment_index % 2 == 0 else 0.9

        elif style == "bulging":
            # Middle segments wider (barrel shape, up to 25% wider at center)
            # Uses sine curve: peaks at middle (t=0.5)
            bulge = 0.25 * math.sin(t * math.pi)
            return 1.0 + bulge

        elif style == "necked":
            # Middle segments narrower (hourglass shape, up to 25% narrower at center)
            # Inverse of bulging
            neck = 0.25 * math.sin(t * math.pi)
            return 1.0 - neck

        return 1.0

    def generate(self) -> List[Brush]:
        ox, oy, oz = self.params.origin
        hw = self.width / 2
        brushes: List[Brush] = []

        # Cap shaft_sides at 8 for performance (higher counts use same geometry)
        effective_sides = min(8, max(3, self.shaft_sides))

        # =====================================================================
        # Height validation and clamping (CLAUDE.md §2: NEVER create degenerate geometry)
        # =====================================================================
        available_height = self.pillar_height

        # Calculate effective heights with clamping
        effective_base_height = 0.0
        effective_capital_height = 0.0

        if self.base_plinth:
            max_base = available_height * self.MAX_BASE_RATIO
            effective_base_height = min(self.base_height, max_base)

        if self.capital:
            max_capital = available_height * self.MAX_CAPITAL_RATIO
            effective_capital_height = min(self.capital_height, max_capital)

        # Ensure total ornament height doesn't exceed available space for shaft
        total_ornament = effective_base_height + effective_capital_height
        max_ornament = available_height - self.MIN_SHAFT_HEIGHT

        if total_ornament > max_ornament and total_ornament > 0:
            # Scale down both proportionally
            scale = max_ornament / total_ornament
            effective_base_height *= scale
            effective_capital_height *= scale

        # =====================================================================
        # Generate base plinth (always matches shaft sides)
        # =====================================================================
        body_bottom = oz
        if self.base_plinth and effective_base_height >= self.MIN_ORNAMENT_HEIGHT:
            base_radius = hw * self.base_width_ratio
            brushes.extend(self._generate_polygonal_solid(
                ox, oy, oz, oz + effective_base_height,
                base_radius, effective_sides, texture=self.texture_structural
            ))
            body_bottom = oz + effective_base_height

        # =====================================================================
        # Calculate shaft dimensions
        # =====================================================================
        body_top = oz + self.pillar_height
        if self.capital and effective_capital_height >= self.MIN_ORNAMENT_HEIGHT:
            body_top -= effective_capital_height

        # Ensure shaft has positive height
        if body_top <= body_bottom:
            body_top = body_bottom + self.MIN_SHAFT_HEIGHT

        shaft_height = body_top - body_bottom

        # =====================================================================
        # Generate shaft (or ruined shaft)
        # =====================================================================
        if self.ruined:
            brushes.extend(self._generate_ruined_shaft(
                ox, oy, body_bottom, body_top, hw, effective_sides
            ))
        else:
            # Use user-specified segments (clamped to 1-6)
            num_segments = max(1, min(6, self.shaft_segments))
            segment_height = shaft_height / num_segments

            for seg in range(num_segments):
                seg_z1 = body_bottom + seg * segment_height
                seg_z2 = body_bottom + (seg + 1) * segment_height

                # Apply segment style width multiplier
                style_ratio = self._get_segment_width_ratio(seg, num_segments, self.segment_style)
                seg_radius = hw * style_ratio

                brushes.extend(self._generate_polygonal_solid(
                    ox, oy, seg_z1, seg_z2,
                    seg_radius, effective_sides, texture=self.texture_structural
                ))

        # =====================================================================
        # Generate capital (only for non-ruined pillars, matches shaft sides)
        # =====================================================================
        if self.capital and not self.ruined and effective_capital_height >= self.MIN_ORNAMENT_HEIGHT:
            capital_radius = hw * self.capital_width_ratio
            brushes.extend(self._generate_polygonal_solid(
                ox, oy, body_top, body_top + effective_capital_height,
                capital_radius, effective_sides, texture=self.texture_structural
            ))

        return brushes

    def _generate_ruined_shaft(
        self,
        cx: float, cy: float, z1: float, z2: float,
        base_radius: float, sides: int
    ) -> List[Brush]:
        """Generate ruined pillar shaft based on ruin_style.

        Optimized for minimal brush count while creating believable ruins.

        Ruin Styles:
        - partial: Clean horizontal cut at reduced height (1 segment)
        - broken_top: Jagged irregular top with 2-3 overlapping chunks
        - tilted: Leaning pillar using box + wedge (2 brushes total)
        """
        import random
        rng = random.Random(self.random_seed)

        brushes: List[Brush] = []
        height = z2 - z1

        # Minimum height for valid geometry
        if height < self.MIN_ORNAMENT_HEIGHT:
            # Just generate a simple solid if height is too small
            brushes.extend(self._generate_polygonal_solid(
                cx, cy, z1, z2, base_radius, sides, texture=self.texture_structural
            ))
            return brushes

        if self.ruin_style == "partial":
            # Clean horizontal cut - pillar is broken off cleanly at a lower height
            # The remaining height is (1.0 - ruin_amount) of the original
            remaining_fraction = max(0.2, 1.0 - self.ruin_amount)  # At least 20% remains
            cut_height = height * remaining_fraction

            brushes.extend(self._generate_polygonal_solid(
                cx, cy, z1, z1 + cut_height,
                base_radius, sides, texture=self.texture_structural
            ))

        elif self.ruin_style == "broken_top":
            # Irregular jagged top - simplified to use fewer brushes
            # Foundation + 2-3 irregular chunks = 3-4 brushes total
            remaining_fraction = max(0.3, 1.0 - self.ruin_amount)
            base_cut = height * remaining_fraction

            # Foundation: 60% of remaining height forms a stable base
            foundation_height = base_cut * 0.6
            brushes.extend(self._generate_polygonal_solid(
                cx, cy, z1, z1 + foundation_height,
                base_radius, sides, texture=self.texture_structural
            ))

            # 2-3 irregular chunks above foundation (fewer than before)
            num_chunks = 2 if sides <= 4 else 3
            for i in range(num_chunks):
                # Each chunk extends 60-90% of remaining height
                chunk_height = base_cut * rng.uniform(0.6, 0.9)
                chunk_radius = base_radius * rng.uniform(0.5, 0.8)

                # Slight random offset from center (max 15% of radius)
                angle = (i / num_chunks) * 2 * math.pi + rng.uniform(-0.3, 0.3)
                offset_dist = base_radius * 0.15 * rng.uniform(0.0, 1.0)
                chunk_cx = cx + math.cos(angle) * offset_dist
                chunk_cy = cy + math.sin(angle) * offset_dist

                # Use simple box for chunks
                hw = chunk_radius * 0.707
                brushes.append(self._structural_box(
                    chunk_cx - hw, chunk_cy - hw, z1 + foundation_height * 0.8,
                    chunk_cx + hw, chunk_cy + hw, z1 + chunk_height
                ))

        elif self.ruin_style == "tilted":
            # Leaning pillar using proper box + wedge geometry (2 brushes)
            # This creates an actual tilted appearance instead of stacked offset boxes
            remaining_fraction = max(0.4, 1.0 - self.ruin_amount * 0.5)
            cut_height = height * remaining_fraction

            # Maximum tilt offset at top (proportional to ruin_amount)
            # At ruin_amount=0.9, top is offset by ~60% of pillar width
            max_offset = base_radius * self.ruin_amount * 1.2

            # Random tilt direction based on seed
            tilt_angle = rng.uniform(0, 2 * math.pi)
            offset_x = math.cos(tilt_angle) * max_offset
            offset_y = math.sin(tilt_angle) * max_offset

            # Use inscribed square half-width for box geometry
            hw = base_radius * 0.707

            # Lower section: standard box (stable foundation)
            lower_height = cut_height * 0.4
            brushes.append(self._structural_box(
                cx - hw, cy - hw, z1,
                cx + hw, cy + hw, z1 + lower_height
            ))

            # Upper section: tilted using wedge
            # The wedge creates a proper parallelogram-like tilt
            upper_z1 = z1 + lower_height
            upper_z2 = z1 + cut_height

            # Create tilted upper section by offsetting top corners
            # We use a box that's shifted at the top, approximated by a wedge
            # For simplicity, use a box centered between base and offset positions
            mid_offset_x = offset_x * 0.5
            mid_offset_y = offset_y * 0.5

            # Slightly smaller upper section (simulates erosion)
            hw_upper = hw * 0.9
            brushes.append(self._structural_box(
                cx + mid_offset_x - hw_upper, cy + mid_offset_y - hw_upper, upper_z1,
                cx + mid_offset_x + hw_upper, cy + mid_offset_y + hw_upper, upper_z2
            ))

        return brushes


class Buttress(GeometricPrimitive):
    """An angled wall reinforcement/support structure.

    Buttresses are architectural elements that provide lateral support to walls.
    They project outward from the wall and typically taper toward the top.

    Features:
    - Base projects from wall, tapers upward
    - Optional stepped profile instead of smooth taper
    - Commonly used along castle/cathedral walls
    """

    width: float = 32.0         # Base width
    height: float = 128.0       # Total height
    depth: float = 48.0         # How far it projects from wall
    taper: float = 0.3          # Top narrowing ratio (0-0.5)
    stepped: bool = False       # Stepped profile vs smooth
    step_count: int = 3         # Steps if stepped=True

    @classmethod
    def get_display_name(cls) -> str:
        return "Buttress"

    @classmethod
    def get_category(cls) -> str:
        return "Structural"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "width": {
                "type": "float", "default": 32.0, "min": 16, "max": 96, "label": "Width",
                "description": "Width of the buttress at its base"
            },
            "height": {
                "type": "float", "default": 128.0, "min": 32, "max": 512, "label": "Height",
                "description": "Total height of the buttress"
            },
            "depth": {
                "type": "float", "default": 48.0, "min": 16, "max": 128, "label": "Depth",
                "description": "How far the buttress projects from the wall"
            },
            "taper": {
                "type": "float", "default": 0.3, "min": 0.0, "max": 0.5, "label": "Taper",
                "description": "How much the buttress narrows at the top (0=none, 0.5=half size)"
            },
            "stepped": {
                "type": "bool", "default": False, "label": "Stepped Profile",
                "description": "Use stepped setbacks instead of smooth taper"
            },
            "step_count": {
                "type": "int", "default": 3, "min": 2, "max": 6, "label": "Step Count",
                "description": "Number of steps when using stepped profile"
            },
        }

    def generate(self) -> List[Brush]:
        ox, oy, oz = self.params.origin
        brushes: List[Brush] = []

        hw = self.width / 2

        if self.stepped:
            # Generate stepped profile
            step_h = self.height / self.step_count
            for i in range(self.step_count):
                # Each step is smaller than the one below
                step_taper = (i / self.step_count) * self.taper
                step_width = hw * (1.0 - step_taper)
                step_depth = self.depth * (1.0 - step_taper)

                step_z1 = oz + i * step_h
                step_z2 = oz + (i + 1) * step_h

                brushes.append(self._structural_box(
                    ox - step_width, oy, step_z1,
                    ox + step_width, oy + step_depth, step_z2
                ))
        else:
            # Generate tapered profile with two sections
            top_hw = hw * (1.0 - self.taper)
            top_depth = self.depth * (1.0 - self.taper)

            mid_z = oz + self.height / 2

            # Lower section (full size)
            brushes.append(self._structural_box(
                ox - hw, oy, oz,
                ox + hw, oy + self.depth, mid_z
            ))

            # Upper section (tapered)
            brushes.append(self._structural_box(
                ox - top_hw, oy, mid_z,
                ox + top_hw, oy + top_depth, oz + self.height
            ))

        return brushes


class Battlement(GeometricPrimitive):
    """Crenellated wall-top defense (merlons and crenels).

    Battlements are the characteristic zigzag pattern at the top of castle walls.
    Merlons are the raised sections; crenels are the gaps between them.

    Features:
    - Alternating merlons and crenels
    - Configurable dimensions
    - Designed to be placed on top of Rampart outer parapet
    """

    length: float = 256.0       # Total length
    merlon_width: float = 32.0  # Width of raised sections
    merlon_height: float = 48.0 # Height of merlons
    crenel_width: float = 24.0  # Width of gaps
    thickness: float = 16.0     # Wall thickness
    base_height: float = 16.0   # Height of base wall

    @classmethod
    def get_display_name(cls) -> str:
        return "Battlement"

    @classmethod
    def get_category(cls) -> str:
        return "Structural"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "length": {
                "type": "float", "default": 256.0, "min": 64, "max": 1024, "label": "Length",
                "description": "Total length of the battlement section"
            },
            "merlon_width": {
                "type": "float", "default": 32.0, "min": 16, "max": 64, "label": "Merlon Width",
                "description": "Width of each raised section (solid blocks)"
            },
            "merlon_height": {
                "type": "float", "default": 48.0, "min": 24, "max": 96, "label": "Merlon Height",
                "description": "Height of raised sections above the base"
            },
            "crenel_width": {
                "type": "float", "default": 24.0, "min": 12, "max": 48, "label": "Crenel Width",
                "description": "Width of gaps between merlons (for firing through)"
            },
            "thickness": {
                "type": "float", "default": 16.0, "min": 8, "max": 32, "label": "Thickness",
                "description": "Wall thickness from front to back"
            },
            "base_height": {
                "type": "float", "default": 16.0, "min": 8, "max": 32, "label": "Base Height",
                "description": "Height of the continuous base wall below merlons"
            },
        }

    def generate(self) -> List[Brush]:
        ox, oy, oz = self.params.origin
        brushes: List[Brush] = []

        # Base wall (continuous)
        brushes.append(self._structural_box(
            ox - self.thickness / 2, oy, oz,
            ox + self.thickness / 2, oy + self.length, oz + self.base_height
        ))

        # Generate merlon/crenel pattern
        pattern_unit = self.merlon_width + self.crenel_width
        num_units = int(self.length / pattern_unit)

        y = oy
        for i in range(num_units):
            # Merlon
            brushes.append(self._structural_box(
                ox - self.thickness / 2, y, oz + self.base_height,
                ox + self.thickness / 2, y + self.merlon_width, oz + self.base_height + self.merlon_height
            ))
            y += self.merlon_width + self.crenel_width

        # Final merlon if there's room
        if y + self.merlon_width <= oy + self.length:
            brushes.append(self._structural_box(
                ox - self.thickness / 2, y, oz + self.base_height,
                ox + self.thickness / 2, y + self.merlon_width, oz + self.base_height + self.merlon_height
            ))

        return brushes
