"""
Hall corridor primitives: StraightHall, SquareCorner, TJunction, Crossroads.

These are SEALED MODULAR PRIMITIVES - they generate complete, leak-free corridor
geometry that can be connected to other halls or rooms.

SEALED GEOMETRY RULES (see CLAUDE.md for full documentation):
1. Floor/ceiling must extend beyond walls by wall thickness (t) in all directions
2. Walls must span the full floor/ceiling extent
3. Brush overlaps at corners must be >= 8 units
4. Portals (openings) are created by splitting a wall into 3 brushes:
   left piece, right piece, and lintel above

Each primitive generates complete, playable corridor geometry with:
- Proper enclosure (floor, ceiling, walls)
- Optional portal openings for connections to other geometry
- 100% sealed geometry (only portals are open)
- Optional stairs/height variation per arm

HEIGHT BASELINE CONVENTION:
- Center/junction is the baseline at Z=origin
- Each arm's height_delta represents change FROM center TO that arm's portal
- Portal floor height = origin_z + height_delta_for_that_arm
"""

from __future__ import annotations
import math
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from quake_levelgenerator.src.conversion.map_writer import Brush
from .base import GeometricPrimitive
from .portal_system import (
    PORTAL_WIDTH, PORTAL_HEIGHT, PortalSpec, generate_portal_wall,
    PortalDirection
)


# Traversability constants for idTech player movement
MAX_CLIMBABLE_STEP = 18.0    # Maximum step height player can climb (comfortable: 16)
MIN_STEP_DEPTH = 16.0        # Minimum step depth for player to stand on
MAX_RAMP_ANGLE = 45.0        # Maximum ramp angle in degrees (idTech limit ~46°)


@dataclass
class ValidationWarning:
    """A warning about hall configuration that may affect playability."""
    arm_name: str           # e.g., "North", "Front", "A"
    message: str            # Human-readable warning
    current_value: float    # Current arm length or other value
    recommended_value: float  # Recommended minimum value
    severity: str = "warning"  # "warning" or "error" (error = untraversable)


@dataclass
class ValidationResult:
    """Result of validating a hall configuration."""
    valid: bool = True                              # True if playable (may have warnings)
    warnings: List[ValidationWarning] = field(default_factory=list)

    def add_warning(self, arm_name: str, message: str, current: float,
                    recommended: float, severity: str = "warning"):
        self.warnings.append(ValidationWarning(
            arm_name=arm_name,
            message=message,
            current_value=current,
            recommended_value=recommended,
            severity=severity
        ))
        if severity == "error":
            self.valid = False

    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    def get_summary(self) -> str:
        """Return a human-readable summary of all warnings."""
        if not self.warnings:
            return "Configuration valid."
        lines = []
        for w in self.warnings:
            lines.append(f"[{w.severity.upper()}] {w.arm_name}: {w.message} "
                        f"(current: {w.current_value:.0f}, recommended: {w.recommended_value:.0f})")
        return "\n".join(lines)


class HallBase(GeometricPrimitive):
    """Abstract base class for hall corridor primitives.

    Provides shared parameters and helper methods for sealed corridor geometry.
    All halls use consistent dimensions for seamless connections.
    """

    # Shared hall dimensions
    hall_width: float = 128.0       # Total interior width
    hall_height: float = 128.0      # Interior ceiling height
    wall_thickness: float = 16.0    # Thickness of walls
    portal_width: float = PORTAL_WIDTH   # Width of portal openings (unified constant)
    portal_height: float = PORTAL_HEIGHT # Height of portal openings (unified constant)

    # Stair generation toggle (default OFF - user places height changes in TrenchBroom)
    generate_stairs: bool = False

    @classmethod
    def get_category(cls) -> str:
        return "Halls"

    def _get_base_schema(self) -> Dict[str, Dict[str, Any]]:
        """Return schema for shared hall parameters."""
        return {
            "hall_width": {
                "type": "float", "default": 128.0, "min": 64, "max": 256, "label": "Hall Width",
                "description": "Interior width of the corridor (player is 32 units wide)"
            },
            "hall_height": {
                "type": "float", "default": 128.0, "min": 64, "max": 256, "label": "Hall Height",
                "description": "Interior ceiling height (player is 56 units tall)"
            },
            "wall_thickness": {
                "type": "float", "default": 16.0, "min": 8, "max": 32, "label": "Wall Thickness",
                "description": "Thickness of walls, floor, and ceiling brushes"
            },
            "portal_width": {
                "type": "float", "default": PORTAL_WIDTH, "min": 48, "max": 192, "label": "Portal Width",
                "description": "Width of portal openings for connections"
            },
            "portal_height": {
                "type": "float", "default": PORTAL_HEIGHT, "min": 64, "max": 192, "label": "Portal Height",
                "description": "Height of portal openings for connections"
            },
            "generate_stairs": {
                "type": "bool", "default": False, "label": "Generate Stairs",
                "description": "Auto-generate stairs for height changes (OFF = flat floors, handle in TrenchBroom)"
            },
        }

    def _get_stair_schema(self) -> Dict[str, Dict[str, Any]]:
        """Return schema for step_height parameter."""
        return {
            "step_height": {
                "type": "float", "default": 12.0, "min": 8, "max": 16, "label": "Step Height",
                "description": "Height of each step when generating stairs (max 18 for traversal)"
            },
        }

    def _calc_min_arm_length(self, height_delta: float, step_height: float) -> float:
        """Calculate minimum arm length needed for climbable stairs.

        Args:
            height_delta: Absolute height change for the arm
            step_height: Desired step height

        Returns:
            Minimum arm length in units for climbable stairs
        """
        if abs(height_delta) < 0.01:
            return 0.0
        num_steps = math.ceil(abs(height_delta) / step_height)
        return num_steps * MIN_STEP_DEPTH

    def _calc_actual_step_height(self, arm_length: float, height_delta: float,
                                  step_height: float) -> float:
        """Calculate actual step height given constraints.

        Returns the step height that would result from the given configuration.
        """
        if abs(height_delta) < 0.01 or arm_length < 1.0:
            return 0.0

        abs_delta = abs(height_delta)
        num_steps = max(1, math.ceil(abs_delta / step_height))
        max_steps = max(1, int(arm_length / MIN_STEP_DEPTH))

        if num_steps > max_steps:
            num_steps = max_steps

        return abs_delta / num_steps

    def _check_arm_traversability(self, arm_name: str, arm_length: float,
                                   height_delta: float, step_height: float,
                                   result: ValidationResult):
        """Check if an arm's stairs are traversable and add warnings if not.

        Args:
            arm_name: Display name for the arm (e.g., "North", "Front")
            arm_length: Length of the arm in units
            height_delta: Height change from center to portal
            step_height: Configured step height
            result: ValidationResult to add warnings to
        """
        if abs(height_delta) < 0.01:
            return  # Flat arm, no issues

        min_length = self._calc_min_arm_length(height_delta, step_height)
        actual_step_h = self._calc_actual_step_height(arm_length, height_delta, step_height)

        if actual_step_h > MAX_CLIMBABLE_STEP:
            # Stairs are unclimbable - will fall back to ramp
            ramp_angle = math.degrees(math.atan2(abs(height_delta), arm_length))
            if ramp_angle > MAX_RAMP_ANGLE:
                # Even ramp is too steep - player cannot traverse
                result.add_warning(
                    arm_name,
                    f"BLOCKED: {ramp_angle:.0f}° slope too steep (max {MAX_RAMP_ANGLE:.0f}°)",
                    arm_length,
                    min_length,
                    severity="error"
                )
            else:
                # Ramp is usable but not ideal
                result.add_warning(
                    arm_name,
                    f"Arm too short for stairs, using {ramp_angle:.0f}° ramp instead",
                    arm_length,
                    min_length,
                    severity="warning"
                )
        elif actual_step_h > step_height + 2:
            # Steps are climbable but steeper than configured
            result.add_warning(
                arm_name,
                f"Steps are {actual_step_h:.0f}u tall (target {step_height:.0f}u)",
                arm_length,
                min_length,
                severity="warning"
            )

    def validate(self) -> ValidationResult:
        """Validate the hall configuration for playability.

        Override in subclasses to check specific arm configurations.
        Returns ValidationResult with any warnings about traversability.
        """
        return ValidationResult()

    def _generate_arm_flat_floor(
        self,
        arm_start: float,      # Coordinate at center end
        arm_end: float,        # Coordinate at portal end
        perp_min: float,       # Min on perpendicular axis
        perp_max: float,       # Max on perpendicular axis
        z_floor: float,        # Z height of floor surface
        axis: str,             # "y" or "x"
        t: float,              # Wall thickness for floor extension
    ) -> List[Brush]:
        """Generate a flat floor brush for a hall arm, ignoring height deltas.

        This is used when generate_stairs=False (default). Users handle height
        changes manually in TrenchBroom.

        Args:
            arm_start: Coordinate at center end of arm
            arm_end: Coordinate at portal end of arm
            perp_min: Minimum extent on perpendicular axis
            perp_max: Maximum extent on perpendicular axis
            z_floor: Z height of floor surface (ignores height deltas)
            axis: "y" for arms along Y axis, "x" for arms along X axis
            t: Wall thickness for floor under-extension
        """
        arm_length = abs(arm_end - arm_start)
        if arm_length < 1.0:
            return []

        if axis == "y":
            y1, y2 = min(arm_start, arm_end), max(arm_start, arm_end)
            return [self._box(
                perp_min, y1, z_floor - t,
                perp_max, y2, z_floor
            )]
        else:  # axis == "x"
            x1, x2 = min(arm_start, arm_end), max(arm_start, arm_end)
            return [self._box(
                x1, perp_min, z_floor - t,
                x2, perp_max, z_floor
            )]

    def _generate_arm_floor(
        self,
        arm_start: float,
        arm_end: float,
        perp_min: float,
        perp_max: float,
        z_start: float,
        z_end: float,
        axis: str,
        step_height: float,
        t: float,
    ) -> List[Brush]:
        """Generate floor for a hall arm, respecting generate_stairs toggle.

        When generate_stairs=True (opt-in): generates stairs/ramps for height changes
        When generate_stairs=False (default): generates flat floor at z_start

        Args: Same as _generate_arm_stairs
        """
        if self.generate_stairs and abs(z_end - z_start) > 0.01:
            # User opted in to stairs and there's a height change
            return self._generate_arm_stairs(
                arm_start, arm_end, perp_min, perp_max,
                z_start, z_end, axis, step_height, t
            )
        else:
            # Default: flat floor at z_start (ignores z_end)
            return self._generate_arm_flat_floor(
                arm_start, arm_end, perp_min, perp_max,
                z_start, axis, t
            )

    def _generate_arm_stairs(
        self,
        arm_start: float,      # Coordinate at center end
        arm_end: float,        # Coordinate at portal end
        perp_min: float,       # Min on perpendicular axis (minus wall extension)
        perp_max: float,       # Max on perpendicular axis (plus wall extension)
        z_start: float,        # Z at center
        z_end: float,          # Z at portal
        axis: str,             # "y" or "x"
        step_height: float,
        t: float,              # Wall thickness for floor extension
    ) -> List[Brush]:
        """Generate stair brushes for a hall arm.

        Returns list of step brushes (or single flat floor brush if z_start == z_end).
        Handles ascending/descending based on sign of height delta.

        Args:
            arm_start: Coordinate at center end of arm
            arm_end: Coordinate at portal end of arm
            perp_min: Minimum extent on perpendicular axis
            perp_max: Maximum extent on perpendicular axis
            z_start: Z height at center (arm_start)
            z_end: Z height at portal (arm_end)
            axis: "y" for arms along Y axis, "x" for arms along X axis
            step_height: Maximum height of each step
            t: Wall thickness for floor under-extension
        """
        brushes: List[Brush] = []

        delta = z_end - z_start
        arm_length = abs(arm_end - arm_start)

        # If arm has zero or negligible length, return empty list
        # The center floor section handles the floor at the junction
        if arm_length < 1.0:
            return brushes

        # If no height change, return single flat floor
        if abs(delta) < 0.01:
            if axis == "y":
                y1, y2 = min(arm_start, arm_end), max(arm_start, arm_end)
                brushes.append(self._box(
                    perp_min, y1, z_start - t,
                    perp_max, y2, z_start
                ))
            else:  # axis == "x"
                x1, x2 = min(arm_start, arm_end), max(arm_start, arm_end)
                brushes.append(self._box(
                    x1, perp_min, z_start - t,
                    x2, perp_max, z_start
                ))
            return brushes

        # Calculate number of steps
        abs_delta = abs(delta)
        num_steps = max(1, math.ceil(abs_delta / step_height))

        # Enforce minimum step depth
        max_steps = max(1, int(arm_length / MIN_STEP_DEPTH))
        if num_steps > max_steps:
            num_steps = max_steps

        actual_step_h = abs_delta / num_steps

        # Check if stairs are climbable; if not, use ramp
        if actual_step_h > MAX_CLIMBABLE_STEP:
            # Fall back to ramp (wedge brush)
            return self._generate_arm_ramp(
                arm_start, arm_end, perp_min, perp_max,
                z_start, z_end, axis, t
            )

        step_depth = arm_length / num_steps

        ascending = delta > 0
        direction = 1 if arm_end > arm_start else -1

        for i in range(num_steps):
            # Calculate step position along the axis
            if direction > 0:
                step_start = arm_start + i * step_depth
                step_end = arm_start + (i + 1) * step_depth
            else:
                step_start = arm_start - i * step_depth
                step_end = arm_start - (i + 1) * step_depth

            # Calculate step Z (top surface)
            if ascending:
                step_z_top = z_start + (i + 1) * actual_step_h
            else:
                step_z_top = z_start - i * actual_step_h

            # Step bottom extends below by wall thickness
            step_z_bottom = step_z_top - t

            if axis == "y":
                y1, y2 = min(step_start, step_end), max(step_start, step_end)
                brushes.append(self._box(
                    perp_min, y1, step_z_bottom,
                    perp_max, y2, step_z_top
                ))
            else:  # axis == "x"
                x1, x2 = min(step_start, step_end), max(step_start, step_end)
                brushes.append(self._box(
                    x1, perp_min, step_z_bottom,
                    x2, perp_max, step_z_top
                ))

        return brushes

    def _generate_arm_ramp(
        self,
        arm_start: float,
        arm_end: float,
        perp_min: float,
        perp_max: float,
        z_start: float,
        z_end: float,
        axis: str,
        t: float,
    ) -> List[Brush]:
        """Generate a ramp (wedge) when stairs would be unclimbable.

        Falls back to this when the height delta is too large for the arm length
        to accommodate climbable stairs.
        """
        brushes: List[Brush] = []

        min_z = min(z_start, z_end)
        max_z = max(z_start, z_end)

        if axis == "y":
            y1, y2 = min(arm_start, arm_end), max(arm_start, arm_end)
            # Determine ramp direction
            if (arm_end > arm_start and z_end > z_start) or \
               (arm_end < arm_start and z_end < z_start):
                # Ascending in +Y direction
                ramp_axis = "y"
            else:
                # Descending in +Y direction (flip)
                ramp_axis = "-y"

            # Use wedge from base class
            brushes.append(self._wedge(
                perp_min, y1, min_z - t,
                perp_max, y2, max_z,
                ramp_axis=ramp_axis
            ))
        else:  # axis == "x"
            x1, x2 = min(arm_start, arm_end), max(arm_start, arm_end)
            # Determine ramp direction
            if (arm_end > arm_start and z_end > z_start) or \
               (arm_end < arm_start and z_end < z_start):
                # Ascending in +X direction
                ramp_axis = "x"
            else:
                # Descending in +X direction (flip)
                ramp_axis = "-x"

            brushes.append(self._wedge(
                x1, perp_min, min_z - t,
                x2, perp_max, max_z,
                ramp_axis=ramp_axis
            ))

        return brushes

    def _wall_with_portal(
        self,
        x1: float, y1: float, z1: float,
        x2: float, y2: float, z2: float,
        portal_enabled: bool,
        portal_axis: str = "x",  # "x" = portal opening spans X, "y" = spans Y
        portal_center: float = None,
    ) -> List[Brush]:
        """Create a wall, optionally with a portal opening.

        Delegates to the unified generate_portal_wall() function from portal_system.py.

        When portal is enabled, creates 3 brushes:
        - Left/front piece (from wall edge to portal edge)
        - Right/back piece (from portal edge to wall edge)
        - Lintel above portal (if portal_height < wall height)

        Args:
            x1, y1, z1: Minimum corner of wall bounds
            x2, y2, z2: Maximum corner of wall bounds
            portal_enabled: If True, create opening; if False, solid wall
            portal_axis: "x" if portal spans X direction, "y" if spans Y
            portal_center: Center of portal along the axis (default: wall center)
        """
        # Create portal specification with instance dimensions
        portal_spec = PortalSpec(
            enabled=portal_enabled,
            width=self.portal_width,
            height=self.portal_height
        )

        # Use unified portal generation function
        brushes, _ = generate_portal_wall(
            box_func=self._box,
            x1=x1, y1=y1, z1=z1,
            x2=x2, y2=y2, z2=z2,
            portal_spec=portal_spec,
            portal_axis=portal_axis,
            portal_center=portal_center,
        )

        return brushes


class StraightHall(HallBase):
    """A simple straight corridor segment with optional stairs.

    Features:
    - Configurable length
    - Optional portal openings at front and back
    - Optional height variation with stairs
    - Sealed floor, ceiling, and side walls

    Floor plan:
    +--[portal]--+
    |            |
    |   HALL     |
    |            |
    +--[portal]--+

    Height baseline: Front end (oy) is the reference at oz.
    height_delta_front = height change from center to front portal
    height_delta_back = height change from center to back portal
    """

    length: float = 256.0
    portal_front: bool = True
    portal_back: bool = True
    height_delta_front: float = 0.0  # Height delta at front portal (from center)
    height_delta_back: float = 0.0   # Height delta at back portal (from center)
    step_height: float = 12.0        # Max step height for stairs

    @classmethod
    def get_display_name(cls) -> str:
        return "Straight Hall"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        instance = cls()
        schema = instance._get_base_schema()
        schema.update(instance._get_stair_schema())
        schema.update({
            "length": {
                "type": "float", "default": 256.0, "min": 64, "max": 1024, "label": "Length",
                "description": "Total length of the corridor from front to back"
            },
            "portal_front": {
                "type": "bool", "default": True, "label": "Front Portal",
                "description": "Enable portal opening at the front (south) end"
            },
            "portal_back": {
                "type": "bool", "default": True, "label": "Back Portal",
                "description": "Enable portal opening at the back (north) end"
            },
            "height_delta_front": {
                "type": "float", "default": 0.0, "min": -256, "max": 256, "label": "Front Height Delta",
                "description": "Height change from center to front portal (negative = descend, positive = ascend)"
            },
            "height_delta_back": {
                "type": "float", "default": 0.0, "min": -256, "max": 256, "label": "Back Height Delta",
                "description": "Height change from center to back portal (negative = descend, positive = ascend)"
            },
        })
        return schema

    def generate(self) -> List[Brush]:
        self._reset_tags()  # Reset tags for this generation
        ox, oy, oz = self.params.origin
        brushes: List[Brush] = []

        hw = self.hall_width / 2
        t = self.wall_thickness
        h = self.hall_height
        length = self.length

        # Calculate Z heights at each end
        # Center is at midpoint (oy + length/2), which is the height baseline
        center_y = oy + length / 2
        center_z = oz  # Center is baseline
        front_z = oz + self.height_delta_front
        back_z = oz + self.height_delta_back

        # Determine height range for walls/ceiling
        min_z = min(center_z, front_z, back_z)
        max_z = max(center_z, front_z, back_z)

        # === CENTER FLOOR BOX ===
        # Like all junction halls, StraightHall needs a center floor box at the midpoint
        # This ensures proper sealing when connected to junctions with different heights
        # Must extend by t in ALL directions for sealed geometry
        brushes.append(self._box(
            ox - hw - t, center_y - hw - t, center_z - t,
            ox + hw + t, center_y + hw + t, center_z
        ))

        # === FRONT ARM (from center to front portal) ===
        # Generate stairs from center to front
        # Floor always extends by t past wall for sealed geometry
        brushes.extend(self._generate_arm_floor(
            arm_start=center_y - hw,  # Start at center edge
            arm_end=oy - t,           # Always extend by t
            perp_min=ox - hw - t,
            perp_max=ox + hw + t,
            z_start=center_z,
            z_end=front_z,
            axis="y",
            step_height=self.step_height,
            t=t
        ))

        # === BACK ARM (from center to back portal) ===
        # Generate stairs from center to back
        # Floor always extends by t past wall for sealed geometry
        brushes.extend(self._generate_arm_floor(
            arm_start=center_y + hw,      # Start at center edge
            arm_end=oy + length + t,      # Always extend by t
            perp_min=ox - hw - t,
            perp_max=ox + hw + t,
            z_start=center_z,
            z_end=back_z,
            axis="y",
            step_height=self.step_height,
            t=t
        ))

        # === CEILING (at max height + hall_height) ===
        # Ceiling always extends by t for proper sealing
        ceiling_z = max_z + h
        brushes.append(self._box(
            ox - hw - t, oy - t, ceiling_z,
            ox + hw + t, oy + length + t, ceiling_z + t
        ))

        # === WALLS (span full height range) ===
        # Walls always extend by t for proper sealing
        wall_bottom = min_z - t

        # Left wall (full Y extent)
        brushes.append(self._box(
            ox - hw - t, oy - t, wall_bottom,
            ox - hw, oy + length + t, ceiling_z
        ))

        # Right wall (full Y extent)
        brushes.append(self._box(
            ox + hw, oy - t, wall_bottom,
            ox + hw + t, oy + length + t, ceiling_z
        ))

        # === END WALLS WITH PORTALS ===
        # End walls must extend to ceiling_z, not just portal_z + h
        # This ensures no gaps when front and back heights differ

        # Front wall at front_z height
        brushes.extend(self._wall_with_portal(
            ox - hw - t, oy - t, front_z,
            ox + hw + t, oy, ceiling_z,
            self.portal_front,
            portal_axis="x"
        ))

        # Back wall at back_z height
        brushes.extend(self._wall_with_portal(
            ox - hw - t, oy + length, back_z,
            ox + hw + t, oy + length + t, ceiling_z,
            self.portal_back,
            portal_axis="x"
        ))

        # === REGISTER PORTAL TAGS ===
        # Register tags at actual portal positions for alignment validation
        # NOTE: In data model, 'front' faces NORTH (portal_back param),
        #       'back' faces SOUTH (portal_front param) - confusing naming!
        # Tags use data model IDs (front/back), not param names

        # Front portal (SOUTH facing, at oy) - data model ID 'back'
        if self.portal_front:
            self._register_portal_tag(
                portal_id="back",  # Data model ID
                center_x=ox,
                center_y=oy,
                center_z=front_z,
                direction=PortalDirection.SOUTH,
            )

        # Back portal (NORTH facing, at oy + length) - data model ID 'front'
        if self.portal_back:
            self._register_portal_tag(
                portal_id="front",  # Data model ID
                center_x=ox,
                center_y=oy + length,
                center_z=back_z,
                direction=PortalDirection.NORTH,
            )

        return brushes

    def validate(self) -> ValidationResult:
        """Validate StraightHall configuration for playability."""
        result = ValidationResult()

        hw = self.hall_width / 2
        # StraightHall now has a center floor box, so each arm is (length/2 - hw)
        # effective length from center edge to portal
        # Use max(0, ...) to handle edge case where hall is shorter than hall_width
        front_arm_length = max(0.0, (self.length / 2) - hw)
        back_arm_length = max(0.0, (self.length / 2) - hw)

        # Check front arm (from center to front portal)
        if abs(self.height_delta_front) > 0.01:
            self._check_arm_traversability(
                "Front",
                front_arm_length,
                abs(self.height_delta_front),
                self.step_height,
                result
            )

        # Check back arm (from center to back portal)
        if abs(self.height_delta_back) > 0.01:
            self._check_arm_traversability(
                "Back",
                back_arm_length,
                abs(self.height_delta_back),
                self.step_height,
                result
            )

        return result


class SquareCorner(HallBase):
    """A 90-degree corner corridor (2×2 footprint).

    Creates an L-shaped corridor with two portals at right angles.
    Arms meet directly at the corner junction.

    Portal Layout (0° rotation):
        Portal A: SOUTH (bottom of vertical arm)
        Portal B: EAST (right end of horizontal arm)

    Floor plan:
        +-----+--[B]--+
        |     |       |
        | hub |  arm  |
        |     |   B   |
        +-----+-------+
        |     |
        |arm A|
        +--[A]+

    Attributes:
        arm_length: Length of each arm (both arms same length)
        portal_a: Enable portal A (south)
        portal_b: Enable portal B (east)
        height_delta_a: Height change from center to portal A
        height_delta_b: Height change from center to portal B
    """

    arm_length: float = 128.0
    portal_a: bool = True
    portal_b: bool = True
    height_delta_a: float = 0.0
    height_delta_b: float = 0.0
    step_height: float = 12.0
    # Layout generator sets these to align junction with footprint cell centers
    _center_x_offset: float = 64.0   # Center at cell (0,1) X center
    _center_y_offset: float = 192.0  # Center at cell (0,1) Y center

    def __init__(self):
        super().__init__()
        self._floor_texture: str = ""
        self._wall_texture: str = ""
        self._ceiling_texture: str = ""

    @property
    def floor_texture(self) -> str:
        return self._floor_texture or self.params.texture

    @property
    def wall_texture(self) -> str:
        return self._wall_texture or self.params.texture

    @property
    def ceiling_texture(self) -> str:
        return self._ceiling_texture or self.params.texture

    @classmethod
    def get_display_name(cls) -> str:
        return "Square Corner"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        instance = cls()
        schema = instance._get_base_schema()
        schema.update(instance._get_stair_schema())
        schema.update({
            "arm_length": {
                "type": "float", "default": 128.0, "min": 64, "max": 512, "label": "Arm Length",
                "description": "Length of each arm from the corner junction"
            },
            "portal_a": {
                "type": "bool", "default": True, "label": "Portal A (Bottom)",
                "description": "Enable portal at the south end of the vertical arm"
            },
            "portal_b": {
                "type": "bool", "default": True, "label": "Portal B (Right)",
                "description": "Enable portal at the east end of the horizontal arm"
            },
            "height_delta_a": {
                "type": "float", "default": 0.0, "min": -256, "max": 256, "label": "Portal A Height Delta",
                "description": "Height change from corner to Portal A (negative = descend)"
            },
            "height_delta_b": {
                "type": "float", "default": 0.0, "min": -256, "max": 256, "label": "Portal B Height Delta",
                "description": "Height change from corner to Portal B (negative = descend)"
            },
        })
        return schema

    def generate(self) -> List[Brush]:
        """Generate square corner geometry.

        Creates an L-shaped 90-degree corner with 2x2 cell footprint.

        Geometry Layout (top-down view at 0° rotation):
        ```
        +-------+-------+
        | SOLID | ARM B |  Cell (0,1) is solid, (1,1) has east arm
        | WALL  | →EAST |
        +-------+---+---+
        | ARM A |   |      Cell (0,0) has south arm, (1,0) is solid
        | ↓SOUTH|   |
        +---+---+---+
            Portal A
        ```

        Portals:
        - Portal A: SOUTH face of cell (0,0)
        - Portal B: EAST face of cell (1,1)

        The center junction is at grid position (1,1) where the two arms meet.
        Solid walls seal the northwest quadrant and west side of center.
        """
        self._reset_tags()  # Reset tags for this generation
        brushes: List[Brush] = []
        t = self.wall_thickness
        hw = self.hall_width / 2
        h = self.hall_height
        ox, oy, oz = self.params.origin

        grid_size = 128.0

        # Center junction point using offset attributes (set by layout generator)
        # Junction should be at cell (0,1) center for proper portal alignment:
        # - Portal 'a' at cell (0,0): X center = 64, portal at Y=0
        # - Portal 'b' at cell (1,1): Y center = 192, portal at X=256
        cx = ox + self._center_x_offset
        cy = oy + self._center_y_offset

        # Store arm end positions for tag registration later
        arm_b_end_x = ox + 2 * grid_size

        # ===== CENTER FLOOR BOX =====
        # Per CLAUDE.md §7: Center floor extends by t in ALL directions
        brushes.append(self._box(
            cx - hw - t, cy - hw - t, oz - t,
            cx + hw + t, cy + hw + t, oz,
            self.floor_texture
        ))

        # ===== CENTER CEILING BOX =====
        brushes.append(self._box(
            cx - hw - t, cy - hw - t, oz + h,
            cx + hw + t, cy + hw + t, oz + h + t,
            self.ceiling_texture
        ))

        # ===== ARM A (SOUTH - extends from center toward portal A) =====
        arm_a_start_y = cy - hw - t
        arm_a_end_y = oy

        # Floor for arm A
        brushes.append(self._box(
            cx - hw - t, arm_a_end_y - t, oz - t,
            cx + hw + t, arm_a_start_y, oz,
            self.floor_texture
        ))
        # Ceiling for arm A
        brushes.append(self._box(
            cx - hw - t, arm_a_end_y - t, oz + h,
            cx + hw + t, arm_a_start_y, oz + h + t,
            self.ceiling_texture
        ))

        # Walls for arm A
        # West wall
        brushes.append(self._box(
            cx - hw - t, arm_a_end_y - t, oz,
            cx - hw, arm_a_start_y, oz + h,
            self.wall_texture
        ))
        # East wall
        brushes.append(self._box(
            cx + hw, arm_a_end_y - t, oz,
            cx + hw + t, arm_a_start_y, oz + h,
            self.wall_texture
        ))

        # ===== ARM B (EAST - extends from center toward portal B) =====
        arm_b_start_x = cx + hw + t
        arm_b_end_x = ox + 2 * grid_size

        # Floor for arm B
        brushes.append(self._box(
            arm_b_start_x, cy - hw - t, oz - t,
            arm_b_end_x + t, cy + hw + t, oz,
            self.floor_texture
        ))
        # Ceiling for arm B
        brushes.append(self._box(
            arm_b_start_x, cy - hw - t, oz + h,
            arm_b_end_x + t, cy + hw + t, oz + h + t,
            self.ceiling_texture
        ))

        # Walls for arm B
        # North wall
        brushes.append(self._box(
            arm_b_start_x, cy + hw, oz,
            arm_b_end_x + t, cy + hw + t, oz + h,
            self.wall_texture
        ))
        # South wall
        brushes.append(self._box(
            arm_b_start_x, cy - hw - t, oz,
            arm_b_end_x + t, cy - hw, oz + h,
            self.wall_texture
        ))

        # ===== CORNER FILL WALLS =====
        # West wall for center area (seals the west side of the L)
        # Extends from end of arm A walls up to the ceiling
        # Only create if there's space between arm A and center
        if arm_a_start_y < cy - hw:
            brushes.append(self._box(
                cx - hw - t, arm_a_start_y, oz,
                cx - hw, cy + hw, oz + h,
                self.wall_texture
            ))

        # Northwest corner fill (seals the area above the corridor in cell column 0)
        # Only create if there's space above cy + hw (corridor ceiling)
        corner_fill_min_y = cy + hw
        corner_fill_max_y = oy + 2 * grid_size
        if corner_fill_max_y > corner_fill_min_y:
            brushes.append(self._box(
                cx - hw - t, corner_fill_min_y, oz - t,
                cx + hw + t, corner_fill_max_y, oz + h + t,
                self.wall_texture
            ))

        # Solid corner fill for cell (1,0) - the unused quadrant
        # Only needed when the junction is at cell (0,1), not (1,1)
        solid_corner_min_x = cx + hw + t
        solid_corner_max_x = ox + 2 * grid_size
        solid_corner_min_y = oy - t
        solid_corner_max_y = cy - hw - t
        if solid_corner_max_x > solid_corner_min_x and solid_corner_max_y > solid_corner_min_y:
            brushes.append(self._box(
                solid_corner_min_x, solid_corner_min_y, oz - t,
                solid_corner_max_x, solid_corner_max_y, oz + h + t,
                self.wall_texture
            ))

        # ===== PORTAL WALLS =====
        # Portal A (SOUTH)
        if self.portal_a:
            # Left wall piece
            brushes.append(self._box(
                cx - hw - t, oy - t, oz,
                cx - hw, oy, oz + h,
                self.wall_texture
            ))
            # Right wall piece
            brushes.append(self._box(
                cx + hw, oy - t, oz,
                cx + hw + t, oy, oz + h,
                self.wall_texture
            ))
        else:
            # Solid wall
            brushes.append(self._box(
                cx - hw - t, oy - t, oz,
                cx + hw + t, oy, oz + h,
                self.wall_texture
            ))

        # Portal B (EAST)
        if self.portal_b:
            # Top wall piece
            brushes.append(self._box(
                arm_b_end_x, cy + hw, oz,
                arm_b_end_x + t, cy + hw + t, oz + h,
                self.wall_texture
            ))
            # Bottom wall piece
            brushes.append(self._box(
                arm_b_end_x, cy - hw - t, oz,
                arm_b_end_x + t, cy - hw, oz + h,
                self.wall_texture
            ))
        else:
            # Solid wall
            brushes.append(self._box(
                arm_b_end_x, cy - hw - t, oz,
                arm_b_end_x + t, cy + hw + t, oz + h,
                self.wall_texture
            ))

        # === REGISTER PORTAL TAGS ===
        # Portal A (SOUTH facing, at Y=oy)
        if self.portal_a:
            self._register_portal_tag(
                portal_id="a",
                center_x=cx,
                center_y=oy,
                center_z=oz,
                direction=PortalDirection.SOUTH,
            )

        # Portal B (EAST facing, at X=arm_b_end_x)
        if self.portal_b:
            self._register_portal_tag(
                portal_id="b",
                center_x=arm_b_end_x,
                center_y=cy,
                center_z=oz,
                direction=PortalDirection.EAST,
            )

        return brushes

    def validate(self) -> ValidationResult:
        """Validate SquareCorner configuration."""
        result = ValidationResult()

        # Validate dimensions
        if self.hall_width < 64:
            result.add_warning("width", "Hall width below minimum", self.hall_width, 64, "error")

        if self.arm_length < 64:
            result.add_warning("arm_length", "Arm length below minimum", self.arm_length, 64, "error")

        # Check traversability for height deltas
        if abs(self.height_delta_a) > 0:
            self._check_arm_traversability("Arm A", self.arm_length, abs(self.height_delta_a), self.step_height, result)

        if abs(self.height_delta_b) > 0:
            self._check_arm_traversability("Arm B", self.arm_length, abs(self.height_delta_b), self.step_height, result)

        return result

    def _check_arm_traversability(
        self,
        arm_name: str,
        arm_length: float,
        height_delta: float,
        step_height: float,
        result: ValidationResult
    ) -> None:
        """Check if arm stairs are traversable per idTech constraints."""
        num_steps = max(1, int(height_delta / step_height))
        actual_step_height = height_delta / num_steps
        step_depth = arm_length / num_steps

        # Per CLAUDE.md §4: max step height 18 units, comfortable 16
        if actual_step_height > 18:
            result.add_warning(arm_name, "Step height exceeds maximum 18 units", actual_step_height, 18, "error")
        elif actual_step_height > 16:
            result.add_warning(arm_name, "Step height above comfortable 16 units", actual_step_height, 16)

        # Per CLAUDE.md §4: min step depth 16 units
        if step_depth < 16:
            result.add_warning(arm_name, "Step depth below minimum 16 units", step_depth, 16, "error")


# NOTE: LJunction and unified Corner classes have been removed.
# Use SquareCorner for all 90-degree corner needs.


class TJunction(HallBase):
    """A three-way T-junction with optional stairs.

    Features:
    - True T-shape: crossbar running WEST-EAST, stem going NORTH
    - Optional portals at each of the three ends (WEST, EAST, NORTH)
    - Optional height variation with stairs per arm
    - Proper sealed geometry

    Floor plan (footprint 3x2 cells):
    Cell layout:
        (0,1) (1,1) (2,1)   <- NORTH portal at (1,1) - top of stem
        (0,0) (1,0) (2,0)   <- WEST portal at (0,0), EAST portal at (2,0) - crossbar ends
              ^--- center junction (no portal)

    Visual T-shape:
        [W]-----+-----[E]   <- crossbar (row 0)
                |
               [N]          <- stem (row 1)

    Height baseline: Center junction is at oz.
    height_delta_west = height change from junction to west portal
    height_delta_east = height change from junction to east portal
    height_delta_north = height change from junction to north portal
    """

    crossbar_length: float = 256.0   # Total WEST-EAST crossbar length
    stem_length: float = 128.0       # NORTH stem length
    portal_west: bool = True
    portal_east: bool = True
    portal_north: bool = True
    height_delta_west: float = 0.0   # Height delta at west portal
    height_delta_east: float = 0.0   # Height delta at east portal
    height_delta_north: float = 0.0  # Height delta at north portal
    step_height: float = 12.0        # Max step height for stairs
    # Internal: set by layout generator for cell alignment
    _center_x_offset: float = 0.0
    _center_y_offset: float = 0.0

    @classmethod
    def get_display_name(cls) -> str:
        return "T Junction"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        instance = cls()
        schema = instance._get_base_schema()
        schema.update(instance._get_stair_schema())
        schema.update({
            "crossbar_length": {
                "type": "float", "default": 256.0, "min": 128, "max": 1024, "label": "Crossbar Length",
                "description": "Total length of the horizontal crossbar (west to east)"
            },
            "stem_length": {
                "type": "float", "default": 128.0, "min": 64, "max": 512, "label": "Stem Length",
                "description": "Length of the vertical stem extending north from the junction"
            },
            "portal_west": {
                "type": "bool", "default": True, "label": "West Portal",
                "description": "Enable portal at the west end of the crossbar"
            },
            "portal_east": {
                "type": "bool", "default": True, "label": "East Portal",
                "description": "Enable portal at the east end of the crossbar"
            },
            "portal_north": {
                "type": "bool", "default": True, "label": "North Portal",
                "description": "Enable portal at the north end of the stem"
            },
            "height_delta_west": {
                "type": "float", "default": 0.0, "min": -256, "max": 256, "label": "West Height Delta",
                "description": "Height change from junction to west portal (negative = descend)"
            },
            "height_delta_east": {
                "type": "float", "default": 0.0, "min": -256, "max": 256, "label": "East Height Delta",
                "description": "Height change from junction to east portal (negative = descend)"
            },
            "height_delta_north": {
                "type": "float", "default": 0.0, "min": -256, "max": 256, "label": "North Height Delta",
                "description": "Height change from junction to north portal (negative = descend)"
            },
        })
        return schema

    def generate(self) -> List[Brush]:
        self._reset_tags()  # Reset tags for this generation
        ox_origin, oy_origin, oz = self.params.origin
        brushes: List[Brush] = []

        hw = self.hall_width / 2
        t = self.wall_thickness
        h = self.hall_height
        crossbar_len = self.crossbar_length
        stem_len = self.stem_length

        # Center position (can be offset by layout generator)
        cx = ox_origin + self._center_x_offset
        cy = oy_origin + self._center_y_offset

        # Half crossbar length (each arm from center)
        half_crossbar = crossbar_len / 2

        # Calculate Z heights at each portal
        center_z = oz  # Center (T intersection) is baseline
        west_z = oz + self.height_delta_west
        east_z = oz + self.height_delta_east
        north_z = oz + self.height_delta_north

        # Height range for walls/ceiling
        min_z = min(center_z, west_z, east_z, north_z)
        max_z = max(center_z, west_z, east_z, north_z)
        ceiling_z = max_z + h
        wall_bottom = min_z - t

        # === FLOOR ===
        # Center floor section (the T intersection area)
        # Must extend by t in ALL directions for sealed geometry
        brushes.append(self._box(
            cx - hw - t, cy - hw - t, center_z - t,
            cx + hw + t, cy + hw + t, center_z
        ))

        # West arm floor (from center toward -X)
        brushes.extend(self._generate_arm_floor(
            arm_start=cx - hw,                    # Start at center edge
            arm_end=cx - half_crossbar - t,       # End at west portal (extend by t)
            perp_min=cy - hw - t,
            perp_max=cy + hw + t,
            z_start=center_z,
            z_end=west_z,
            axis="x",
            step_height=self.step_height,
            t=t
        ))

        # East arm floor (from center toward +X)
        brushes.extend(self._generate_arm_floor(
            arm_start=cx + hw,                    # Start at center edge
            arm_end=cx + half_crossbar + t,       # End at east portal (extend by t)
            perp_min=cy - hw - t,
            perp_max=cy + hw + t,
            z_start=center_z,
            z_end=east_z,
            axis="x",
            step_height=self.step_height,
            t=t
        ))

        # North arm floor (from center toward +Y to match footprint row 1)
        # In footprint coords: row 0 = crossbar, row 1 = stem with NORTH portal
        brushes.extend(self._generate_arm_floor(
            arm_start=cy + hw,                    # Start at center edge (+Y side)
            arm_end=cy + stem_len + t,            # End at north portal (extend by t)
            perp_min=cx - hw - t,
            perp_max=cx + hw + t,
            z_start=center_z,
            z_end=north_z,
            axis="y",
            step_height=self.step_height,
            t=t
        ))

        # === CEILING ===
        # Crossbar ceiling (full WEST-EAST span)
        brushes.append(self._box(
            cx - half_crossbar - t, cy - hw - t, ceiling_z,
            cx + half_crossbar + t, cy + hw + t, ceiling_z + t
        ))
        # Stem ceiling (NORTH arm, toward +Y)
        brushes.append(self._box(
            cx - hw - t, cy + hw, ceiling_z,
            cx + hw + t, cy + stem_len + t, ceiling_z + t
        ))

        # === WALLS ===
        # Back wall of crossbar (full length, no opening - this is the flat back of the T)
        # Located at -Y side (row 0's back edge)
        brushes.append(self._box(
            cx - half_crossbar - t, cy - hw - t, wall_bottom,
            cx + half_crossbar + t, cy - hw, ceiling_z
        ))

        # Front wall of crossbar (with opening for stem toward +Y)
        # West section (before stem opening)
        brushes.append(self._box(
            cx - half_crossbar - t, cy + hw, wall_bottom,
            cx - hw, cy + hw + t, ceiling_z
        ))
        # East section (after stem opening)
        brushes.append(self._box(
            cx + hw, cy + hw, wall_bottom,
            cx + half_crossbar + t, cy + hw + t, ceiling_z
        ))

        # West wall of stem (toward +Y)
        brushes.append(self._box(
            cx - hw - t, cy + hw, wall_bottom,
            cx - hw, cy + stem_len + t, ceiling_z
        ))

        # East wall of stem (toward +Y)
        brushes.append(self._box(
            cx + hw, cy + hw, wall_bottom,
            cx + hw + t, cy + stem_len + t, ceiling_z
        ))

        # === END WALLS WITH PORTALS ===
        # West end wall
        brushes.extend(self._wall_with_portal(
            cx - half_crossbar - t, cy - hw - t, west_z,
            cx - half_crossbar, cy + hw + t, ceiling_z,
            self.portal_west,
            portal_axis="y",
            portal_center=cy
        ))

        # East end wall
        brushes.extend(self._wall_with_portal(
            cx + half_crossbar, cy - hw - t, east_z,
            cx + half_crossbar + t, cy + hw + t, ceiling_z,
            self.portal_east,
            portal_axis="y",
            portal_center=cy
        ))

        # North end wall (stem end, at +Y)
        brushes.extend(self._wall_with_portal(
            cx - hw - t, cy + stem_len, north_z,
            cx + hw + t, cy + stem_len + t, ceiling_z,
            self.portal_north,
            portal_axis="x"
        ))

        # === REGISTER PORTAL TAGS ===
        # West portal (WEST facing, at X = cx - half_crossbar)
        if self.portal_west:
            self._register_portal_tag(
                portal_id="west",
                center_x=cx - half_crossbar,
                center_y=cy,
                center_z=west_z,
                direction=PortalDirection.WEST,
            )

        # East portal (EAST facing, at X = cx + half_crossbar)
        if self.portal_east:
            self._register_portal_tag(
                portal_id="east",
                center_x=cx + half_crossbar,
                center_y=cy,
                center_z=east_z,
                direction=PortalDirection.EAST,
            )

        # North portal (NORTH facing, at Y = cy + stem_len)
        if self.portal_north:
            self._register_portal_tag(
                portal_id="north",
                center_x=cx,
                center_y=cy + stem_len,
                center_z=north_z,
                direction=PortalDirection.NORTH,
            )

        return brushes

    def validate(self) -> ValidationResult:
        """Validate TJunction configuration for playability."""
        result = ValidationResult()

        hw = self.hall_width / 2
        # Effective arm lengths from center to portals
        west_arm_length = (self.crossbar_length / 2) - hw
        east_arm_length = (self.crossbar_length / 2) - hw
        north_arm_length = self.stem_length - hw

        # Check West arm
        if abs(self.height_delta_west) > 0.01:
            self._check_arm_traversability(
                "West",
                west_arm_length,
                abs(self.height_delta_west),
                self.step_height,
                result
            )

        # Check East arm
        if abs(self.height_delta_east) > 0.01:
            self._check_arm_traversability(
                "East",
                east_arm_length,
                abs(self.height_delta_east),
                self.step_height,
                result
            )

        # Check North arm
        if abs(self.height_delta_north) > 0.01:
            self._check_arm_traversability(
                "North",
                north_arm_length,
                abs(self.height_delta_north),
                self.step_height,
                result
            )

        return result


class Crossroads(HallBase):
    """A four-way intersection with optional stairs.

    Features:
    - Four arms extending from a central hub
    - Optional portals at each arm end
    - Optional height variation with stairs per arm
    - Corner pillars to fill inside corners

    Floor plan:
         +--[N]--+
         |       |
    +--[W]+     +[E]--+
         |       |
         +--[S]--+

    Height baseline: Center hub is at oz.
    height_delta_north/south/east/west = height change from center to each portal
    """

    arm_north_length: float = 128.0  # Length of north arm
    arm_south_length: float = 128.0  # Length of south arm
    arm_east_length: float = 128.0   # Length of east arm
    arm_west_length: float = 128.0   # Length of west arm
    portal_north: bool = True
    portal_south: bool = True
    portal_east: bool = True
    portal_west: bool = True
    height_delta_north: float = 0.0   # Height delta at north portal
    height_delta_south: float = 0.0   # Height delta at south portal
    height_delta_east: float = 0.0    # Height delta at east portal
    height_delta_west: float = 0.0    # Height delta at west portal
    step_height: float = 12.0         # Max step height for stairs

    @classmethod
    def get_display_name(cls) -> str:
        return "Crossroads"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        instance = cls()
        schema = instance._get_base_schema()
        schema.update(instance._get_stair_schema())
        schema.update({
            "arm_north_length": {
                "type": "float", "default": 128.0, "min": 64, "max": 512, "label": "North Arm Length",
                "description": "Length of the north arm from center hub"
            },
            "arm_south_length": {
                "type": "float", "default": 128.0, "min": 64, "max": 512, "label": "South Arm Length",
                "description": "Length of the south arm from center hub"
            },
            "arm_east_length": {
                "type": "float", "default": 128.0, "min": 64, "max": 512, "label": "East Arm Length",
                "description": "Length of the east arm from center hub"
            },
            "arm_west_length": {
                "type": "float", "default": 128.0, "min": 64, "max": 512, "label": "West Arm Length",
                "description": "Length of the west arm from center hub"
            },
            "portal_north": {
                "type": "bool", "default": True, "label": "North Portal",
                "description": "Enable portal at the north end"
            },
            "portal_south": {
                "type": "bool", "default": True, "label": "South Portal",
                "description": "Enable portal at the south end"
            },
            "portal_east": {
                "type": "bool", "default": True, "label": "East Portal",
                "description": "Enable portal at the east end"
            },
            "portal_west": {
                "type": "bool", "default": True, "label": "West Portal",
                "description": "Enable portal at the west end"
            },
            "height_delta_north": {
                "type": "float", "default": 0.0, "min": -256, "max": 256, "label": "North Height Delta",
                "description": "Height change from center to north portal"
            },
            "height_delta_south": {
                "type": "float", "default": 0.0, "min": -256, "max": 256, "label": "South Height Delta",
                "description": "Height change from center to south portal"
            },
            "height_delta_east": {
                "type": "float", "default": 0.0, "min": -256, "max": 256, "label": "East Height Delta",
                "description": "Height change from center to east portal"
            },
            "height_delta_west": {
                "type": "float", "default": 0.0, "min": -256, "max": 256, "label": "West Height Delta",
                "description": "Height change from center to west portal"
            },
        })
        return schema

    def generate(self) -> List[Brush]:
        self._reset_tags()  # Reset tags for fresh generation
        ox, oy, oz = self.params.origin
        brushes: List[Brush] = []

        hw = self.hall_width / 2
        t = self.wall_thickness
        h = self.hall_height
        arm_n = self.arm_north_length
        arm_s = self.arm_south_length
        arm_e = self.arm_east_length
        arm_w = self.arm_west_length

        # Cross shape: center at origin, arms extend in all 4 directions
        # N/S arms along Y axis, E/W arms along X axis

        # Calculate Z heights at each portal
        center_z = oz  # Center hub is baseline
        north_z = oz + self.height_delta_north
        south_z = oz + self.height_delta_south
        east_z = oz + self.height_delta_east
        west_z = oz + self.height_delta_west

        # Height range for walls/ceiling
        min_z = min(center_z, north_z, south_z, east_z, west_z)
        max_z = max(center_z, north_z, south_z, east_z, west_z)
        ceiling_z = max_z + h
        wall_bottom = min_z - t

        # === FLOOR ===
        # Center hub floor section
        # Must extend by t in ALL directions for sealed geometry
        brushes.append(self._box(
            ox - hw - t, oy - hw - t, center_z - t,
            ox + hw + t, oy + hw + t, center_z
        ))

        # North arm stairs (from center to north portal)
        # Floor always extends by t past wall for sealed geometry
        brushes.extend(self._generate_arm_floor(
            arm_start=oy + hw,        # Start at center edge
            arm_end=oy + arm_n + t,   # Always extend by t
            perp_min=ox - hw - t,
            perp_max=ox + hw + t,
            z_start=center_z,
            z_end=north_z,
            axis="y",
            step_height=self.step_height,
            t=t
        ))

        # South arm stairs (from center to south portal)
        # Floor always extends by t past wall for sealed geometry
        brushes.extend(self._generate_arm_floor(
            arm_start=oy - hw,        # Start at center edge
            arm_end=oy - arm_s - t,   # Always extend by t
            perp_min=ox - hw - t,
            perp_max=ox + hw + t,
            z_start=center_z,
            z_end=south_z,
            axis="y",
            step_height=self.step_height,
            t=t
        ))

        # East arm stairs (from center to east portal)
        # Floor always extends by t past wall for sealed geometry
        brushes.extend(self._generate_arm_floor(
            arm_start=ox + hw,        # Start at center edge
            arm_end=ox + arm_e + t,   # Always extend by t
            perp_min=oy - hw - t,
            perp_max=oy + hw + t,
            z_start=center_z,
            z_end=east_z,
            axis="x",
            step_height=self.step_height,
            t=t
        ))

        # West arm stairs (from center to west portal)
        # Floor always extends by t past wall for sealed geometry
        brushes.extend(self._generate_arm_floor(
            arm_start=ox - hw,        # Start at center edge
            arm_end=ox - arm_w - t,   # Always extend by t
            perp_min=oy - hw - t,
            perp_max=oy + hw + t,
            z_start=center_z,
            z_end=west_z,
            axis="x",
            step_height=self.step_height,
            t=t
        ))

        # === CEILING (cross-shaped, at max height) ===
        # N-S corridor ceiling (spans from south end to north end)
        brushes.append(self._box(
            ox - hw - t, oy - arm_s - t, ceiling_z,
            ox + hw + t, oy + arm_n + t, ceiling_z + t
        ))
        # West arm ceiling
        brushes.append(self._box(
            ox - arm_w - t, oy - hw - t, ceiling_z,
            ox - hw, oy + hw + t, ceiling_z + t
        ))
        # East arm ceiling
        brushes.append(self._box(
            ox + hw, oy - hw - t, ceiling_z,
            ox + arm_e + t, oy + hw + t, ceiling_z + t
        ))

        # === WALLS (span full height range) ===
        # North arm - left and right walls
        brushes.append(self._box(ox - hw - t, oy + hw, wall_bottom, ox - hw, oy + arm_n + t, ceiling_z))
        brushes.append(self._box(ox + hw, oy + hw, wall_bottom, ox + hw + t, oy + arm_n + t, ceiling_z))

        # South arm - left and right walls
        brushes.append(self._box(ox - hw - t, oy - arm_s - t, wall_bottom, ox - hw, oy - hw, ceiling_z))
        brushes.append(self._box(ox + hw, oy - arm_s - t, wall_bottom, ox + hw + t, oy - hw, ceiling_z))

        # West arm - top and bottom walls
        brushes.append(self._box(ox - arm_w - t, oy + hw, wall_bottom, ox - hw, oy + hw + t, ceiling_z))
        brushes.append(self._box(ox - arm_w - t, oy - hw - t, wall_bottom, ox - hw, oy - hw, ceiling_z))

        # East arm - top and bottom walls
        brushes.append(self._box(ox + hw, oy + hw, wall_bottom, ox + arm_e + t, oy + hw + t, ceiling_z))
        brushes.append(self._box(ox + hw, oy - hw - t, wall_bottom, ox + arm_e + t, oy - hw, ceiling_z))

        # === END WALLS WITH PORTALS ===
        # End walls must extend to ceiling_z, not just portal_z + h
        # This ensures no gaps at multi-height junctions

        # North end
        brushes.extend(self._wall_with_portal(
            ox - hw - t, oy + arm_n, north_z,
            ox + hw + t, oy + arm_n + t, ceiling_z,
            self.portal_north,
            portal_axis="x"
        ))

        # South end
        brushes.extend(self._wall_with_portal(
            ox - hw - t, oy - arm_s - t, south_z,
            ox + hw + t, oy - arm_s, ceiling_z,
            self.portal_south,
            portal_axis="x"
        ))

        # East end
        brushes.extend(self._wall_with_portal(
            ox + arm_e, oy - hw - t, east_z,
            ox + arm_e + t, oy + hw + t, ceiling_z,
            self.portal_east,
            portal_axis="y"
        ))

        # West end
        brushes.extend(self._wall_with_portal(
            ox - arm_w - t, oy - hw - t, west_z,
            ox - arm_w, oy + hw + t, ceiling_z,
            self.portal_west,
            portal_axis="y"
        ))

        # === REGISTER PORTAL TAGS ===
        # Register tags at actual portal positions for tag-based validation
        if self.portal_north:
            self._register_portal_tag(
                portal_id="north",
                center_x=ox,
                center_y=oy + arm_n,
                center_z=north_z,
                direction=PortalDirection.NORTH,
            )
        if self.portal_south:
            self._register_portal_tag(
                portal_id="south",
                center_x=ox,
                center_y=oy - arm_s,
                center_z=south_z,
                direction=PortalDirection.SOUTH,
            )
        if self.portal_east:
            self._register_portal_tag(
                portal_id="east",
                center_x=ox + arm_e,
                center_y=oy,
                center_z=east_z,
                direction=PortalDirection.EAST,
            )
        if self.portal_west:
            self._register_portal_tag(
                portal_id="west",
                center_x=ox - arm_w,
                center_y=oy,
                center_z=west_z,
                direction=PortalDirection.WEST,
            )

        return brushes

    def validate(self) -> ValidationResult:
        """Validate Crossroads configuration for playability."""
        result = ValidationResult()

        hw = self.hall_width / 2
        # Effective arm lengths from center hub to portal
        # Each arm has its own length now
        arms = [
            ("North", self.arm_north_length - hw, self.height_delta_north),
            ("South", self.arm_south_length - hw, self.height_delta_south),
            ("East", self.arm_east_length - hw, self.height_delta_east),
            ("West", self.arm_west_length - hw, self.height_delta_west),
        ]

        for arm_name, effective_length, height_delta in arms:
            if abs(height_delta) > 0.01:
                self._check_arm_traversability(
                    arm_name,
                    effective_length,
                    abs(height_delta),
                    self.step_height,
                    result
                )

        return result


class VerticalStairHall(HallBase):
    """A vertical staircase corridor connecting two floor levels.

    This primitive allows dungeons to span multiple floors by providing
    a stairwell that connects a lower portal to an upper portal.

    Features:
    - BOTTOM portal at z_offset (lower floor level)
    - TOP portal at z_offset + height_change (upper floor level)
    - Enclosed stairwell with landing at each level
    - Configurable stair dimensions

    Floor plan (top view):
    +--[TOP]--+
    |         |
    |  STAIR  |
    |  WELL   |
    |         |
    +--[BOT]--+

    Vertical section (side view):
    +---TOP---+  <- upper portal at z + height_change
    |  /----  |
    | /       |
    |/        |
    +---BOT---+  <- lower portal at z

    Portals:
    - bottom: SOUTH facing at z_offset (lower level)
    - top: NORTH facing at z_offset + height_change (upper level)

    Note: When placed, the primitive's z_offset determines the BOTTOM floor level.
    The TOP portal is automatically at z_offset + height_change.
    """

    # Stairwell dimensions
    stair_run: float = 320.0         # Horizontal length for stairs (Y direction) - increased for 160-unit rise
    height_change: float = 160.0     # Z difference between levels (one floor = 160 units)
    portal_bottom: bool = True       # Enable bottom portal (SOUTH)
    portal_top: bool = True          # Enable top portal (NORTH)
    step_height: float = 12.0        # Maximum step height

    @classmethod
    def get_display_name(cls) -> str:
        return "Vertical Stair Hall"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        instance = cls()
        schema = instance._get_base_schema()
        schema.update(instance._get_stair_schema())
        schema.update({
            "stair_run": {
                "type": "float",
                "default": 320.0,
                "min": 128,
                "max": 640,
                "label": "Stair Run (Length)",
                "description": "Horizontal distance for the stair section (affects slope)"
            },
            "height_change": {
                "type": "float",
                "default": 160.0,
                "min": 64,
                "max": 320,
                "label": "Height Change",
                "description": "Vertical rise from bottom to top portal (one floor = 160)"
            },
            "portal_bottom": {
                "type": "bool",
                "default": True,
                "label": "Bottom Portal",
                "description": "Enable portal at the lower floor level (south facing)"
            },
            "portal_top": {
                "type": "bool",
                "default": True,
                "label": "Top Portal",
                "description": "Enable portal at the upper floor level (north facing)"
            },
        })
        return schema

    def generate(self) -> List[Brush]:
        """Generate the vertical stairwell geometry.

        Structure:
        1. Bottom landing with BOTTOM portal (SOUTH facing)
        2. Stair section rising height_change over stair_run
        3. Top landing with TOP portal (NORTH facing)
        4. Sealed walls/ceiling enclosing the stairwell
        """
        self._reset_tags()  # Reset tags for fresh generation
        ox, oy, oz = self.params.origin
        brushes: List[Brush] = []

        hw = self.hall_width / 2
        t = self.wall_thickness
        h = self.hall_height
        run = self.stair_run
        rise = self.height_change

        # Heights
        bottom_z = oz                    # Bottom landing floor
        top_z = oz + rise                # Top landing floor
        ceiling_z = top_z + h            # Ceiling above top landing

        # Landing sizes (half hall_width in Y direction each)
        landing_depth = hw  # Each landing is hw deep in Y

        # Y coordinates
        bottom_landing_y1 = oy           # Start of bottom landing
        bottom_landing_y2 = oy + landing_depth
        stair_y1 = bottom_landing_y2     # Start of stairs
        stair_y2 = stair_y1 + run        # End of stairs
        top_landing_y1 = stair_y2        # Start of top landing
        top_landing_y2 = stair_y2 + landing_depth  # End of top landing

        # Total Y extent for walls - INSIDE footprint bounds
        # Portal openings must be at footprint edges for proper alignment
        total_y1 = oy
        total_y2 = top_landing_y2

        # === BOTTOM LANDING FLOOR ===
        # Floor extends by t in X (perpendicular to portal) but NOT in Y (portal direction)
        # This keeps portal at footprint edge (Y=0) for proper alignment
        brushes.append(self._box(
            ox - hw - t, bottom_landing_y1, bottom_z - t,
            ox + hw + t, bottom_landing_y2, bottom_z
        ))

        # === STAIR SECTION ===
        # VerticalStairHall ALWAYS generates stairs - it's the dedicated floor connector
        # (ignores generate_stairs toggle which applies to other halls)
        brushes.extend(self._generate_arm_stairs(
            arm_start=stair_y1,
            arm_end=stair_y2,
            perp_min=ox - hw - t,
            perp_max=ox + hw + t,
            z_start=bottom_z,
            z_end=top_z,
            axis="y",
            step_height=self.step_height,
            t=t
        ))

        # === TOP LANDING FLOOR ===
        # Floor extends by t in X but NOT in Y (portal direction)
        # This keeps portal at footprint edge for proper alignment
        brushes.append(self._box(
            ox - hw - t, top_landing_y1, top_z - t,
            ox + hw + t, top_landing_y2, top_z
        ))

        # === CEILING ===
        # Single ceiling spanning the entire stairwell at ceiling_z
        brushes.append(self._box(
            ox - hw - t, total_y1, ceiling_z,
            ox + hw + t, total_y2, ceiling_z + t
        ))

        # === SIDE WALLS ===
        # Left wall (spans full height from bottom floor to ceiling)
        brushes.append(self._box(
            ox - hw - t, total_y1, bottom_z - t,
            ox - hw, total_y2, ceiling_z
        ))

        # Right wall (spans full height from bottom floor to ceiling)
        brushes.append(self._box(
            ox + hw, total_y1, bottom_z - t,
            ox + hw + t, total_y2, ceiling_z
        ))

        # === END WALLS WITH PORTALS ===
        # Walls are INSIDE footprint bounds so portals align with footprint edges

        # Bottom wall (SOUTH) with portal at bottom_z
        # Wall from Y=0 to Y=t, portal opening at Y=0 (footprint south edge)
        brushes.extend(self._wall_with_portal(
            ox - hw - t, bottom_landing_y1, bottom_z,
            ox + hw + t, bottom_landing_y1 + t, ceiling_z,
            self.portal_bottom,
            portal_axis="x"
        ))

        # Top wall (NORTH) with portal at top_z
        # Wall from Y=top-t to Y=top, portal opening at Y=top (footprint north edge)
        brushes.extend(self._wall_with_portal(
            ox - hw - t, top_landing_y2 - t, top_z,
            ox + hw + t, top_landing_y2, ceiling_z,
            self.portal_top,
            portal_axis="x"
        ))

        # === FILL WALLS UNDER STAIRS ===
        # Need to seal the space under the staircase
        # This creates a sloping fill from bottom landing to top landing base

        # Bottom fill: wall from bottom floor up to where stairs start
        # This seals the bottom of the stairwell
        brushes.append(self._box(
            ox - hw - t, stair_y1, bottom_z - t,
            ox + hw + t, stair_y2, bottom_z
        ))

        # === REGISTER PORTAL TAGS ===
        # Register tags at actual portal positions for tag-based validation
        # VerticalStairHall has bottom (SOUTH) and top (NORTH) portals at different Z levels
        if self.portal_bottom:
            self._register_portal_tag(
                portal_id="bottom",
                center_x=ox,
                center_y=oy,  # bottom_landing_y1 = oy
                center_z=bottom_z,
                direction=PortalDirection.SOUTH,
            )
        if self.portal_top:
            self._register_portal_tag(
                portal_id="top",
                center_x=ox,
                center_y=top_landing_y2,
                center_z=top_z,
                direction=PortalDirection.NORTH,
            )

        return brushes

    def validate(self) -> ValidationResult:
        """Validate VerticalStairHall configuration for playability."""
        result = ValidationResult()

        # Check if stairs are traversable
        if abs(self.height_change) > 0.01:
            self._check_arm_traversability(
                "Stairwell",
                self.stair_run,
                abs(self.height_change),
                self.step_height,
                result
            )

        return result


class SecretHall(HallBase):
    """A corridor with CLIP-textured side walls for walk-through secret passages.

    Features:
    - Same geometry as StraightHall
    - Side walls use CLIP texture (walk-through from outside)
    - Floor, ceiling, and end walls remain solid
    - Ideal for hidden shortcuts between areas

    Floor plan:
    +--[portal]--+
    |            |  <- CLIP wall (walk-through)
    |   HALL     |
    |            |  <- CLIP wall (walk-through)
    +--[portal]--+

    The CLIP texture allows players to walk through the side walls
    from outside, creating hidden passage entrances.
    """

    length: float = 256.0
    portal_front: bool = True
    portal_back: bool = True
    height_delta_front: float = 0.0  # Height delta at front portal (from center)
    height_delta_back: float = 0.0   # Height delta at back portal (from center)
    step_height: float = 12.0        # Max step height for stairs

    CLIP_TEXTURE: str = "CLIP"  # Walk-through texture

    @classmethod
    def get_display_name(cls) -> str:
        return "Secret Hall"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        instance = cls()
        schema = instance._get_base_schema()
        schema.update(instance._get_stair_schema())
        schema.update({
            "length": {
                "type": "float", "default": 256.0, "min": 64, "max": 1024, "label": "Length",
                "description": "Total length of the corridor from front to back"
            },
            "portal_front": {
                "type": "bool", "default": True, "label": "Front Portal",
                "description": "Enable portal opening at the front (south) end"
            },
            "portal_back": {
                "type": "bool", "default": True, "label": "Back Portal",
                "description": "Enable portal opening at the back (north) end"
            },
            "height_delta_front": {
                "type": "float", "default": 0.0, "min": -256, "max": 256, "label": "Front Height Delta",
                "description": "Height change from center to front portal (negative = descend, positive = ascend)"
            },
            "height_delta_back": {
                "type": "float", "default": 0.0, "min": -256, "max": 256, "label": "Back Height Delta",
                "description": "Height change from center to back portal (negative = descend, positive = ascend)"
            },
        })
        return schema

    def generate(self) -> List[Brush]:
        self._reset_tags()  # Reset tags for fresh generation
        ox, oy, oz = self.params.origin
        brushes: List[Brush] = []

        hw = self.hall_width / 2
        t = self.wall_thickness
        h = self.hall_height
        length = self.length

        # Calculate Z heights at each end
        # Center is at midpoint (oy + length/2), which is the height baseline
        center_y = oy + length / 2
        center_z = oz  # Center is baseline
        front_z = oz + self.height_delta_front
        back_z = oz + self.height_delta_back

        # Determine height range for walls/ceiling
        min_z = min(center_z, front_z, back_z)
        max_z = max(center_z, front_z, back_z)

        # === CENTER FLOOR BOX ===
        # Like all junction halls, SecretHall needs a center floor box at the midpoint
        # Must extend by t in ALL directions for sealed geometry
        brushes.append(self._box(
            ox - hw - t, center_y - hw - t, center_z - t,
            ox + hw + t, center_y + hw + t, center_z
        ))

        # === FRONT ARM (from center to front portal) ===
        brushes.extend(self._generate_arm_floor(
            arm_start=center_y - hw,  # Start at center edge
            arm_end=oy - t,           # Always extend by t
            perp_min=ox - hw - t,
            perp_max=ox + hw + t,
            z_start=center_z,
            z_end=front_z,
            axis="y",
            step_height=self.step_height,
            t=t
        ))

        # === BACK ARM (from center to back portal) ===
        brushes.extend(self._generate_arm_floor(
            arm_start=center_y + hw,      # Start at center edge
            arm_end=oy + length + t,      # Always extend by t
            perp_min=ox - hw - t,
            perp_max=ox + hw + t,
            z_start=center_z,
            z_end=back_z,
            axis="y",
            step_height=self.step_height,
            t=t
        ))

        # === CEILING (at max height + hall_height) ===
        # Ceiling always extends by t for proper sealing
        ceiling_z = max_z + h
        brushes.append(self._box(
            ox - hw - t, oy - t, ceiling_z,
            ox + hw + t, oy + length + t, ceiling_z + t
        ))

        # === WALLS WITH CLIP TEXTURE ===
        # Side walls use CLIP texture for walk-through from outside
        wall_bottom = min_z - t

        # Left wall (CLIP texture for walk-through)
        brushes.append(self._box(
            ox - hw - t, oy - t, wall_bottom,
            ox - hw, oy + length + t, ceiling_z,
            texture=self.CLIP_TEXTURE
        ))

        # Right wall (CLIP texture for walk-through)
        brushes.append(self._box(
            ox + hw, oy - t, wall_bottom,
            ox + hw + t, oy + length + t, ceiling_z,
            texture=self.CLIP_TEXTURE
        ))

        # === END WALLS WITH PORTALS ===
        # End walls remain solid (not CLIP)

        # Front wall at front_z height
        brushes.extend(self._wall_with_portal(
            ox - hw - t, oy - t, front_z,
            ox + hw + t, oy, ceiling_z,
            self.portal_front,
            portal_axis="x"
        ))

        # Back wall at back_z height
        brushes.extend(self._wall_with_portal(
            ox - hw - t, oy + length, back_z,
            ox + hw + t, oy + length + t, ceiling_z,
            self.portal_back,
            portal_axis="x"
        ))

        # === REGISTER PORTAL TAGS ===
        # Register tags at actual portal positions for tag-based validation
        # SecretHall uses same convention as StraightHall: front=SOUTH, back=NORTH
        if self.portal_front:
            self._register_portal_tag(
                portal_id="back",  # Data model uses 'back' for SOUTH portal
                center_x=ox,
                center_y=oy,
                center_z=front_z,
                direction=PortalDirection.SOUTH,
            )
        if self.portal_back:
            self._register_portal_tag(
                portal_id="front",  # Data model uses 'front' for NORTH portal
                center_x=ox,
                center_y=oy + length,
                center_z=back_z,
                direction=PortalDirection.NORTH,
            )

        return brushes

    def validate(self) -> ValidationResult:
        """Validate SecretHall configuration for playability."""
        result = ValidationResult()

        hw = self.hall_width / 2
        # SecretHall has a center floor box, so each arm is (length/2 - hw)
        front_arm_length = max(0.0, (self.length / 2) - hw)
        back_arm_length = max(0.0, (self.length / 2) - hw)

        # Check front arm (from center to front portal)
        if abs(self.height_delta_front) > 0.01:
            self._check_arm_traversability(
                "Front",
                front_arm_length,
                abs(self.height_delta_front),
                self.step_height,
                result
            )

        # Check back arm (from center to back portal)
        if abs(self.height_delta_back) > 0.01:
            self._check_arm_traversability(
                "Back",
                back_arm_length,
                abs(self.height_delta_back),
                self.step_height,
                result
            )

        return result
