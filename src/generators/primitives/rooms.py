"""
Room-scale geometry primitives: Sanctuary, Tomb, Tower.

These are SEALED MODULAR PRIMITIVES - they generate complete, leak-free geometry
that can be dropped into a map without checking for BSP leaks.

SEALED GEOMETRY RULES (see CLAUDE.md for full documentation):
1. Floor/ceiling must extend beyond walls by wall thickness (t) in all directions
2. Walls must span the full floor/ceiling Y-extent: (oy - t) to (oy + length + t)
3. Junction fills required where geometry width changes (e.g., nave to apse)
4. Brush overlaps at corners must be >= 8 units
5. Only designated entrances/exits are open - everything else is solid

Each primitive generates complete, playable geometry with:
- Proper enclosure (floor, ceiling, walls)
- Entrances/exits for player access
- Variation through random seed for unique instances
- 100% sealed geometry (only entrances are open)
"""

from __future__ import annotations
import math
import random
from enum import Enum
from typing import Any, Dict, List

from quake_levelgenerator.src.conversion.map_writer import Brush
from .base import GeometricPrimitive
from .portal_system import (
    PORTAL_WIDTH, PORTAL_HEIGHT, PortalSpec, generate_portal_wall,
    PortalDirection
)


class SanctuaryType(Enum):
    """Distinct sanctuary floor plan types."""
    SINGLE_NAVE = "single_nave"      # Simple rectangular, no aisles
    BASILICA = "basilica"            # Three-nave with side aisles and clerestory
    CRUCIFORM = "cruciform"          # Cross-shaped with transept
    HALL_CHURCH = "hall_church"      # Nave and aisles same height (no clerestory)
    ROTUNDA = "rotunda"              # Octagonal central space


class Sanctuary(GeometricPrimitive):
    """A sanctuary with multiple distinct floor plan types.

    Features:
    - Five architectural types: single_nave, basilica, cruciform, hall_church, rotunda
    - 100% sealed geometry (only front entrance is open when has_entrance=True)
    - Variation via random seed (dimensions AND type selection when seed=0)
    - Optional apse (semicircular back wall)
    """

    # Core dimensions
    nave_width: float = 256.0
    nave_length: float = 512.0
    nave_height: float = 192.0

    # Aisle dimensions (for basilica/hall_church)
    aisle_width: float = 96.0
    aisle_height: float = 128.0

    # Transept dimensions (for cruciform)
    transept_width: float = 384.0
    transept_depth: float = 96.0

    # Options
    apse: bool = True
    sanctuary_type: str = "random"  # single_nave, basilica, cruciform, hall_church, rotunda, random
    shell_sides: int = 4  # Number of sides for polygonal mode (4=square, 6=hex, 8=octagon)
    random_seed: int = 0

    # Pillar customization
    pillar_style: str = "square"  # square, hexagonal, octagonal, round
    pillar_capital: bool = False  # Add capitals to pillars

    # Portal control - set by layout generator based on connections
    has_entrance: bool = True

    # Wall thickness constant
    WALL_THICKNESS: float = 16.0

    @classmethod
    def get_display_name(cls) -> str:
        return "Sanctuary"

    @classmethod
    def get_category(cls) -> str:
        return "Rooms"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "sanctuary_type": {
                "type": "choice",
                "default": "random",
                "choices": ["random", "single_nave", "basilica", "cruciform", "hall_church", "rotunda"],
                "label": "Sanctuary Type",
                "description": "Floor plan style (single_nave=simple, basilica=aisles, cruciform=cross-shaped, hall_church=equal height, rotunda=octagonal)"
            },
            "nave_width": {
                "type": "float", "default": 256.0, "min": 128, "max": 512, "label": "Nave Width",
                "description": "Width of the central nave space"
            },
            "nave_length": {
                "type": "float", "default": 512.0, "min": 256, "max": 1024, "label": "Nave Length",
                "description": "Length of the nave from entrance to altar/apse"
            },
            "nave_height": {
                "type": "float", "default": 192.0, "min": 128, "max": 384, "label": "Nave Height",
                "description": "Ceiling height of the central nave"
            },
            "aisle_width": {
                "type": "float", "default": 96.0, "min": 0, "max": 256, "label": "Aisle Width",
                "description": "Width of side aisles (for basilica/hall_church types)"
            },
            "aisle_height": {
                "type": "float", "default": 128.0, "min": 64, "max": 256, "label": "Aisle Height",
                "description": "Ceiling height of side aisles (lower than nave for clerestory)"
            },
            "transept_width": {
                "type": "float", "default": 384.0, "min": 192, "max": 768, "label": "Transept Width",
                "description": "Total width of the cross-arms (for cruciform type)"
            },
            "transept_depth": {
                "type": "float", "default": 96.0, "min": 48, "max": 192, "label": "Transept Depth",
                "description": "How far the transept arms extend from the nave"
            },
            "apse": {
                "type": "bool", "default": True, "label": "Add Apse",
                "description": "Add semicircular apse at the altar end"
            },
            "shell_sides": {
                "type": "int",
                "default": 4,
                "min": 3,
                "max": 16,
                "label": "Shell Sides",
                "description": "Number of sides for polygonal mode (4=square, 6=hex, 8=octagon). Non-4 overrides sanctuary_type."
            },
            "pillar_style": {
                "type": "choice",
                "default": "square",
                "choices": ["square", "hexagonal", "octagonal", "round"],
                "label": "Pillar Style",
                "description": "Cross-section shape of arcade pillars"
            },
            "pillar_capital": {
                "type": "bool", "default": False, "label": "Pillar Capitals",
                "description": "Add decorative capitals atop pillars"
            },
            "has_entrance": {
                "type": "bool", "default": True, "label": "Enable Entrance Portal",
                "description": "Enable main entrance portal at the front"
            },
            "random_seed": {
                "type": "int", "default": 0, "min": 0, "max": 999999, "label": "Random Seed (0=random)",
                "description": "Seed for reproducible variation (0 = random each time)"
            },
        }

    def generate(self) -> List[Brush]:
        """Generate sanctuary geometry based on selected type."""
        self._reset_tags()  # Reset tags for fresh generation

        # Initialize RNG
        if self.random_seed == 0:
            rng = random.Random()
        else:
            rng = random.Random(self.random_seed)

        ox, oy, oz = self.params.origin

        # Use polygonal shell if sides != 4
        # Note: Overrides sanctuary_type - uses simple polygonal shell
        if self.shell_sides != 4:
            brushes = self._generate_polygonal_sanctuary(rng)
            # Register portal tag at footprint edge (grid-aligned)
            # Not at polygon interior - ensures proper alignment with adjoining halls
            if self.has_entrance:
                self._register_portal_tag(
                    portal_id="entrance",
                    center_x=ox,  # Room center X for grid alignment
                    center_y=oy,  # Footprint south edge (Y=origin)
                    center_z=oz,
                    direction=PortalDirection.SOUTH,
                )
            return brushes

        # Determine sanctuary type
        sanctuary_type = self.sanctuary_type.lower()
        if sanctuary_type == "random":
            sanctuary_type = rng.choice(["single_nave", "basilica", "cruciform", "hall_church", "rotunda"])

        # Dispatch to type-specific generator
        if sanctuary_type == "single_nave":
            brushes = self._generate_single_nave(rng)
        elif sanctuary_type == "basilica":
            brushes = self._generate_basilica(rng)
        elif sanctuary_type == "cruciform":
            brushes = self._generate_cruciform(rng)
        elif sanctuary_type == "hall_church":
            brushes = self._generate_hall_church(rng)
        elif sanctuary_type == "rotunda":
            brushes = self._generate_rotunda(rng)
        else:
            brushes = self._generate_single_nave(rng)

        # Register portal tag for rectangular sanctuary
        # Entrance is at the SOUTH wall, centered at X=ox
        if self.has_entrance:
            self._register_portal_tag(
                portal_id="entrance",
                center_x=ox,
                center_y=oy,
                center_z=oz,
                direction=PortalDirection.SOUTH,
            )

        return brushes

    # =========================================================================
    # HELPER: SEALED APSE
    # =========================================================================

    def _sealed_apse(
        self,
        center_x: float,
        attach_y: float,
        floor_z: float,
        radius: float,
        depth: float,
        height: float,
        segments: int = 6,
        nave_half_width: float = None,
    ) -> List[Brush]:
        """Create a sealed apse (semicircular back extension).

        Uses thick overlapping boxes to ensure no gaps.
        nave_half_width: if provided, adds fill brushes to seal gap between apse and nave
        """
        brushes: List[Brush] = []
        t = self.WALL_THICKNESS

        # Apse floor - extends into nave area for overlap
        brushes.append(self._box(
            center_x - radius - t, attach_y - t, floor_z - t,
            center_x + radius + t, attach_y + depth + t, floor_z
        ))

        # Apse ceiling - extends into nave area for overlap
        brushes.append(self._box(
            center_x - radius - t, attach_y - t, floor_z + height,
            center_x + radius + t, attach_y + depth + t, floor_z + height + t
        ))

        # Back wall segments - use thick overlapping boxes with extra margin
        for i in range(segments):
            a0 = math.pi * i / segments
            a1 = math.pi * (i + 1) / segments

            # Calculate segment box bounds with extra thickness for overlap
            cx0 = center_x + math.cos(a0) * radius
            cy0 = attach_y + math.sin(a0) * depth
            cx1 = center_x + math.cos(a1) * radius
            cy1 = attach_y + math.sin(a1) * depth

            # Outer points for wall thickness (extra thick for overlap)
            outer_t = t * 2
            ox0 = center_x + math.cos(a0) * (radius + outer_t)
            oy0 = attach_y + math.sin(a0) * (depth + outer_t)
            ox1 = center_x + math.cos(a1) * (radius + outer_t)
            oy1 = attach_y + math.sin(a1) * (depth + outer_t)

            # Create bounding box for segment with generous overlap
            bx1 = min(cx0, cx1, ox0, ox1) - t
            bx2 = max(cx0, cx1, ox0, ox1) + t
            by1 = min(cy0, cy1, oy0, oy1) - t
            by2 = max(cy0, cy1, oy0, oy1) + t

            # Ensure minimum size
            if bx2 - bx1 < t * 2:
                bx2 = bx1 + t * 2
            if by2 - by1 < t * 2:
                by2 = by1 + t * 2

            brushes.append(self._box(bx1, by1, floor_z, bx2, by2, floor_z + height))

        # Corner fills - connect apse to main structure (thicker overlap)
        brushes.append(self._box(
            center_x - radius - t, attach_y - t, floor_z,
            center_x - radius + t, attach_y + t * 2, floor_z + height
        ))
        brushes.append(self._box(
            center_x + radius - t, attach_y - t, floor_z,
            center_x + radius + t, attach_y + t * 2, floor_z + height
        ))

        # If nave is wider than apse, add fill walls on sides
        if nave_half_width is not None and nave_half_width > radius:
            # Left fill wall
            brushes.append(self._box(
                center_x - nave_half_width - t, attach_y - t, floor_z,
                center_x - radius + t, attach_y + t, floor_z + height
            ))
            # Right fill wall
            brushes.append(self._box(
                center_x + radius - t, attach_y - t, floor_z,
                center_x + nave_half_width + t, attach_y + t, floor_z + height
            ))

        return brushes

    # =========================================================================
    # POLYGONAL SHELL MODE
    # =========================================================================

    def _generate_polygonal_sanctuary(self, rng: random.Random) -> List[Brush]:
        """Generate polygonal sanctuary shell (overrides sanctuary_type).

        Supports interior pillars arranged in a ring pattern.
        """
        ox, oy, oz = self.params.origin
        t = self.WALL_THICKNESS
        brushes: List[Brush] = []

        hw = self.nave_width / 2
        nl = self.nave_length
        nh = self.nave_height

        # Calculate radius from room dimensions, clamped to fit footprint
        # The polygon must fit within both dimensions: hw (half-width) and nl/2 (half-length)
        # Using min() ensures non-square footprints don't overflow in the short dimension
        max_radius = min(hw, nl / 2)
        radius = max(max_radius, self._get_polygon_min_radius(self.shell_sides))

        # Center the polygon at the room center
        cx = ox
        cy = oy + nl / 2

        # Determine portal segment
        portal_segment = self._find_portal_segment(self.shell_sides, "SOUTH") if self.has_entrance else -1

        # Compute vestibule clip zone to avoid floor/ceiling overlap
        vestibule_clip = self._compute_vestibule_clip_zone(
            cx, cy, radius, self.shell_sides, portal_segment,
            PORTAL_WIDTH, t,
            room_origin=self.params.origin,
            room_length=nl,
            room_width=self.nave_width
        )

        # Generate floor (with clip zone to avoid vestibule overlap)
        brushes.extend(self._generate_polygonal_floor(
            cx, cy, oz, radius, self.shell_sides, t,
            is_ceiling=False,
            vestibule_clip_zone=vestibule_clip
        ))

        # Generate ceiling (with clip zone to avoid vestibule overlap)
        brushes.extend(self._generate_polygonal_floor(
            cx, cy, oz + nh, radius, self.shell_sides, t,
            is_ceiling=True,
            vestibule_clip_zone=vestibule_clip
        ))

        # Set portal target for grid alignment before generating walls
        self._set_portal_target_for_polygonal(cx, cy, radius, self.shell_sides, portal_segment)

        # Generate walls
        brushes.extend(self._generate_polygonal_walls(
            cx, cy, oz, oz + nh, radius, self.shell_sides, t,
            portal_segment=portal_segment,
            portal_width=PORTAL_WIDTH,
            portal_height=PORTAL_HEIGHT,
            portal_target_x=getattr(self, '_portal_target_x', None),
            portal_target_y=getattr(self, '_portal_target_y', None)
        ))

        # Generate connector corridor from portal to grid center
        if self.has_entrance:
            brushes.extend(self._generate_polygonal_portal_connector(
                cx, cy, oz, radius, self.shell_sides, portal_segment,
                PORTAL_WIDTH, PORTAL_HEIGHT, t, room_height=nh,
                room_origin=self.params.origin,
                room_length=nl,
                room_width=self.nave_width
            ))

        # Add interior pillars in a ring pattern
        # Number of pillars based on room size (6-8 for large sanctuaries)
        pillar_size = 24.0
        pillar_radius = radius * 0.6  # Inner ring at 60% of room radius
        num_pillars = min(8, max(4, self.shell_sides))

        for i in range(num_pillars):
            # Skip pillar at entrance position
            angle = 2 * math.pi * i / num_pillars
            # Rotate so pillars don't overlap with entrance (portal is at south)
            angle += math.pi / num_pillars  # Offset by half a segment

            px = cx + pillar_radius * math.sin(angle)
            py = cy - pillar_radius * math.cos(angle)
            brushes.extend(self._generate_room_pillar(
                px, py, oz, oz + nh,
                pillar_size, self.pillar_style, self.pillar_capital
            ))

        return brushes

    # =========================================================================
    # FLOOR PLAN TYPE: SINGLE NAVE
    # =========================================================================

    def _generate_single_nave(self, rng: random.Random) -> List[Brush]:
        """Simple rectangular chapel with no side aisles.

        Floor plan:
        +------------------+
        |                  |
        |      NAVE        |
        |                  |
        +------[  ]--------+  <- Entrance
        """
        ox, oy, oz = self.params.origin
        t = self.WALL_THICKNESS

        # Structural dimensions - MUST be exact for portal alignment
        nw = self.nave_width / 2
        nl = self.nave_length
        nh = self.nave_height

        brushes: List[Brush] = []

        # FLOOR - extends beyond all walls
        brushes.append(self._box(ox - nw - t, oy - t, oz - t, ox + nw + t, oy + nl + t, oz,
                                 texture=self.texture_floor))

        # CEILING - extends beyond all walls
        brushes.append(self._box(ox - nw - t, oy - t, oz + nh, ox + nw + t, oy + nl + t, oz + nh + t,
                                 texture=self.texture_ceiling))

        # LEFT WALL - full Y extent to connect with floor/ceiling
        brushes.append(self._box(ox - nw - t, oy - t, oz, ox - nw, oy + nl + t, oz + nh,
                                 texture=self.texture_wall))

        # RIGHT WALL - full Y extent to connect with floor/ceiling
        brushes.append(self._box(ox + nw, oy - t, oz, ox + nw + t, oy + nl + t, oz + nh,
                                 texture=self.texture_wall))

        # FRONT WALL with optional entrance - using unified portal system
        portal_spec = PortalSpec(enabled=self.has_entrance)
        # Create box function that uses wall texture
        def wall_box(x1, y1, z1, x2, y2, z2):
            return self._box(x1, y1, z1, x2, y2, z2, texture=self.texture_wall)
        wall_brushes, _ = generate_portal_wall(
            box_func=wall_box,
            x1=ox - nw - t, y1=oy - t, z1=oz,
            x2=ox + nw + t, y2=oy, z2=oz + nh,
            portal_spec=portal_spec,
            portal_axis="x",
            portal_center=ox,
        )
        brushes.extend(wall_brushes)

        # BACK WALL or APSE
        if self.apse:
            apse_r = nw * rng.uniform(0.8, 0.95)
            apse_d = apse_r * rng.uniform(0.5, 0.7)
            apse_segs = rng.choice([5, 6, 7, 8])
            # Pass nave_half_width so apse can generate its own fill brushes
            brushes.extend(self._sealed_apse(ox, oy + nl, oz, apse_r, apse_d, nh, apse_segs, nw))
        else:
            brushes.append(self._box(ox - nw - t, oy + nl, oz, ox + nw + t, oy + nl + t, oz + nh,
                                     texture=self.texture_wall))

        return brushes

    # =========================================================================
    # FLOOR PLAN TYPE: BASILICA (three-nave with clerestory)
    # =========================================================================

    def _generate_basilica(self, rng: random.Random) -> List[Brush]:
        """Three-nave basilica with side aisles and clerestory.

        Floor plan:
        +-----+--------+-----+
        |AISLE|  NAVE  |AISLE|
        +--[]-+--[  ]--+--[]-+  <- Three entrances

        Section:
              +----+
              |NAVE|  <- Higher ceiling (clerestory)
        +-----+    +-----+
        |AISLE|    |AISLE|  <- Lower ceiling
        +-----+----+-----+
        """
        ox, oy, oz = self.params.origin
        t = self.WALL_THICKNESS

        # Structural dimensions - MUST be exact for portal alignment
        nw = self.nave_width / 2
        nl = self.nave_length
        nh = self.nave_height
        # Aisle dimensions can vary (interior decorative)
        aw = self.aisle_width * rng.uniform(0.85, 1.15)
        ah = min(self.aisle_height * rng.uniform(0.9, 1.1), nh - 32)

        aisle_door_width = 48.0
        aisle_door_height = min(PORTAL_HEIGHT, ah - 8)  # Aisle door fits within aisle height

        brushes: List[Brush] = []

        total_hw = nw + aw  # Total half-width including aisles

        # === UNIFIED FLOOR (covers entire footprint including aisles) ===
        brushes.append(self._box(
            ox - total_hw - t, oy - t, oz - t,
            ox + total_hw + t, oy + nl + t, oz
        ))

        # === NAVE CEILING ===
        brushes.append(self._box(
            ox - nw - t, oy - t, oz + nh,
            ox + nw + t, oy + nl + t, oz + nh + t
        ))

        # Nave front wall with optional entrance - using unified portal system
        portal_spec = PortalSpec(enabled=self.has_entrance)
        wall_brushes, _ = generate_portal_wall(
            box_func=self._box,
            x1=ox - nw, y1=oy - t, z1=oz,
            x2=ox + nw, y2=oy, z2=oz + nh,
            portal_spec=portal_spec,
            portal_axis="x",
            portal_center=ox,
        )
        brushes.extend(wall_brushes)

        # === AISLES (both sides) ===
        for side in (-1, 1):
            if side == -1:
                ax1, ax2 = ox - total_hw, ox - nw  # Left aisle
                outer_x = ox - total_hw - t
            else:
                ax1, ax2 = ox + nw, ox + total_hw  # Right aisle
                outer_x = ox + total_hw

            acx = (ax1 + ax2) / 2
            adw2 = aisle_door_width / 2

            # Aisle ceiling (floor is unified, no separate aisle floor needed)
            brushes.append(self._box(
                ax1 - t if side == -1 else ax1, oy - t, oz + ah,
                ax2 if side == -1 else ax2 + t, oy + nl + t, oz + ah + t
            ))

            # Aisle outer wall (full height to aisle ceiling)
            if side == -1:
                brushes.append(self._box(outer_x, oy - t, oz, ax1, oy + nl + t, oz + ah))
            else:
                brushes.append(self._box(ax2, oy - t, oz, outer_x + t, oy + nl + t, oz + ah))

            # Aisle front wall with entrance
            if side == -1:
                brushes.append(self._box(outer_x, oy - t, oz, acx - adw2, oy, oz + ah))
                brushes.append(self._box(acx + adw2, oy - t, oz, ax2, oy, oz + ah))
            else:
                brushes.append(self._box(ax1, oy - t, oz, acx - adw2, oy, oz + ah))
                brushes.append(self._box(acx + adw2, oy - t, oz, outer_x + t, oy, oz + ah))
            brushes.append(self._box(acx - adw2, oy - t, oz + aisle_door_height, acx + adw2, oy, oz + ah))

            # Aisle back wall (extends full width and connects to outer wall)
            if side == -1:
                brushes.append(self._box(outer_x, oy + nl, oz, ax2, oy + nl + t, oz + ah))
            else:
                brushes.append(self._box(ax1, oy + nl, oz, outer_x + t, oy + nl + t, oz + ah))

            # === CLERESTORY WALL (aisle ceiling to nave ceiling) - full extent ===
            if side == -1:
                brushes.append(self._box(ox - nw - t, oy - t, oz + ah, ox - nw, oy + nl + t, oz + nh))
            else:
                brushes.append(self._box(ox + nw, oy - t, oz + ah, ox + nw + t, oy + nl + t, oz + nh))

            # === ARCADE WALL (floor to aisle ceiling) ===
            if side == -1:
                brushes.append(self._box(ox - nw, oy - t, oz, ox - nw + t, oy + nl + t, oz + ah))
            else:
                brushes.append(self._box(ox + nw - t, oy - t, oz, ox + nw, oy + nl + t, oz + ah))

        # === BACK WALL or APSE ===
        if self.apse:
            apse_r = nw * rng.uniform(0.8, 0.95)
            apse_d = apse_r * rng.uniform(0.5, 0.7)
            apse_segs = rng.choice([5, 6, 7, 8])
            # Pass nave_half_width so apse generates fill brushes
            brushes.extend(self._sealed_apse(ox, oy + nl, oz, apse_r, apse_d, nh, apse_segs, nw))
        else:
            brushes.append(self._box(ox - nw - t, oy + nl, oz, ox + nw + t, oy + nl + t, oz + nh))

        return brushes

    # =========================================================================
    # FLOOR PLAN TYPE: CRUCIFORM (cross-shaped)
    # =========================================================================

    def _generate_cruciform(self, rng: random.Random) -> List[Brush]:
        """Cross-shaped chapel with transept.

        Floor plan:
               +----+
               |    |
        +------+    +------+
        |      TRANSEPT    |
        +------+    +------+
               |NAVE|
               +[  ]+  <- Entrance
        """
        ox, oy, oz = self.params.origin
        t = self.WALL_THICKNESS

        # Structural dimensions - MUST be exact for portal alignment
        nw = self.nave_width / 2
        nl = self.nave_length
        nh = self.nave_height
        # Transept dimensions can vary (interior decorative)
        tw = (self.transept_width / 2) * rng.uniform(0.85, 1.15)
        td = self.transept_depth * rng.uniform(0.85, 1.15)

        # Transept position (2/3 up the nave)
        transept_y = oy + nl * rng.uniform(0.6, 0.75)

        brushes: List[Brush] = []

        # === UNIFIED FLOOR (nave + transept area) ===
        # Nave floor
        brushes.append(self._box(ox - nw - t, oy - t, oz - t, ox + nw + t, oy + nl + t, oz))
        # Transept floor extensions
        brushes.append(self._box(ox - tw - t, transept_y - td/2, oz - t, ox - nw, transept_y + td/2, oz))
        brushes.append(self._box(ox + nw, transept_y - td/2, oz - t, ox + tw + t, transept_y + td/2, oz))

        # === UNIFIED CEILING ===
        # Nave ceiling
        brushes.append(self._box(ox - nw - t, oy - t, oz + nh, ox + nw + t, oy + nl + t, oz + nh + t))
        # Transept ceiling extensions
        brushes.append(self._box(ox - tw - t, transept_y - td/2, oz + nh, ox - nw, transept_y + td/2, oz + nh + t))
        brushes.append(self._box(ox + nw, transept_y - td/2, oz + nh, ox + tw + t, transept_y + td/2, oz + nh + t))

        # === NAVE WALLS ===
        # Front wall with entrance - using unified portal system
        portal_spec = PortalSpec(enabled=self.has_entrance)
        wall_brushes, _ = generate_portal_wall(
            box_func=self._box,
            x1=ox - nw - t, y1=oy - t, z1=oz,
            x2=ox + nw + t, y2=oy, z2=oz + nh,
            portal_spec=portal_spec,
            portal_axis="x",
            portal_center=ox,
        )
        brushes.extend(wall_brushes)

        # Nave side walls (before transept)
        brushes.append(self._box(ox - nw - t, oy, oz, ox - nw, transept_y - td/2, oz + nh))
        brushes.append(self._box(ox + nw, oy, oz, ox + nw + t, transept_y - td/2, oz + nh))

        # Nave side walls (after transept)
        brushes.append(self._box(ox - nw - t, transept_y + td/2, oz, ox - nw, oy + nl, oz + nh))
        brushes.append(self._box(ox + nw, transept_y + td/2, oz, ox + nw + t, oy + nl, oz + nh))

        # === TRANSEPT END WALLS ===
        brushes.append(self._box(ox - tw - t, transept_y - td/2, oz, ox - tw, transept_y + td/2, oz + nh))
        brushes.append(self._box(ox + tw, transept_y - td/2, oz, ox + tw + t, transept_y + td/2, oz + nh))

        # === TRANSEPT FRONT/BACK WALLS (outside nave) - extend to cover corners ===
        # Left arm front/back walls
        brushes.append(self._box(ox - tw - t, transept_y - td/2 - t, oz, ox - nw + t, transept_y - td/2, oz + nh))
        brushes.append(self._box(ox - tw - t, transept_y + td/2, oz, ox - nw + t, transept_y + td/2 + t, oz + nh))
        # Right arm front/back walls
        brushes.append(self._box(ox + nw - t, transept_y - td/2 - t, oz, ox + tw + t, transept_y - td/2, oz + nh))
        brushes.append(self._box(ox + nw - t, transept_y + td/2, oz, ox + tw + t, transept_y + td/2 + t, oz + nh))

        # === BACK WALL or APSE ===
        if self.apse:
            apse_r = nw * rng.uniform(0.8, 0.95)
            apse_d = apse_r * rng.uniform(0.5, 0.7)
            apse_segs = rng.choice([5, 6, 7, 8])
            # Pass nave_half_width so apse generates fill brushes
            brushes.extend(self._sealed_apse(ox, oy + nl, oz, apse_r, apse_d, nh, apse_segs, nw))
        else:
            brushes.append(self._box(ox - nw - t, oy + nl, oz, ox + nw + t, oy + nl + t, oz + nh))

        return brushes

    # =========================================================================
    # FLOOR PLAN TYPE: HALL CHURCH (same height throughout)
    # =========================================================================

    def _generate_hall_church(self, rng: random.Random) -> List[Brush]:
        """Hall church: nave and aisles at same height (no clerestory).

        Floor plan:
        +-----+--------+-----+
        |AISLE|  NAVE  |AISLE|
        +--[]-+--[  ]--+--[]-+

        Section (same height throughout):
        +-----+--------+-----+
        |     |        |     |
        +-----+--------+-----+
        """
        ox, oy, oz = self.params.origin
        t = self.WALL_THICKNESS

        # Structural dimensions - MUST be exact for portal alignment
        nw = self.nave_width / 2
        nl = self.nave_length
        nh = self.nave_height
        # Aisle width can vary (interior decorative)
        aw = self.aisle_width * rng.uniform(0.85, 1.15)

        # Smaller aisle door spec
        aisle_door_spec = PortalSpec(enabled=True, width=48.0, height=PORTAL_HEIGHT)

        total_hw = nw + aw

        # Pillars between nave and aisles
        num_pillars = rng.randint(4, 6)
        pillar_size = rng.uniform(24.0, 32.0)

        brushes: List[Brush] = []

        # === UNIFIED FLOOR ===
        brushes.append(self._box(ox - total_hw - t, oy - t, oz - t, ox + total_hw + t, oy + nl + t, oz))

        # === UNIFIED CEILING ===
        brushes.append(self._box(ox - total_hw - t, oy - t, oz + nh, ox + total_hw + t, oy + nl + t, oz + nh + t))

        # === OUTER WALLS (full Y extent to connect with floor/ceiling) ===
        brushes.append(self._box(ox - total_hw - t, oy - t, oz, ox - total_hw, oy + nl + t, oz + nh))
        brushes.append(self._box(ox + total_hw, oy - t, oz, ox + total_hw + t, oy + nl + t, oz + nh))

        # === FRONT WALL WITH THREE DOORS - using unified portal system ===
        # For hall church, we need multiple portals in one wall
        # Main center entrance
        main_portal = PortalSpec(enabled=self.has_entrance)
        left_cx = ox - nw - aw/2
        right_cx = ox + nw + aw/2

        # Left aisle section with smaller door
        left_wall_brushes, _ = generate_portal_wall(
            box_func=self._box,
            x1=ox - total_hw - t, y1=oy - t, z1=oz,
            x2=left_cx + aisle_door_spec.width/2 + 16, y2=oy, z2=oz + nh,
            portal_spec=aisle_door_spec,
            portal_axis="x",
            portal_center=left_cx,
        )
        brushes.extend(left_wall_brushes)

        # Center nave section with main entrance
        center_wall_brushes, _ = generate_portal_wall(
            box_func=self._box,
            x1=left_cx + aisle_door_spec.width/2, y1=oy - t, z1=oz,
            x2=right_cx - aisle_door_spec.width/2, y2=oy, z2=oz + nh,
            portal_spec=main_portal,
            portal_axis="x",
            portal_center=ox,
        )
        brushes.extend(center_wall_brushes)

        # Right aisle section with smaller door
        right_wall_brushes, _ = generate_portal_wall(
            box_func=self._box,
            x1=right_cx - aisle_door_spec.width/2 - 16, y1=oy - t, z1=oz,
            x2=ox + total_hw + t, y2=oy, z2=oz + nh,
            portal_spec=aisle_door_spec,
            portal_axis="x",
            portal_center=right_cx,
        )
        brushes.extend(right_wall_brushes)

        # === BACK WALL or APSE ===
        if self.apse:
            apse_r = nw * rng.uniform(0.8, 0.95)
            apse_d = apse_r * rng.uniform(0.5, 0.7)
            apse_segs = rng.choice([5, 6, 7, 8])
            # Pass total_hw so apse generates fill brushes for the wider hall church
            brushes.extend(self._sealed_apse(ox, oy + nl, oz, apse_r, apse_d, nh, apse_segs, total_hw))
        else:
            brushes.append(self._box(ox - total_hw - t, oy + nl, oz, ox + total_hw + t, oy + nl + t, oz + nh))

        # === INTERIOR PILLARS ===
        pillar_spacing = nl / (num_pillars + 1)
        for i in range(num_pillars):
            py = oy + (i + 1) * pillar_spacing
            # Left row
            brushes.extend(self._generate_room_pillar(
                ox - nw, py, oz, oz + nh,
                pillar_size, self.pillar_style, self.pillar_capital
            ))
            # Right row
            brushes.extend(self._generate_room_pillar(
                ox + nw, py, oz, oz + nh,
                pillar_size, self.pillar_style, self.pillar_capital
            ))

        return brushes

    # =========================================================================
    # FLOOR PLAN TYPE: ROTUNDA (octagonal)
    # =========================================================================

    def _generate_rotunda(self, rng: random.Random) -> List[Brush]:
        """Octagonal rotunda using overlapping rectangles.

        Floor plan:
            +------+
           /        \\
          |   MAIN   |
           \\        /
            +--[  ]--+
        """
        ox, oy, oz = self.params.origin
        t = self.WALL_THICKNESS

        # Structural dimensions - MUST be exact for portal alignment
        r = self.nave_width / 2
        nh = self.nave_height

        # Inner dimension for octagon (cos 45)
        inner = r * 0.707

        brushes: List[Brush] = []

        # === FLOOR (two overlapping rectangles) ===
        brushes.append(self._box(ox - r - t, oy - inner - t, oz - t, ox + r + t, oy + inner + t, oz))
        brushes.append(self._box(ox - inner - t, oy - r - t, oz - t, ox + inner + t, oy + r + t, oz))

        # === CEILING (two overlapping rectangles) ===
        brushes.append(self._box(ox - r - t, oy - inner - t, oz + nh, ox + r + t, oy + inner + t, oz + nh + t))
        brushes.append(self._box(ox - inner - t, oy - r - t, oz + nh, ox + inner + t, oy + r + t, oz + nh + t))

        # === WALLS ===
        # N-S rectangle walls (X = +/- r) - extend Y to overlap with corners
        brushes.append(self._box(ox - r - t, oy - inner - t, oz, ox - r, oy + inner + t, oz + nh))
        brushes.append(self._box(ox + r, oy - inner - t, oz, ox + r + t, oy + inner + t, oz + nh))

        # E-W rectangle walls (Y = +/- r)
        # Back wall (solid) - extend X to overlap with corners
        brushes.append(self._box(ox - inner - t, oy + r, oz, ox + inner + t, oy + r + t, oz + nh))

        # Front wall (with entrance) - using unified portal system
        portal_spec = PortalSpec(enabled=self.has_entrance)
        wall_brushes, _ = generate_portal_wall(
            box_func=self._box,
            x1=ox - inner - t, y1=oy - r - t, z1=oz,
            x2=ox + inner + t, y2=oy - r, z2=oz + nh,
            portal_spec=portal_spec,
            portal_axis="x",
            portal_center=ox,
        )
        brushes.extend(wall_brushes)

        # === CORNER WALLS (seal octagon corners) - thick overlap ===
        brushes.append(self._box(ox - r - t, oy + inner - t, oz, ox - inner + t, oy + r + t, oz + nh))
        brushes.append(self._box(ox + inner - t, oy + inner - t, oz, ox + r + t, oy + r + t, oz + nh))
        brushes.append(self._box(ox - r - t, oy - r - t, oz, ox - inner + t, oy - inner + t, oz + nh))
        brushes.append(self._box(ox + inner - t, oy - r - t, oz, ox + r + t, oy - inner + t, oz + nh))

        # === OPTIONAL APSE ===
        if self.apse:
            apse_r = inner * rng.uniform(0.65, 0.85)
            apse_d = apse_r * rng.uniform(0.4, 0.6)
            apse_segs = rng.choice([4, 5, 6])
            # Pass inner dimension so apse generates fill brushes
            brushes.extend(self._sealed_apse(ox, oy + r, oz, apse_r, apse_d, nh, apse_segs, inner))

        return brushes


class Tomb(GeometricPrimitive):
    """A tomb: low-ceiling room with alcoves and coffin/slab platforms.

    Features:
    - Low oppressive ceiling (or polygonal with shell_sides != 4)
    - Side alcoves for atmosphere (disabled for polygonal shapes)
    - Configurable number of coffins/slabs arranged in rows
    - Central sarcophagus platform (optional)
    - Entrance doorway at front, optional exit at back
    - 100% sealed geometry
    """

    width: float = 256.0
    length: float = 384.0
    tomb_height: float = 96.0
    shell_sides: int = 4        # Number of sides (4=square, 6=hex, 8=octagon)
    alcove_depth: float = 48.0
    alcove_count: int = 3
    coffin_count: int = 0          # Number of coffins/slabs (0 = just central platform)
    coffin_layout: str = "rows"    # rows, alcoves, perimeter
    central_platform: bool = True  # Include central sarcophagus
    random_seed: int = 0

    # Portal control - set by layout generator based on connections
    has_entrance: bool = True      # Front (SOUTH) entrance
    has_exit: bool = True          # Back (NORTH) exit

    WALL_THICKNESS: float = 16.0

    @classmethod
    def get_display_name(cls) -> str:
        return "Tomb"

    @classmethod
    def get_category(cls) -> str:
        return "Rooms"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "width": {
                "type": "float", "default": 256.0, "min": 128, "max": 512, "label": "Width",
                "description": "Interior width of the tomb chamber"
            },
            "length": {
                "type": "float", "default": 384.0, "min": 192, "max": 768, "label": "Length",
                "description": "Interior length from entrance to back wall"
            },
            "tomb_height": {
                "type": "float", "default": 96.0, "min": 64, "max": 192, "label": "Ceiling Height",
                "description": "Low oppressive ceiling height for tomb atmosphere"
            },
            "shell_sides": {
                "type": "int",
                "default": 4,
                "min": 3,
                "max": 16,
                "label": "Shell Sides",
                "description": "Number of sides for room shape (4=square, 6=hex). Non-4 disables alcoves."
            },
            "alcove_depth": {
                "type": "float", "default": 48.0, "min": 16, "max": 96, "label": "Alcove Depth",
                "description": "How far alcoves extend into the walls"
            },
            "alcove_count": {
                "type": "int", "default": 3, "min": 0, "max": 6, "label": "Alcoves per Side",
                "description": "Number of alcoves along each side wall"
            },
            "coffin_count": {
                "type": "int", "default": 0, "min": 0, "max": 12, "label": "Coffin Count",
                "description": "Number of coffin platforms (0 = central platform only)"
            },
            "coffin_layout": {
                "type": "choice", "default": "rows", "choices": ["rows", "alcoves", "perimeter"], "label": "Coffin Layout",
                "description": "Arrangement of coffins (rows=center, alcoves=wall niches, perimeter=edges)"
            },
            "central_platform": {
                "type": "bool", "default": True, "label": "Central Sarcophagus",
                "description": "Add central raised sarcophagus platform"
            },
            "has_entrance": {
                "type": "bool", "default": True, "label": "Enable Entrance Portal",
                "description": "Enable entrance portal at the front (south)"
            },
            "has_exit": {
                "type": "bool", "default": True, "label": "Enable Exit Portal",
                "description": "Enable exit portal at the back (north)"
            },
            "random_seed": {
                "type": "int", "default": 0, "min": 0, "max": 999999, "label": "Random Seed (0=random)",
                "description": "Seed for reproducible variation (0 = random each time)"
            },
        }

    def generate(self) -> List[Brush]:
        self._reset_tags()  # Reset tags for fresh generation
        ox, oy, oz = self.params.origin
        brushes: List[Brush] = []
        t = self.WALL_THICKNESS

        # Initialize random for decorative variation
        if self.random_seed == 0:
            rng = random.Random()
        else:
            rng = random.Random(self.random_seed)

        # Structural dimensions - MUST be exact for portal alignment
        hw = self.width / 2
        length = self.length
        height = self.tomb_height

        # Use polygonal shell if sides != 4
        # Note: Alcoves and coffin layouts are disabled for polygonal shapes
        if self.shell_sides != 4:
            brushes = self._generate_polygonal_tomb(ox, oy, oz, hw, length, height, t, rng)
            # Register portal tags at footprint edges (grid-aligned)
            # Not at polygon interior - ensures proper alignment with adjoining halls
            if self.has_entrance:
                self._register_portal_tag(
                    portal_id="entrance",
                    center_x=ox,  # Room center X for grid alignment
                    center_y=oy,  # Footprint south edge (Y=origin)
                    center_z=oz,
                    direction=PortalDirection.SOUTH,
                )
            if self.has_exit:
                self._register_portal_tag(
                    portal_id="exit",
                    center_x=ox,  # Room center X for grid alignment
                    center_y=oy + length,  # Footprint north edge
                    center_z=oz,
                    direction=PortalDirection.NORTH,
                )
            return brushes

        # Standard rectangular generation (sides == 4)
        # Check if using alcove layout - alcoves need extended floor/ceiling
        using_alcoves = (self.coffin_count > 0 and self.coffin_layout.lower() == "alcoves")
        alcove_depth = self.alcove_depth if using_alcoves else 0

        # Floor - extends beyond walls, and into alcove depth if using alcoves
        floor_x1 = ox - hw - t - alcove_depth
        floor_x2 = ox + hw + t + alcove_depth
        brushes.append(self._box(floor_x1, oy - t, oz - t, floor_x2, oy + length + t, oz))

        # Ceiling - extends beyond walls, and into alcove depth if using alcoves
        brushes.append(self._box(floor_x1, oy - t, oz + height, floor_x2, oy + length + t, oz + height + t))

        # Side walls - only generate solid walls if NOT using alcoves
        # When using alcoves, _coffins_in_alcoves generates segmented walls
        if not using_alcoves:
            brushes.append(self._box(ox - hw - t, oy - t, oz, ox - hw, oy + length + t, oz + height))
            brushes.append(self._box(ox + hw, oy - t, oz, ox + hw + t, oy + length + t, oz + height))

        # Front wall with optional entrance - using unified portal system
        entrance_portal = PortalSpec(enabled=self.has_entrance)
        front_wall_brushes, _ = generate_portal_wall(
            box_func=self._box,
            x1=floor_x1, y1=oy - t, z1=oz,
            x2=floor_x2, y2=oy, z2=oz + height,
            portal_spec=entrance_portal,
            portal_axis="x",
            portal_center=ox,
        )
        brushes.extend(front_wall_brushes)

        # Back wall with optional exit - using unified portal system
        exit_portal = PortalSpec(enabled=self.has_exit)
        back_wall_brushes, _ = generate_portal_wall(
            box_func=self._box,
            x1=floor_x1, y1=oy + length, z1=oz,
            x2=floor_x2, y2=oy + length + t, z2=oz + height,
            portal_spec=exit_portal,
            portal_axis="x",
            portal_center=ox,
        )
        brushes.extend(back_wall_brushes)

        # Central sarcophagus platform (optional)
        if self.central_platform:
            pw = hw * rng.uniform(0.35, 0.45)
            pl = length * rng.uniform(0.15, 0.25)
            platform_h = rng.uniform(20, 28)
            brushes.append(self._box(
                ox - pw, oy + length / 2 - pl, oz,
                ox + pw, oy + length / 2 + pl, oz + platform_h,
            ))

        # Coffins/slabs
        if self.coffin_count > 0:
            brushes.extend(self._generate_coffins(ox, oy, oz, hw, length, height, rng))

        # Register portal tags for rectangular tomb
        if self.has_entrance:
            self._register_portal_tag(
                portal_id="entrance",
                center_x=ox,
                center_y=oy,
                center_z=oz,
                direction=PortalDirection.SOUTH,
            )
        if self.has_exit:
            self._register_portal_tag(
                portal_id="exit",
                center_x=ox,
                center_y=oy + length,
                center_z=oz,
                direction=PortalDirection.NORTH,
            )

        return brushes

    def _generate_polygonal_tomb(
        self,
        ox: float, oy: float, oz: float,
        hw: float, length: float, height: float,
        t: float, rng: random.Random
    ) -> List[Brush]:
        """Generate tomb with polygonal (non-rectangular) shell.

        Creates an N-sided tomb. Alcoves and complex coffin layouts are disabled;
        only central sarcophagus platform is preserved.

        Multi-portal support: Both entrance (SOUTH) and exit (NORTH) portals are
        generated when enabled, using the vestibule approach for grid-aligned
        connections.
        """
        brushes: List[Brush] = []

        # Calculate radius from room dimensions, clamped to fit footprint
        # The polygon must fit within both dimensions: hw (half-width) and length/2 (half-length)
        # Using min() ensures non-square footprints don't overflow in the short dimension
        max_radius = min(hw, length / 2)
        radius = max(max_radius, self._get_polygon_min_radius(self.shell_sides))

        # Center the polygon at the room center
        cx = ox
        cy = oy + length / 2

        # Determine portal segments for entrance and exit
        entrance_segment = self._find_portal_segment(self.shell_sides, "SOUTH") if self.has_entrance else -1
        exit_segment = self._find_portal_segment(self.shell_sides, "NORTH") if self.has_exit else -1

        # Compute vestibule clip zones for all active portals
        room_origin = (ox, oy, oz)
        vestibule_clips = []
        if entrance_segment >= 0:
            clip = self._compute_vestibule_clip_zone(
                cx, cy, radius, self.shell_sides, entrance_segment,
                PORTAL_WIDTH, t,
                room_origin=room_origin, room_length=length, room_width=hw * 2
            )
            if clip:
                vestibule_clips.append(clip)
        if exit_segment >= 0:
            clip = self._compute_vestibule_clip_zone(
                cx, cy, radius, self.shell_sides, exit_segment,
                PORTAL_WIDTH, t,
                room_origin=room_origin, room_length=length, room_width=hw * 2
            )
            if clip:
                vestibule_clips.append(clip)

        # Generate floor (with clip zones to avoid vestibule overlap)
        brushes.extend(self._generate_polygonal_floor(
            cx, cy, oz, radius, self.shell_sides, t,
            is_ceiling=False,
            vestibule_clip_zones=vestibule_clips if vestibule_clips else None
        ))

        # Generate ceiling (with clip zones to avoid vestibule overlap)
        brushes.extend(self._generate_polygonal_floor(
            cx, cy, oz + height, radius, self.shell_sides, t,
            is_ceiling=True,
            vestibule_clip_zones=vestibule_clips if vestibule_clips else None
        ))

        # Build portal segments list for multi-portal wall generation
        portal_segments = []
        used_segments = set()

        if entrance_segment >= 0 and entrance_segment not in used_segments:
            # Set portal target for entrance (grid-aligned at room center X)
            self._set_portal_target_for_polygonal(cx, cy, radius, self.shell_sides, entrance_segment)
            portal_segments.append({
                'segment': entrance_segment,
                'width': PORTAL_WIDTH,
                'height': PORTAL_HEIGHT,
                'target_x': getattr(self, '_portal_target_x', None),
                'target_y': getattr(self, '_portal_target_y', None),
                'direction': 'SOUTH'
            })
            used_segments.add(entrance_segment)

        if exit_segment >= 0 and exit_segment not in used_segments:
            # Set portal target for exit (grid-aligned at room center X)
            self._set_portal_target_for_polygonal(cx, cy, radius, self.shell_sides, exit_segment)
            portal_segments.append({
                'segment': exit_segment,
                'width': PORTAL_WIDTH,
                'height': PORTAL_HEIGHT,
                'target_x': getattr(self, '_portal_target_x', None),
                'target_y': getattr(self, '_portal_target_y', None),
                'direction': 'NORTH'
            })
            used_segments.add(exit_segment)

        # Generate walls with ALL portal openings
        brushes.extend(self._generate_polygonal_walls(
            cx, cy, oz, oz + height, radius, self.shell_sides, t,
            portal_segments=portal_segments if portal_segments else None
        ))

        # Generate vestibule corridor for EACH enabled portal
        # Pass room origin and dimensions for accurate footprint edge calculation
        room_origin = (ox, oy, oz)
        if self.has_entrance and entrance_segment >= 0:
            brushes.extend(self._generate_polygonal_portal_connector(
                cx, cy, oz, radius, self.shell_sides, entrance_segment,
                PORTAL_WIDTH, PORTAL_HEIGHT, t, room_height=height,
                room_origin=room_origin, room_length=length, room_width=hw * 2
            ))

        if self.has_exit and exit_segment >= 0:
            brushes.extend(self._generate_polygonal_portal_connector(
                cx, cy, oz, radius, self.shell_sides, exit_segment,
                PORTAL_WIDTH, PORTAL_HEIGHT, t, room_height=height,
                room_origin=room_origin, room_length=length, room_width=hw * 2
            ))

        # Central sarcophagus platform (preserved for polygonal shapes)
        if self.central_platform:
            # Use smaller platform for polygonal shapes
            pw = radius * rng.uniform(0.25, 0.35)
            pl = radius * rng.uniform(0.15, 0.25)
            platform_h = rng.uniform(20, 28)
            brushes.append(self._box(
                cx - pw, cy - pl, oz,
                cx + pw, cy + pl, oz + platform_h,
            ))

        return brushes

    def _generate_coffins(
        self,
        ox: float, oy: float, oz: float,
        hw: float, length: float, height: float,
        rng: random.Random
    ) -> List[Brush]:
        """Generate coffin/slab platforms based on layout type."""
        brushes: List[Brush] = []

        # Standard coffin dimensions
        coffin_w = rng.uniform(28, 36)   # Width
        coffin_l = rng.uniform(72, 88)   # Length
        coffin_h = rng.uniform(16, 24)   # Height (raised platform)

        layout = self.coffin_layout.lower()

        if layout == "rows":
            # Arrange coffins in parallel rows along the length
            brushes.extend(self._coffins_in_rows(
                ox, oy, oz, hw, length, coffin_w, coffin_l, coffin_h, rng
            ))
        elif layout == "alcoves":
            # Place coffins in wall alcoves (recessed niches)
            brushes.extend(self._coffins_in_alcoves(
                ox, oy, oz, hw, length, height, coffin_w, coffin_l, coffin_h, rng
            ))
        elif layout == "perimeter":
            # Arrange coffins along the walls
            brushes.extend(self._coffins_along_perimeter(
                ox, oy, oz, hw, length, coffin_w, coffin_l, coffin_h, rng
            ))

        return brushes

    def _coffins_in_rows(
        self,
        ox: float, oy: float, oz: float,
        hw: float, length: float,
        coffin_w: float, coffin_l: float, coffin_h: float,
        rng: random.Random
    ) -> List[Brush]:
        """Place coffins in two parallel rows along the crypt length."""
        brushes: List[Brush] = []
        count = self.coffin_count

        # Calculate spacing - coffins oriented lengthwise (head toward back wall)
        # Leave space for central platform if enabled
        start_y = oy + 48  # Leave entrance clearance
        end_y = oy + length - 48

        # Determine row positions (left and right of center)
        row_offset = hw * 0.55  # Distance from center to row

        # Calculate how many per row
        coffins_per_row = (count + 1) // 2
        spacing = (end_y - start_y - coffin_l) / max(1, coffins_per_row - 1) if coffins_per_row > 1 else 0
        spacing = max(spacing, coffin_l + 16)  # Minimum gap between coffins

        placed = 0
        for row_side in [-1, 1]:
            rx = ox + row_side * row_offset
            for i in range(coffins_per_row):
                if placed >= count:
                    break
                cy = start_y + i * spacing
                # Skip if would overlap central platform area
                if self.central_platform:
                    center_y = oy + length / 2
                    if abs(cy + coffin_l / 2 - center_y) < coffin_l * 0.8:
                        continue
                brushes.append(self._box(
                    rx - coffin_w / 2, cy, oz,
                    rx + coffin_w / 2, cy + coffin_l, oz + coffin_h
                ))
                placed += 1

        return brushes

    def _coffins_in_alcoves(
        self,
        ox: float, oy: float, oz: float,
        hw: float, length: float, height: float,
        coffin_w: float, coffin_l: float, coffin_h: float,
        rng: random.Random
    ) -> List[Brush]:
        """Place coffins in actual wall alcoves (recessed niches).

        Creates proper alcove architecture:
        - Removes solid walls in generate() and replaces with segmented walls
        - Each alcove is a recessed space carved into the wall
        - Back wall closes the alcove
        - Floor/ceiling extend into alcove depth
        - Coffin platform sits inside the alcove

        Note: This method generates BOTH the alcove structure AND the coffins.
        The main generate() method must NOT generate solid side walls when
        coffin_layout=="alcoves" - it should call this method for walls instead.
        """
        brushes: List[Brush] = []
        t = self.WALL_THICKNESS
        count = self.coffin_count

        # Alcove dimensions
        alcove_depth = self.alcove_depth
        alcove_w = coffin_l + 32  # Width of opening (slightly wider than coffin)
        alcove_h = min(height * 0.75, height - 16)  # Leave ceiling thickness
        alcove_floor_h = 24  # Raised floor in alcove

        # Pilaster (wall segment between alcoves) width
        pilaster_w = 32

        # Distribute alcoves along side walls
        start_y = oy + 48 + pilaster_w  # Leave room for front pilaster
        end_y = oy + length - 48 - pilaster_w  # Leave room for back pilaster
        available_length = end_y - start_y

        # Calculate alcove positions
        alcoves_per_side = max(1, (count + 1) // 2)
        total_alcove_width = alcoves_per_side * alcove_w
        total_pilaster_width = (alcoves_per_side - 1) * pilaster_w

        # Adjust if alcoves won't fit
        if total_alcove_width + total_pilaster_width > available_length:
            alcove_w = (available_length - total_pilaster_width) / alcoves_per_side

        spacing = alcove_w + pilaster_w

        placed = 0
        for side in [-1, 1]:
            if side == -1:
                # Left wall: alcoves extend from -hw to -(hw + alcove_depth)
                wall_inner_x = ox - hw
                wall_outer_x = ox - hw - t
                alcove_back_x = ox - hw - alcove_depth
            else:
                # Right wall: alcoves extend from +hw to +(hw + alcove_depth)
                wall_inner_x = ox + hw
                wall_outer_x = ox + hw + t
                alcove_back_x = ox + hw + alcove_depth

            # Front pilaster (solid wall segment before first alcove)
            if side == -1:
                brushes.append(self._box(
                    wall_outer_x, oy, oz,
                    wall_inner_x, start_y, oz + height
                ))
            else:
                brushes.append(self._box(
                    wall_inner_x, oy, oz,
                    wall_outer_x, start_y, oz + height
                ))

            for i in range(alcoves_per_side):
                if placed >= count:
                    # No more coffins - fill remaining wall as solid
                    remaining_start = start_y + i * spacing
                    if side == -1:
                        brushes.append(self._box(
                            wall_outer_x, remaining_start, oz,
                            wall_inner_x, end_y + pilaster_w, oz + height
                        ))
                    else:
                        brushes.append(self._box(
                            wall_inner_x, remaining_start, oz,
                            wall_outer_x, end_y + pilaster_w, oz + height
                        ))
                    break

                alcove_y1 = start_y + i * spacing
                alcove_y2 = alcove_y1 + alcove_w

                # === ALCOVE OPENING STRUCTURE ===

                # Wall segment ABOVE alcove opening (lintel)
                if side == -1:
                    brushes.append(self._box(
                        wall_outer_x, alcove_y1, oz + alcove_h,
                        wall_inner_x, alcove_y2, oz + height
                    ))
                else:
                    brushes.append(self._box(
                        wall_inner_x, alcove_y1, oz + alcove_h,
                        wall_outer_x, alcove_y2, oz + height
                    ))

                # Wall segment BELOW alcove (threshold, if alcove floor is raised)
                if alcove_floor_h > 0:
                    if side == -1:
                        brushes.append(self._box(
                            wall_outer_x, alcove_y1, oz,
                            wall_inner_x, alcove_y2, oz + alcove_floor_h
                        ))
                    else:
                        brushes.append(self._box(
                            wall_inner_x, alcove_y1, oz,
                            wall_outer_x, alcove_y2, oz + alcove_floor_h
                        ))

                # === ALCOVE INTERIOR ===

                # Back wall of alcove
                if side == -1:
                    brushes.append(self._box(
                        alcove_back_x - t, alcove_y1 - t, oz,
                        alcove_back_x, alcove_y2 + t, oz + alcove_h
                    ))
                else:
                    brushes.append(self._box(
                        alcove_back_x, alcove_y1 - t, oz,
                        alcove_back_x + t, alcove_y2 + t, oz + alcove_h
                    ))

                # Alcove floor (extends into alcove depth)
                if side == -1:
                    brushes.append(self._box(
                        alcove_back_x, alcove_y1, oz + alcove_floor_h - t,
                        wall_inner_x, alcove_y2, oz + alcove_floor_h
                    ))
                else:
                    brushes.append(self._box(
                        wall_inner_x, alcove_y1, oz + alcove_floor_h - t,
                        alcove_back_x, alcove_y2, oz + alcove_floor_h
                    ))

                # Alcove ceiling
                if side == -1:
                    brushes.append(self._box(
                        alcove_back_x, alcove_y1, oz + alcove_h,
                        wall_inner_x, alcove_y2, oz + alcove_h + t
                    ))
                else:
                    brushes.append(self._box(
                        wall_inner_x, alcove_y1, oz + alcove_h,
                        alcove_back_x, alcove_y2, oz + alcove_h + t
                    ))

                # Side jambs (walls at alcove opening edges)
                # Front jamb
                if side == -1:
                    brushes.append(self._box(
                        alcove_back_x, alcove_y1 - t, oz + alcove_floor_h,
                        wall_inner_x, alcove_y1, oz + alcove_h
                    ))
                    # Back jamb
                    brushes.append(self._box(
                        alcove_back_x, alcove_y2, oz + alcove_floor_h,
                        wall_inner_x, alcove_y2 + t, oz + alcove_h
                    ))
                else:
                    brushes.append(self._box(
                        wall_inner_x, alcove_y1 - t, oz + alcove_floor_h,
                        alcove_back_x, alcove_y1, oz + alcove_h
                    ))
                    # Back jamb
                    brushes.append(self._box(
                        wall_inner_x, alcove_y2, oz + alcove_floor_h,
                        alcove_back_x, alcove_y2 + t, oz + alcove_h
                    ))

                # === COFFIN INSIDE ALCOVE ===
                # Coffin oriented with head toward back wall
                coffin_cx = (alcove_back_x + wall_inner_x) / 2 if side == -1 else (wall_inner_x + alcove_back_x) / 2
                coffin_cy = (alcove_y1 + alcove_y2) / 2

                brushes.append(self._box(
                    coffin_cx - coffin_w / 2, coffin_cy - coffin_l / 2, oz + alcove_floor_h,
                    coffin_cx + coffin_w / 2, coffin_cy + coffin_l / 2, oz + alcove_floor_h + coffin_h
                ))

                # === PILASTER (wall segment after this alcove) ===
                if i < alcoves_per_side - 1:
                    pilaster_y1 = alcove_y2
                    pilaster_y2 = alcove_y2 + pilaster_w
                    if side == -1:
                        brushes.append(self._box(
                            wall_outer_x, pilaster_y1, oz,
                            wall_inner_x, pilaster_y2, oz + height
                        ))
                    else:
                        brushes.append(self._box(
                            wall_inner_x, pilaster_y1, oz,
                            wall_outer_x, pilaster_y2, oz + height
                        ))

                placed += 1

            # Back pilaster (solid wall segment after last alcove)
            last_alcove_end = start_y + min(alcoves_per_side, (count + 1) // 2) * spacing - pilaster_w
            if last_alcove_end < end_y + pilaster_w:
                if side == -1:
                    brushes.append(self._box(
                        wall_outer_x, last_alcove_end, oz,
                        wall_inner_x, oy + length, oz + height
                    ))
                else:
                    brushes.append(self._box(
                        wall_inner_x, last_alcove_end, oz,
                        wall_outer_x, oy + length, oz + height
                    ))

        return brushes

    def _coffins_along_perimeter(
        self,
        ox: float, oy: float, oz: float,
        hw: float, length: float,
        coffin_w: float, coffin_l: float, coffin_h: float,
        rng: random.Random
    ) -> List[Brush]:
        """Place coffins along the walls (heads toward walls)."""
        brushes: List[Brush] = []
        count = self.coffin_count
        wall_offset = 24  # Distance from wall

        placed = 0

        # Back wall coffins (heads toward back)
        back_count = min(count - placed, 3)
        back_spacing = (hw * 2 - coffin_w * 2) / max(1, back_count)
        for i in range(back_count):
            if placed >= count:
                break
            cx = ox - hw + coffin_w + i * back_spacing
            cy = oy + length - wall_offset - coffin_l
            brushes.append(self._box(
                cx - coffin_w / 2, cy, oz,
                cx + coffin_w / 2, cy + coffin_l, oz + coffin_h
            ))
            placed += 1

        # Side wall coffins (heads toward side walls, perpendicular)
        remaining = count - placed
        per_side = (remaining + 1) // 2
        side_start_y = oy + 80
        side_end_y = oy + length - coffin_w - 48
        side_spacing = (side_end_y - side_start_y) / max(1, per_side) if per_side > 1 else 0

        for side in [-1, 1]:
            for i in range(per_side):
                if placed >= count:
                    break
                # Coffin perpendicular to wall (rotated 90 degrees)
                cy = side_start_y + i * side_spacing
                if side == -1:
                    cx = ox - hw + wall_offset + coffin_l / 2
                else:
                    cx = ox + hw - wall_offset - coffin_l / 2
                # Place as rotated (swap w and l for perpendicular orientation)
                brushes.append(self._box(
                    cx - coffin_l / 2, cy - coffin_w / 2, oz,
                    cx + coffin_l / 2, cy + coffin_w / 2, oz + coffin_h
                ))
                placed += 1

        return brushes


class Tower(GeometricPrimitive):
    """An octagonal tower approximation with multi-level interior and stairs.

    Features:
    - Octagonal shape approximated with overlapping rectangles
    - Multiple floors/levels with stairwell openings
    - Spiral or straight stairs connecting all levels
    - Ground floor entrance (controllable via has_entrance)
    - 100% sealed geometry
    """

    tower_radius: float = 128.0
    tower_height: float = 384.0
    levels: int = 3
    shell_sides: int = 8  # Default 8 for traditional octagonal tower
    stair_type: str = "straight"    # straight only (spiral removed)
    random_seed: int = 0

    # Portal control - set by layout generator based on connections
    has_entrance: bool = True

    # Portal position offset - set by layout generator to match footprint cell positions
    _entrance_x_offset: float = 0.0  # X offset for entrance (negative = left of center)

    WALL_THICKNESS: float = 16.0

    @classmethod
    def get_display_name(cls) -> str:
        return "Tower"

    @classmethod
    def get_category(cls) -> str:
        return "Rooms"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "tower_radius": {
                "type": "float", "default": 128.0, "min": 64, "max": 256, "label": "Radius",
                "description": "Outer radius of the tower from center to wall"
            },
            "tower_height": {
                "type": "float", "default": 384.0, "min": 128, "max": 1024, "label": "Height",
                "description": "Total height of the tower from base to roof"
            },
            "levels": {
                "type": "int", "default": 3, "min": 1, "max": 6, "label": "Levels",
                "description": "Number of interior floors connected by stairs"
            },
            "shell_sides": {
                "type": "int",
                "default": 8,
                "min": 3,
                "max": 16,
                "label": "Shell Sides",
                "description": "Number of sides (4=square, 6=hex, 8=octagon traditional)"
            },
            "stair_type": {
                "type": "choice", "default": "straight", "choices": ["straight"], "label": "Stair Type",
                "description": "Type of interior staircase between levels"
            },
            "has_entrance": {
                "type": "bool", "default": True, "label": "Enable Entrance Portal",
                "description": "Enable ground floor entrance portal"
            },
            "random_seed": {
                "type": "int", "default": 0, "min": 0, "max": 999999, "label": "Random Seed (0=random)",
                "description": "Seed for reproducible variation (0 = random each time)"
            },
        }

    def generate(self) -> List[Brush]:
        self._reset_tags()  # Reset tags for fresh generation
        ox, oy, oz = self.params.origin
        brushes: List[Brush] = []
        t = self.WALL_THICKNESS

        # Initialize random for decorative variation
        if self.random_seed == 0:
            rng = random.Random()
        else:
            rng = random.Random(self.random_seed)

        # Structural dimensions - MUST be exact for portal alignment
        r = self.tower_radius
        total_h = self.tower_height
        level_h = total_h / self.levels

        # Use polygonal shell if sides != 8 (Tower default is octagonal)
        # Note: Multi-level interior and stairs disabled for non-octagonal
        if self.shell_sides != 8:
            brushes = self._generate_polygonal_tower(ox, oy, oz, r, total_h, t)
            # Register portal tag at footprint edge (grid-aligned)
            # Tower is centered at (ox, oy), footprint south edge is oy - r
            if self.has_entrance:
                self._register_portal_tag(
                    portal_id="entrance",
                    center_x=ox,  # Room center X for grid alignment
                    center_y=oy - r,  # Footprint south edge
                    center_z=oz,
                    direction=PortalDirection.SOUTH,
                )
            return brushes

        # Octagonal approximation inner dimension
        # Use fixed ratio (cos 45° ≈ 0.707) rounded to ensure integer coordinates
        inner = int(r * 0.707)

        # Stairwell dimensions (positioned in back-right quadrant)
        stair_w = 48.0   # Width of stairwell
        stair_d = 64.0   # Depth of stairwell

        # Ground floor
        # Floor (two overlapping rectangles)
        brushes.append(self._box(ox - r - t, oy - inner - t, oz - t, ox + r + t, oy + inner + t, oz))
        brushes.append(self._box(ox - inner - t, oy - r - t, oz - t, ox + inner + t, oy + r + t, oz))

        # Top ceiling (two overlapping rectangles)
        brushes.append(self._box(ox - r - t, oy - inner - t, oz + total_h, ox + r + t, oy + inner + t, oz + total_h + t))
        brushes.append(self._box(ox - inner - t, oy - r - t, oz + total_h, ox + inner + t, oy + r + t, oz + total_h + t))

        # Walls - extend Y to overlap with corners for proper sealing
        # N-S rectangle walls
        brushes.append(self._box(ox - r - t, oy - inner - t, oz, ox - r, oy + inner + t, oz + total_h))
        brushes.append(self._box(ox + r, oy - inner - t, oz, ox + r + t, oy + inner + t, oz + total_h))

        # E-W rectangle walls - extend X to overlap with corners
        # Back wall (solid)
        brushes.append(self._box(ox - inner - t, oy + r, oz, ox + inner + t, oy + r + t, oz + total_h))

        # Front wall (with optional entrance on ground floor) - using unified portal system
        entrance_x = ox + self._entrance_x_offset  # Apply portal offset
        portal_spec = PortalSpec(enabled=self.has_entrance)
        wall_brushes, _ = generate_portal_wall(
            box_func=self._box,
            x1=ox - inner - t, y1=oy - r - t, z1=oz,
            x2=ox + inner + t, y2=oy - r, z2=oz + total_h,
            portal_spec=portal_spec,
            portal_axis="x",
            portal_center=entrance_x,
        )
        brushes.extend(wall_brushes)

        # Corner walls (seal octagon corners) - thick overlap
        brushes.append(self._box(ox - r - t, oy + inner - t, oz, ox - inner + t, oy + r + t, oz + total_h))
        brushes.append(self._box(ox + inner - t, oy + inner - t, oz, ox + r + t, oy + r + t, oz + total_h))
        brushes.append(self._box(ox - r - t, oy - r - t, oz, ox - inner + t, oy - inner + t, oz + total_h))
        brushes.append(self._box(ox + inner - t, oy - r - t, oz, ox + r + t, oy - inner + t, oz + total_h))

        # Interior level floors with stairwell openings
        # Opening location depends on stair type
        if self.stair_type.lower() == "spiral":
            # Spiral stairs: central opening
            stair_cx = ox + r * 0.4
            stair_cy = oy + inner * 0.4
            stairwell_r = min(r * 0.6, inner * 0.7) + 16

            for lvl in range(1, self.levels):
                fz = oz + lvl * level_h
                # Create floor with circular stairwell cutout
                brushes.append(self._box(
                    ox - r, oy - inner, fz - t,
                    ox + r, stair_cy - stairwell_r, fz
                ))
                brushes.append(self._box(
                    ox - r, stair_cy + stairwell_r, fz - t,
                    ox + r, oy + inner, fz
                ))
                brushes.append(self._box(
                    ox - r, stair_cy - stairwell_r, fz - t,
                    stair_cx - stairwell_r, stair_cy + stairwell_r, fz
                ))
                brushes.append(self._box(
                    stair_cx + stairwell_r, stair_cy - stairwell_r, fz - t,
                    ox + r, stair_cy + stairwell_r, fz
                ))
                # E-W floor sections for octagon corners
                brushes.append(self._box(ox - inner, oy - r, fz - t, ox + inner, oy - inner, fz))
                brushes.append(self._box(ox - inner, oy + inner, fz - t, ox + inner, oy + r, fz))
        else:
            # Straight/perimeter stairs: create proper floors with stair openings
            # Stairs wrap around: wall 0 (right/+X) → wall 1 (back/+Y) → wall 2 (left/-X) → wall 3 (front/-Y)
            stair_width = 48.0   # Match _generate_straight_stairs
            wall_gap = 8.0       # Match _generate_straight_stairs
            stair_zone = stair_width + wall_gap + 16  # Zone to keep clear for stairs

            for lvl in range(1, self.levels):
                fz = oz + lvl * level_h
                wall_index = (lvl - 1) % 4  # Which wall has stairs arriving at this floor

                # Create main floor pieces (two overlapping rectangles for octagon)
                # but with cutouts for the stair arrival zone

                if wall_index == 0:
                    # Stairs arrive from right wall (+X side)
                    # Main N-S rectangle with cutout on right
                    brushes.append(self._box(ox - r, oy - inner, fz - t, ox + r - stair_zone, oy + inner, fz))
                    # Fill above/below stair zone on right
                    brushes.append(self._box(ox + r - stair_zone, oy - inner, fz - t, ox + r, oy - stair_zone, fz))
                    brushes.append(self._box(ox + r - stair_zone, oy + stair_zone, fz - t, ox + r, oy + inner, fz))
                    # E-W rectangle (full, stairs don't penetrate this direction)
                    brushes.append(self._box(ox - inner, oy - r, fz - t, ox + inner, oy - inner, fz))
                    brushes.append(self._box(ox - inner, oy + inner, fz - t, ox + inner, oy + r, fz))

                elif wall_index == 1:
                    # Stairs arrive from back wall (+Y side)
                    # Main N-S rectangle (full, stairs don't penetrate this direction)
                    brushes.append(self._box(ox - r, oy - inner, fz - t, ox + r, oy + inner - stair_zone, fz))
                    # E-W rectangle with cutout on back
                    brushes.append(self._box(ox - inner, oy - r, fz - t, ox + inner, oy - inner, fz))
                    brushes.append(self._box(ox - inner, oy + inner, fz - t, ox - stair_zone, oy + r, fz))
                    brushes.append(self._box(ox + stair_zone, oy + inner, fz - t, ox + inner, oy + r, fz))
                    # Fill the gap from main rect to back
                    brushes.append(self._box(ox - r, oy + inner - stair_zone, fz - t, ox - stair_zone, oy + inner, fz))
                    brushes.append(self._box(ox + stair_zone, oy + inner - stair_zone, fz - t, ox + r, oy + inner, fz))

                elif wall_index == 2:
                    # Stairs arrive from left wall (-X side)
                    # Main N-S rectangle with cutout on left
                    brushes.append(self._box(ox - r + stair_zone, oy - inner, fz - t, ox + r, oy + inner, fz))
                    # Fill above/below stair zone on left
                    brushes.append(self._box(ox - r, oy - inner, fz - t, ox - r + stair_zone, oy - stair_zone, fz))
                    brushes.append(self._box(ox - r, oy + stair_zone, fz - t, ox - r + stair_zone, oy + inner, fz))
                    # E-W rectangle (full)
                    brushes.append(self._box(ox - inner, oy - r, fz - t, ox + inner, oy - inner, fz))
                    brushes.append(self._box(ox - inner, oy + inner, fz - t, ox + inner, oy + r, fz))

                else:
                    # Stairs arrive from front wall (-Y side)
                    # Main N-S rectangle (full)
                    brushes.append(self._box(ox - r, oy - inner + stair_zone, fz - t, ox + r, oy + inner, fz))
                    # E-W rectangle with cutout on front
                    brushes.append(self._box(ox - inner, oy - r, fz - t, ox - stair_zone, oy - inner, fz))
                    brushes.append(self._box(ox + stair_zone, oy - r, fz - t, ox + inner, oy - inner, fz))
                    brushes.append(self._box(ox - inner, oy + inner, fz - t, ox + inner, oy + r, fz))
                    # Fill the gap from main rect to front
                    brushes.append(self._box(ox - r, oy - inner, fz - t, ox - stair_zone, oy - inner + stair_zone, fz))
                    brushes.append(self._box(ox + stair_zone, oy - inner, fz - t, ox + r, oy - inner + stair_zone, fz))

        # Generate stairs
        if self.levels > 1:
            if self.stair_type.lower() == "spiral":
                brushes.extend(self._generate_spiral_stairs(
                    ox, oy, oz, r, inner, total_h, level_h, rng
                ))
            else:
                brushes.extend(self._generate_straight_stairs(
                    ox, oy, oz, r, inner, total_h, level_h, rng
                ))

        # Register portal tag for octagonal tower
        if self.has_entrance:
            self._register_portal_tag(
                portal_id="entrance",
                center_x=ox,
                center_y=oy - r,  # South side of octagonal tower
                center_z=oz,
                direction=PortalDirection.SOUTH,
            )

        return brushes

    def _generate_spiral_stairs(
        self,
        ox: float, oy: float, oz: float,
        r: float, inner: float,
        total_h: float, level_h: float,
        rng: random.Random
    ) -> List[Brush]:
        """Generate spiral stairs wrapping around a central post.

        Creates a proper spiral staircase with pie-slice steps radiating
        from a central column. Each step is a wedge that connects to
        the next, forming a continuous ascending spiral.
        """
        brushes: List[Brush] = []

        # Center the spiral in the back-right area of the tower
        # Use a larger radius for a visible spiral
        cx = ox + r * 0.4  # Offset toward back-right
        cy = oy + inner * 0.4

        # Central post (runs full height)
        post_r = 16.0
        brushes.append(self._box(
            cx - post_r, cy - post_r, oz,
            cx + post_r, cy + post_r, oz + total_h
        ))

        # Spiral parameters - use a large radius for visibility
        spiral_inner_r = post_r + 4  # Small gap from post
        spiral_outer_r = min(r * 0.6, inner * 0.7)  # Large enough to see

        # Climbable step height (max 16 units for comfortable climbing)
        step_h = 12.0  # Conservative for easy climbing
        steps_per_rotation = 12  # Steps to complete one full circle
        angle_per_step = (2 * math.pi) / steps_per_rotation

        # Generate steps for full tower height
        total_steps = int(total_h / step_h)

        for step in range(total_steps):
            # Each step is a pie-slice wedge
            angle1 = step * angle_per_step
            angle2 = (step + 1) * angle_per_step
            step_z = oz + step * step_h

            # Use radial segment for proper pie-slice geometry
            brushes.append(self._radial_segment(
                cx, cy, step_z, step_z + step_h,
                spiral_inner_r, spiral_outer_r,
                angle1, angle2
            ))

        return brushes

    def _generate_straight_stairs(
        self,
        ox: float, oy: float, oz: float,
        r: float, inner: float,
        total_h: float, level_h: float,
        rng: random.Random
    ) -> List[Brush]:
        """Generate perimeter stairs that connect tower floor levels.

        Creates stairs that properly connect each floor level:
        - Stairs run along one wall per level
        - Each flight starts at a floor level and ends at the next floor
        - Landing platforms at each floor for access
        """
        brushes: List[Brush] = []

        # Stair dimensions
        stair_width = 48.0   # Width of stair treads
        step_h = 16.0        # Height per step (max for comfortable climbing)
        wall_gap = 8.0       # Gap between stairs and outer wall

        # Calculate steps needed per floor
        steps_per_level = int(level_h / step_h)
        actual_step_h = level_h / steps_per_level  # Exact step height to reach floor

        # Stair run positions along each wall (cycle through walls for each level)
        # Wall 0: Right wall (+X side), stairs run in +Y
        # Wall 1: Back wall (+Y side), stairs run in -X
        # Wall 2: Left wall (-X side), stairs run in -Y
        # Wall 3: Front wall (-Y side), stairs run in +X

        for level in range(self.levels - 1):
            floor_z = oz + level * level_h
            next_floor_z = oz + (level + 1) * level_h
            wall_index = level % 4

            # Calculate stair depth per step (to fit all steps along the wall)
            wall_length = 2 * inner - 2 * stair_width  # Available length along wall
            step_d = wall_length / steps_per_level

            # Generate stairs for this level
            if wall_index == 0:
                # Right wall: stairs along +Y direction
                start_x = ox + r - wall_gap - stair_width
                start_y = oy - inner + stair_width

                for step in range(steps_per_level):
                    sy = start_y + step * step_d
                    sz = floor_z + step * actual_step_h
                    brushes.append(self._box(
                        start_x, sy, sz,
                        start_x + stair_width, sy + step_d, sz + actual_step_h
                    ))

                # Landing at next floor level (at end of stairs)
                brushes.append(self._box(
                    start_x - 16, start_y + wall_length - 16, next_floor_z - 8,
                    start_x + stair_width + 16, start_y + wall_length + stair_width, next_floor_z
                ))

            elif wall_index == 1:
                # Back wall: stairs along -X direction
                start_x = ox + inner - stair_width
                start_y = oy + r - wall_gap - stair_width

                for step in range(steps_per_level):
                    sx = start_x - step * step_d
                    sz = floor_z + step * actual_step_h
                    brushes.append(self._box(
                        sx - step_d, start_y, sz,
                        sx, start_y + stair_width, sz + actual_step_h
                    ))

                # Landing at next floor level
                brushes.append(self._box(
                    start_x - wall_length - stair_width, start_y - 16, next_floor_z - 8,
                    start_x - wall_length + 16, start_y + stair_width + 16, next_floor_z
                ))

            elif wall_index == 2:
                # Left wall: stairs along -Y direction
                start_x = ox - r + wall_gap
                start_y = oy + inner - stair_width

                for step in range(steps_per_level):
                    sy = start_y - step * step_d
                    sz = floor_z + step * actual_step_h
                    brushes.append(self._box(
                        start_x, sy - step_d, sz,
                        start_x + stair_width, sy, sz + actual_step_h
                    ))

                # Landing at next floor level
                brushes.append(self._box(
                    start_x - 16, start_y - wall_length - stair_width, next_floor_z - 8,
                    start_x + stair_width + 16, start_y - wall_length + 16, next_floor_z
                ))

            else:
                # Front wall: stairs along +X direction
                start_x = ox - inner + stair_width
                start_y = oy - r + wall_gap

                for step in range(steps_per_level):
                    sx = start_x + step * step_d
                    sz = floor_z + step * actual_step_h
                    brushes.append(self._box(
                        sx, start_y, sz,
                        sx + step_d, start_y + stair_width, sz + actual_step_h
                    ))

                # Landing at next floor level
                brushes.append(self._box(
                    start_x + wall_length - 16, start_y - 16, next_floor_z - 8,
                    start_x + wall_length + stair_width + 16, start_y + stair_width + 16, next_floor_z
                ))

        return brushes

    def _generate_polygonal_tower(
        self, ox: float, oy: float, oz: float,
        radius: float, total_h: float, t: float
    ) -> List[Brush]:
        """Generate polygonal tower shell (no multi-level interior)."""
        brushes: List[Brush] = []

        # Ensure minimum radius for polygon
        radius = max(radius, self._get_polygon_min_radius(self.shell_sides))

        # Center is at origin
        cx = ox
        cy = oy

        # Determine portal segment
        portal_segment = self._find_portal_segment(self.shell_sides, "SOUTH") if self.has_entrance else -1

        # Compute vestibule clip zone to avoid floor/ceiling overlap
        vestibule_clip = self._compute_vestibule_clip_zone(
            cx, cy, radius, self.shell_sides, portal_segment,
            PORTAL_WIDTH, t,
            room_origin=(ox, oy - radius, oz),
            room_length=radius * 2,
            room_width=radius * 2
        )

        # Generate floor (with clip zone to avoid vestibule overlap)
        brushes.extend(self._generate_polygonal_floor(
            cx, cy, oz, radius, self.shell_sides, t,
            is_ceiling=False,
            vestibule_clip_zone=vestibule_clip
        ))

        # Generate ceiling (with clip zone to avoid vestibule overlap)
        brushes.extend(self._generate_polygonal_floor(
            cx, cy, oz + total_h, radius, self.shell_sides, t,
            is_ceiling=True,
            vestibule_clip_zone=vestibule_clip
        ))

        # Set portal target for grid alignment before generating walls
        self._set_portal_target_for_polygonal(cx, cy, radius, self.shell_sides, portal_segment)

        # Generate walls
        brushes.extend(self._generate_polygonal_walls(
            cx, cy, oz, oz + total_h, radius, self.shell_sides, t,
            portal_segment=portal_segment,
            portal_width=PORTAL_WIDTH,
            portal_height=PORTAL_HEIGHT,
            portal_target_x=getattr(self, '_portal_target_x', None),
            portal_target_y=getattr(self, '_portal_target_y', None)
        ))

        # Generate connector corridor from portal to grid center
        # Tower is centered at origin, so room_length/width derived from radius
        if self.has_entrance:
            brushes.extend(self._generate_polygonal_portal_connector(
                cx, cy, oz, radius, self.shell_sides, portal_segment,
                PORTAL_WIDTH, PORTAL_HEIGHT, t, room_height=total_h,
                room_origin=(ox, oy - radius, oz),  # Approximate room bounds
                room_length=radius * 2,
                room_width=radius * 2
            ))

        return brushes


class Chamber(GeometricPrimitive):
    """A generic configurable room for any purpose.

    The simplest room primitive - serves as the template for others.

    Features:
    - Simple rectangular layout (or polygonal with shell_sides != 4)
    - Configurable entrance position (front, left, right, back)
    - Optional corner pillars
    - Optional wall base trim
    - 100% sealed geometry
    """

    width: float = 192.0        # Half-width (total = 2×width)
    length: float = 256.0       # Front-to-back length
    height: float = 128.0       # Ceiling height
    shell_sides: int = 4        # Number of sides (4=square, 6=hex, 8=octagon)
    entrance_position: str = "front"  # front, left, right, back
    pillar_count: int = 0       # Corner pillars (0, 2, or 4)
    wall_trim: bool = False     # Add base trim around walls
    random_seed: int = 0

    # Pillar customization
    pillar_style: str = "square"  # square, hexagonal, octagonal, round
    pillar_capital: bool = False  # Add capitals to pillars

    # Portal control - set by layout generator based on connections
    has_entrance: bool = True

    WALL_THICKNESS: float = 16.0

    @classmethod
    def get_display_name(cls) -> str:
        return "Chamber"

    @classmethod
    def get_category(cls) -> str:
        return "Rooms"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "width": {
                "type": "float", "default": 192.0, "min": 64, "max": 384, "label": "Half-Width",
                "description": "Half-width of the room (total width = 2x this value)"
            },
            "length": {
                "type": "float", "default": 256.0, "min": 128, "max": 512, "label": "Length",
                "description": "Length from front to back of the room"
            },
            "height": {
                "type": "float", "default": 128.0, "min": 64, "max": 256, "label": "Height",
                "description": "Interior ceiling height"
            },
            "shell_sides": {
                "type": "int",
                "default": 4,
                "min": 3,
                "max": 16,
                "label": "Shell Sides",
                "description": "Number of sides for room shape (4=square, 6=hex, 8=octagon)"
            },
            "entrance_position": {
                "type": "choice",
                "default": "front",
                "choices": ["front", "left", "right", "back"],
                "label": "Entrance Position",
                "description": "Which wall to place the entrance portal on"
            },
            "pillar_count": {
                "type": "int", "default": 0, "min": 0, "max": 4, "label": "Corner Pillars",
                "description": "Number of decorative pillars (0, 2 front, or 4 corners)"
            },
            "pillar_style": {
                "type": "choice",
                "default": "square",
                "choices": ["square", "hexagonal", "octagonal", "round"],
                "label": "Pillar Style",
                "description": "Cross-section shape of corner pillars"
            },
            "pillar_capital": {
                "type": "bool", "default": False, "label": "Pillar Capitals",
                "description": "Add decorative capitals atop pillars"
            },
            "wall_trim": {
                "type": "bool", "default": False, "label": "Wall Base Trim",
                "description": "Add decorative trim along the base of walls"
            },
            "has_entrance": {
                "type": "bool", "default": True, "label": "Enable Entrance Portal",
                "description": "Enable the entrance portal opening"
            },
            "random_seed": {
                "type": "int", "default": 0, "min": 0, "max": 999999, "label": "Random Seed (0=random)",
                "description": "Seed for reproducible variation (0 = random each time)"
            },
        }

    def generate(self) -> List[Brush]:
        self._reset_tags()  # Reset tags for fresh generation
        ox, oy, oz = self.params.origin
        brushes: List[Brush] = []
        t = self.WALL_THICKNESS

        if self.random_seed == 0:
            rng = random.Random()
        else:
            rng = random.Random(self.random_seed)

        # Structural dimensions - MUST be exact for portal alignment
        hw = self.width
        nl = self.length
        nh = self.height

        # Use polygonal shell if sides != 4
        if self.shell_sides != 4:
            brushes = self._generate_polygonal_chamber(ox, oy, oz, hw, nl, nh, t)
            # Register portal tag at footprint edge (grid-aligned)
            # Not at polygon interior - ensures proper alignment with adjoining halls
            if self.has_entrance:
                entrance = self.entrance_position.lower()
                portal_dir = {"front": PortalDirection.SOUTH, "back": PortalDirection.NORTH,
                              "left": PortalDirection.WEST, "right": PortalDirection.EAST}.get(entrance, PortalDirection.SOUTH)
                # Compute footprint edge positions based on portal direction
                if portal_dir == PortalDirection.SOUTH:
                    tag_center_x, tag_center_y = ox, oy
                elif portal_dir == PortalDirection.NORTH:
                    tag_center_x, tag_center_y = ox, oy + nl
                elif portal_dir == PortalDirection.WEST:
                    tag_center_x, tag_center_y = ox - hw, oy + nl / 2
                else:  # EAST
                    tag_center_x, tag_center_y = ox + hw, oy + nl / 2
                self._register_portal_tag(
                    portal_id="entrance",
                    center_x=tag_center_x,
                    center_y=tag_center_y,
                    center_z=oz,
                    direction=portal_dir,
                )
            return brushes

        # Standard rectangular generation (sides == 4)
        # Floor - extends beyond all walls
        brushes.append(self._box(ox - hw - t, oy - t, oz - t, ox + hw + t, oy + nl + t, oz))

        # Ceiling - extends beyond all walls
        brushes.append(self._box(ox - hw - t, oy - t, oz + nh, ox + hw + t, oy + nl + t, oz + nh + t))

        entrance = self.entrance_position.lower() if self.has_entrance else None
        door_center_y = oy + nl / 2

        # Generate walls with entrance opening at the specified position (if has_entrance=True)
        # Using unified portal system for all entrance locations

        # Left wall
        left_portal = PortalSpec(enabled=(entrance == "left"))
        left_brushes, _ = generate_portal_wall(
            box_func=self._box,
            x1=ox - hw - t, y1=oy - t, z1=oz,
            x2=ox - hw, y2=oy + nl + t, z2=oz + nh,
            portal_spec=left_portal,
            portal_axis="y",
            portal_center=door_center_y,
        )
        brushes.extend(left_brushes)

        # Right wall
        right_portal = PortalSpec(enabled=(entrance == "right"))
        right_brushes, _ = generate_portal_wall(
            box_func=self._box,
            x1=ox + hw, y1=oy - t, z1=oz,
            x2=ox + hw + t, y2=oy + nl + t, z2=oz + nh,
            portal_spec=right_portal,
            portal_axis="y",
            portal_center=door_center_y,
        )
        brushes.extend(right_brushes)

        # Front wall
        front_portal = PortalSpec(enabled=(entrance == "front"))
        front_brushes, _ = generate_portal_wall(
            box_func=self._box,
            x1=ox - hw - t, y1=oy - t, z1=oz,
            x2=ox + hw + t, y2=oy, z2=oz + nh,
            portal_spec=front_portal,
            portal_axis="x",
            portal_center=ox,
        )
        brushes.extend(front_brushes)

        # Back wall
        back_portal = PortalSpec(enabled=(entrance == "back"))
        back_brushes, _ = generate_portal_wall(
            box_func=self._box,
            x1=ox - hw - t, y1=oy + nl, z1=oz,
            x2=ox + hw + t, y2=oy + nl + t, z2=oz + nh,
            portal_spec=back_portal,
            portal_axis="x",
            portal_center=ox,
        )
        brushes.extend(back_brushes)

        # Corner pillars
        pillar_size = 24.0
        pillar_inset = 16.0
        if self.pillar_count >= 2:
            # Front corners
            brushes.extend(self._generate_room_pillar(
                ox - hw + pillar_inset, oy + pillar_inset, oz, oz + nh,
                pillar_size, self.pillar_style, self.pillar_capital
            ))
            brushes.extend(self._generate_room_pillar(
                ox + hw - pillar_inset, oy + pillar_inset, oz, oz + nh,
                pillar_size, self.pillar_style, self.pillar_capital
            ))
        if self.pillar_count >= 4:
            # Back corners
            brushes.extend(self._generate_room_pillar(
                ox - hw + pillar_inset, oy + nl - pillar_inset, oz, oz + nh,
                pillar_size, self.pillar_style, self.pillar_capital
            ))
            brushes.extend(self._generate_room_pillar(
                ox + hw - pillar_inset, oy + nl - pillar_inset, oz, oz + nh,
                pillar_size, self.pillar_style, self.pillar_capital
            ))

        # Wall base trim
        if self.wall_trim:
            trim_h = 8.0
            trim_depth = 4.0
            # Left trim (skip entrance)
            if entrance != "left":
                brushes.append(self._box(ox - hw, oy, oz, ox - hw + trim_depth, oy + nl, oz + trim_h))
            # Right trim
            if entrance != "right":
                brushes.append(self._box(ox + hw - trim_depth, oy, oz, ox + hw, oy + nl, oz + trim_h))
            # Front trim
            if entrance != "front":
                brushes.append(self._box(ox - hw, oy, oz, ox + hw, oy + trim_depth, oz + trim_h))
            # Back trim
            if entrance != "back":
                brushes.append(self._box(ox - hw, oy + nl - trim_depth, oz, ox + hw, oy + nl, oz + trim_h))

        # Register portal tag for rectangular chamber
        if self.has_entrance:
            portal_positions = {
                "front": (ox, oy, PortalDirection.SOUTH),
                "back": (ox, oy + nl, PortalDirection.NORTH),
                "left": (ox - hw, oy + nl / 2, PortalDirection.WEST),
                "right": (ox + hw, oy + nl / 2, PortalDirection.EAST),
            }
            pos = portal_positions.get(entrance, portal_positions["front"])
            self._register_portal_tag(
                portal_id="entrance",
                center_x=pos[0],
                center_y=pos[1],
                center_z=oz,
                direction=pos[2],
            )

        return brushes

    def _generate_polygonal_chamber(
        self,
        ox: float, oy: float, oz: float,
        hw: float, nl: float, nh: float,
        t: float
    ) -> List[Brush]:
        """Generate chamber with polygonal (non-rectangular) shell.

        Creates an N-sided room centered at origin with configurable entrance.
        Supports pillars within the polygonal space.
        """
        brushes: List[Brush] = []

        # Calculate radius from room dimensions, clamped to fit footprint
        # The polygon must fit within both dimensions: hw (half-width) and nl/2 (half-length)
        # Using min() ensures non-square footprints don't overflow in the short dimension
        max_radius = min(hw, nl / 2)
        radius = max(max_radius, self._get_polygon_min_radius(self.shell_sides))

        # Center the polygon at the room center
        cx = ox
        cy = oy + nl / 2

        # Determine portal segment based on entrance position
        entrance = self.entrance_position.lower() if self.has_entrance else None
        portal_segment = -1
        if entrance:
            direction_map = {
                "front": "SOUTH",
                "back": "NORTH",
                "left": "WEST",
                "right": "EAST"
            }
            portal_segment = self._find_portal_segment(
                self.shell_sides,
                direction_map.get(entrance, "SOUTH")
            )

        # Compute vestibule clip zone to avoid floor/ceiling overlap
        vestibule_clip = self._compute_vestibule_clip_zone(
            cx, cy, radius, self.shell_sides, portal_segment,
            PORTAL_WIDTH, t,
            room_origin=self.params.origin,
            room_length=nl,
            room_width=hw * 2
        )

        # Generate floor (with clip zone to avoid vestibule overlap)
        brushes.extend(self._generate_polygonal_floor(
            cx, cy, oz, radius, self.shell_sides, t,
            is_ceiling=False,
            vestibule_clip_zone=vestibule_clip
        ))

        # Generate ceiling (with clip zone to avoid vestibule overlap)
        brushes.extend(self._generate_polygonal_floor(
            cx, cy, oz + nh, radius, self.shell_sides, t,
            is_ceiling=True,
            vestibule_clip_zone=vestibule_clip
        ))

        # Set portal target for grid alignment before generating walls
        effective_portal_segment = portal_segment if self.has_entrance else -1
        self._set_portal_target_for_polygonal(cx, cy, radius, self.shell_sides, effective_portal_segment)

        # Generate walls
        brushes.extend(self._generate_polygonal_walls(
            cx, cy, oz, oz + nh, radius, self.shell_sides, t,
            portal_segment=effective_portal_segment,
            portal_width=PORTAL_WIDTH,
            portal_height=PORTAL_HEIGHT,
            portal_target_x=getattr(self, '_portal_target_x', None),
            portal_target_y=getattr(self, '_portal_target_y', None)
        ))

        # Generate connector corridor from portal to grid center
        # Pass room origin and dimensions for accurate footprint edge calculation
        if self.has_entrance:
            brushes.extend(self._generate_polygonal_portal_connector(
                cx, cy, oz, radius, self.shell_sides, portal_segment,
                PORTAL_WIDTH, PORTAL_HEIGHT, t, room_height=nh,
                room_origin=(ox, oy, oz), room_length=nl, room_width=hw * 2
            ))

        # Add corner pillars within polygonal space
        if self.pillar_count > 0:
            pillar_size = 24.0
            # Place pillars at corners inside the polygon
            # Use 70% of radius to keep pillars away from walls
            pillar_radius = radius * 0.7
            import math

            if self.pillar_count >= 2:
                # Front pillars (near entrance side)
                for angle_offset in [-0.3, 0.3]:
                    px = cx + pillar_radius * math.sin(angle_offset)
                    py = cy - pillar_radius * math.cos(angle_offset)
                    brushes.extend(self._generate_room_pillar(
                        px, py, oz, oz + nh,
                        pillar_size, self.pillar_style, self.pillar_capital
                    ))

            if self.pillar_count >= 4:
                # Back pillars
                for angle_offset in [math.pi - 0.3, math.pi + 0.3]:
                    px = cx + pillar_radius * math.sin(angle_offset)
                    py = cy - pillar_radius * math.cos(angle_offset)
                    brushes.extend(self._generate_room_pillar(
                        px, py, oz, oz + nh,
                        pillar_size, self.pillar_style, self.pillar_capital
                    ))

        return brushes


class Storage(GeometricPrimitive):
    """Generic storage room with configurable storage type.

    Features:
    - Configurable ceiling height (low/normal/tall)
    - Multiple storage types: barrels, shelves, crates, mixed
    - Optional floor props
    - Optional support posts (for cellar-style)
    - Configurable shell shape (4=square, 6=hex, 8=octagon)
    - 100% sealed geometry

    Note: This room consolidates the former Cellar and Storeroom types.
    """

    width: float = 128.0        # Half-width
    length: float = 192.0       # Length
    height: float = 112.0       # Ceiling height
    shell_sides: int = 4        # Number of sides (4=square, 6=hex, 8=octagon)
    storage_type: str = "mixed" # barrels, shelves, crates, mixed
    ceiling_height: str = "normal"  # low (80), normal (112), tall (144)
    alcove_count: int = 4       # Wall alcoves (0-8)
    crate_clusters: int = 2     # Floor crate piles (0-4)
    support_posts: int = 0      # Central support posts (0-4)
    random_seed: int = 0

    # Portal control - set by layout generator based on connections
    has_entrance: bool = True

    # Portal position offset - set by layout generator to match footprint cell positions
    _entrance_x_offset: float = 0.0  # X offset for entrance (negative = left of center)

    WALL_THICKNESS: float = 16.0

    # Height presets
    HEIGHT_PRESETS = {"low": 80.0, "normal": 112.0, "tall": 144.0}

    @classmethod
    def get_display_name(cls) -> str:
        return "Storage"

    @classmethod
    def get_category(cls) -> str:
        return "Rooms"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "width": {
                "type": "float", "default": 128.0, "min": 64, "max": 256, "label": "Half-Width",
                "description": "Half-width of the storage room"
            },
            "length": {
                "type": "float", "default": 192.0, "min": 128, "max": 384, "label": "Length",
                "description": "Length from front to back"
            },
            "shell_sides": {
                "type": "int",
                "default": 4,
                "min": 3,
                "max": 16,
                "label": "Shell Sides",
                "description": "Number of sides for room shape (4=square, 6=hex, 8=octagon)"
            },
            "storage_type": {
                "type": "choice",
                "default": "mixed",
                "choices": ["barrels", "shelves", "crates", "mixed"],
                "label": "Storage Type",
                "description": "Type of storage items to generate (affects alcoves and floor props)"
            },
            "ceiling_height": {
                "type": "choice",
                "default": "normal",
                "choices": ["low", "normal", "tall"],
                "label": "Ceiling Height",
                "description": "Ceiling height preset (low=80, normal=112, tall=144 units)"
            },
            "alcove_count": {
                "type": "int", "default": 4, "min": 0, "max": 8, "label": "Wall Alcoves",
                "description": "Number of alcove niches along the walls"
            },
            "crate_clusters": {
                "type": "int", "default": 2, "min": 0, "max": 4, "label": "Crate Clusters",
                "description": "Number of crate/barrel clusters on the floor"
            },
            "support_posts": {
                "type": "int", "default": 0, "min": 0, "max": 4, "label": "Support Posts",
                "description": "Central support posts (for cellar-style rooms)"
            },
            "has_entrance": {
                "type": "bool", "default": True, "label": "Enable Entrance Portal",
                "description": "Enable entrance portal opening"
            },
            "random_seed": {
                "type": "int", "default": 0, "min": 0, "max": 999999, "label": "Random Seed (0=random)",
                "description": "Seed for reproducible variation (0 = random each time)"
            },
        }

    def generate(self) -> List[Brush]:
        self._reset_tags()  # Reset tags for fresh generation
        ox, oy, oz = self.params.origin
        brushes: List[Brush] = []
        t = self.WALL_THICKNESS

        if self.random_seed == 0:
            rng = random.Random()
        else:
            rng = random.Random(self.random_seed)

        # Structural dimensions - MUST be exact for portal alignment
        hw = self.width
        nl = self.length
        # Use ceiling_height preset if available, else use height directly
        nh = self.HEIGHT_PRESETS.get(self.ceiling_height, self.height)

        # Use polygonal shell if sides != 4
        if self.shell_sides != 4:
            brushes = self._generate_polygonal_storage(ox, oy, oz, hw, nl, nh, t, rng)
            # Register portal tag at footprint edge (grid-aligned)
            if self.has_entrance:
                self._register_portal_tag(
                    portal_id="entrance",
                    center_x=ox,  # Room center X for grid alignment
                    center_y=oy,  # Footprint south edge (Y=origin)
                    center_z=oz,
                    direction=PortalDirection.SOUTH,
                )
            return brushes

        # Standard rectangular generation (sides == 4)
        # Determine storage behavior based on type
        storage_type = self.storage_type.lower()
        use_shelf_alcoves = storage_type in ("shelves", "mixed") and self.alcove_count > 0
        use_barrel_alcoves = storage_type == "barrels" and self.alcove_count > 0

        # Calculate alcove depth based on type
        if use_shelf_alcoves:
            alcove_depth = 24.0
        elif use_barrel_alcoves:
            alcove_depth = 32.0
        else:
            alcove_depth = 0

        # Floor
        brushes.append(self._box(
            ox - hw - t - alcove_depth, oy - t, oz - t,
            ox + hw + t + alcove_depth, oy + nl + t, oz
        ))

        # Ceiling
        brushes.append(self._box(
            ox - hw - t - alcove_depth, oy - t, oz + nh,
            ox + hw + t + alcove_depth, oy + nl + t, oz + nh + t
        ))

        # Side walls with optional alcoves based on storage type
        if use_shelf_alcoves:
            brushes.extend(self._generate_shelf_walls(ox, oy, oz, hw, nl, nh, alcove_depth, rng))
        elif use_barrel_alcoves:
            brushes.extend(self._generate_barrel_alcove_walls(ox, oy, oz, hw, nl, nh, alcove_depth, rng))
        else:
            brushes.append(self._box(ox - hw - t, oy - t, oz, ox - hw, oy + nl + t, oz + nh))
            brushes.append(self._box(ox + hw, oy - t, oz, ox + hw + t, oy + nl + t, oz + nh))

        # Front wall with optional entrance - using unified portal system
        entrance_x = ox + self._entrance_x_offset  # Apply portal offset
        portal_spec = PortalSpec(enabled=self.has_entrance)
        wall_brushes, _ = generate_portal_wall(
            box_func=self._box,
            x1=ox - hw - t - alcove_depth, y1=oy - t, z1=oz,
            x2=ox + hw + t + alcove_depth, y2=oy, z2=oz + nh,
            portal_spec=portal_spec,
            portal_axis="x",
            portal_center=entrance_x,
        )
        brushes.extend(wall_brushes)

        # Back wall
        brushes.append(self._box(ox - hw - t - alcove_depth, oy + nl, oz, ox + hw + t + alcove_depth, oy + nl + t, oz + nh))

        # Crate clusters (for crates or mixed storage types)
        if self.crate_clusters > 0 and storage_type in ("crates", "mixed"):
            brushes.extend(self._generate_crates(ox, oy, oz, hw, nl, rng))

        # Support posts (for low-ceiling cellar-style storage)
        if self.support_posts > 0:
            post_size = 16.0
            ps2 = post_size / 2
            post_spacing = nl / (self.support_posts + 1)
            for i in range(self.support_posts):
                py = oy + (i + 1) * post_spacing
                brushes.append(self._box(ox - ps2, py - ps2, oz, ox + ps2, py + ps2, oz + nh))

        # Register portal tag for rectangular storage
        if self.has_entrance:
            entrance_x = ox + self._entrance_x_offset
            self._register_portal_tag(
                portal_id="entrance",
                center_x=entrance_x,
                center_y=oy,
                center_z=oz,
                direction=PortalDirection.SOUTH,
            )

        return brushes

    def _generate_polygonal_storage(
        self,
        ox: float, oy: float, oz: float,
        hw: float, nl: float, nh: float,
        t: float, rng: random.Random
    ) -> List[Brush]:
        """Generate storage room with polygonal (non-rectangular) shell.

        Alcoves are disabled for polygonal shapes; crate clusters and support
        posts are preserved as simple interior features.
        """
        brushes: List[Brush] = []

        # Calculate radius from room dimensions, clamped to fit footprint
        # The polygon must fit within both dimensions: hw (half-width) and nl/2 (half-length)
        # Using min() ensures non-square footprints don't overflow in the short dimension
        max_radius = min(hw, nl / 2)
        radius = max(max_radius, self._get_polygon_min_radius(self.shell_sides))

        # Center the polygon at the room center
        cx = ox
        cy = oy + nl / 2

        # Determine portal segment
        portal_segment = self._find_portal_segment(self.shell_sides, "SOUTH") if self.has_entrance else -1

        # Compute vestibule clip zone to avoid floor/ceiling overlap
        vestibule_clip = self._compute_vestibule_clip_zone(
            cx, cy, radius, self.shell_sides, portal_segment,
            PORTAL_WIDTH, t,
            room_origin=(ox, oy, oz), room_length=nl, room_width=hw * 2
        )

        # Generate floor (with clip zone to avoid vestibule overlap)
        brushes.extend(self._generate_polygonal_floor(
            cx, cy, oz, radius, self.shell_sides, t,
            is_ceiling=False,
            vestibule_clip_zone=vestibule_clip
        ))

        # Generate ceiling (with clip zone to avoid vestibule overlap)
        brushes.extend(self._generate_polygonal_floor(
            cx, cy, oz + nh, radius, self.shell_sides, t,
            is_ceiling=True,
            vestibule_clip_zone=vestibule_clip
        ))

        # Set portal target for grid alignment before generating walls
        self._set_portal_target_for_polygonal(cx, cy, radius, self.shell_sides, portal_segment)

        # Generate walls
        brushes.extend(self._generate_polygonal_walls(
            cx, cy, oz, oz + nh, radius, self.shell_sides, t,
            portal_segment=portal_segment,
            portal_width=PORTAL_WIDTH,
            portal_height=PORTAL_HEIGHT,
            portal_target_x=getattr(self, '_portal_target_x', None),
            portal_target_y=getattr(self, '_portal_target_y', None)
        ))

        # Generate connector corridor from portal to grid center
        # Pass room origin and dimensions for accurate footprint edge calculation
        if self.has_entrance:
            brushes.extend(self._generate_polygonal_portal_connector(
                cx, cy, oz, radius, self.shell_sides, portal_segment,
                PORTAL_WIDTH, PORTAL_HEIGHT, t, room_height=nh,
                room_origin=(ox, oy, oz), room_length=nl, room_width=hw * 2
            ))

        # Support posts (preserved for polygonal shapes)
        if self.support_posts > 0:
            post_size = 16.0
            ps2 = post_size / 2
            # Arrange posts in circle around center
            post_radius = radius * 0.5
            for i in range(self.support_posts):
                angle = 2 * math.pi * i / self.support_posts
                px = cx + post_radius * math.cos(angle)
                py = cy + post_radius * math.sin(angle)
                brushes.append(self._box(px - ps2, py - ps2, oz, px + ps2, py + ps2, oz + nh))

        return brushes

    def _generate_shelf_walls(
        self,
        ox: float, oy: float, oz: float,
        hw: float, nl: float, nh: float,
        alcove_depth: float,
        rng: random.Random
    ) -> List[Brush]:
        """Generate side walls with shelf alcoves."""
        brushes: List[Brush] = []
        t = self.WALL_THICKNESS

        alcove_w = 40.0
        alcove_h = 80.0
        pilaster_w = 20.0
        shelf_h = 8.0
        shelf_count = 2

        start_y = oy + 40
        end_y = oy + nl - 40
        spacing = alcove_w + pilaster_w
        alcoves_per_side = self.alcove_count // 2

        for side in [-1, 1]:
            if side == -1:
                wall_inner = ox - hw
                wall_outer = ox - hw - t
                alcove_back = ox - hw - alcove_depth
            else:
                wall_inner = ox + hw
                wall_outer = ox + hw + t
                alcove_back = ox + hw + alcove_depth

            # Front section
            if side == -1:
                brushes.append(self._box(wall_outer, oy, oz, wall_inner, start_y, oz + nh))
            else:
                brushes.append(self._box(wall_inner, oy, oz, wall_outer, start_y, oz + nh))

            for i in range(alcoves_per_side):
                alcove_y1 = start_y + i * spacing
                alcove_y2 = alcove_y1 + alcove_w

                if alcove_y2 > end_y:
                    if side == -1:
                        brushes.append(self._box(wall_outer, alcove_y1, oz, wall_inner, oy + nl, oz + nh))
                    else:
                        brushes.append(self._box(wall_inner, alcove_y1, oz, wall_outer, oy + nl, oz + nh))
                    break

                # Lintel
                if side == -1:
                    brushes.append(self._box(wall_outer, alcove_y1, oz + alcove_h, wall_inner, alcove_y2, oz + nh))
                else:
                    brushes.append(self._box(wall_inner, alcove_y1, oz + alcove_h, wall_outer, alcove_y2, oz + nh))

                # Back wall
                if side == -1:
                    brushes.append(self._box(alcove_back - t, alcove_y1 - t, oz, alcove_back, alcove_y2 + t, oz + alcove_h))
                else:
                    brushes.append(self._box(alcove_back, alcove_y1 - t, oz, alcove_back + t, alcove_y2 + t, oz + alcove_h))

                # Side jambs
                if side == -1:
                    brushes.append(self._box(alcove_back, alcove_y1 - t, oz, wall_inner, alcove_y1, oz + alcove_h))
                    brushes.append(self._box(alcove_back, alcove_y2, oz, wall_inner, alcove_y2 + t, oz + alcove_h))
                else:
                    brushes.append(self._box(wall_inner, alcove_y1 - t, oz, alcove_back, alcove_y1, oz + alcove_h))
                    brushes.append(self._box(wall_inner, alcove_y2, oz, alcove_back, alcove_y2 + t, oz + alcove_h))

                # Ceiling
                if side == -1:
                    brushes.append(self._box(alcove_back, alcove_y1, oz + alcove_h, wall_inner, alcove_y2, oz + alcove_h + t))
                else:
                    brushes.append(self._box(wall_inner, alcove_y1, oz + alcove_h, alcove_back, alcove_y2, oz + alcove_h + t))

                # Shelves inside alcove
                for s in range(shelf_count):
                    shelf_z = oz + 24 + s * 28
                    if shelf_z + shelf_h > oz + alcove_h:
                        break
                    if side == -1:
                        brushes.append(self._box(alcove_back + 2, alcove_y1 + 2, shelf_z, wall_inner - 2, alcove_y2 - 2, shelf_z + shelf_h))
                    else:
                        brushes.append(self._box(wall_inner + 2, alcove_y1 + 2, shelf_z, alcove_back - 2, alcove_y2 - 2, shelf_z + shelf_h))

                # Pilaster
                if i < alcoves_per_side - 1:
                    if side == -1:
                        brushes.append(self._box(wall_outer, alcove_y2, oz, wall_inner, alcove_y2 + pilaster_w, oz + nh))
                    else:
                        brushes.append(self._box(wall_inner, alcove_y2, oz, wall_outer, alcove_y2 + pilaster_w, oz + nh))

            # Back section
            last_y = start_y + alcoves_per_side * spacing - pilaster_w
            if last_y < oy + nl:
                if side == -1:
                    brushes.append(self._box(wall_outer, last_y, oz, wall_inner, oy + nl, oz + nh))
                else:
                    brushes.append(self._box(wall_inner, last_y, oz, wall_outer, oy + nl, oz + nh))

        return brushes

    def _generate_barrel_alcove_walls(
        self,
        ox: float, oy: float, oz: float,
        hw: float, nl: float, nh: float,
        alcove_depth: float,
        rng: random.Random
    ) -> List[Brush]:
        """Generate side walls with barrel alcoves (cellar-style)."""
        brushes: List[Brush] = []
        t = self.WALL_THICKNESS

        alcove_w = 48.0  # Width of alcove opening
        alcove_h = 64.0  # Height of alcove
        pilaster_w = 24.0  # Wall between alcoves

        start_y = oy + 48
        end_y = oy + nl - 48
        spacing = alcove_w + pilaster_w
        alcoves_per_side = self.alcove_count // 2

        for side in [-1, 1]:
            if side == -1:
                wall_inner = ox - hw
                wall_outer = ox - hw - t
                alcove_back = ox - hw - alcove_depth
            else:
                wall_inner = ox + hw
                wall_outer = ox + hw + t
                alcove_back = ox + hw + alcove_depth

            # Front pilaster
            if side == -1:
                brushes.append(self._box(wall_outer, oy, oz, wall_inner, start_y, oz + nh))
            else:
                brushes.append(self._box(wall_inner, oy, oz, wall_outer, start_y, oz + nh))

            for i in range(alcoves_per_side):
                alcove_y1 = start_y + i * spacing
                alcove_y2 = alcove_y1 + alcove_w

                if alcove_y2 > end_y:
                    # Fill remaining as solid wall
                    if side == -1:
                        brushes.append(self._box(wall_outer, alcove_y1, oz, wall_inner, oy + nl, oz + nh))
                    else:
                        brushes.append(self._box(wall_inner, alcove_y1, oz, wall_outer, oy + nl, oz + nh))
                    break

                # Lintel above alcove
                if side == -1:
                    brushes.append(self._box(wall_outer, alcove_y1, oz + alcove_h, wall_inner, alcove_y2, oz + nh))
                else:
                    brushes.append(self._box(wall_inner, alcove_y1, oz + alcove_h, wall_outer, alcove_y2, oz + nh))

                # Alcove back wall
                if side == -1:
                    brushes.append(self._box(alcove_back - t, alcove_y1 - t, oz, alcove_back, alcove_y2 + t, oz + alcove_h))
                else:
                    brushes.append(self._box(alcove_back, alcove_y1 - t, oz, alcove_back + t, alcove_y2 + t, oz + alcove_h))

                # Alcove side jambs
                if side == -1:
                    brushes.append(self._box(alcove_back, alcove_y1 - t, oz, wall_inner, alcove_y1, oz + alcove_h))
                    brushes.append(self._box(alcove_back, alcove_y2, oz, wall_inner, alcove_y2 + t, oz + alcove_h))
                else:
                    brushes.append(self._box(wall_inner, alcove_y1 - t, oz, alcove_back, alcove_y1, oz + alcove_h))
                    brushes.append(self._box(wall_inner, alcove_y2, oz, alcove_back, alcove_y2 + t, oz + alcove_h))

                # Alcove ceiling
                if side == -1:
                    brushes.append(self._box(alcove_back, alcove_y1, oz + alcove_h, wall_inner, alcove_y2, oz + alcove_h + t))
                else:
                    brushes.append(self._box(wall_inner, alcove_y1, oz + alcove_h, alcove_back, alcove_y2, oz + alcove_h + t))

                # Pilaster after alcove
                if i < alcoves_per_side - 1:
                    pilaster_y1 = alcove_y2
                    pilaster_y2 = alcove_y2 + pilaster_w
                    if side == -1:
                        brushes.append(self._box(wall_outer, pilaster_y1, oz, wall_inner, pilaster_y2, oz + nh))
                    else:
                        brushes.append(self._box(wall_inner, pilaster_y1, oz, wall_outer, pilaster_y2, oz + nh))

            # Back pilaster
            last_y = start_y + alcoves_per_side * spacing - pilaster_w
            if last_y < oy + nl:
                if side == -1:
                    brushes.append(self._box(wall_outer, last_y, oz, wall_inner, oy + nl, oz + nh))
                else:
                    brushes.append(self._box(wall_inner, last_y, oz, wall_outer, oy + nl, oz + nh))

        return brushes

    def _generate_crates(
        self,
        ox: float, oy: float, oz: float,
        hw: float, nl: float,
        rng: random.Random
    ) -> List[Brush]:
        """Generate crate clusters on the floor."""
        brushes: List[Brush] = []
        placed_zones: List[tuple] = []

        for c in range(self.crate_clusters):
            # Find position avoiding entrance and other clusters
            cx, cy = None, None
            for _ in range(10):  # Try 10 times to place
                tcx = rng.uniform(ox - hw + 48, ox + hw - 48)
                tcy = rng.uniform(oy + 64, oy + nl - 48)
                if tcy < oy + 80:  # Skip entrance area
                    continue
                # Check against existing clusters
                too_close = False
                for px, py in placed_zones:
                    if abs(tcx - px) < 64 and abs(tcy - py) < 64:
                        too_close = True
                        break
                if not too_close:
                    cx, cy = tcx, tcy
                    placed_zones.append((cx, cy))
                    break

            if cx is None:
                continue

            # Generate 2-4 crates in cluster
            crate_count = rng.randint(2, 4)
            for _ in range(crate_count):
                cw = rng.uniform(20, 32)
                ch = rng.uniform(20, 36)
                ccx = cx + rng.uniform(-24, 24)
                ccy = cy + rng.uniform(-24, 24)
                brushes.append(self._box(ccx - cw/2, ccy - cw/2, oz, ccx + cw/2, ccy + cw/2, oz + ch))

        return brushes


class GreatHall(GeometricPrimitive):
    """Large rectangular feasting/gathering hall.

    High visual impact room for important scenes - feasts, audiences, etc.

    Features:
    - Large open space with high ceiling
    - Optional long tables (evenly spaced)
    - Optional raised dais at back for throne/high table
    - Optional fireplace alcove with raised hearth
    - Optional support pillar rows
    - 100% sealed geometry
    """

    width: float = 256.0        # Half-width
    length: float = 512.0       # Length
    height: float = 192.0       # High ceiling
    shell_sides: int = 4        # Number of sides (4=square, 6=hex, 8=octagon)
    table_count: int = 2        # Long tables (0-4)
    dais: bool = True           # Raised platform at back
    dais_height: float = 24.0   # Dais step height
    fireplace: bool = True      # Alcove with raised hearth
    pillar_rows: int = 2        # Support pillar rows (0, 2, 4)
    random_seed: int = 0

    # Pillar customization
    pillar_style: str = "square"  # square, hexagonal, octagonal, round
    pillar_capital: bool = False  # Add capitals to pillars

    # Portal control - set by layout generator based on connections
    has_entrance: bool = True   # Front (SOUTH) entrance
    has_side: bool = True       # Right (EAST) side entrance

    # Portal position offsets - set by layout generator to match footprint cell positions
    # These offset the portal from room center to align with the expected cell position
    _entrance_x_offset: float = 0.0  # X offset for entrance (negative = left of center)
    _side_y_offset: float = 0.0      # Y offset for side portal from room center

    WALL_THICKNESS: float = 16.0

    @classmethod
    def get_display_name(cls) -> str:
        return "Great Hall"

    @classmethod
    def get_category(cls) -> str:
        return "Rooms"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "width": {
                "type": "float", "default": 256.0, "min": 128, "max": 512, "label": "Half-Width",
                "description": "Half-width of the great hall"
            },
            "length": {
                "type": "float", "default": 512.0, "min": 256, "max": 1024, "label": "Length",
                "description": "Length from entrance to high table area"
            },
            "height": {
                "type": "float", "default": 192.0, "min": 128, "max": 320, "label": "Height",
                "description": "Ceiling height of the hall"
            },
            "shell_sides": {
                "type": "int",
                "default": 4,
                "min": 3,
                "max": 16,
                "label": "Shell Sides",
                "description": "Number of sides for room shape (4=square, 6=hex, 8=octagon)"
            },
            "table_count": {
                "type": "int", "default": 2, "min": 0, "max": 4, "label": "Table Count",
                "description": "Number of long feasting tables in the hall"
            },
            "dais": {
                "type": "bool", "default": True, "label": "Dais Platform",
                "description": "Add raised platform at the back for the high table"
            },
            "dais_height": {
                "type": "float", "default": 24.0, "min": 16, "max": 48, "label": "Dais Height",
                "description": "Height of the raised dais platform"
            },
            "fireplace": {
                "type": "bool", "default": True, "label": "Fireplace",
                "description": "Add decorative fireplace alcove at the back wall"
            },
            "pillar_rows": {
                "type": "int", "default": 2, "min": 0, "max": 4, "label": "Pillar Rows",
                "description": "Number of pillar rows along the length of the hall"
            },
            "pillar_style": {
                "type": "choice",
                "default": "square",
                "choices": ["square", "hexagonal", "octagonal", "round"],
                "label": "Pillar Style",
                "description": "Cross-section shape of support pillars"
            },
            "pillar_capital": {
                "type": "bool", "default": False, "label": "Pillar Capitals",
                "description": "Add decorative capitals atop pillars"
            },
            "has_entrance": {
                "type": "bool", "default": True, "label": "Enable Entrance Portal",
                "description": "Enable main entrance portal at the front"
            },
            "has_side": {
                "type": "bool", "default": True, "label": "Enable Side Portal",
                "description": "Enable side entrance portal on the right wall"
            },
            "random_seed": {
                "type": "int", "default": 0, "min": 0, "max": 999999, "label": "Random Seed (0=random)",
                "description": "Seed for reproducible variation (0 = random each time)"
            },
        }

    def generate(self) -> List[Brush]:
        self._reset_tags()  # Reset tags for fresh generation
        ox, oy, oz = self.params.origin
        brushes: List[Brush] = []
        t = self.WALL_THICKNESS

        if self.random_seed == 0:
            rng = random.Random()
        else:
            rng = random.Random(self.random_seed)

        # Structural dimensions - MUST be exact for portal alignment
        hw = self.width
        nl = self.length
        nh = self.height

        # Use polygonal shell if sides != 4
        if self.shell_sides != 4:
            brushes = self._generate_polygonal_greathall(ox, oy, oz, hw, nl, nh, t)
            # Register portal tags at footprint edges (grid-aligned)
            # Not at polygon interior - ensures proper alignment with adjoining halls
            if self.has_entrance:
                self._register_portal_tag(
                    portal_id="entrance",
                    center_x=ox,  # Room center X for grid alignment
                    center_y=oy,  # Footprint south edge (Y=origin)
                    center_z=oz,
                    direction=PortalDirection.SOUTH,
                )
            if self.has_side:
                self._register_portal_tag(
                    portal_id="side",
                    center_x=ox + hw,  # Footprint east edge
                    center_y=oy + nl / 2,  # Room center Y
                    center_z=oz,
                    direction=PortalDirection.EAST,
                )
            return brushes

        # Fireplace alcove dimensions
        fp_width = 96.0 if self.fireplace else 0
        fp_depth = 48.0 if self.fireplace else 0

        # Floor
        brushes.append(self._box(
            ox - hw - t, oy - t, oz - t,
            ox + hw + t, oy + nl + t + fp_depth, oz
        ))

        # Ceiling
        brushes.append(self._box(
            ox - hw - t, oy - t, oz + nh,
            ox + hw + t, oy + nl + t + fp_depth, oz + nh + t
        ))

        # Side walls
        # Left wall is always solid
        brushes.append(self._box(ox - hw - t, oy - t, oz, ox - hw, oy + nl + t + fp_depth, oz + nh))

        # Right wall has optional side portal - using unified portal system
        wall_center_y = oy + nl / 2 + self._side_y_offset
        side_portal = PortalSpec(enabled=self.has_side)
        right_wall_brushes, _ = generate_portal_wall(
            box_func=self._box,
            x1=ox + hw, y1=oy - t, z1=oz,
            x2=ox + hw + t, y2=oy + nl + t + fp_depth, z2=oz + nh,
            portal_spec=side_portal,
            portal_axis="y",
            portal_center=wall_center_y,
        )
        brushes.extend(right_wall_brushes)

        # Front wall with optional grand entrance - using unified portal system
        entrance_x = ox + self._entrance_x_offset
        entrance_portal = PortalSpec(enabled=self.has_entrance)
        front_wall_brushes, _ = generate_portal_wall(
            box_func=self._box,
            x1=ox - hw - t, y1=oy - t, z1=oz,
            x2=ox + hw + t, y2=oy, z2=oz + nh,
            portal_spec=entrance_portal,
            portal_axis="x",
            portal_center=entrance_x,
        )
        brushes.extend(front_wall_brushes)

        # Back wall with optional fireplace alcove
        if self.fireplace:
            # Wall sections on either side of fireplace
            brushes.append(self._box(ox - hw - t, oy + nl, oz, ox - fp_width/2, oy + nl + t, oz + nh))
            brushes.append(self._box(ox + fp_width/2, oy + nl, oz, ox + hw + t, oy + nl + t, oz + nh))
            # Fireplace lintel above
            brushes.append(self._box(ox - fp_width/2, oy + nl, oz + nh * 0.6, ox + fp_width/2, oy + nl + t, oz + nh))
            # Fireplace back wall
            brushes.append(self._box(ox - fp_width/2 - t, oy + nl + fp_depth, oz, ox + fp_width/2 + t, oy + nl + fp_depth + t, oz + nh * 0.6))
            # Fireplace side walls
            brushes.append(self._box(ox - fp_width/2 - t, oy + nl, oz, ox - fp_width/2, oy + nl + fp_depth + t, oz + nh * 0.6))
            brushes.append(self._box(ox + fp_width/2, oy + nl, oz, ox + fp_width/2 + t, oy + nl + fp_depth + t, oz + nh * 0.6))
            # Fireplace ceiling
            brushes.append(self._box(ox - fp_width/2, oy + nl, oz + nh * 0.6, ox + fp_width/2, oy + nl + fp_depth, oz + nh * 0.6 + t))
            # Raised hearth
            hearth_h = 16.0
            brushes.append(self._box(ox - fp_width/2 + 8, oy + nl - 8, oz, ox + fp_width/2 - 8, oy + nl + fp_depth - 8, oz + hearth_h))
        else:
            brushes.append(self._box(ox - hw - t, oy + nl, oz, ox + hw + t, oy + nl + t, oz + nh))

        # Dais at back
        if self.dais:
            dais_length = nl * 0.2
            dais_y = oy + nl - dais_length
            brushes.append(self._box(
                ox - hw + 16, dais_y, oz,
                ox + hw - 16, oy + nl - 16, oz + self.dais_height
            ))

        # Pillar rows
        if self.pillar_rows >= 2:
            pillar_size = 32.0
            pillar_x = hw * 0.6
            pillar_count = max(3, int(nl / 128))
            pillar_spacing = (nl - 128) / (pillar_count - 1)
            for i in range(pillar_count):
                py = oy + 64 + i * pillar_spacing
                for side in [-1, 1]:
                    px = ox + side * pillar_x
                    brushes.extend(self._generate_room_pillar(
                        px, py, oz, oz + nh,
                        pillar_size, self.pillar_style, self.pillar_capital
                    ))

        if self.pillar_rows >= 4:
            # Inner row of pillars
            pillar_x = hw * 0.3
            for i in range(pillar_count):
                py = oy + 64 + i * pillar_spacing
                for side in [-1, 1]:
                    px = ox + side * pillar_x
                    brushes.extend(self._generate_room_pillar(
                        px, py, oz, oz + nh,
                        pillar_size, self.pillar_style, self.pillar_capital
                    ))

        # Long tables
        if self.table_count > 0:
            table_w = 32.0
            table_h = 28.0
            table_l = nl * 0.5
            table_spacing = (hw * 2 - 96) / (self.table_count + 1)
            table_y = oy + 80
            for i in range(self.table_count):
                tx = ox - hw + 48 + (i + 1) * table_spacing
                brushes.append(self._box(tx - table_w/2, table_y, oz, tx + table_w/2, table_y + table_l, oz + table_h))

        # Register portal tags for rectangular greathall
        if self.has_entrance:
            entrance_x = ox + self._entrance_x_offset
            self._register_portal_tag(
                portal_id="entrance",
                center_x=entrance_x,
                center_y=oy,
                center_z=oz,
                direction=PortalDirection.SOUTH,
            )
        if self.has_side:
            wall_center_y = oy + nl / 2 + self._side_y_offset
            self._register_portal_tag(
                portal_id="side",
                center_x=ox + hw,
                center_y=wall_center_y,
                center_z=oz,
                direction=PortalDirection.EAST,
            )

        return brushes

    def _generate_polygonal_greathall(
        self, ox: float, oy: float, oz: float,
        hw: float, nl: float, nh: float, t: float
    ) -> List[Brush]:
        """Generate polygonal GreatHall shell with interior pillars.

        Multi-portal support: Both entrance (SOUTH) and side (EAST) portals are
        generated when enabled, using the vestibule approach for grid-aligned
        connections.
        """
        brushes: List[Brush] = []

        # Calculate radius from room dimensions, clamped to fit footprint
        # The polygon must fit within both dimensions: hw (half-width) and nl/2 (half-length)
        # Using min() ensures non-square footprints don't overflow in the short dimension
        max_radius = min(hw, nl / 2)
        radius = max(max_radius, self._get_polygon_min_radius(self.shell_sides))

        # Center the polygon at the room center
        cx = ox
        cy = oy + nl / 2

        # Determine portal segments for entrance and side
        entrance_segment = self._find_portal_segment(self.shell_sides, "SOUTH") if self.has_entrance else -1
        side_segment = self._find_portal_segment(self.shell_sides, "EAST") if self.has_side else -1

        # Compute vestibule clip zones for all active portals
        room_origin = (ox, oy, oz)
        vestibule_clips = []
        if entrance_segment >= 0:
            clip = self._compute_vestibule_clip_zone(
                cx, cy, radius, self.shell_sides, entrance_segment,
                PORTAL_WIDTH, t,
                room_origin=room_origin, room_length=nl, room_width=hw * 2
            )
            if clip:
                vestibule_clips.append(clip)
        if side_segment >= 0:
            clip = self._compute_vestibule_clip_zone(
                cx, cy, radius, self.shell_sides, side_segment,
                PORTAL_WIDTH, t,
                room_origin=room_origin, room_length=nl, room_width=hw * 2
            )
            if clip:
                vestibule_clips.append(clip)

        # Generate floor (with clip zones to avoid vestibule overlap)
        brushes.extend(self._generate_polygonal_floor(
            cx, cy, oz, radius, self.shell_sides, t,
            is_ceiling=False,
            vestibule_clip_zones=vestibule_clips if vestibule_clips else None
        ))

        # Generate ceiling (with clip zones to avoid vestibule overlap)
        brushes.extend(self._generate_polygonal_floor(
            cx, cy, oz + nh, radius, self.shell_sides, t,
            is_ceiling=True,
            vestibule_clip_zones=vestibule_clips if vestibule_clips else None
        ))

        # Build portal segments list for multi-portal wall generation
        portal_segments = []
        used_segments = set()

        if entrance_segment >= 0 and entrance_segment not in used_segments:
            # Set portal target for entrance (grid-aligned at room center X)
            self._set_portal_target_for_polygonal(cx, cy, radius, self.shell_sides, entrance_segment)
            portal_segments.append({
                'segment': entrance_segment,
                'width': PORTAL_WIDTH,
                'height': PORTAL_HEIGHT,
                'target_x': getattr(self, '_portal_target_x', None),
                'target_y': getattr(self, '_portal_target_y', None),
                'direction': 'SOUTH'
            })
            used_segments.add(entrance_segment)

        if side_segment >= 0 and side_segment not in used_segments:
            # Set portal target for side portal (grid-aligned at room center Y)
            self._set_portal_target_for_polygonal(cx, cy, radius, self.shell_sides, side_segment)
            portal_segments.append({
                'segment': side_segment,
                'width': PORTAL_WIDTH,
                'height': PORTAL_HEIGHT,
                'target_x': getattr(self, '_portal_target_x', None),
                'target_y': getattr(self, '_portal_target_y', None),
                'direction': 'EAST'
            })
            used_segments.add(side_segment)

        # Generate walls with ALL portal openings
        brushes.extend(self._generate_polygonal_walls(
            cx, cy, oz, oz + nh, radius, self.shell_sides, t,
            portal_segments=portal_segments if portal_segments else None
        ))

        # Generate vestibule corridor for EACH enabled portal
        # Pass room origin and dimensions for accurate footprint edge calculation
        room_origin = (ox, oy, oz)
        if self.has_entrance and entrance_segment >= 0:
            brushes.extend(self._generate_polygonal_portal_connector(
                cx, cy, oz, radius, self.shell_sides, entrance_segment,
                PORTAL_WIDTH, PORTAL_HEIGHT, t, room_height=nh,
                room_origin=room_origin, room_length=nl, room_width=hw * 2
            ))

        if self.has_side and side_segment >= 0:
            brushes.extend(self._generate_polygonal_portal_connector(
                cx, cy, oz, radius, self.shell_sides, side_segment,
                PORTAL_WIDTH, PORTAL_HEIGHT, t, room_height=nh,
                room_origin=room_origin, room_length=nl, room_width=hw * 2
            ))

        # Add two rows of pillars along the hall length
        pillar_size = 24.0
        pillar_x_offset = radius * 0.5  # Pillars at 50% of radius from center

        # Place 4-6 pillars per row based on room size
        num_pillars = min(6, max(3, int(nl / 128)))
        pillar_spacing = nl / (num_pillars + 1)

        for i in range(num_pillars):
            py = oy + (i + 1) * pillar_spacing
            # Left row
            brushes.extend(self._generate_room_pillar(
                cx - pillar_x_offset, py, oz, oz + nh,
                pillar_size, self.pillar_style, self.pillar_capital
            ))
            # Right row
            brushes.extend(self._generate_room_pillar(
                cx + pillar_x_offset, py, oz, oz + nh,
                pillar_size, self.pillar_style, self.pillar_capital
            ))

        return brushes


class Prison(GeometricPrimitive):
    """Prison/detention area with cells.

    Oppressive atmosphere with low ceiling and cell alcoves.

    Features:
    - Low ceiling for oppressive feel
    - Cell alcoves on sides (with optional bar geometry)
    - Optional central drain grate
    - 100% sealed geometry
    """

    width: float = 192.0        # Half-width
    length: float = 256.0       # Length
    height: float = 96.0        # Low oppressive ceiling
    shell_sides: int = 4        # Number of sides (4=square, 6=hex, 8=octagon)
    cell_count: int = 4         # Cell alcoves (2-8, split left/right)
    cell_depth: float = 64.0    # How deep cells recess
    central_drain: bool = True  # Floor drain grate in center
    bars: bool = True           # Bar geometry across cell openings
    random_seed: int = 0

    # Portal control - set by layout generator based on connections
    has_entrance: bool = True   # Front (SOUTH) entrance

    # Portal position offset - set by layout generator to match footprint cell positions
    _entrance_x_offset: float = 0.0  # X offset for entrance (negative = left of center)

    WALL_THICKNESS: float = 16.0

    @classmethod
    def get_display_name(cls) -> str:
        return "Prison"

    @classmethod
    def get_category(cls) -> str:
        return "Rooms"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "width": {
                "type": "float", "default": 192.0, "min": 128, "max": 320, "label": "Half-Width",
                "description": "Half-width of the prison block"
            },
            "length": {
                "type": "float", "default": 256.0, "min": 192, "max": 512, "label": "Length",
                "description": "Length from entrance to back wall"
            },
            "height": {
                "type": "float", "default": 96.0, "min": 72, "max": 128, "label": "Height",
                "description": "Low ceiling height for oppressive atmosphere"
            },
            "shell_sides": {
                "type": "int",
                "default": 4,
                "min": 3,
                "max": 16,
                "label": "Shell Sides",
                "description": "Number of sides for room shape (4=square, 6=hex, 8=octagon)"
            },
            "cell_count": {
                "type": "int", "default": 4, "min": 2, "max": 8, "label": "Cell Count",
                "description": "Total cell alcoves (split between left and right walls)"
            },
            "cell_depth": {
                "type": "float", "default": 64.0, "min": 48, "max": 96, "label": "Cell Depth",
                "description": "How deep the cell alcoves extend into the walls"
            },
            "central_drain": {
                "type": "bool", "default": True, "label": "Central Drain",
                "description": "Add floor drain grate in the center"
            },
            "bars": {
                "type": "bool", "default": True, "label": "Cell Bars",
                "description": "Add bar geometry across cell openings"
            },
            "has_entrance": {
                "type": "bool", "default": True, "label": "Enable Entrance Portal",
                "description": "Enable entrance portal at the front"
            },
            "random_seed": {
                "type": "int", "default": 0, "min": 0, "max": 999999, "label": "Random Seed (0=random)",
                "description": "Seed for reproducible variation (0 = random each time)"
            },
        }

    def generate(self) -> List[Brush]:
        self._reset_tags()  # Reset tags for fresh generation
        ox, oy, oz = self.params.origin
        brushes: List[Brush] = []
        t = self.WALL_THICKNESS

        if self.random_seed == 0:
            rng = random.Random()
        else:
            rng = random.Random(self.random_seed)

        # Structural dimensions - MUST be exact for portal alignment
        hw = self.width
        nl = self.length
        nh = self.height

        # Use polygonal shell if sides != 4
        if self.shell_sides != 4:
            brushes = self._generate_polygonal_prison(ox, oy, oz, hw, nl, nh, t)
            # Register portal tag at footprint edge (grid-aligned)
            if self.has_entrance:
                self._register_portal_tag(
                    portal_id="entrance",
                    center_x=ox,  # Room center X for grid alignment
                    center_y=oy,  # Footprint south edge (Y=origin)
                    center_z=oz,
                    direction=PortalDirection.SOUTH,
                )
            return brushes

        cell_depth = self.cell_depth

        # Floor - extends into cell alcoves
        brushes.append(self._box(
            ox - hw - t - cell_depth, oy - t, oz - t,
            ox + hw + t + cell_depth, oy + nl + t, oz
        ))

        # Ceiling
        brushes.append(self._box(
            ox - hw - t - cell_depth, oy - t, oz + nh,
            ox + hw + t + cell_depth, oy + nl + t, oz + nh + t
        ))

        # Generate cell walls
        brushes.extend(self._generate_cell_walls(ox, oy, oz, hw, nl, nh, cell_depth, rng))

        # Front wall with optional entrance - using unified portal system
        entrance_x = ox + self._entrance_x_offset  # Apply portal offset
        portal_spec = PortalSpec(enabled=self.has_entrance)
        wall_brushes, _ = generate_portal_wall(
            box_func=self._box,
            x1=ox - hw - t - cell_depth, y1=oy - t, z1=oz,
            x2=ox + hw + t + cell_depth, y2=oy, z2=oz + nh,
            portal_spec=portal_spec,
            portal_axis="x",
            portal_center=entrance_x,
        )
        brushes.extend(wall_brushes)

        # Back wall
        brushes.append(self._box(ox - hw - t - cell_depth, oy + nl, oz, ox + hw + t + cell_depth, oy + nl + t, oz + nh))

        # Central drain
        if self.central_drain:
            drain_size = 24.0
            drain_depth = 8.0
            drain_y = oy + nl / 2
            # Recessed drain
            brushes.append(self._box(
                ox - drain_size/2, drain_y - drain_size/2, oz - drain_depth,
                ox + drain_size/2, drain_y + drain_size/2, oz - t
            ))
            # Grate bars - minimum 8 units per CLAUDE.md §6.1
            bar_w = 8.0
            for i in range(3):
                bar_x = ox - drain_size/2 + (i + 1) * (drain_size / 4)
                brushes.append(self._box(
                    bar_x - bar_w/2, drain_y - drain_size/2, oz - t,
                    bar_x + bar_w/2, drain_y + drain_size/2, oz
                ))

        # Register portal tag for rectangular prison
        if self.has_entrance:
            entrance_x = ox + self._entrance_x_offset
            self._register_portal_tag(
                portal_id="entrance",
                center_x=entrance_x,
                center_y=oy,
                center_z=oz,
                direction=PortalDirection.SOUTH,
            )

        return brushes

    def _generate_cell_walls(
        self,
        ox: float, oy: float, oz: float,
        hw: float, nl: float, nh: float,
        cell_depth: float,
        rng: random.Random
    ) -> List[Brush]:
        """Generate side walls with cell alcoves and optional bars."""
        brushes: List[Brush] = []
        t = self.WALL_THICKNESS

        cell_w = 56.0  # Width of cell opening
        cell_h = min(72.0, nh - 8)  # Cell height
        pilaster_w = 24.0

        cells_per_side = self.cell_count // 2
        start_y = oy + 48
        end_y = oy + nl - 48
        spacing = cell_w + pilaster_w

        for side in [-1, 1]:
            if side == -1:
                wall_inner = ox - hw
                wall_outer = ox - hw - t
                cell_back = ox - hw - cell_depth
            else:
                wall_inner = ox + hw
                wall_outer = ox + hw + t
                cell_back = ox + hw + cell_depth

            # Front section
            if side == -1:
                brushes.append(self._box(wall_outer, oy, oz, wall_inner, start_y, oz + nh))
            else:
                brushes.append(self._box(wall_inner, oy, oz, wall_outer, start_y, oz + nh))

            for i in range(cells_per_side):
                cell_y1 = start_y + i * spacing
                cell_y2 = cell_y1 + cell_w

                if cell_y2 > end_y:
                    if side == -1:
                        brushes.append(self._box(wall_outer, cell_y1, oz, wall_inner, oy + nl, oz + nh))
                    else:
                        brushes.append(self._box(wall_inner, cell_y1, oz, wall_outer, oy + nl, oz + nh))
                    break

                # Lintel above cell
                if side == -1:
                    brushes.append(self._box(wall_outer, cell_y1, oz + cell_h, wall_inner, cell_y2, oz + nh))
                else:
                    brushes.append(self._box(wall_inner, cell_y1, oz + cell_h, wall_outer, cell_y2, oz + nh))

                # Cell back wall
                if side == -1:
                    brushes.append(self._box(cell_back - t, cell_y1 - t, oz, cell_back, cell_y2 + t, oz + cell_h))
                else:
                    brushes.append(self._box(cell_back, cell_y1 - t, oz, cell_back + t, cell_y2 + t, oz + cell_h))

                # Cell side walls
                if side == -1:
                    brushes.append(self._box(cell_back, cell_y1 - t, oz, wall_inner, cell_y1, oz + cell_h))
                    brushes.append(self._box(cell_back, cell_y2, oz, wall_inner, cell_y2 + t, oz + cell_h))
                else:
                    brushes.append(self._box(wall_inner, cell_y1 - t, oz, cell_back, cell_y1, oz + cell_h))
                    brushes.append(self._box(wall_inner, cell_y2, oz, cell_back, cell_y2 + t, oz + cell_h))

                # Cell ceiling
                if side == -1:
                    brushes.append(self._box(cell_back, cell_y1, oz + cell_h, wall_inner, cell_y2, oz + cell_h + t))
                else:
                    brushes.append(self._box(wall_inner, cell_y1, oz + cell_h, cell_back, cell_y2, oz + cell_h + t))

                # Cell bars - minimum 8 units per CLAUDE.md §6.1
                if self.bars:
                    bar_w = 8.0
                    bar_spacing = cell_w / 4
                    for b in range(3):
                        bar_y = cell_y1 + (b + 1) * bar_spacing
                        if side == -1:
                            brushes.append(self._box(
                                wall_inner - t, bar_y - bar_w/2, oz,
                                wall_inner, bar_y + bar_w/2, oz + cell_h
                            ))
                        else:
                            brushes.append(self._box(
                                wall_inner, bar_y - bar_w/2, oz,
                                wall_inner + t, bar_y + bar_w/2, oz + cell_h
                            ))

                # Pilaster after cell
                if i < cells_per_side - 1:
                    if side == -1:
                        brushes.append(self._box(wall_outer, cell_y2, oz, wall_inner, cell_y2 + pilaster_w, oz + nh))
                    else:
                        brushes.append(self._box(wall_inner, cell_y2, oz, wall_outer, cell_y2 + pilaster_w, oz + nh))

            # Back section
            last_y = start_y + cells_per_side * spacing - pilaster_w
            if last_y < oy + nl:
                if side == -1:
                    brushes.append(self._box(wall_outer, last_y, oz, wall_inner, oy + nl, oz + nh))
                else:
                    brushes.append(self._box(wall_inner, last_y, oz, wall_outer, oy + nl, oz + nh))

        return brushes

    def _generate_polygonal_prison(
        self, ox: float, oy: float, oz: float,
        hw: float, nl: float, nh: float, t: float
    ) -> List[Brush]:
        """Generate polygonal Prison shell (simplified for non-rectangular)."""
        brushes: List[Brush] = []

        # Calculate radius from room dimensions, clamped to fit footprint
        # The polygon must fit within both dimensions: hw (half-width) and nl/2 (half-length)
        # Using min() ensures non-square footprints don't overflow in the short dimension
        max_radius = min(hw, nl / 2)
        radius = max(max_radius, self._get_polygon_min_radius(self.shell_sides))

        # Center the polygon at the room center
        cx = ox
        cy = oy + nl / 2

        # Determine portal segment
        portal_segment = self._find_portal_segment(self.shell_sides, "SOUTH") if self.has_entrance else -1

        # Compute vestibule clip zone to avoid floor/ceiling overlap
        vestibule_clip = self._compute_vestibule_clip_zone(
            cx, cy, radius, self.shell_sides, portal_segment,
            PORTAL_WIDTH, t,
            room_origin=(ox, oy, oz), room_length=nl, room_width=hw * 2
        )

        # Generate floor (with clip zone to avoid vestibule overlap)
        brushes.extend(self._generate_polygonal_floor(
            cx, cy, oz, radius, self.shell_sides, t,
            is_ceiling=False,
            vestibule_clip_zone=vestibule_clip
        ))

        # Generate ceiling (with clip zone to avoid vestibule overlap)
        brushes.extend(self._generate_polygonal_floor(
            cx, cy, oz + nh, radius, self.shell_sides, t,
            is_ceiling=True,
            vestibule_clip_zone=vestibule_clip
        ))

        # Set portal target for grid alignment before generating walls
        self._set_portal_target_for_polygonal(cx, cy, radius, self.shell_sides, portal_segment)

        # Generate walls
        brushes.extend(self._generate_polygonal_walls(
            cx, cy, oz, oz + nh, radius, self.shell_sides, t,
            portal_segment=portal_segment,
            portal_width=PORTAL_WIDTH,
            portal_height=PORTAL_HEIGHT,
            portal_target_x=getattr(self, '_portal_target_x', None),
            portal_target_y=getattr(self, '_portal_target_y', None)
        ))

        # Generate connector corridor from portal to grid center
        # Pass room origin and dimensions for accurate footprint edge calculation
        if self.has_entrance:
            brushes.extend(self._generate_polygonal_portal_connector(
                cx, cy, oz, radius, self.shell_sides, portal_segment,
                PORTAL_WIDTH, PORTAL_HEIGHT, t, room_height=nh,
                room_origin=(ox, oy, oz), room_length=nl, room_width=hw * 2
            ))

        return brushes


class Armory(GeometricPrimitive):
    """Weapon and armor storage room.

    Features:
    - Moderate ceiling height
    - Optional wall weapon racks
    - Optional freestanding armor display stands
    - Optional central work/display table
    - 100% sealed geometry
    """

    width: float = 160.0        # Half-width
    length: float = 224.0       # Length
    height: float = 128.0       # Ceiling height
    shell_sides: int = 4        # Number of sides (4=square, 6=hex, 8=octagon)
    rack_count: int = 4         # Wall weapon racks (0-8)
    armor_stands: int = 2       # Armor displays (0-4)
    central_table: bool = True  # Central work table
    random_seed: int = 0

    # Portal control - set by layout generator based on connections
    has_entrance: bool = True   # Front (SOUTH) entrance

    WALL_THICKNESS: float = 16.0

    @classmethod
    def get_display_name(cls) -> str:
        return "Armory"

    @classmethod
    def get_category(cls) -> str:
        return "Rooms"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "width": {
                "type": "float", "default": 160.0, "min": 96, "max": 256, "label": "Half-Width",
                "description": "Half-width of the armory"
            },
            "length": {
                "type": "float", "default": 224.0, "min": 160, "max": 384, "label": "Length",
                "description": "Length from entrance to back wall"
            },
            "height": {
                "type": "float", "default": 128.0, "min": 96, "max": 192, "label": "Height",
                "description": "Ceiling height of the armory"
            },
            "shell_sides": {
                "type": "int",
                "default": 4,
                "min": 3,
                "max": 16,
                "label": "Shell Sides",
                "description": "Number of sides for room shape (4=square, 6=hex, 8=octagon)"
            },
            "rack_count": {
                "type": "int", "default": 4, "min": 0, "max": 8, "label": "Weapon Racks",
                "description": "Number of wall-mounted weapon racks"
            },
            "armor_stands": {
                "type": "int", "default": 2, "min": 0, "max": 4, "label": "Armor Stands",
                "description": "Number of freestanding armor display stands"
            },
            "central_table": {
                "type": "bool", "default": True, "label": "Central Table",
                "description": "Add central work/display table"
            },
            "has_entrance": {
                "type": "bool", "default": True, "label": "Enable Entrance Portal",
                "description": "Enable entrance portal at the front"
            },
            "random_seed": {
                "type": "int", "default": 0, "min": 0, "max": 999999, "label": "Random Seed (0=random)",
                "description": "Seed for reproducible variation (0 = random each time)"
            },
        }

    def generate(self) -> List[Brush]:
        self._reset_tags()  # Reset tags for fresh generation
        ox, oy, oz = self.params.origin
        brushes: List[Brush] = []
        t = self.WALL_THICKNESS

        if self.random_seed == 0:
            rng = random.Random()
        else:
            rng = random.Random(self.random_seed)

        # Structural dimensions - MUST be exact for portal alignment
        hw = self.width
        nl = self.length
        nh = self.height

        # Use polygonal shell if sides != 4
        if self.shell_sides != 4:
            brushes = self._generate_polygonal_armory(ox, oy, oz, hw, nl, nh, t)
            # Register portal tag at footprint edge (grid-aligned)
            if self.has_entrance:
                self._register_portal_tag(
                    portal_id="entrance",
                    center_x=ox,  # Room center X for grid alignment
                    center_y=oy,  # Footprint south edge (Y=origin)
                    center_z=oz,
                    direction=PortalDirection.SOUTH,
                )
            return brushes

        # Floor
        brushes.append(self._box(ox - hw - t, oy - t, oz - t, ox + hw + t, oy + nl + t, oz))

        # Ceiling
        brushes.append(self._box(ox - hw - t, oy - t, oz + nh, ox + hw + t, oy + nl + t, oz + nh + t))

        # Side walls
        brushes.append(self._box(ox - hw - t, oy - t, oz, ox - hw, oy + nl + t, oz + nh))
        brushes.append(self._box(ox + hw, oy - t, oz, ox + hw + t, oy + nl + t, oz + nh))

        # Front wall with optional entrance - using unified portal system
        portal_spec = PortalSpec(enabled=self.has_entrance)
        wall_brushes, _ = generate_portal_wall(
            box_func=self._box,
            x1=ox - hw - t, y1=oy - t, z1=oz,
            x2=ox + hw + t, y2=oy, z2=oz + nh,
            portal_spec=portal_spec,
            portal_axis="x",
            portal_center=ox,
        )
        brushes.extend(wall_brushes)

        # Back wall
        brushes.append(self._box(ox - hw - t, oy + nl, oz, ox + hw + t, oy + nl + t, oz + nh))

        # Weapon racks on walls
        if self.rack_count > 0:
            rack_w = 32.0
            rack_h = 48.0
            rack_d = 8.0
            racks_per_side = self.rack_count // 2
            rack_spacing = (nl - 80) / (racks_per_side + 1)
            for side in [-1, 1]:
                for i in range(racks_per_side):
                    rack_y = oy + 40 + (i + 1) * rack_spacing
                    if side == -1:
                        rack_x = ox - hw + 2
                    else:
                        rack_x = ox + hw - rack_d - 2
                    # Rack backing
                    brushes.append(self._box(
                        rack_x, rack_y - rack_w/2, oz + 32,
                        rack_x + rack_d, rack_y + rack_w/2, oz + 32 + rack_h
                    ))
                    # Shelf/hooks
                    brushes.append(self._box(
                        rack_x, rack_y - rack_w/2, oz + 48,
                        rack_x + rack_d + 4, rack_y + rack_w/2, oz + 56
                    ))

        # Armor stands
        if self.armor_stands > 0:
            stand_size = 24.0
            stand_h = 64.0
            ss2 = stand_size / 2
            stand_spacing = hw / (self.armor_stands + 1)
            for i in range(self.armor_stands):
                sx = ox - hw + (i + 1) * stand_spacing * 2
                sy = oy + nl - 64
                # Base
                brushes.append(self._box(sx - ss2, sy - ss2, oz, sx + ss2, sy + ss2, oz + 8))
                # Stand post
                brushes.append(self._box(sx - 8, sy - 8, oz + 8, sx + 8, sy + 8, oz + stand_h))

        # Central table
        if self.central_table:
            table_w = 48.0
            table_l = 64.0
            table_h = 28.0
            brushes.append(self._box(
                ox - table_w/2, oy + nl/2 - table_l/2, oz,
                ox + table_w/2, oy + nl/2 + table_l/2, oz + table_h
            ))

        # Register portal tag for rectangular armory
        if self.has_entrance:
            self._register_portal_tag(
                portal_id="entrance",
                center_x=ox,
                center_y=oy,
                center_z=oz,
                direction=PortalDirection.SOUTH,
            )

        return brushes

    def _generate_polygonal_armory(
        self, ox: float, oy: float, oz: float,
        hw: float, nl: float, nh: float, t: float
    ) -> List[Brush]:
        """Generate polygonal Armory shell (simplified for non-rectangular)."""
        brushes: List[Brush] = []

        # Calculate radius from room dimensions, clamped to fit footprint
        # The polygon must fit within both dimensions: hw (half-width) and nl/2 (half-length)
        # Using min() ensures non-square footprints don't overflow in the short dimension
        max_radius = min(hw, nl / 2)
        radius = max(max_radius, self._get_polygon_min_radius(self.shell_sides))

        # Center the polygon at the room center
        cx = ox
        cy = oy + nl / 2

        # Determine portal segment
        portal_segment = self._find_portal_segment(self.shell_sides, "SOUTH") if self.has_entrance else -1

        # Compute vestibule clip zone to avoid floor/ceiling overlap
        vestibule_clip = self._compute_vestibule_clip_zone(
            cx, cy, radius, self.shell_sides, portal_segment,
            PORTAL_WIDTH, t,
            room_origin=(ox, oy, oz), room_length=nl, room_width=hw * 2
        )

        # Generate floor (with clip zone to avoid vestibule overlap)
        brushes.extend(self._generate_polygonal_floor(
            cx, cy, oz, radius, self.shell_sides, t,
            is_ceiling=False,
            vestibule_clip_zone=vestibule_clip
        ))

        # Generate ceiling (with clip zone to avoid vestibule overlap)
        brushes.extend(self._generate_polygonal_floor(
            cx, cy, oz + nh, radius, self.shell_sides, t,
            is_ceiling=True,
            vestibule_clip_zone=vestibule_clip
        ))

        # Set portal target for grid alignment before generating walls
        self._set_portal_target_for_polygonal(cx, cy, radius, self.shell_sides, portal_segment)

        # Generate walls
        brushes.extend(self._generate_polygonal_walls(
            cx, cy, oz, oz + nh, radius, self.shell_sides, t,
            portal_segment=portal_segment,
            portal_width=PORTAL_WIDTH,
            portal_height=PORTAL_HEIGHT,
            portal_target_x=getattr(self, '_portal_target_x', None),
            portal_target_y=getattr(self, '_portal_target_y', None)
        ))

        # Generate connector corridor from portal to grid center
        # Pass room origin and dimensions for accurate footprint edge calculation
        if self.has_entrance:
            brushes.extend(self._generate_polygonal_portal_connector(
                cx, cy, oz, radius, self.shell_sides, portal_segment,
                PORTAL_WIDTH, PORTAL_HEIGHT, t, room_height=nh,
                room_origin=(ox, oy, oz), room_length=nl, room_width=hw * 2
            ))

        return brushes


class Cistern(GeometricPrimitive):
    """Partially flooded water storage chamber.

    Cisterns store water for castle/fortress use. The central area is
    sunken (representing water), with a raised walkway around the perimeter.

    Features:
    - Sunken central area (water depth visual)
    - Raised perimeter walkway
    - Support pillars in water area
    - 100% sealed geometry
    """

    width: float = 192.0        # Half-width
    length: float = 256.0       # Length
    height: float = 128.0       # Ceiling height
    shell_sides: int = 4        # Number of sides (4=square, 6=hex, 8=octagon)
    water_depth: float = 32.0   # Sunken floor depth
    pillar_count: int = 4       # Support pillars in water
    walkway: bool = True        # Raised perimeter walkway
    walkway_width: float = 48.0 # Width of walkway
    random_seed: int = 0

    # Pillar customization
    pillar_style: str = "square"  # square, hexagonal, octagonal, round
    pillar_capital: bool = False  # Add capitals to pillars

    # Portal control - set by layout generator based on connections
    has_entrance: bool = True   # Front (SOUTH) entrance

    WALL_THICKNESS: float = 16.0

    @classmethod
    def get_display_name(cls) -> str:
        return "Cistern"

    @classmethod
    def get_category(cls) -> str:
        return "Rooms"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "width": {
                "type": "float", "default": 192.0, "min": 128, "max": 320, "label": "Half-Width",
                "description": "Half-width of the cistern chamber"
            },
            "length": {
                "type": "float", "default": 256.0, "min": 192, "max": 512, "label": "Length",
                "description": "Length from entrance to back wall"
            },
            "height": {
                "type": "float", "default": 128.0, "min": 96, "max": 192, "label": "Height",
                "description": "Ceiling height above the walkway level"
            },
            "shell_sides": {
                "type": "int",
                "default": 4,
                "min": 3,
                "max": 16,
                "label": "Shell Sides",
                "description": "Number of sides for room shape (4=square, 6=hex, 8=octagon)"
            },
            "water_depth": {
                "type": "float", "default": 32.0, "min": 16, "max": 64, "label": "Water Depth",
                "description": "How deep the water pool is below walkway level"
            },
            "pillar_count": {
                "type": "int", "default": 4, "min": 0, "max": 8, "label": "Pillar Count",
                "description": "Number of support pillars standing in the water"
            },
            "pillar_style": {
                "type": "choice",
                "default": "square",
                "choices": ["square", "hexagonal", "octagonal", "round"],
                "label": "Pillar Style",
                "description": "Cross-section shape of support pillars"
            },
            "pillar_capital": {
                "type": "bool", "default": False, "label": "Pillar Capitals",
                "description": "Add decorative capitals atop pillars"
            },
            "walkway": {
                "type": "bool", "default": True, "label": "Perimeter Walkway",
                "description": "Add raised walkway around the water pool edge"
            },
            "walkway_width": {
                "type": "float", "default": 48.0, "min": 32, "max": 64, "label": "Walkway Width",
                "description": "Width of the perimeter walkway (player is 32 units wide)"
            },
            "has_entrance": {
                "type": "bool", "default": True, "label": "Enable Entrance Portal",
                "description": "Enable entrance portal at the front"
            },
            "random_seed": {
                "type": "int", "default": 0, "min": 0, "max": 999999, "label": "Random Seed (0=random)",
                "description": "Seed for reproducible variation (0 = random each time)"
            },
        }

    def generate(self) -> List[Brush]:
        self._reset_tags()  # Reset tags for fresh generation
        ox, oy, oz = self.params.origin
        brushes: List[Brush] = []
        t = self.WALL_THICKNESS

        if self.random_seed == 0:
            rng = random.Random()
        else:
            rng = random.Random(self.random_seed)

        # Structural dimensions - MUST be exact for portal alignment
        hw = self.width
        nl = self.length
        nh = self.height

        # Use polygonal shell if sides != 4
        if self.shell_sides != 4:
            brushes = self._generate_polygonal_cistern(ox, oy, oz, hw, nl, nh, t)
            # Register portal tag at footprint edge (grid-aligned)
            if self.has_entrance:
                self._register_portal_tag(
                    portal_id="entrance",
                    center_x=ox,  # Room center X for grid alignment
                    center_y=oy,  # Footprint south edge (Y=origin)
                    center_z=oz,
                    direction=PortalDirection.SOUTH,
                )
            return brushes

        ww = self.walkway_width if self.walkway else 0
        water_z = oz - self.water_depth

        # Base floor under everything (below water level)
        brushes.append(self._box(
            ox - hw - t, oy - t, water_z - t,
            ox + hw + t, oy + nl + t, water_z
        ))

        # Ceiling
        brushes.append(self._box(
            ox - hw - t, oy - t, oz + nh,
            ox + hw + t, oy + nl + t, oz + nh + t
        ))

        # Side walls (from base to ceiling)
        brushes.append(self._box(ox - hw - t, oy - t, water_z, ox - hw, oy + nl + t, oz + nh))
        brushes.append(self._box(ox + hw, oy - t, water_z, ox + hw + t, oy + nl + t, oz + nh))

        # Front wall with optional entrance - using unified portal system
        # Note: Cistern front wall starts at water_z, but portal starts at oz (walkway level)
        portal_spec = PortalSpec(enabled=self.has_entrance)
        # Lower section below portal (water_z to oz)
        brushes.append(self._box(ox - hw - t, oy - t, water_z, ox + hw + t, oy, oz))
        # Upper section with portal (oz to ceiling)
        wall_brushes, _ = generate_portal_wall(
            box_func=self._box,
            x1=ox - hw - t, y1=oy - t, z1=oz,
            x2=ox + hw + t, y2=oy, z2=oz + nh,
            portal_spec=portal_spec,
            portal_axis="x",
            portal_center=ox,
        )
        brushes.extend(wall_brushes)

        # Back wall
        brushes.append(self._box(ox - hw - t, oy + nl, water_z, ox + hw + t, oy + nl + t, oz + nh))

        # Walkway and pool walls
        if self.walkway:
            # Raised walkway floor around perimeter
            # Left walkway
            brushes.append(self._box(ox - hw, oy, oz - t, ox - hw + ww, oy + nl, oz))
            # Right walkway
            brushes.append(self._box(ox + hw - ww, oy, oz - t, ox + hw, oy + nl, oz))
            # Front walkway
            brushes.append(self._box(ox - hw + ww, oy, oz - t, ox + hw - ww, oy + ww, oz))
            # Back walkway
            brushes.append(self._box(ox - hw + ww, oy + nl - ww, oz - t, ox + hw - ww, oy + nl, oz))

            # Pool retaining walls (inner edge of walkway down to water level)
            # Left pool wall
            brushes.append(self._box(ox - hw + ww - t, oy + ww, water_z, ox - hw + ww, oy + nl - ww, oz))
            # Right pool wall
            brushes.append(self._box(ox + hw - ww, oy + ww, water_z, ox + hw - ww + t, oy + nl - ww, oz))
            # Front pool wall
            brushes.append(self._box(ox - hw + ww, oy + ww - t, water_z, ox + hw - ww, oy + ww, oz))
            # Back pool wall
            brushes.append(self._box(ox - hw + ww, oy + nl - ww, water_z, ox + hw - ww, oy + nl - ww + t, oz))
        else:
            # No walkway - just fill above floor to origin level
            brushes.append(self._box(ox - hw, oy, oz - t, ox + hw, oy + nl, oz))

        # Support pillars in water area
        if self.pillar_count > 0:
            pillar_size = 24.0
            pool_hw = hw - ww if self.walkway else hw
            pool_nl = nl - 2 * ww if self.walkway else nl
            pool_start_y = oy + ww if self.walkway else oy

            # Grid layout for pillars
            cols = max(1, int(math.sqrt(self.pillar_count)))
            rows = max(1, (self.pillar_count + cols - 1) // cols)
            col_spacing = (pool_hw * 2 - 48) / max(1, cols + 1)
            row_spacing = (pool_nl - 48) / max(1, rows + 1)

            for r in range(rows):
                for c in range(cols):
                    if r * cols + c >= self.pillar_count:
                        break
                    px = ox - pool_hw + 24 + (c + 1) * col_spacing
                    py = pool_start_y + 24 + (r + 1) * row_spacing
                    brushes.extend(self._generate_room_pillar(
                        px, py, water_z, oz + nh,
                        pillar_size, self.pillar_style, self.pillar_capital
                    ))

        # Register portal tag for rectangular cistern
        if self.has_entrance:
            self._register_portal_tag(
                portal_id="entrance",
                center_x=ox,
                center_y=oy,
                center_z=oz,
                direction=PortalDirection.SOUTH,
            )

        return brushes

    def _generate_polygonal_cistern(
        self, ox: float, oy: float, oz: float,
        hw: float, nl: float, nh: float, t: float
    ) -> List[Brush]:
        """Generate polygonal Cistern shell with support pillars."""
        brushes: List[Brush] = []

        # Calculate radius from room dimensions, clamped to fit footprint
        # The polygon must fit within both dimensions: hw (half-width) and nl/2 (half-length)
        # Using min() ensures non-square footprints don't overflow in the short dimension
        max_radius = min(hw, nl / 2)
        radius = max(max_radius, self._get_polygon_min_radius(self.shell_sides))

        # Center the polygon at the room center
        cx = ox
        cy = oy + nl / 2

        # Determine portal segment
        portal_segment = self._find_portal_segment(self.shell_sides, "SOUTH") if self.has_entrance else -1

        # Compute vestibule clip zone to avoid floor/ceiling overlap
        vestibule_clip = self._compute_vestibule_clip_zone(
            cx, cy, radius, self.shell_sides, portal_segment,
            PORTAL_WIDTH, t,
            room_origin=(ox, oy, oz), room_length=nl, room_width=hw * 2
        )

        # Generate floor (with clip zone to avoid vestibule overlap)
        brushes.extend(self._generate_polygonal_floor(
            cx, cy, oz, radius, self.shell_sides, t,
            is_ceiling=False,
            vestibule_clip_zone=vestibule_clip
        ))

        # Generate ceiling (with clip zone to avoid vestibule overlap)
        brushes.extend(self._generate_polygonal_floor(
            cx, cy, oz + nh, radius, self.shell_sides, t,
            is_ceiling=True,
            vestibule_clip_zone=vestibule_clip
        ))

        # Set portal target for grid alignment before generating walls
        self._set_portal_target_for_polygonal(cx, cy, radius, self.shell_sides, portal_segment)

        # Generate walls
        brushes.extend(self._generate_polygonal_walls(
            cx, cy, oz, oz + nh, radius, self.shell_sides, t,
            portal_segment=portal_segment,
            portal_width=PORTAL_WIDTH,
            portal_height=PORTAL_HEIGHT,
            portal_target_x=getattr(self, '_portal_target_x', None),
            portal_target_y=getattr(self, '_portal_target_y', None)
        ))

        # Generate connector corridor from portal to grid center
        # Pass room origin and dimensions for accurate footprint edge calculation
        if self.has_entrance:
            brushes.extend(self._generate_polygonal_portal_connector(
                cx, cy, oz, radius, self.shell_sides, portal_segment,
                PORTAL_WIDTH, PORTAL_HEIGHT, t, room_height=nh,
                room_origin=(ox, oy, oz), room_length=nl, room_width=hw * 2
            ))

        # Add support pillars in the center area
        if self.pillar_count > 0:
            pillar_size = 24.0
            pillar_radius = radius * 0.5  # Pillars at 50% radius (central area)

            for i in range(self.pillar_count):
                angle = 2 * math.pi * i / self.pillar_count
                # Offset so pillars don't block entrance
                angle += math.pi / self.pillar_count

                px = cx + pillar_radius * math.sin(angle)
                py = cy - pillar_radius * math.cos(angle)
                brushes.extend(self._generate_room_pillar(
                    px, py, oz, oz + nh,
                    pillar_size, self.pillar_style, self.pillar_capital
                ))

        return brushes


class Stronghold(GeometricPrimitive):
    """Central fortified stronghold structure (multi-level tower).

    A multi-story fortified tower with thick walls.
    Works for castles, bunkers, compounds.

    Features:
    - Multiple floors with interior stairs
    - Extra thick walls for defense
    - Optional arrow slits per level
    - Optional battlements on roof
    - 100% sealed geometry
    """

    width: float = 192.0        # Half-width
    length: float = 256.0       # Length
    levels: int = 3             # Number of floors (2-5)
    level_height: float = 128.0 # Height per level
    shell_sides: int = 4        # Number of sides (4=square, 6=hex, 8=octagon)
    wall_thickness: float = 32.0  # Extra thick walls
    stair_type: str = "straight"  # straight only (spiral removed)
    battlements: bool = True    # Crenellated roof
    arrow_slits: int = 2        # Slits per wall per level (0-4)
    random_seed: int = 0

    # Portal control - set by layout generator based on connections
    has_entrance: bool = True   # Front (SOUTH) entrance

    @classmethod
    def get_display_name(cls) -> str:
        return "Stronghold"

    @classmethod
    def get_category(cls) -> str:
        return "Rooms"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "width": {
                "type": "float", "default": 192.0, "min": 128, "max": 320, "label": "Half-Width",
                "description": "Half-width of the stronghold base"
            },
            "length": {
                "type": "float", "default": 256.0, "min": 192, "max": 512, "label": "Length",
                "description": "Length from entrance to back wall"
            },
            "levels": {
                "type": "int", "default": 3, "min": 2, "max": 5, "label": "Levels",
                "description": "Number of interior floors"
            },
            "level_height": {
                "type": "float", "default": 128.0, "min": 96, "max": 192, "label": "Level Height",
                "description": "Ceiling height of each floor level"
            },
            "shell_sides": {
                "type": "int",
                "default": 4,
                "min": 3,
                "max": 16,
                "label": "Shell Sides",
                "description": "Number of sides for room shape (4=square, 6=hex, 8=octagon)"
            },
            "wall_thickness": {
                "type": "float", "default": 32.0, "min": 16, "max": 48, "label": "Wall Thickness",
                "description": "Extra thick defensive wall thickness"
            },
            "stair_type": {
                "type": "choice",
                "default": "straight",
                "choices": ["straight"],
                "label": "Stair Type",
                "description": "Type of interior staircase between levels"
            },
            "battlements": {
                "type": "bool", "default": True, "label": "Battlements",
                "description": "Add crenellated battlements on the roof"
            },
            "arrow_slits": {
                "type": "int", "default": 2, "min": 0, "max": 4, "label": "Arrow Slits/Wall/Level",
                "description": "Number of defensive arrow slits per wall per level"
            },
            "has_entrance": {
                "type": "bool", "default": True, "label": "Enable Entrance Portal",
                "description": "Enable ground floor entrance portal"
            },
            "random_seed": {
                "type": "int", "default": 0, "min": 0, "max": 999999, "label": "Random Seed (0=random)",
                "description": "Seed for reproducible variation (0 = random each time)"
            },
        }

    def generate(self) -> List[Brush]:
        self._reset_tags()  # Reset tags for fresh generation
        ox, oy, oz = self.params.origin
        brushes: List[Brush] = []

        if self.random_seed == 0:
            rng = random.Random()
        else:
            rng = random.Random(self.random_seed)

        # Structural dimensions - MUST be exact for portal alignment
        hw = self.width
        nl = self.length
        lh = self.level_height
        t = self.wall_thickness

        total_h = self.levels * lh

        # Use polygonal shell if sides != 4
        if self.shell_sides != 4:
            brushes = self._generate_polygonal_stronghold(ox, oy, oz, hw, nl, total_h, t)
            # Register portal tag at footprint edge (grid-aligned)
            if self.has_entrance:
                self._register_portal_tag(
                    portal_id="entrance",
                    center_x=ox,  # Room center X for grid alignment
                    center_y=oy,  # Footprint south edge (Y=origin)
                    center_z=oz,
                    direction=PortalDirection.SOUTH,
                )
            return brushes

        # Base floor
        brushes.append(self._box(ox - hw - t, oy - t, oz - t, ox + hw + t, oy + nl + t, oz))

        # Exterior walls (full height, thick)
        # Left wall
        brushes.append(self._box(ox - hw - t, oy - t, oz, ox - hw, oy + nl + t, oz + total_h))
        # Right wall
        brushes.append(self._box(ox + hw, oy - t, oz, ox + hw + t, oy + nl + t, oz + total_h))
        # Front wall with optional entrance on ground floor - using unified portal system
        portal_spec = PortalSpec(enabled=self.has_entrance)
        wall_brushes, _ = generate_portal_wall(
            box_func=self._box,
            x1=ox - hw - t, y1=oy - t, z1=oz,
            x2=ox + hw + t, y2=oy, z2=oz + total_h,
            portal_spec=portal_spec,
            portal_axis="x",
            portal_center=ox,
        )
        brushes.extend(wall_brushes)
        # Back wall
        brushes.append(self._box(ox - hw - t, oy + nl, oz, ox + hw + t, oy + nl + t, oz + total_h))

        # Interior floors and stairwell
        stair_size = 64.0  # Stair footprint
        stair_corner_x = ox + hw - stair_size - 16
        stair_corner_y = oy + nl - stair_size - 16

        for level in range(self.levels):
            floor_z = oz + level * lh

            if level > 0:
                # Floor with hole for stairwell
                # Main floor section
                brushes.append(self._box(
                    ox - hw, oy, floor_z - 8,
                    stair_corner_x, oy + nl, floor_z
                ))
                # Section past stairwell
                brushes.append(self._box(
                    stair_corner_x, oy, floor_z - 8,
                    ox + hw, stair_corner_y, floor_z
                ))
                # Section behind stairwell
                brushes.append(self._box(
                    stair_corner_x + stair_size, stair_corner_y, floor_z - 8,
                    ox + hw, oy + nl, floor_z
                ))

            # Stairs to next level (except on top floor)
            if level < self.levels - 1:
                next_floor_z = oz + (level + 1) * lh
                # Calculate step count ensuring minimum 8-unit step depth per CLAUDE.md §6.1
                max_steps_for_min_depth = int(stair_size / 8)  # Maximum steps with 8-unit depth
                step_count = min(max(1, int(lh / 12)), max_steps_for_min_depth)
                step_h = lh / step_count
                step_d = stair_size / step_count

                if self.stair_type == "spiral":
                    # Spiral staircase (approximated with quarter-turn sections)
                    center_x = stair_corner_x + stair_size / 2
                    center_y = stair_corner_y + stair_size / 2
                    post_r = 16.0
                    brushes.append(self._box(
                        center_x - post_r, center_y - post_r, floor_z,
                        center_x + post_r, center_y + post_r, next_floor_z
                    ))
                    # Spiral steps
                    inner_r = post_r + 4
                    outer_r = stair_size / 2
                    for i in range(step_count):
                        angle1 = (2 * math.pi * i) / step_count
                        angle2 = (2 * math.pi * (i + 1)) / step_count
                        step_z = floor_z + i * step_h
                        brushes.append(self._radial_segment(
                            center_x, center_y, step_z, step_z + step_h,
                            inner_r, outer_r, angle1, angle2
                        ))
                else:
                    # Straight staircase along Y
                    for i in range(step_count):
                        step_y = stair_corner_y + i * step_d
                        step_z = floor_z + i * step_h
                        brushes.append(self._box(
                            stair_corner_x, step_y, step_z,
                            stair_corner_x + stair_size, step_y + step_d, step_z + step_h
                        ))

        # Roof
        roof_z = oz + total_h
        if self.battlements:
            # Parapet base
            parapet_h = 16.0
            merlon_h = 48.0
            merlon_w = 32.0
            crenel_w = 24.0
            parapet_t = 16.0

            # Parapet walls around roof edge
            brushes.append(self._box(ox - hw - t, oy - t, roof_z, ox - hw - t + parapet_t, oy + nl + t, roof_z + parapet_h))
            brushes.append(self._box(ox + hw + t - parapet_t, oy - t, roof_z, ox + hw + t, oy + nl + t, roof_z + parapet_h))
            brushes.append(self._box(ox - hw - t, oy - t, roof_z, ox + hw + t, oy - t + parapet_t, roof_z + parapet_h))
            brushes.append(self._box(ox - hw - t, oy + nl + t - parapet_t, roof_z, ox + hw + t, oy + nl + t, roof_z + parapet_h))

            # Merlons along each side
            pattern = merlon_w + crenel_w
            # Left and right sides
            y = oy
            while y + merlon_w <= oy + nl:
                brushes.append(self._box(
                    ox - hw - t, y, roof_z + parapet_h,
                    ox - hw - t + parapet_t, y + merlon_w, roof_z + parapet_h + merlon_h
                ))
                brushes.append(self._box(
                    ox + hw + t - parapet_t, y, roof_z + parapet_h,
                    ox + hw + t, y + merlon_w, roof_z + parapet_h + merlon_h
                ))
                y += pattern
            # Front and back sides
            x = ox - hw
            while x + merlon_w <= ox + hw:
                brushes.append(self._box(
                    x, oy - t, roof_z + parapet_h,
                    x + merlon_w, oy - t + parapet_t, roof_z + parapet_h + merlon_h
                ))
                brushes.append(self._box(
                    x, oy + nl + t - parapet_t, roof_z + parapet_h,
                    x + merlon_w, oy + nl + t, roof_z + parapet_h + merlon_h
                ))
                x += pattern
        else:
            # Simple flat roof
            brushes.append(self._box(ox - hw - t, oy - t, roof_z, ox + hw + t, oy + nl + t, roof_z + 8))

        # Arrow slits (embedded in walls)
        if self.arrow_slits > 0:
            slit_w = 8.0
            slit_h = 48.0
            for level in range(self.levels):
                floor_z = oz + level * lh
                slit_z = floor_z + lh / 2 - slit_h / 2
                # Distribute slits along each wall
                slit_spacing_y = nl / (self.arrow_slits + 1)
                slit_spacing_x = (hw * 2) / (self.arrow_slits + 1)
                # Left and right walls
                for i in range(self.arrow_slits):
                    slit_y = oy + (i + 1) * slit_spacing_y
                    # Left wall slit recess (interior widening)
                    brushes.append(self._box(
                        ox - hw, slit_y - 16, slit_z,
                        ox - hw + 16, slit_y + 16, slit_z + slit_h
                    ))
                    # Right wall slit recess
                    brushes.append(self._box(
                        ox + hw - 16, slit_y - 16, slit_z,
                        ox + hw, slit_y + 16, slit_z + slit_h
                    ))

        # Register portal tag for rectangular stronghold
        if self.has_entrance:
            self._register_portal_tag(
                portal_id="entrance",
                center_x=ox,
                center_y=oy,
                center_z=oz,
                direction=PortalDirection.SOUTH,
            )

        return brushes

    def _generate_polygonal_stronghold(
        self, ox: float, oy: float, oz: float,
        hw: float, nl: float, nh: float, t: float
    ) -> List[Brush]:
        """Generate polygonal Stronghold shell (simplified for non-rectangular)."""
        brushes: List[Brush] = []

        # Calculate radius from room dimensions, clamped to fit footprint
        # The polygon must fit within both dimensions: hw (half-width) and nl/2 (half-length)
        # Using min() ensures non-square footprints don't overflow in the short dimension
        max_radius = min(hw, nl / 2)
        radius = max(max_radius, self._get_polygon_min_radius(self.shell_sides))

        # Center the polygon at the room center
        cx = ox
        cy = oy + nl / 2

        # Determine portal segment
        portal_segment = self._find_portal_segment(self.shell_sides, "SOUTH") if self.has_entrance else -1

        # Compute vestibule clip zone to avoid floor/ceiling overlap
        vestibule_clip = self._compute_vestibule_clip_zone(
            cx, cy, radius, self.shell_sides, portal_segment,
            PORTAL_WIDTH, t,
            room_origin=(ox, oy, oz), room_length=nl, room_width=hw * 2
        )

        # Generate floor (with clip zone to avoid vestibule overlap)
        brushes.extend(self._generate_polygonal_floor(
            cx, cy, oz, radius, self.shell_sides, t,
            is_ceiling=False,
            vestibule_clip_zone=vestibule_clip
        ))

        # Generate ceiling (with clip zone to avoid vestibule overlap)
        brushes.extend(self._generate_polygonal_floor(
            cx, cy, oz + nh, radius, self.shell_sides, t,
            is_ceiling=True,
            vestibule_clip_zone=vestibule_clip
        ))

        # Set portal target for grid alignment before generating walls
        self._set_portal_target_for_polygonal(cx, cy, radius, self.shell_sides, portal_segment)

        # Generate walls
        brushes.extend(self._generate_polygonal_walls(
            cx, cy, oz, oz + nh, radius, self.shell_sides, t,
            portal_segment=portal_segment,
            portal_width=PORTAL_WIDTH,
            portal_height=PORTAL_HEIGHT,
            portal_target_x=getattr(self, '_portal_target_x', None),
            portal_target_y=getattr(self, '_portal_target_y', None)
        ))

        # Generate connector corridor from portal to grid center
        # Pass room origin and dimensions for accurate footprint edge calculation
        if self.has_entrance:
            brushes.extend(self._generate_polygonal_portal_connector(
                cx, cy, oz, radius, self.shell_sides, portal_segment,
                PORTAL_WIDTH, PORTAL_HEIGHT, t, room_height=nh,
                room_origin=(ox, oy, oz), room_length=nl, room_width=hw * 2
            ))

        return brushes


class Courtyard(GeometricPrimitive):
    """Open-air courtyard with sky ceiling.

    In idTech engines, outdoor areas are rendered by applying sky textures
    (e.g., SKY1) to ceiling brushes. The geometry remains sealed per
    CLAUDE.md [BINDING], but the sky texture creates the visual illusion
    of open sky.

    Features:
    - Sky texture on ceiling creates outdoor effect
    - Optional low perimeter walls with battlements
    - Optional corner towers (buttress-style)
    - Optional central well (sunken floor section)
    - 100% sealed geometry (sky ceiling still seals)
    """

    width: float = 192.0        # Half-width
    length: float = 256.0       # Length
    height: float = 192.0       # Sky ceiling height
    shell_sides: int = 4        # Number of sides (4=square, 6=hex, 8=octagon)
    wall_height: float = 64.0   # Perimeter wall height (lower than ceiling)
    corner_towers: bool = False  # Add corner buttresses
    central_well: bool = False   # Central sunken area
    sky_texture: str = "SKY1"   # Sky texture for ceiling
    random_seed: int = 0

    # Portal control - set by layout generator based on connections
    has_entrance: bool = True

    WALL_THICKNESS: float = 16.0

    @classmethod
    def get_display_name(cls) -> str:
        return "Courtyard"

    @classmethod
    def get_category(cls) -> str:
        return "Rooms"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "width": {
                "type": "float", "default": 192.0, "min": 128, "max": 384, "label": "Half-Width",
                "description": "Half-width of the courtyard"
            },
            "length": {
                "type": "float", "default": 256.0, "min": 192, "max": 512, "label": "Length",
                "description": "Length from entrance to back wall"
            },
            "height": {
                "type": "float", "default": 192.0, "min": 128, "max": 384, "label": "Sky Height",
                "description": "Height of the sky ceiling above ground"
            },
            "shell_sides": {
                "type": "int",
                "default": 4,
                "min": 3,
                "max": 16,
                "label": "Shell Sides",
                "description": "Number of sides for courtyard shape (4=square, 6=hex, 8=octagon)"
            },
            "wall_height": {
                "type": "float", "default": 64.0, "min": 32, "max": 128, "label": "Wall Height",
                "description": "Height of perimeter walls below sky level"
            },
            "corner_towers": {
                "type": "bool", "default": False, "label": "Corner Towers",
                "description": "Add decorative corner buttress towers"
            },
            "central_well": {
                "type": "bool", "default": False, "label": "Central Well",
                "description": "Add sunken well area in the center"
            },
            "sky_texture": {
                "type": "str", "default": "SKY1", "label": "Sky Texture",
                "description": "Texture name for the sky ceiling (e.g., SKY1, SKY4)"
            },
            "has_entrance": {
                "type": "bool", "default": True, "label": "Enable Entrance Portal",
                "description": "Enable entrance portal at the front"
            },
            "random_seed": {
                "type": "int", "default": 0, "min": 0, "max": 999999, "label": "Random Seed (0=random)",
                "description": "Seed for reproducible variation (0 = random each time)"
            },
        }

    def generate(self) -> List[Brush]:
        self._reset_tags()  # Reset tags for fresh generation
        ox, oy, oz = self.params.origin
        brushes: List[Brush] = []
        t = self.WALL_THICKNESS

        if self.random_seed == 0:
            rng = random.Random()
        else:
            rng = random.Random(self.random_seed)

        # Structural dimensions - MUST be exact for portal alignment
        hw = self.width
        nl = self.length
        nh = self.height
        wh = self.wall_height  # Perimeter wall height

        # Use polygonal shell if sides != 4
        if self.shell_sides != 4:
            brushes = self._generate_polygonal_courtyard(ox, oy, oz, hw, nl, nh, t)
            # Register portal tag at footprint edge (grid-aligned)
            if self.has_entrance:
                self._register_portal_tag(
                    portal_id="entrance",
                    center_x=ox,  # Room center X for grid alignment
                    center_y=oy,  # Footprint south edge (Y=origin)
                    center_z=oz,
                    direction=PortalDirection.SOUTH,
                )
            return brushes

        # === FLOOR ===
        # Floor extends beyond all walls (sealed geometry rule)
        if self.central_well:
            # Central well creates sunken area in middle
            well_size = min(hw, nl / 2) * 0.4  # 40% of smaller dimension
            well_depth = 32.0
            well_cx = ox
            well_cy = oy + nl / 2

            # Floor around well (4 sections)
            # Front section
            brushes.append(self._box(
                ox - hw - t, oy - t, oz - t,
                ox + hw + t, well_cy - well_size, oz
            ))
            # Back section
            brushes.append(self._box(
                ox - hw - t, well_cy + well_size, oz - t,
                ox + hw + t, oy + nl + t, oz
            ))
            # Left section
            brushes.append(self._box(
                ox - hw - t, well_cy - well_size, oz - t,
                well_cx - well_size, well_cy + well_size, oz
            ))
            # Right section
            brushes.append(self._box(
                well_cx + well_size, well_cy - well_size, oz - t,
                ox + hw + t, well_cy + well_size, oz
            ))
            # Well floor (sunken)
            brushes.append(self._box(
                well_cx - well_size, well_cy - well_size, oz - well_depth - t,
                well_cx + well_size, well_cy + well_size, oz - well_depth
            ))
            # Well walls
            brushes.append(self._box(
                well_cx - well_size - t, well_cy - well_size, oz - well_depth,
                well_cx - well_size, well_cy + well_size, oz
            ))
            brushes.append(self._box(
                well_cx + well_size, well_cy - well_size, oz - well_depth,
                well_cx + well_size + t, well_cy + well_size, oz
            ))
            brushes.append(self._box(
                well_cx - well_size, well_cy - well_size - t, oz - well_depth,
                well_cx + well_size, well_cy - well_size, oz
            ))
            brushes.append(self._box(
                well_cx - well_size, well_cy + well_size, oz - well_depth,
                well_cx + well_size, well_cy + well_size + t, oz
            ))
        else:
            # Simple flat floor
            brushes.append(self._box(ox - hw - t, oy - t, oz - t, ox + hw + t, oy + nl + t, oz))

        # === SKY CEILING ===
        # Ceiling uses sky texture - creates outdoor effect in idTech
        # The texture is applied via the primitive's texture parameter system
        # The ceiling still seals the geometry (required by CLAUDE.md)
        brushes.append(self._box(
            ox - hw - t, oy - t, oz + nh,
            ox + hw + t, oy + nl + t, oz + nh + t,
            texture=self.sky_texture
        ))

        # === PERIMETER WALLS (low walls with optional crenellation) ===
        # Left wall
        brushes.append(self._box(ox - hw - t, oy - t, oz, ox - hw, oy + nl + t, oz + wh))
        # Right wall
        brushes.append(self._box(ox + hw, oy - t, oz, ox + hw + t, oy + nl + t, oz + wh))

        # Front wall with optional entrance - using unified portal system
        portal_spec = PortalSpec(enabled=self.has_entrance)
        wall_brushes, _ = generate_portal_wall(
            box_func=self._box,
            x1=ox - hw - t, y1=oy - t, z1=oz,
            x2=ox + hw + t, y2=oy, z2=oz + wh,
            portal_spec=portal_spec,
            portal_axis="x",
            portal_center=ox,
        )
        brushes.extend(wall_brushes)

        # Back wall
        brushes.append(self._box(ox - hw - t, oy + nl, oz, ox + hw + t, oy + nl + t, oz + wh))

        # === CORNER TOWERS (optional) ===
        if self.corner_towers:
            tower_size = 48.0
            tower_height = wh + 48.0  # Taller than perimeter walls
            ts2 = tower_size / 2

            # Four corner towers (buttress-style)
            corners = [
                (ox - hw + ts2, oy + ts2),              # Front-left
                (ox + hw - ts2, oy + ts2),              # Front-right
                (ox - hw + ts2, oy + nl - ts2),         # Back-left
                (ox + hw - ts2, oy + nl - ts2),         # Back-right
            ]
            for cx, cy in corners:
                brushes.append(self._box(
                    cx - ts2, cy - ts2, oz,
                    cx + ts2, cy + ts2, oz + tower_height
                ))

        # === BATTLEMENT CRENELLATION (optional, on top of low walls) ===
        # Add simple merlons along perimeter walls if height is sufficient
        if wh >= 48:
            merlon_h = 24.0
            merlon_w = 24.0
            crenel_w = 16.0
            pattern = merlon_w + crenel_w

            # Merlons along left and right walls
            y = oy
            while y + merlon_w <= oy + nl:
                brushes.append(self._box(
                    ox - hw - t, y, oz + wh,
                    ox - hw, y + merlon_w, oz + wh + merlon_h
                ))
                brushes.append(self._box(
                    ox + hw, y, oz + wh,
                    ox + hw + t, y + merlon_w, oz + wh + merlon_h
                ))
                y += pattern

            # Merlons along front and back walls
            x = ox - hw
            while x + merlon_w <= ox + hw:
                # Front wall merlons (skip entrance area if enabled)
                if not self.has_entrance or abs(x + merlon_w / 2 - ox) > 48:
                    brushes.append(self._box(
                        x, oy - t, oz + wh,
                        x + merlon_w, oy, oz + wh + merlon_h
                    ))
                brushes.append(self._box(
                    x, oy + nl, oz + wh,
                    x + merlon_w, oy + nl + t, oz + wh + merlon_h
                ))
                x += pattern

        # Register portal tag for rectangular courtyard
        if self.has_entrance:
            self._register_portal_tag(
                portal_id="entrance",
                center_x=ox,
                center_y=oy,
                center_z=oz,
                direction=PortalDirection.SOUTH,
            )

        return brushes

    def _generate_polygonal_courtyard(
        self, ox: float, oy: float, oz: float,
        hw: float, nl: float, nh: float, t: float
    ) -> List[Brush]:
        """Generate polygonal Courtyard shell (simplified for non-rectangular)."""
        brushes: List[Brush] = []

        # Calculate radius from room dimensions, clamped to fit footprint
        # The polygon must fit within both dimensions: hw (half-width) and nl/2 (half-length)
        # Using min() ensures non-square footprints don't overflow in the short dimension
        max_radius = min(hw, nl / 2)
        radius = max(max_radius, self._get_polygon_min_radius(self.shell_sides))

        # Center the polygon at the room center
        cx = ox
        cy = oy + nl / 2

        # Determine portal segment
        portal_segment = self._find_portal_segment(self.shell_sides, "SOUTH") if self.has_entrance else -1

        # Compute vestibule clip zone to avoid floor/ceiling overlap
        vestibule_clip = self._compute_vestibule_clip_zone(
            cx, cy, radius, self.shell_sides, portal_segment,
            PORTAL_WIDTH, t,
            room_origin=(ox, oy, oz), room_length=nl, room_width=hw * 2
        )

        # Generate floor (with clip zone to avoid vestibule overlap)
        brushes.extend(self._generate_polygonal_floor(
            cx, cy, oz, radius, self.shell_sides, t,
            is_ceiling=False,
            vestibule_clip_zone=vestibule_clip
        ))

        # Generate ceiling (with sky texture, and clip zone to avoid vestibule overlap)
        brushes.extend(self._generate_polygonal_floor(
            cx, cy, oz + nh, radius, self.shell_sides, t,
            is_ceiling=True,
            vestibule_clip_zone=vestibule_clip
        ))

        # Set portal target for grid alignment before generating walls
        self._set_portal_target_for_polygonal(cx, cy, radius, self.shell_sides, portal_segment)

        # Generate walls
        brushes.extend(self._generate_polygonal_walls(
            cx, cy, oz, oz + nh, radius, self.shell_sides, t,
            portal_segment=portal_segment,
            portal_width=PORTAL_WIDTH,
            portal_height=PORTAL_HEIGHT,
            portal_target_x=getattr(self, '_portal_target_x', None),
            portal_target_y=getattr(self, '_portal_target_y', None)
        ))

        # Generate connector corridor from portal to grid center
        # Pass room origin and dimensions for accurate footprint edge calculation
        if self.has_entrance:
            brushes.extend(self._generate_polygonal_portal_connector(
                cx, cy, oz, radius, self.shell_sides, portal_segment,
                PORTAL_WIDTH, PORTAL_HEIGHT, t, room_height=nh,
                room_origin=(ox, oy, oz), room_length=nl, room_width=hw * 2
            ))

        return brushes


class Arena(GeometricPrimitive):
    """Combat arena with sunken central pit and raised spectator gallery.

    Features:
    - Sunken central combat pit (negative space)
    - Raised walkway/gallery around perimeter
    - Configurable shell shape (4=square, 6=hex, 8=octagon)
    - Multiple entrance options (gate-style portals)
    - Optional pillars in arena floor
    - 100% sealed geometry

    Unique geometry: Multi-level with central depression (inverse of Cistern).
    """

    width: float = 256.0        # Half-width of arena
    length: float = 256.0       # Length
    height: float = 128.0       # Ceiling height
    shell_sides: int = 4        # Number of sides (4=square, 6=hex, 8=octagon)
    pit_depth: float = 48.0     # How deep the pit floor is below gallery
    gallery_width: float = 64.0  # Width of raised walkway
    arena_pillars: int = 0      # Optional pillars in pit (0-4)
    random_seed: int = 0

    # Portal control
    has_entrance: bool = True

    # Portal position offset
    _entrance_x_offset: float = 0.0

    WALL_THICKNESS: float = 16.0

    @classmethod
    def get_display_name(cls) -> str:
        return "Arena"

    @classmethod
    def get_category(cls) -> str:
        return "Rooms"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "width": {
                "type": "float", "default": 256.0, "min": 128, "max": 384, "label": "Half-Width",
                "description": "Half-width of the arena from center to side walls"
            },
            "length": {
                "type": "float", "default": 256.0, "min": 128, "max": 512, "label": "Length",
                "description": "Total length of the arena from entrance to back wall"
            },
            "height": {
                "type": "float", "default": 128.0, "min": 96, "max": 192, "label": "Ceiling Height",
                "description": "Height from gallery floor to ceiling (player is 56 units tall)"
            },
            "shell_sides": {
                "type": "int",
                "default": 4,
                "min": 3,
                "max": 16,
                "label": "Shell Sides",
                "description": "Number of sides for room shape (4=square, 6=hex, 8=octagon)"
            },
            "pit_depth": {
                "type": "float", "default": 48.0, "min": 24, "max": 96, "label": "Pit Depth",
                "description": "How far the central combat pit drops below gallery level"
            },
            "gallery_width": {
                "type": "float", "default": 64.0, "min": 32, "max": 128, "label": "Gallery Width",
                "description": "Width of raised spectator walkway around the pit perimeter"
            },
            "arena_pillars": {
                "type": "int", "default": 0, "min": 0, "max": 4, "label": "Pit Pillars",
                "description": "Number of pillars in the combat pit for cover (0-4)"
            },
            "has_entrance": {
                "type": "bool", "default": True, "label": "Enable Entrance Portal",
                "description": "Create an opening in the front wall for entry"
            },
            "random_seed": {
                "type": "int", "default": 0, "min": 0, "max": 999999, "label": "Random Seed",
                "description": "Seed for deterministic pillar placement (0 = random)"
            },
        }

    def generate(self) -> List[Brush]:
        """Generate arena geometry with sunken pit and raised gallery."""
        self._reset_tags()  # Reset tags for fresh generation
        ox, oy, oz = self.params.origin
        brushes: List[Brush] = []
        t = self.WALL_THICKNESS

        if self.random_seed == 0:
            rng = random.Random()
        else:
            rng = random.Random(self.random_seed)

        hw = self.width
        nl = self.length
        nh = self.height
        pd = self.pit_depth
        gw = self.gallery_width

        # Use polygonal shell if sides != 4
        # Note: Pit, gallery, and pillars are disabled for polygonal shapes
        if self.shell_sides != 4:
            brushes = self._generate_polygonal_arena(ox, oy, oz, hw, nl, nh, t)
            # Register portal tag at footprint edge (grid-aligned)
            if self.has_entrance:
                self._register_portal_tag(
                    portal_id="entrance",
                    center_x=ox,  # Room center X for grid alignment
                    center_y=oy,  # Footprint south edge (Y=origin)
                    center_z=oz,
                    direction=PortalDirection.SOUTH,
                )
            return brushes

        # Pit dimensions (central area)
        pit_hw = hw - gw  # Half-width of pit
        pit_y1 = oy + gw
        pit_y2 = oy + nl - gw

        # === FLOOR ===
        # Gallery floor (raised level, at oz)
        # Extends by t in all directions for sealing
        brushes.append(self._box(
            ox - hw - t, oy - t, oz - t,
            ox + hw + t, oy + nl + t, oz
        ))

        # Pit floor (sunken, at oz - pit_depth)
        # This creates the depression
        brushes.append(self._box(
            ox - pit_hw, pit_y1, oz - pd - t,
            ox + pit_hw, pit_y2, oz - pd
        ))

        # === CEILING ===
        brushes.append(self._box(
            ox - hw - t, oy - t, oz + nh,
            ox + hw + t, oy + nl + t, oz + nh + t
        ))

        # === OUTER WALLS ===
        # Left wall
        brushes.append(self._box(
            ox - hw - t, oy - t, oz,
            ox - hw, oy + nl + t, oz + nh
        ))
        # Right wall
        brushes.append(self._box(
            ox + hw, oy - t, oz,
            ox + hw + t, oy + nl + t, oz + nh
        ))
        # Back wall
        brushes.append(self._box(
            ox - hw - t, oy + nl, oz,
            ox + hw + t, oy + nl + t, oz + nh
        ))
        # Front wall (with portal opening)
        if self.has_entrance:
            pw, ph = PORTAL_WIDTH, PORTAL_HEIGHT
            entrance_x = ox + self._entrance_x_offset
            # Left section
            brushes.append(self._box(
                ox - hw - t, oy - t, oz,
                entrance_x - pw / 2, oy, oz + nh
            ))
            # Right section
            brushes.append(self._box(
                entrance_x + pw / 2, oy - t, oz,
                ox + hw + t, oy, oz + nh
            ))
            # Lintel
            brushes.append(self._box(
                entrance_x - pw / 2, oy - t, oz + ph,
                entrance_x + pw / 2, oy, oz + nh
            ))
        else:
            brushes.append(self._box(
                ox - hw - t, oy - t, oz,
                ox + hw + t, oy, oz + nh
            ))

        # === PIT RETAINING WALLS ===
        # These walls separate the pit from the gallery
        # Left pit wall
        brushes.append(self._box(
            ox - pit_hw - t, pit_y1 - t, oz - pd,
            ox - pit_hw, pit_y2 + t, oz
        ))
        # Right pit wall
        brushes.append(self._box(
            ox + pit_hw, pit_y1 - t, oz - pd,
            ox + pit_hw + t, pit_y2 + t, oz
        ))
        # Front pit wall
        brushes.append(self._box(
            ox - pit_hw - t, pit_y1 - t, oz - pd,
            ox + pit_hw + t, pit_y1, oz
        ))
        # Back pit wall
        brushes.append(self._box(
            ox - pit_hw - t, pit_y2, oz - pd,
            ox + pit_hw + t, pit_y2 + t, oz
        ))

        # === OPTIONAL PILLARS IN PIT ===
        if self.arena_pillars > 0:
            pillar_r = 16.0  # Pillar half-width
            positions = [
                (ox - pit_hw / 2, (pit_y1 + pit_y2) / 2),
                (ox + pit_hw / 2, (pit_y1 + pit_y2) / 2),
                (ox, pit_y1 + (pit_y2 - pit_y1) * 0.33),
                (ox, pit_y1 + (pit_y2 - pit_y1) * 0.67),
            ]
            for i in range(min(self.arena_pillars, 4)):
                px, py = positions[i]
                brushes.append(self._box(
                    px - pillar_r, py - pillar_r, oz - pd,
                    px + pillar_r, py + pillar_r, oz + nh
                ))

        # Register portal tag for rectangular arena
        if self.has_entrance:
            entrance_x = ox + self._entrance_x_offset
            self._register_portal_tag(
                portal_id="entrance",
                center_x=entrance_x,
                center_y=oy,
                center_z=oz,
                direction=PortalDirection.SOUTH,
            )

        return brushes

    def _generate_polygonal_arena(
        self, ox: float, oy: float, oz: float,
        hw: float, nl: float, nh: float, t: float
    ) -> List[Brush]:
        """Generate polygonal arena shell (no pit/gallery for non-rectangular)."""
        brushes: List[Brush] = []

        # Calculate radius from room dimensions, clamped to fit footprint
        # The polygon must fit within both dimensions: hw (half-width) and nl/2 (half-length)
        # Using min() ensures non-square footprints don't overflow in the short dimension
        max_radius = min(hw, nl / 2)
        radius = max(max_radius, self._get_polygon_min_radius(self.shell_sides))

        # Center the polygon at the room center
        cx = ox
        cy = oy + nl / 2

        # Determine portal segment
        portal_segment = self._find_portal_segment(self.shell_sides, "SOUTH") if self.has_entrance else -1

        # Compute vestibule clip zone to avoid floor/ceiling overlap
        vestibule_clip = self._compute_vestibule_clip_zone(
            cx, cy, radius, self.shell_sides, portal_segment,
            PORTAL_WIDTH, t,
            room_origin=(ox, oy, oz), room_length=nl, room_width=hw * 2
        )

        # Generate floor (with clip zone to avoid vestibule overlap)
        brushes.extend(self._generate_polygonal_floor(
            cx, cy, oz, radius, self.shell_sides, t,
            is_ceiling=False,
            vestibule_clip_zone=vestibule_clip
        ))

        # Generate ceiling (with clip zone to avoid vestibule overlap)
        brushes.extend(self._generate_polygonal_floor(
            cx, cy, oz + nh, radius, self.shell_sides, t,
            is_ceiling=True,
            vestibule_clip_zone=vestibule_clip
        ))

        # Set portal target for grid alignment before generating walls
        self._set_portal_target_for_polygonal(cx, cy, radius, self.shell_sides, portal_segment)

        # Generate walls
        brushes.extend(self._generate_polygonal_walls(
            cx, cy, oz, oz + nh, radius, self.shell_sides, t,
            portal_segment=portal_segment,
            portal_width=PORTAL_WIDTH,
            portal_height=PORTAL_HEIGHT,
            portal_target_x=getattr(self, '_portal_target_x', None),
            portal_target_y=getattr(self, '_portal_target_y', None)
        ))

        # Generate connector corridor from portal to grid center
        # Pass room origin and dimensions for accurate footprint edge calculation
        if self.has_entrance:
            brushes.extend(self._generate_polygonal_portal_connector(
                cx, cy, oz, radius, self.shell_sides, portal_segment,
                PORTAL_WIDTH, PORTAL_HEIGHT, t, room_height=nh,
                room_origin=(ox, oy, oz), room_length=nl, room_width=hw * 2
            ))

        return brushes


class Laboratory(GeometricPrimitive):
    """Workshop/laboratory space with work surfaces and storage.

    Features:
    - Work tables (raised platforms)
    - Wall alcoves for equipment/shelves
    - Optional sunken drainage channel
    - Configurable shell shape (4=square, 6=hex, 8=octagon)
    - 100% sealed geometry

    Fills the common "mad scientist" dungeon role.
    """

    width: float = 192.0        # Half-width
    length: float = 256.0       # Length
    height: float = 128.0       # Ceiling height
    shell_sides: int = 4        # Number of sides (4=square, 6=hex, 8=octagon)
    work_tables: int = 2        # Number of work table platforms (0-4)
    alcove_count: int = 4       # Wall alcoves (0-8)
    has_drain: bool = False     # Sunken central drain channel
    random_seed: int = 0

    # Portal control
    has_entrance: bool = True
    _entrance_x_offset: float = 0.0

    WALL_THICKNESS: float = 16.0

    @classmethod
    def get_display_name(cls) -> str:
        return "Laboratory"

    @classmethod
    def get_category(cls) -> str:
        return "Rooms"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "width": {
                "type": "float", "default": 192.0, "min": 96, "max": 256, "label": "Half-Width",
                "description": "Half-width of the laboratory from center to walls"
            },
            "length": {
                "type": "float", "default": 256.0, "min": 192, "max": 384, "label": "Length",
                "description": "Total length from entrance to back wall"
            },
            "height": {
                "type": "float", "default": 128.0, "min": 96, "max": 192, "label": "Ceiling Height",
                "description": "Height from floor to ceiling (player is 56 units tall)"
            },
            "shell_sides": {
                "type": "int",
                "default": 4,
                "min": 3,
                "max": 16,
                "label": "Shell Sides",
                "description": "Number of sides for room shape (4=square, 6=hex, 8=octagon)"
            },
            "work_tables": {
                "type": "int", "default": 2, "min": 0, "max": 4, "label": "Work Tables",
                "description": "Number of raised work surface platforms (disabled for polygonal shapes)"
            },
            "alcove_count": {
                "type": "int", "default": 4, "min": 0, "max": 8, "label": "Wall Alcoves",
                "description": "Number of wall niches for equipment storage (disabled for polygonal shapes)"
            },
            "has_drain": {
                "type": "bool", "default": False, "label": "Central Drain",
                "description": "Add sunken drainage channel running through center"
            },
            "has_entrance": {
                "type": "bool", "default": True, "label": "Enable Entrance Portal",
                "description": "Create an opening in the front wall for entry"
            },
            "random_seed": {
                "type": "int", "default": 0, "min": 0, "max": 999999, "label": "Random Seed",
                "description": "Seed for deterministic table placement (0 = random)"
            },
        }

    def generate(self) -> List[Brush]:
        """Generate laboratory geometry."""
        self._reset_tags()  # Reset tags for fresh generation
        ox, oy, oz = self.params.origin
        brushes: List[Brush] = []
        t = self.WALL_THICKNESS

        if self.random_seed == 0:
            rng = random.Random()
        else:
            rng = random.Random(self.random_seed)

        hw = self.width
        nl = self.length
        nh = self.height

        # Use polygonal shell if sides != 4
        # Note: Work tables, alcoves, and drain are disabled for polygonal shapes
        if self.shell_sides != 4:
            brushes = self._generate_polygonal_laboratory(ox, oy, oz, hw, nl, nh, t)
            # Register portal tag at footprint edge (grid-aligned)
            if self.has_entrance:
                self._register_portal_tag(
                    portal_id="entrance",
                    center_x=ox,  # Room center X for grid alignment
                    center_y=oy,  # Footprint south edge (Y=origin)
                    center_z=oz,
                    direction=PortalDirection.SOUTH,
                )
            return brushes

        # === FLOOR ===
        brushes.append(self._box(
            ox - hw - t, oy - t, oz - t,
            ox + hw + t, oy + nl + t, oz
        ))

        # === CEILING ===
        brushes.append(self._box(
            ox - hw - t, oy - t, oz + nh,
            ox + hw + t, oy + nl + t, oz + nh + t
        ))

        # === WALLS ===
        # Left wall
        brushes.append(self._box(
            ox - hw - t, oy - t, oz,
            ox - hw, oy + nl + t, oz + nh
        ))
        # Right wall
        brushes.append(self._box(
            ox + hw, oy - t, oz,
            ox + hw + t, oy + nl + t, oz + nh
        ))
        # Back wall
        brushes.append(self._box(
            ox - hw - t, oy + nl, oz,
            ox + hw + t, oy + nl + t, oz + nh
        ))
        # Front wall (with portal opening)
        if self.has_entrance:
            pw, ph = PORTAL_WIDTH, PORTAL_HEIGHT
            entrance_x = ox + self._entrance_x_offset
            brushes.append(self._box(
                ox - hw - t, oy - t, oz,
                entrance_x - pw / 2, oy, oz + nh
            ))
            brushes.append(self._box(
                entrance_x + pw / 2, oy - t, oz,
                ox + hw + t, oy, oz + nh
            ))
            brushes.append(self._box(
                entrance_x - pw / 2, oy - t, oz + ph,
                entrance_x + pw / 2, oy, oz + nh
            ))
        else:
            brushes.append(self._box(
                ox - hw - t, oy - t, oz,
                ox + hw + t, oy, oz + nh
            ))

        # === WORK TABLES (raised platforms) ===
        table_h = 32.0  # Height of tables
        table_w = 48.0  # Half-width
        table_l = 64.0  # Length
        if self.work_tables > 0:
            positions = [
                (ox - hw / 2, oy + nl * 0.3),
                (ox + hw / 2, oy + nl * 0.3),
                (ox - hw / 2, oy + nl * 0.7),
                (ox + hw / 2, oy + nl * 0.7),
            ]
            for i in range(min(self.work_tables, 4)):
                tx, ty = positions[i]
                brushes.append(self._box(
                    tx - table_w, ty - table_l / 2, oz,
                    tx + table_w, ty + table_l / 2, oz + table_h
                ))

        # === OPTIONAL DRAIN CHANNEL ===
        if self.has_drain:
            drain_w = 16.0  # Half-width of drain
            drain_d = 8.0   # Depth
            brushes.append(self._box(
                ox - drain_w, oy + 32, oz - drain_d - t,
                ox + drain_w, oy + nl - 32, oz - drain_d
            ))

        # Register portal tag for rectangular laboratory
        if self.has_entrance:
            entrance_x = ox + self._entrance_x_offset
            self._register_portal_tag(
                portal_id="entrance",
                center_x=entrance_x,
                center_y=oy,
                center_z=oz,
                direction=PortalDirection.SOUTH,
            )

        return brushes

    def _generate_polygonal_laboratory(
        self, ox: float, oy: float, oz: float,
        hw: float, nl: float, nh: float, t: float
    ) -> List[Brush]:
        """Generate polygonal laboratory shell (no tables/alcoves for non-rectangular)."""
        brushes: List[Brush] = []

        # Calculate radius from room dimensions, clamped to fit footprint
        # The polygon must fit within both dimensions: hw (half-width) and nl/2 (half-length)
        # Using min() ensures non-square footprints don't overflow in the short dimension
        max_radius = min(hw, nl / 2)
        radius = max(max_radius, self._get_polygon_min_radius(self.shell_sides))

        # Center the polygon at the room center
        cx = ox
        cy = oy + nl / 2

        # Determine portal segment
        portal_segment = self._find_portal_segment(self.shell_sides, "SOUTH") if self.has_entrance else -1

        # Compute vestibule clip zone to avoid floor/ceiling overlap
        vestibule_clip = self._compute_vestibule_clip_zone(
            cx, cy, radius, self.shell_sides, portal_segment,
            PORTAL_WIDTH, t,
            room_origin=(ox, oy, oz), room_length=nl, room_width=hw * 2
        )

        # Generate floor (with clip zone to avoid vestibule overlap)
        brushes.extend(self._generate_polygonal_floor(
            cx, cy, oz, radius, self.shell_sides, t,
            is_ceiling=False,
            vestibule_clip_zone=vestibule_clip
        ))

        # Generate ceiling (with clip zone to avoid vestibule overlap)
        brushes.extend(self._generate_polygonal_floor(
            cx, cy, oz + nh, radius, self.shell_sides, t,
            is_ceiling=True,
            vestibule_clip_zone=vestibule_clip
        ))

        # Set portal target for grid alignment before generating walls
        self._set_portal_target_for_polygonal(cx, cy, radius, self.shell_sides, portal_segment)

        # Generate walls
        brushes.extend(self._generate_polygonal_walls(
            cx, cy, oz, oz + nh, radius, self.shell_sides, t,
            portal_segment=portal_segment,
            portal_width=PORTAL_WIDTH,
            portal_height=PORTAL_HEIGHT,
            portal_target_x=getattr(self, '_portal_target_x', None),
            portal_target_y=getattr(self, '_portal_target_y', None)
        ))

        # Generate connector corridor from portal to grid center
        # Pass room origin and dimensions for accurate footprint edge calculation
        if self.has_entrance:
            brushes.extend(self._generate_polygonal_portal_connector(
                cx, cy, oz, radius, self.shell_sides, portal_segment,
                PORTAL_WIDTH, PORTAL_HEIGHT, t, room_height=nh,
                room_origin=(ox, oy, oz), room_length=nl, room_width=hw * 2
            ))

        return brushes


class Vault(GeometricPrimitive):
    """Secure storage vault with extra thick walls.

    Features:
    - Extra thick walls (like Stronghold)
    - Single narrow entrance
    - Optional alcove niches for display
    - Raised platform for treasure
    - Configurable shell shape (4=square, 6=hex, 8=octagon)
    - 100% sealed geometry

    Defensive design focused on small footprint.
    """

    width: float = 128.0        # Half-width
    length: float = 192.0       # Length
    height: float = 112.0       # Ceiling height
    shell_sides: int = 4        # Number of sides (4=square, 6=hex, 8=octagon)
    has_alcoves: bool = True    # Niches in walls
    has_pedestal: bool = True   # Raised display platform
    random_seed: int = 0

    # Portal control
    has_entrance: bool = True
    _entrance_x_offset: float = 0.0

    WALL_THICKNESS: float = 24.0  # Extra thick walls

    @classmethod
    def get_display_name(cls) -> str:
        return "Vault"

    @classmethod
    def get_category(cls) -> str:
        return "Rooms"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "width": {
                "type": "float", "default": 128.0, "min": 64, "max": 192, "label": "Half-Width",
                "description": "Half-width of the vault (compact for secure storage)"
            },
            "length": {
                "type": "float", "default": 192.0, "min": 128, "max": 256, "label": "Length",
                "description": "Total length from narrow entrance to back wall"
            },
            "height": {
                "type": "float", "default": 112.0, "min": 80, "max": 144, "label": "Ceiling Height",
                "description": "Height from floor to ceiling (lower for confined feel)"
            },
            "shell_sides": {
                "type": "int",
                "default": 4,
                "min": 3,
                "max": 16,
                "label": "Shell Sides",
                "description": "Number of sides for room shape (4=square, 6=hex, 8=octagon)"
            },
            "has_alcoves": {
                "type": "bool", "default": True, "label": "Wall Niches",
                "description": "Add recessed wall niches for display items"
            },
            "has_pedestal": {
                "type": "bool", "default": True, "label": "Display Pedestal",
                "description": "Add raised platform at back for treasure display"
            },
            "has_entrance": {
                "type": "bool", "default": True, "label": "Enable Entrance Portal",
                "description": "Create narrow security entrance (75% standard width)"
            },
            "random_seed": {
                "type": "int", "default": 0, "min": 0, "max": 999999, "label": "Random Seed",
                "description": "Seed for deterministic generation (0 = random)"
            },
        }

    def generate(self) -> List[Brush]:
        """Generate vault geometry with thick walls."""
        self._reset_tags()  # Reset tags for fresh generation
        ox, oy, oz = self.params.origin
        brushes: List[Brush] = []
        t = self.WALL_THICKNESS

        hw = self.width
        nl = self.length
        nh = self.height

        # Use polygonal shell if sides != 4
        if self.shell_sides != 4:
            brushes = self._generate_polygonal_vault(ox, oy, oz, hw, nl, nh, t)
            # Register portal tag at footprint edge (grid-aligned)
            if self.has_entrance:
                self._register_portal_tag(
                    portal_id="entrance",
                    center_x=ox,  # Room center X for grid alignment
                    center_y=oy,  # Footprint south edge (Y=origin)
                    center_z=oz,
                    direction=PortalDirection.SOUTH,
                )
            return brushes

        # === FLOOR ===
        brushes.append(self._box(
            ox - hw - t, oy - t, oz - t,
            ox + hw + t, oy + nl + t, oz
        ))

        # === CEILING ===
        brushes.append(self._box(
            ox - hw - t, oy - t, oz + nh,
            ox + hw + t, oy + nl + t, oz + nh + t
        ))

        # === WALLS ===
        # Left wall
        brushes.append(self._box(
            ox - hw - t, oy - t, oz,
            ox - hw, oy + nl + t, oz + nh
        ))
        # Right wall
        brushes.append(self._box(
            ox + hw, oy - t, oz,
            ox + hw + t, oy + nl + t, oz + nh
        ))
        # Back wall
        brushes.append(self._box(
            ox - hw - t, oy + nl, oz,
            ox + hw + t, oy + nl + t, oz + nh
        ))
        # Front wall (with narrow portal)
        if self.has_entrance:
            # Vault uses narrower portal width for security feel
            pw = PORTAL_WIDTH * 0.75  # Narrower
            ph = PORTAL_HEIGHT
            entrance_x = ox + self._entrance_x_offset
            brushes.append(self._box(
                ox - hw - t, oy - t, oz,
                entrance_x - pw / 2, oy, oz + nh
            ))
            brushes.append(self._box(
                entrance_x + pw / 2, oy - t, oz,
                ox + hw + t, oy, oz + nh
            ))
            brushes.append(self._box(
                entrance_x - pw / 2, oy - t, oz + ph,
                entrance_x + pw / 2, oy, oz + nh
            ))
        else:
            brushes.append(self._box(
                ox - hw - t, oy - t, oz,
                ox + hw + t, oy, oz + nh
            ))

        # === PEDESTAL (display platform at back) ===
        if self.has_pedestal:
            ped_h = 24.0
            ped_w = hw * 0.5
            ped_l = 48.0
            brushes.append(self._box(
                ox - ped_w, oy + nl - ped_l - 16, oz,
                ox + ped_w, oy + nl - 16, oz + ped_h
            ))

        # Register portal tag for rectangular vault
        if self.has_entrance:
            entrance_x = ox + self._entrance_x_offset
            self._register_portal_tag(
                portal_id="entrance",
                center_x=entrance_x,
                center_y=oy,
                center_z=oz,
                direction=PortalDirection.SOUTH,
            )

        return brushes

    def _generate_polygonal_vault(
        self, ox: float, oy: float, oz: float,
        hw: float, nl: float, nh: float, t: float
    ) -> List[Brush]:
        """Generate polygonal Vault shell (simplified for non-rectangular)."""
        brushes: List[Brush] = []

        # Calculate radius from room dimensions, clamped to fit footprint
        # The polygon must fit within both dimensions: hw (half-width) and nl/2 (half-length)
        # Using min() ensures non-square footprints don't overflow in the short dimension
        max_radius = min(hw, nl / 2)
        radius = max(max_radius, self._get_polygon_min_radius(self.shell_sides))

        # Center the polygon at the room center
        cx = ox
        cy = oy + nl / 2

        # Determine portal segment
        portal_segment = self._find_portal_segment(self.shell_sides, "SOUTH") if self.has_entrance else -1

        # Compute vestibule clip zone to avoid floor/ceiling overlap
        vestibule_clip = self._compute_vestibule_clip_zone(
            cx, cy, radius, self.shell_sides, portal_segment,
            PORTAL_WIDTH, t,
            room_origin=(ox, oy, oz), room_length=nl, room_width=hw * 2
        )

        # Generate floor (with clip zone to avoid vestibule overlap)
        brushes.extend(self._generate_polygonal_floor(
            cx, cy, oz, radius, self.shell_sides, t,
            is_ceiling=False,
            vestibule_clip_zone=vestibule_clip
        ))

        # Generate ceiling (with clip zone to avoid vestibule overlap)
        brushes.extend(self._generate_polygonal_floor(
            cx, cy, oz + nh, radius, self.shell_sides, t,
            is_ceiling=True,
            vestibule_clip_zone=vestibule_clip
        ))

        # Set portal target for grid alignment before generating walls
        self._set_portal_target_for_polygonal(cx, cy, radius, self.shell_sides, portal_segment)

        # Generate walls
        brushes.extend(self._generate_polygonal_walls(
            cx, cy, oz, oz + nh, radius, self.shell_sides, t,
            portal_segment=portal_segment,
            portal_width=PORTAL_WIDTH,
            portal_height=PORTAL_HEIGHT,
            portal_target_x=getattr(self, '_portal_target_x', None),
            portal_target_y=getattr(self, '_portal_target_y', None)
        ))

        # Generate connector corridor from portal to grid center
        # Pass room origin and dimensions for accurate footprint edge calculation
        if self.has_entrance:
            brushes.extend(self._generate_polygonal_portal_connector(
                cx, cy, oz, radius, self.shell_sides, portal_segment,
                PORTAL_WIDTH, PORTAL_HEIGHT, t, room_height=nh,
                room_origin=(ox, oy, oz), room_length=nl, room_width=hw * 2
            ))

        return brushes


class Barracks(GeometricPrimitive):
    """Sleeping quarters with bed alcoves along walls.

    Features:
    - Rows of alcoves (bed niches) on both walls
    - Central aisle for movement
    - Optional footlockers (low platforms)
    - 100% sealed geometry

    Highly symmetrical with repeating alcove pattern.
    """

    width: float = 192.0        # Half-width
    length: float = 320.0       # Length
    height: float = 112.0       # Ceiling height
    shell_sides: int = 4        # Number of sides (4=square, 6=hex, 8=octagon)
    bed_alcoves: int = 4        # Alcoves per side (0-6)
    has_footlockers: bool = True  # Low platforms at bed foot
    random_seed: int = 0

    # Portal control
    has_entrance: bool = True
    _entrance_x_offset: float = 0.0

    WALL_THICKNESS: float = 16.0

    @classmethod
    def get_display_name(cls) -> str:
        return "Barracks"

    @classmethod
    def get_category(cls) -> str:
        return "Rooms"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "width": {
                "type": "float", "default": 192.0, "min": 128, "max": 256, "label": "Half-Width",
                "description": "Half-width of the barracks (needs room for beds on both sides)"
            },
            "length": {
                "type": "float", "default": 320.0, "min": 256, "max": 512, "label": "Length",
                "description": "Total length (longer to accommodate multiple bed rows)"
            },
            "height": {
                "type": "float", "default": 112.0, "min": 80, "max": 144, "label": "Ceiling Height",
                "description": "Height from floor to ceiling (player is 56 units tall)"
            },
            "shell_sides": {
                "type": "int",
                "default": 4,
                "min": 3,
                "max": 16,
                "label": "Shell Sides",
                "description": "Number of sides for room shape (4=square, 6=hex, 8=octagon)"
            },
            "bed_alcoves": {
                "type": "int", "default": 4, "min": 0, "max": 6, "label": "Beds Per Side",
                "description": "Number of sleeping platforms on each wall (total = 2x this)"
            },
            "has_footlockers": {
                "type": "bool", "default": True, "label": "Footlockers",
                "description": "Add low storage platforms at the foot of each bed"
            },
            "has_entrance": {
                "type": "bool", "default": True, "label": "Enable Entrance Portal",
                "description": "Create an opening in the front wall for entry"
            },
            "random_seed": {
                "type": "int", "default": 0, "min": 0, "max": 999999, "label": "Random Seed",
                "description": "Seed for deterministic generation (0 = random)"
            },
        }

    def generate(self) -> List[Brush]:
        """Generate barracks geometry with bed alcoves."""
        self._reset_tags()  # Reset tags for fresh generation
        ox, oy, oz = self.params.origin
        brushes: List[Brush] = []
        t = self.WALL_THICKNESS

        hw = self.width
        nl = self.length
        nh = self.height

        # Use polygonal shell if sides != 4
        if self.shell_sides != 4:
            brushes = self._generate_polygonal_barracks(ox, oy, oz, hw, nl, nh, t)
            # Register portal tag at footprint edge (grid-aligned)
            if self.has_entrance:
                self._register_portal_tag(
                    portal_id="entrance",
                    center_x=ox,  # Room center X for grid alignment
                    center_y=oy,  # Footprint south edge (Y=origin)
                    center_z=oz,
                    direction=PortalDirection.SOUTH,
                )
            return brushes

        # === FLOOR ===
        brushes.append(self._box(
            ox - hw - t, oy - t, oz - t,
            ox + hw + t, oy + nl + t, oz
        ))

        # === CEILING ===
        brushes.append(self._box(
            ox - hw - t, oy - t, oz + nh,
            ox + hw + t, oy + nl + t, oz + nh + t
        ))

        # === WALLS ===
        # Left wall - will have alcoves cut into it
        # Right wall - will have alcoves cut into it
        # We generate walls as segments with alcoves

        # Back wall (solid)
        brushes.append(self._box(
            ox - hw - t, oy + nl, oz,
            ox + hw + t, oy + nl + t, oz + nh
        ))

        # Front wall (with portal opening)
        if self.has_entrance:
            pw, ph = PORTAL_WIDTH, PORTAL_HEIGHT
            entrance_x = ox + self._entrance_x_offset
            brushes.append(self._box(
                ox - hw - t, oy - t, oz,
                entrance_x - pw / 2, oy, oz + nh
            ))
            brushes.append(self._box(
                entrance_x + pw / 2, oy - t, oz,
                ox + hw + t, oy, oz + nh
            ))
            brushes.append(self._box(
                entrance_x - pw / 2, oy - t, oz + ph,
                entrance_x + pw / 2, oy, oz + nh
            ))
        else:
            brushes.append(self._box(
                ox - hw - t, oy - t, oz,
                ox + hw + t, oy, oz + nh
            ))

        # Side walls (simple solid for now)
        brushes.append(self._box(
            ox - hw - t, oy - t, oz,
            ox - hw, oy + nl + t, oz + nh
        ))
        brushes.append(self._box(
            ox + hw, oy - t, oz,
            ox + hw + t, oy + nl + t, oz + nh
        ))

        # === BED PLATFORMS (raised sleeping areas) ===
        if self.bed_alcoves > 0:
            bed_h = 24.0   # Height of bed platform
            bed_w = 40.0   # Width
            bed_l = 64.0   # Length
            gap = 16.0     # Gap between beds

            total_length = self.bed_alcoves * (bed_l + gap)
            start_y = oy + (nl - total_length) / 2

            for i in range(self.bed_alcoves):
                y = start_y + i * (bed_l + gap)
                # Left side bed
                brushes.append(self._box(
                    ox - hw + 8, y, oz,
                    ox - hw + 8 + bed_w, y + bed_l, oz + bed_h
                ))
                # Right side bed
                brushes.append(self._box(
                    ox + hw - 8 - bed_w, y, oz,
                    ox + hw - 8, y + bed_l, oz + bed_h
                ))

                # Footlockers at foot of beds
                if self.has_footlockers:
                    locker_h = 16.0
                    locker_l = 24.0
                    brushes.append(self._box(
                        ox - hw + 8, y + bed_l, oz,
                        ox - hw + 8 + bed_w, y + bed_l + locker_l, oz + locker_h
                    ))
                    brushes.append(self._box(
                        ox + hw - 8 - bed_w, y + bed_l, oz,
                        ox + hw - 8, y + bed_l + locker_l, oz + locker_h
                    ))

        # Register portal tag for rectangular barracks
        if self.has_entrance:
            entrance_x = ox + self._entrance_x_offset
            self._register_portal_tag(
                portal_id="entrance",
                center_x=entrance_x,
                center_y=oy,
                center_z=oz,
                direction=PortalDirection.SOUTH,
            )

        return brushes

    def _generate_polygonal_barracks(
        self, ox: float, oy: float, oz: float,
        hw: float, nl: float, nh: float, t: float
    ) -> List[Brush]:
        """Generate polygonal Barracks shell (simplified for non-rectangular)."""
        brushes: List[Brush] = []

        # Calculate radius from room dimensions, clamped to fit footprint
        # The polygon must fit within both dimensions: hw (half-width) and nl/2 (half-length)
        # Using min() ensures non-square footprints don't overflow in the short dimension
        max_radius = min(hw, nl / 2)
        radius = max(max_radius, self._get_polygon_min_radius(self.shell_sides))

        # Center the polygon at the room center
        cx = ox
        cy = oy + nl / 2

        # Determine portal segment
        portal_segment = self._find_portal_segment(self.shell_sides, "SOUTH") if self.has_entrance else -1

        # Compute vestibule clip zone to avoid floor/ceiling overlap
        vestibule_clip = self._compute_vestibule_clip_zone(
            cx, cy, radius, self.shell_sides, portal_segment,
            PORTAL_WIDTH, t,
            room_origin=(ox, oy, oz), room_length=nl, room_width=hw * 2
        )

        # Generate floor (with clip zone to avoid vestibule overlap)
        brushes.extend(self._generate_polygonal_floor(
            cx, cy, oz, radius, self.shell_sides, t,
            is_ceiling=False,
            vestibule_clip_zone=vestibule_clip
        ))

        # Generate ceiling (with clip zone to avoid vestibule overlap)
        brushes.extend(self._generate_polygonal_floor(
            cx, cy, oz + nh, radius, self.shell_sides, t,
            is_ceiling=True,
            vestibule_clip_zone=vestibule_clip
        ))

        # Set portal target for grid alignment before generating walls
        self._set_portal_target_for_polygonal(cx, cy, radius, self.shell_sides, portal_segment)

        # Generate walls
        brushes.extend(self._generate_polygonal_walls(
            cx, cy, oz, oz + nh, radius, self.shell_sides, t,
            portal_segment=portal_segment,
            portal_width=PORTAL_WIDTH,
            portal_height=PORTAL_HEIGHT,
            portal_target_x=getattr(self, '_portal_target_x', None),
            portal_target_y=getattr(self, '_portal_target_y', None)
        ))

        # Generate connector corridor from portal to grid center
        # Pass room origin and dimensions for accurate footprint edge calculation
        if self.has_entrance:
            brushes.extend(self._generate_polygonal_portal_connector(
                cx, cy, oz, radius, self.shell_sides, portal_segment,
                PORTAL_WIDTH, PORTAL_HEIGHT, t, room_height=nh,
                room_origin=(ox, oy, oz), room_length=nl, room_width=hw * 2
            ))

        return brushes


class Shrine(GeometricPrimitive):
    """Small worship space with raised altar.

    Features:
    - Raised altar platform at back
    - Optional alcove behind altar
    - Simpler and smaller than Sanctuary
    - 100% sealed geometry

    Compact worship space for small dungeons.
    """

    width: float = 128.0        # Half-width
    length: float = 128.0       # Length
    height: float = 112.0       # Ceiling height
    shell_sides: int = 4        # Number of sides (4=square, 6=hex, 8=octagon)
    altar_style: str = "raised"  # raised, sunken, flat
    has_alcove: bool = True     # Alcove behind altar
    random_seed: int = 0

    # Portal control
    has_entrance: bool = True
    _entrance_x_offset: float = 0.0

    WALL_THICKNESS: float = 16.0

    @classmethod
    def get_display_name(cls) -> str:
        return "Shrine"

    @classmethod
    def get_category(cls) -> str:
        return "Rooms"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "width": {
                "type": "float", "default": 128.0, "min": 64, "max": 192, "label": "Half-Width",
                "description": "Half-width of the shrine (compact worship space)"
            },
            "length": {
                "type": "float", "default": 128.0, "min": 96, "max": 192, "label": "Length",
                "description": "Total length from entrance to altar"
            },
            "height": {
                "type": "float", "default": 112.0, "min": 80, "max": 144, "label": "Ceiling Height",
                "description": "Height from floor to ceiling (player is 56 units tall)"
            },
            "shell_sides": {
                "type": "int",
                "default": 4,
                "min": 3,
                "max": 16,
                "label": "Shell Sides",
                "description": "Number of sides for room shape (4=square, 6=hex, 8=octagon)"
            },
            "altar_style": {
                "type": "choice",
                "default": "raised",
                "choices": ["raised", "sunken", "flat"],
                "label": "Altar Style",
                "description": "Altar elevation: raised platform, sunken pit, or floor level"
            },
            "has_alcove": {
                "type": "bool", "default": True, "label": "Altar Alcove",
                "description": "Add recessed niche behind the altar area"
            },
            "has_entrance": {
                "type": "bool", "default": True, "label": "Enable Entrance Portal",
                "description": "Create an opening in the front wall for entry"
            },
            "random_seed": {
                "type": "int", "default": 0, "min": 0, "max": 999999, "label": "Random Seed",
                "description": "Seed for deterministic generation (0 = random)"
            },
        }

    def generate(self) -> List[Brush]:
        """Generate shrine geometry."""
        self._reset_tags()  # Reset tags for fresh generation
        ox, oy, oz = self.params.origin
        brushes: List[Brush] = []
        t = self.WALL_THICKNESS

        hw = self.width
        nl = self.length
        nh = self.height

        # Use polygonal shell if sides != 4
        if self.shell_sides != 4:
            brushes = self._generate_polygonal_shrine(ox, oy, oz, hw, nl, nh, t)
            # Register portal tag at footprint edge (grid-aligned)
            if self.has_entrance:
                self._register_portal_tag(
                    portal_id="entrance",
                    center_x=ox,  # Room center X for grid alignment
                    center_y=oy,  # Footprint south edge (Y=origin)
                    center_z=oz,
                    direction=PortalDirection.SOUTH,
                )
            return brushes

        # === FLOOR ===
        brushes.append(self._box(
            ox - hw - t, oy - t, oz - t,
            ox + hw + t, oy + nl + t, oz
        ))

        # === CEILING ===
        brushes.append(self._box(
            ox - hw - t, oy - t, oz + nh,
            ox + hw + t, oy + nl + t, oz + nh + t
        ))

        # === WALLS ===
        brushes.append(self._box(
            ox - hw - t, oy - t, oz,
            ox - hw, oy + nl + t, oz + nh
        ))
        brushes.append(self._box(
            ox + hw, oy - t, oz,
            ox + hw + t, oy + nl + t, oz + nh
        ))
        brushes.append(self._box(
            ox - hw - t, oy + nl, oz,
            ox + hw + t, oy + nl + t, oz + nh
        ))
        # Front wall
        if self.has_entrance:
            pw, ph = PORTAL_WIDTH, PORTAL_HEIGHT
            entrance_x = ox + self._entrance_x_offset
            brushes.append(self._box(
                ox - hw - t, oy - t, oz,
                entrance_x - pw / 2, oy, oz + nh
            ))
            brushes.append(self._box(
                entrance_x + pw / 2, oy - t, oz,
                ox + hw + t, oy, oz + nh
            ))
            brushes.append(self._box(
                entrance_x - pw / 2, oy - t, oz + ph,
                entrance_x + pw / 2, oy, oz + nh
            ))
        else:
            brushes.append(self._box(
                ox - hw - t, oy - t, oz,
                ox + hw + t, oy, oz + nh
            ))

        # === ALTAR ===
        altar_w = hw * 0.6
        altar_l = 32.0
        altar_y = oy + nl - altar_l - 24

        if self.altar_style == "raised":
            altar_h = 32.0
            brushes.append(self._box(
                ox - altar_w, altar_y, oz,
                ox + altar_w, altar_y + altar_l, oz + altar_h
            ))
        elif self.altar_style == "sunken":
            # Sunken area in front of altar
            pit_h = 16.0
            brushes.append(self._box(
                ox - altar_w, oy + 48, oz - pit_h - t,
                ox + altar_w, altar_y, oz - pit_h
            ))

        # Register portal tag for rectangular shrine
        if self.has_entrance:
            entrance_x = ox + self._entrance_x_offset
            self._register_portal_tag(
                portal_id="entrance",
                center_x=entrance_x,
                center_y=oy,
                center_z=oz,
                direction=PortalDirection.SOUTH,
            )

        return brushes

    def _generate_polygonal_shrine(
        self, ox: float, oy: float, oz: float,
        hw: float, nl: float, nh: float, t: float
    ) -> List[Brush]:
        """Generate polygonal Shrine shell (simplified for non-rectangular)."""
        brushes: List[Brush] = []

        # Calculate radius from room dimensions, clamped to fit footprint
        # The polygon must fit within both dimensions: hw (half-width) and nl/2 (half-length)
        # Using min() ensures non-square footprints don't overflow in the short dimension
        max_radius = min(hw, nl / 2)
        radius = max(max_radius, self._get_polygon_min_radius(self.shell_sides))

        # Center the polygon at the room center
        cx = ox
        cy = oy + nl / 2

        # Determine portal segment
        portal_segment = self._find_portal_segment(self.shell_sides, "SOUTH") if self.has_entrance else -1

        # Compute vestibule clip zone to avoid floor/ceiling overlap
        vestibule_clip = self._compute_vestibule_clip_zone(
            cx, cy, radius, self.shell_sides, portal_segment,
            PORTAL_WIDTH, t,
            room_origin=(ox, oy, oz), room_length=nl, room_width=hw * 2
        )

        # Generate floor (with clip zone to avoid vestibule overlap)
        brushes.extend(self._generate_polygonal_floor(
            cx, cy, oz, radius, self.shell_sides, t,
            is_ceiling=False,
            vestibule_clip_zone=vestibule_clip
        ))

        # Generate ceiling (with clip zone to avoid vestibule overlap)
        brushes.extend(self._generate_polygonal_floor(
            cx, cy, oz + nh, radius, self.shell_sides, t,
            is_ceiling=True,
            vestibule_clip_zone=vestibule_clip
        ))

        # Set portal target for grid alignment before generating walls
        self._set_portal_target_for_polygonal(cx, cy, radius, self.shell_sides, portal_segment)

        # Generate walls
        brushes.extend(self._generate_polygonal_walls(
            cx, cy, oz, oz + nh, radius, self.shell_sides, t,
            portal_segment=portal_segment,
            portal_width=PORTAL_WIDTH,
            portal_height=PORTAL_HEIGHT,
            portal_target_x=getattr(self, '_portal_target_x', None),
            portal_target_y=getattr(self, '_portal_target_y', None)
        ))

        # Generate connector corridor from portal to grid center
        # Pass room origin and dimensions for accurate footprint edge calculation
        if self.has_entrance:
            brushes.extend(self._generate_polygonal_portal_connector(
                cx, cy, oz, radius, self.shell_sides, portal_segment,
                PORTAL_WIDTH, PORTAL_HEIGHT, t, room_height=nh,
                room_origin=(ox, oy, oz), room_length=nl, room_width=hw * 2
            ))

        return brushes


class Pit(GeometricPrimitive):
    """Vertical hazard room with deep central pit.

    Features:
    - Deep central pit extending below floor
    - Narrow walkway around edge
    - Optional bridges across
    - 100% sealed geometry

    Unique geometry with negative space extending downward.
    """

    width: float = 128.0        # Half-width
    length: float = 128.0       # Length
    height: float = 96.0        # Ceiling height above floor
    shell_sides: int = 4        # Number of sides (4=square, 6=hex, 8=octagon)
    pit_depth: float = 96.0     # Depth of pit below floor
    walkway_width: float = 32.0  # Width of walkway around pit
    has_bridge: bool = False    # Bridge across pit
    random_seed: int = 0

    # Portal control
    has_entrance: bool = True
    _entrance_x_offset: float = 0.0

    WALL_THICKNESS: float = 16.0

    @classmethod
    def get_display_name(cls) -> str:
        return "Pit"

    @classmethod
    def get_category(cls) -> str:
        return "Rooms"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "width": {
                "type": "float", "default": 128.0, "min": 64, "max": 192, "label": "Half-Width",
                "description": "Half-width of the pit room from center to walls"
            },
            "length": {
                "type": "float", "default": 128.0, "min": 96, "max": 192, "label": "Length",
                "description": "Total length from entrance to back wall"
            },
            "height": {
                "type": "float", "default": 96.0, "min": 64, "max": 128, "label": "Ceiling Height",
                "description": "Height from walkway floor to ceiling above"
            },
            "shell_sides": {
                "type": "int",
                "default": 4,
                "min": 3,
                "max": 16,
                "label": "Shell Sides",
                "description": "Number of sides for room shape (4=square, 6=hex, 8=octagon)"
            },
            "pit_depth": {
                "type": "float", "default": 96.0, "min": 48, "max": 192, "label": "Pit Depth",
                "description": "How far the central pit drops below walkway level (lethal depth)"
            },
            "walkway_width": {
                "type": "float", "default": 32.0, "min": 24, "max": 64, "label": "Walkway Width",
                "description": "Width of safe perimeter ledge around the pit (player is 32 wide)"
            },
            "has_bridge": {
                "type": "bool", "default": False, "label": "Bridge",
                "description": "Add a narrow bridge spanning the pit"
            },
            "has_entrance": {
                "type": "bool", "default": True, "label": "Enable Entrance Portal",
                "description": "Create an opening in the front wall for entry"
            },
            "random_seed": {
                "type": "int", "default": 0, "min": 0, "max": 999999, "label": "Random Seed",
                "description": "Seed for deterministic generation (0 = random)"
            },
        }

    def generate(self) -> List[Brush]:
        """Generate pit geometry."""
        self._reset_tags()  # Reset tags for fresh generation
        ox, oy, oz = self.params.origin
        brushes: List[Brush] = []
        t = self.WALL_THICKNESS

        hw = self.width
        nl = self.length
        nh = self.height
        pd = self.pit_depth
        ww = self.walkway_width

        # Use polygonal shell if sides != 4
        if self.shell_sides != 4:
            brushes = self._generate_polygonal_pit(ox, oy, oz, hw, nl, nh, t)
            # Register portal tag at footprint edge (grid-aligned)
            if self.has_entrance:
                self._register_portal_tag(
                    portal_id="entrance",
                    center_x=ox,  # Room center X for grid alignment
                    center_y=oy,  # Footprint south edge (Y=origin)
                    center_z=oz,
                    direction=PortalDirection.SOUTH,
                )
            return brushes

        # Pit dimensions (inner edges)
        pit_x1 = ox - hw + ww
        pit_x2 = ox + hw - ww
        pit_y1 = oy + ww
        pit_y2 = oy + nl - ww

        # === WALKWAY FLOOR (outer ring) ===
        # Front walkway
        brushes.append(self._box(
            ox - hw - t, oy - t, oz - t,
            ox + hw + t, pit_y1, oz
        ))
        # Back walkway
        brushes.append(self._box(
            ox - hw - t, pit_y2, oz - t,
            ox + hw + t, oy + nl + t, oz
        ))
        # Left walkway
        brushes.append(self._box(
            ox - hw - t, pit_y1, oz - t,
            pit_x1, pit_y2, oz
        ))
        # Right walkway
        brushes.append(self._box(
            pit_x2, pit_y1, oz - t,
            ox + hw + t, pit_y2, oz
        ))

        # === PIT BOTTOM FLOOR ===
        brushes.append(self._box(
            pit_x1, pit_y1, oz - pd - t,
            pit_x2, pit_y2, oz - pd
        ))

        # === PIT WALLS (vertical walls of the pit) ===
        brushes.append(self._box(
            pit_x1 - t, pit_y1, oz - pd,
            pit_x1, pit_y2, oz
        ))
        brushes.append(self._box(
            pit_x2, pit_y1, oz - pd,
            pit_x2 + t, pit_y2, oz
        ))
        brushes.append(self._box(
            pit_x1, pit_y1 - t, oz - pd,
            pit_x2, pit_y1, oz
        ))
        brushes.append(self._box(
            pit_x1, pit_y2, oz - pd,
            pit_x2, pit_y2 + t, oz
        ))

        # === CEILING ===
        brushes.append(self._box(
            ox - hw - t, oy - t, oz + nh,
            ox + hw + t, oy + nl + t, oz + nh + t
        ))

        # === OUTER WALLS ===
        brushes.append(self._box(
            ox - hw - t, oy - t, oz,
            ox - hw, oy + nl + t, oz + nh
        ))
        brushes.append(self._box(
            ox + hw, oy - t, oz,
            ox + hw + t, oy + nl + t, oz + nh
        ))
        brushes.append(self._box(
            ox - hw - t, oy + nl, oz,
            ox + hw + t, oy + nl + t, oz + nh
        ))
        # Front wall
        if self.has_entrance:
            pw, ph = PORTAL_WIDTH, PORTAL_HEIGHT
            entrance_x = ox + self._entrance_x_offset
            brushes.append(self._box(
                ox - hw - t, oy - t, oz,
                entrance_x - pw / 2, oy, oz + nh
            ))
            brushes.append(self._box(
                entrance_x + pw / 2, oy - t, oz,
                ox + hw + t, oy, oz + nh
            ))
            brushes.append(self._box(
                entrance_x - pw / 2, oy - t, oz + ph,
                entrance_x + pw / 2, oy, oz + nh
            ))
        else:
            brushes.append(self._box(
                ox - hw - t, oy - t, oz,
                ox + hw + t, oy, oz + nh
            ))

        # === OPTIONAL BRIDGE ===
        if self.has_bridge:
            bridge_w = 24.0
            bridge_h = 8.0
            brushes.append(self._box(
                ox - bridge_w, pit_y1, oz,
                ox + bridge_w, pit_y2, oz + bridge_h
            ))

        # Register portal tag for rectangular pit
        if self.has_entrance:
            entrance_x = ox + self._entrance_x_offset
            self._register_portal_tag(
                portal_id="entrance",
                center_x=entrance_x,
                center_y=oy,
                center_z=oz,
                direction=PortalDirection.SOUTH,
            )

        return brushes

    def _generate_polygonal_pit(
        self, ox: float, oy: float, oz: float,
        hw: float, nl: float, nh: float, t: float
    ) -> List[Brush]:
        """Generate polygonal Pit shell (simplified for non-rectangular)."""
        brushes: List[Brush] = []

        # Calculate radius from room dimensions, clamped to fit footprint
        # The polygon must fit within both dimensions: hw (half-width) and nl/2 (half-length)
        # Using min() ensures non-square footprints don't overflow in the short dimension
        max_radius = min(hw, nl / 2)
        radius = max(max_radius, self._get_polygon_min_radius(self.shell_sides))

        # Center the polygon at the room center
        cx = ox
        cy = oy + nl / 2

        # Determine portal segment
        portal_segment = self._find_portal_segment(self.shell_sides, "SOUTH") if self.has_entrance else -1

        # Compute vestibule clip zone to avoid floor/ceiling overlap
        vestibule_clip = self._compute_vestibule_clip_zone(
            cx, cy, radius, self.shell_sides, portal_segment,
            PORTAL_WIDTH, t,
            room_origin=(ox, oy, oz), room_length=nl, room_width=hw * 2
        )

        # Generate floor (with clip zone to avoid vestibule overlap)
        brushes.extend(self._generate_polygonal_floor(
            cx, cy, oz, radius, self.shell_sides, t,
            is_ceiling=False,
            vestibule_clip_zone=vestibule_clip
        ))

        # Generate ceiling (with clip zone to avoid vestibule overlap)
        brushes.extend(self._generate_polygonal_floor(
            cx, cy, oz + nh, radius, self.shell_sides, t,
            is_ceiling=True,
            vestibule_clip_zone=vestibule_clip
        ))

        # Set portal target for grid alignment before generating walls
        self._set_portal_target_for_polygonal(cx, cy, radius, self.shell_sides, portal_segment)

        # Generate walls
        brushes.extend(self._generate_polygonal_walls(
            cx, cy, oz, oz + nh, radius, self.shell_sides, t,
            portal_segment=portal_segment,
            portal_width=PORTAL_WIDTH,
            portal_height=PORTAL_HEIGHT,
            portal_target_x=getattr(self, '_portal_target_x', None),
            portal_target_y=getattr(self, '_portal_target_y', None)
        ))

        # Generate connector corridor from portal to grid center
        # Pass room origin and dimensions for accurate footprint edge calculation
        if self.has_entrance:
            brushes.extend(self._generate_polygonal_portal_connector(
                cx, cy, oz, radius, self.shell_sides, portal_segment,
                PORTAL_WIDTH, PORTAL_HEIGHT, t, room_height=nh,
                room_origin=(ox, oy, oz), room_length=nl, room_width=hw * 2
            ))

        return brushes


class Antechamber(GeometricPrimitive):
    """Transitional room serving as buffer or waiting area.

    Features:
    - Simple rectangular with multiple portal options
    - Optional pillars flanking entrance
    - Serves as hub or buffer room
    - 2-4 portal positions available
    - 100% sealed geometry

    Multi-portal hub (smaller than Crossroads hall).
    """

    width: float = 128.0        # Half-width
    length: float = 128.0       # Length
    height: float = 112.0       # Ceiling height
    shell_sides: int = 4        # Number of sides (4=square, 6=hex, 8=octagon)
    portal_count: int = 2       # Number of active portals (2-4)
    has_pillars: bool = True    # Pillars flanking entrance
    random_seed: int = 0

    # Portal control - up to 4 portals
    has_entrance: bool = True   # South (front)
    has_exit: bool = False      # North (back)
    has_side_east: bool = False # East
    has_side_west: bool = False # West

    _entrance_x_offset: float = 0.0

    WALL_THICKNESS: float = 16.0

    @classmethod
    def get_display_name(cls) -> str:
        return "Antechamber"

    @classmethod
    def get_category(cls) -> str:
        return "Rooms"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "width": {
                "type": "float", "default": 128.0, "min": 64, "max": 192, "label": "Half-Width",
                "description": "Half-width of the antechamber from center to walls"
            },
            "length": {
                "type": "float", "default": 128.0, "min": 96, "max": 192, "label": "Length",
                "description": "Total length from south to north wall"
            },
            "height": {
                "type": "float", "default": 112.0, "min": 80, "max": 144, "label": "Ceiling Height",
                "description": "Height from floor to ceiling (player is 56 units tall)"
            },
            "shell_sides": {
                "type": "int",
                "default": 4,
                "min": 3,
                "max": 16,
                "label": "Shell Sides",
                "description": "Number of sides for room shape (4=square, 6=hex, 8=octagon)"
            },
            "has_pillars": {
                "type": "bool", "default": True, "label": "Entrance Pillars",
                "description": "Add decorative pillars flanking the south entrance"
            },
            "has_entrance": {
                "type": "bool", "default": True, "label": "South Portal",
                "description": "Create portal opening in the south (front) wall"
            },
            "has_exit": {
                "type": "bool", "default": False, "label": "North Portal",
                "description": "Create portal opening in the north (back) wall"
            },
            "has_side_east": {
                "type": "bool", "default": False, "label": "East Portal",
                "description": "Create portal opening in the east (right) wall"
            },
            "has_side_west": {
                "type": "bool", "default": False, "label": "West Portal",
                "description": "Create portal opening in the west (left) wall"
            },
            "random_seed": {
                "type": "int", "default": 0, "min": 0, "max": 999999, "label": "Random Seed",
                "description": "Seed for deterministic generation (0 = random)"
            },
        }

    def generate(self) -> List[Brush]:
        """Generate antechamber geometry."""
        self._reset_tags()  # Reset tags for fresh generation
        ox, oy, oz = self.params.origin
        brushes: List[Brush] = []
        t = self.WALL_THICKNESS
        pw, ph = PORTAL_WIDTH, PORTAL_HEIGHT

        hw = self.width
        nl = self.length
        nh = self.height

        # Use polygonal shell if sides != 4
        if self.shell_sides != 4:
            brushes = self._generate_polygonal_antechamber(ox, oy, oz, hw, nl, nh, t)
            # Register portal tags for ALL enabled portals at grid-boundary positions
            # Portal positions are at footprint edges (not polygon interior)
            # This matches the rectangular implementation and ensures proper alignment
            if self.has_entrance:
                self._register_portal_tag(
                    portal_id="entrance",
                    center_x=ox,  # Room center X for grid alignment
                    center_y=oy,  # Footprint south edge (Y=origin)
                    center_z=oz,
                    direction=PortalDirection.SOUTH,
                )
            if self.has_exit:
                self._register_portal_tag(
                    portal_id="exit",
                    center_x=ox,  # Room center X
                    center_y=oy + nl,  # Footprint north edge
                    center_z=oz,
                    direction=PortalDirection.NORTH,
                )
            if self.has_side_east:
                self._register_portal_tag(
                    portal_id="side_east",
                    center_x=ox + hw,  # Footprint east edge
                    center_y=oy + nl / 2,  # Room center Y
                    center_z=oz,
                    direction=PortalDirection.EAST,
                )
            if self.has_side_west:
                self._register_portal_tag(
                    portal_id="side_west",
                    center_x=ox - hw,  # Footprint west edge
                    center_y=oy + nl / 2,  # Room center Y
                    center_z=oz,
                    direction=PortalDirection.WEST,
                )
            return brushes

        # === FLOOR ===
        brushes.append(self._floor_box(
            ox - hw - t, oy - t, oz - t,
            ox + hw + t, oy + nl + t, oz
        ))

        # === CEILING ===
        brushes.append(self._ceiling_box(
            ox - hw - t, oy - t, oz + nh,
            ox + hw + t, oy + nl + t, oz + nh + t
        ))

        # === WALLS WITH PORTAL OPENINGS ===

        # Front (South) wall
        if self.has_entrance:
            entrance_x = ox + self._entrance_x_offset
            brushes.append(self._wall_box(
                ox - hw - t, oy - t, oz,
                entrance_x - pw / 2, oy, oz + nh
            ))
            brushes.append(self._wall_box(
                entrance_x + pw / 2, oy - t, oz,
                ox + hw + t, oy, oz + nh
            ))
            brushes.append(self._wall_box(
                entrance_x - pw / 2, oy - t, oz + ph,
                entrance_x + pw / 2, oy, oz + nh
            ))
        else:
            brushes.append(self._wall_box(
                ox - hw - t, oy - t, oz,
                ox + hw + t, oy, oz + nh
            ))

        # Back (North) wall
        if self.has_exit:
            exit_x = ox
            brushes.append(self._wall_box(
                ox - hw - t, oy + nl, oz,
                exit_x - pw / 2, oy + nl + t, oz + nh
            ))
            brushes.append(self._wall_box(
                exit_x + pw / 2, oy + nl, oz,
                ox + hw + t, oy + nl + t, oz + nh
            ))
            brushes.append(self._wall_box(
                exit_x - pw / 2, oy + nl, oz + ph,
                exit_x + pw / 2, oy + nl + t, oz + nh
            ))
        else:
            brushes.append(self._wall_box(
                ox - hw - t, oy + nl, oz,
                ox + hw + t, oy + nl + t, oz + nh
            ))

        # Left (West) wall
        if self.has_side_west:
            side_y = oy + nl / 2
            brushes.append(self._wall_box(
                ox - hw - t, oy - t, oz,
                ox - hw, side_y - pw / 2, oz + nh
            ))
            brushes.append(self._wall_box(
                ox - hw - t, side_y + pw / 2, oz,
                ox - hw, oy + nl + t, oz + nh
            ))
            brushes.append(self._wall_box(
                ox - hw - t, side_y - pw / 2, oz + ph,
                ox - hw, side_y + pw / 2, oz + nh
            ))
        else:
            brushes.append(self._wall_box(
                ox - hw - t, oy - t, oz,
                ox - hw, oy + nl + t, oz + nh
            ))

        # Right (East) wall
        if self.has_side_east:
            side_y = oy + nl / 2
            brushes.append(self._wall_box(
                ox + hw, oy - t, oz,
                ox + hw + t, side_y - pw / 2, oz + nh
            ))
            brushes.append(self._wall_box(
                ox + hw, side_y + pw / 2, oz,
                ox + hw + t, oy + nl + t, oz + nh
            ))
            brushes.append(self._wall_box(
                ox + hw, side_y - pw / 2, oz + ph,
                ox + hw + t, side_y + pw / 2, oz + nh
            ))
        else:
            brushes.append(self._wall_box(
                ox + hw, oy - t, oz,
                ox + hw + t, oy + nl + t, oz + nh
            ))

        # === OPTIONAL PILLARS ===
        if self.has_pillars and self.has_entrance:
            pillar_r = 12.0
            entrance_x = ox + self._entrance_x_offset
            # Left pillar
            brushes.append(self._structural_box(
                entrance_x - pw / 2 - pillar_r * 2 - 8, oy + 8, oz,
                entrance_x - pw / 2 - 8, oy + 8 + pillar_r * 2, oz + nh - 8
            ))
            # Right pillar
            brushes.append(self._structural_box(
                entrance_x + pw / 2 + 8, oy + 8, oz,
                entrance_x + pw / 2 + pillar_r * 2 + 8, oy + 8 + pillar_r * 2, oz + nh - 8
            ))

        # Register portal tags for rectangular antechamber (multi-portal room)
        if self.has_entrance:
            entrance_x = ox + self._entrance_x_offset
            self._register_portal_tag(
                portal_id="entrance",
                center_x=entrance_x,
                center_y=oy,
                center_z=oz,
                direction=PortalDirection.SOUTH,
            )
        if self.has_exit:
            self._register_portal_tag(
                portal_id="exit",
                center_x=ox,
                center_y=oy + nl,
                center_z=oz,
                direction=PortalDirection.NORTH,
            )
        if self.has_side_east:
            self._register_portal_tag(
                portal_id="side_east",
                center_x=ox + hw,
                center_y=oy + nl / 2,
                center_z=oz,
                direction=PortalDirection.EAST,
            )
        if self.has_side_west:
            self._register_portal_tag(
                portal_id="side_west",
                center_x=ox - hw,
                center_y=oy + nl / 2,
                center_z=oz,
                direction=PortalDirection.WEST,
            )

        return brushes

    def _generate_polygonal_antechamber(
        self, ox: float, oy: float, oz: float,
        hw: float, nl: float, nh: float, t: float
    ) -> List[Brush]:
        """Generate polygonal Antechamber shell (simplified for non-rectangular).

        Multi-portal support: All 4 portals (entrance/SOUTH, exit/NORTH, side_east/EAST,
        side_west/WEST) are generated when enabled, using the vestibule approach for
        grid-aligned connections.
        """
        brushes: List[Brush] = []

        # Calculate radius from room dimensions, clamped to fit footprint
        # The polygon must fit within both dimensions: hw (half-width) and nl/2 (half-length)
        # Using min() ensures non-square footprints don't overflow in the short dimension
        max_radius = min(hw, nl / 2)
        radius = max(max_radius, self._get_polygon_min_radius(self.shell_sides))

        # Center the polygon at the room center
        cx = ox
        cy = oy + nl / 2

        # Determine portal segments for all 4 directions
        entrance_segment = self._find_portal_segment(self.shell_sides, "SOUTH") if self.has_entrance else -1
        exit_segment = self._find_portal_segment(self.shell_sides, "NORTH") if self.has_exit else -1
        east_segment = self._find_portal_segment(self.shell_sides, "EAST") if self.has_side_east else -1
        west_segment = self._find_portal_segment(self.shell_sides, "WEST") if self.has_side_west else -1

        # Compute vestibule clip zones for all active portals
        room_origin = (ox, oy, oz)
        vestibule_clips = []
        for seg in [entrance_segment, exit_segment, east_segment, west_segment]:
            if seg >= 0:
                clip = self._compute_vestibule_clip_zone(
                    cx, cy, radius, self.shell_sides, seg,
                    PORTAL_WIDTH, t,
                    room_origin=room_origin, room_length=nl, room_width=hw * 2
                )
                if clip:
                    vestibule_clips.append(clip)

        # Generate floor (with clip zones to avoid vestibule overlap)
        brushes.extend(self._generate_polygonal_floor(
            cx, cy, oz, radius, self.shell_sides, t,
            is_ceiling=False,
            vestibule_clip_zones=vestibule_clips if vestibule_clips else None
        ))

        # Generate ceiling (with clip zones to avoid vestibule overlap)
        brushes.extend(self._generate_polygonal_floor(
            cx, cy, oz + nh, radius, self.shell_sides, t,
            is_ceiling=True,
            vestibule_clip_zones=vestibule_clips if vestibule_clips else None
        ))

        # Build portal segments list for multi-portal wall generation
        # Use a set to track used segments (handles low shell_sides where multiple
        # directions might map to the same segment)
        portal_segments = []
        used_segments = set()

        # Process portals in priority order: SOUTH, NORTH, EAST, WEST
        portal_configs = [
            (entrance_segment, 'SOUTH', self.has_entrance),
            (exit_segment, 'NORTH', self.has_exit),
            (east_segment, 'EAST', self.has_side_east),
            (west_segment, 'WEST', self.has_side_west),
        ]

        for seg, direction, enabled in portal_configs:
            if enabled and seg >= 0 and seg not in used_segments:
                # Set portal target for this portal (grid-aligned at room center)
                self._set_portal_target_for_polygonal(cx, cy, radius, self.shell_sides, seg)
                portal_segments.append({
                    'segment': seg,
                    'width': PORTAL_WIDTH,
                    'height': PORTAL_HEIGHT,
                    'target_x': getattr(self, '_portal_target_x', None),
                    'target_y': getattr(self, '_portal_target_y', None),
                    'direction': direction
                })
                used_segments.add(seg)

        # Generate walls with ALL portal openings
        brushes.extend(self._generate_polygonal_walls(
            cx, cy, oz, oz + nh, radius, self.shell_sides, t,
            portal_segments=portal_segments if portal_segments else None
        ))

        # Generate vestibule corridor for EACH enabled portal
        # Pass room origin and dimensions for accurate footprint edge calculation
        if self.has_entrance and entrance_segment >= 0:
            brushes.extend(self._generate_polygonal_portal_connector(
                cx, cy, oz, radius, self.shell_sides, entrance_segment,
                PORTAL_WIDTH, PORTAL_HEIGHT, t, room_height=nh,
                room_origin=(ox, oy, oz), room_length=nl, room_width=hw * 2
            ))

        if self.has_exit and exit_segment >= 0 and exit_segment not in {entrance_segment}:
            brushes.extend(self._generate_polygonal_portal_connector(
                cx, cy, oz, radius, self.shell_sides, exit_segment,
                PORTAL_WIDTH, PORTAL_HEIGHT, t, room_height=nh,
                room_origin=(ox, oy, oz), room_length=nl, room_width=hw * 2
            ))

        if self.has_side_east and east_segment >= 0 and east_segment not in {entrance_segment, exit_segment}:
            brushes.extend(self._generate_polygonal_portal_connector(
                cx, cy, oz, radius, self.shell_sides, east_segment,
                PORTAL_WIDTH, PORTAL_HEIGHT, t, room_height=nh,
                room_origin=(ox, oy, oz), room_length=nl, room_width=hw * 2
            ))

        if self.has_side_west and west_segment >= 0 and west_segment not in {entrance_segment, exit_segment, east_segment}:
            brushes.extend(self._generate_polygonal_portal_connector(
                cx, cy, oz, radius, self.shell_sides, west_segment,
                PORTAL_WIDTH, PORTAL_HEIGHT, t, room_height=nh,
                room_origin=(ox, oy, oz), room_length=nl, room_width=hw * 2
            ))

        return brushes


class SecretChamber(GeometricPrimitive):
    """A sealed room with a hidden walk-through wall using CLIP texture.

    Features:
    - Standard entrance portal (south wall)
    - One wall uses CLIP texture for secret walk-through access
    - 100% sealed geometry (only entrance portal is open)
    - Secret direction configurable: north, east, or west
    - Geometry-only: no entities, triggers, or doors

    The CLIP texture creates a walk-through wall in idTech engines.
    Users connect an adjacent room/hall to the secret wall exterior
    to create a functional secret area.
    """

    width: float = 128.0        # Half-width
    length: float = 256.0       # Length
    height: float = 128.0       # Ceiling height
    shell_sides: int = 4        # Number of sides (4=square, 6=hex, 8=octagon)
    secret_direction: str = "north"  # Which wall is the secret: north, east, west
    random_seed: int = 0

    # Portal control
    has_entrance: bool = True

    WALL_THICKNESS: float = 16.0
    CLIP_TEXTURE: str = "CLIP"  # Walk-through texture

    @classmethod
    def get_display_name(cls) -> str:
        return "Secret Chamber"

    @classmethod
    def get_category(cls) -> str:
        return "Rooms"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "width": {
                "type": "float", "default": 128.0, "min": 64, "max": 256, "label": "Half-Width",
                "description": "Half-width of the chamber"
            },
            "length": {
                "type": "float", "default": 256.0, "min": 128, "max": 384, "label": "Length",
                "description": "Total length from entrance to back wall"
            },
            "height": {
                "type": "float", "default": 128.0, "min": 96, "max": 192, "label": "Ceiling Height",
                "description": "Height from floor to ceiling"
            },
            "shell_sides": {
                "type": "int",
                "default": 4,
                "min": 3,
                "max": 16,
                "label": "Shell Sides",
                "description": "Number of sides for room shape (4=square, 6=hex, 8=octagon)"
            },
            "secret_direction": {
                "type": "choice",
                "default": "north",
                "choices": ["north", "east", "west"],
                "label": "Secret Wall",
                "description": "Which wall has the CLIP texture for walk-through (entrance is always south)"
            },
            "has_entrance": {
                "type": "bool", "default": True, "label": "Enable Entrance Portal",
                "description": "Enable the main entrance portal on south wall"
            },
            "random_seed": {
                "type": "int", "default": 0, "min": 0, "max": 999999, "label": "Random Seed",
                "description": "Seed for deterministic generation (0 = random)"
            },
        }

    def generate(self) -> List[Brush]:
        """Generate secret chamber geometry with CLIP wall."""
        self._reset_tags()  # Reset tags for fresh generation
        ox, oy, oz = self.params.origin
        brushes: List[Brush] = []
        t = self.WALL_THICKNESS

        hw = self.width
        nl = self.length
        nh = self.height

        # Use polygonal shell if sides != 4
        if self.shell_sides != 4:
            brushes = self._generate_polygonal_secret_chamber(ox, oy, oz, hw, nl, nh, t)
            if self.has_entrance:
                # Register portal tag at footprint edge (grid-aligned)
                # Not at polygon interior - matches rectangular implementation
                self._register_portal_tag(
                    portal_id="entrance",
                    center_x=ox,  # Room center X for grid alignment
                    center_y=oy,  # Footprint south edge (Y=origin)
                    center_z=oz,
                    direction=PortalDirection.SOUTH,
                )
            return brushes

        # === FLOOR (extends by t in all directions) ===
        brushes.append(self._box(
            ox - hw - t, oy - t, oz - t,
            ox + hw + t, oy + nl + t, oz
        ))

        # === CEILING (extends by t in all directions) ===
        brushes.append(self._box(
            ox - hw - t, oy - t, oz + nh,
            ox + hw + t, oy + nl + t, oz + nh + t
        ))

        # === WALLS ===
        # Determine which wall gets the CLIP texture
        secret_dir = self.secret_direction.lower()

        # LEFT (WEST) WALL
        if secret_dir == "west":
            brushes.append(self._box(
                ox - hw - t, oy - t, oz,
                ox - hw, oy + nl + t, oz + nh,
                texture=self.CLIP_TEXTURE
            ))
        else:
            brushes.append(self._box(
                ox - hw - t, oy - t, oz,
                ox - hw, oy + nl + t, oz + nh
            ))

        # RIGHT (EAST) WALL
        if secret_dir == "east":
            brushes.append(self._box(
                ox + hw, oy - t, oz,
                ox + hw + t, oy + nl + t, oz + nh,
                texture=self.CLIP_TEXTURE
            ))
        else:
            brushes.append(self._box(
                ox + hw, oy - t, oz,
                ox + hw + t, oy + nl + t, oz + nh
            ))

        # BACK (NORTH) WALL
        if secret_dir == "north":
            brushes.append(self._box(
                ox - hw - t, oy + nl, oz,
                ox + hw + t, oy + nl + t, oz + nh,
                texture=self.CLIP_TEXTURE
            ))
        else:
            brushes.append(self._box(
                ox - hw - t, oy + nl, oz,
                ox + hw + t, oy + nl + t, oz + nh
            ))

        # FRONT (SOUTH) WALL with portal
        if self.has_entrance:
            pw = PORTAL_WIDTH
            ph = PORTAL_HEIGHT
            entrance_x = ox  # Entrance at center

            # Left portion of front wall
            brushes.append(self._box(
                ox - hw - t, oy - t, oz,
                entrance_x - pw / 2, oy, oz + nh
            ))
            # Right portion of front wall
            brushes.append(self._box(
                entrance_x + pw / 2, oy - t, oz,
                ox + hw + t, oy, oz + nh
            ))
            # Lintel above portal
            brushes.append(self._box(
                entrance_x - pw / 2, oy - t, oz + ph,
                entrance_x + pw / 2, oy, oz + nh
            ))
        else:
            # Solid front wall
            brushes.append(self._box(
                ox - hw - t, oy - t, oz,
                ox + hw + t, oy, oz + nh
            ))

        # Register portal tag for rectangular secret chamber
        if self.has_entrance:
            self._register_portal_tag(
                portal_id="entrance",
                center_x=ox,
                center_y=oy,
                center_z=oz,
                direction=PortalDirection.SOUTH,
            )

        return brushes

    def _generate_polygonal_secret_chamber(
        self, ox: float, oy: float, oz: float,
        hw: float, nl: float, nh: float, t: float
    ) -> List[Brush]:
        """Generate polygonal SecretChamber shell (simplified for non-rectangular).

        Note: In polygonal mode, secret wall direction is not applied.
        All walls use the default texture. This is a limitation of polygonal mode.
        """
        brushes: List[Brush] = []

        # Calculate radius from room dimensions, clamped to fit footprint
        # The polygon must fit within both dimensions: hw (half-width) and nl/2 (half-length)
        # Using min() ensures non-square footprints don't overflow in the short dimension
        max_radius = min(hw, nl / 2)
        radius = max(max_radius, self._get_polygon_min_radius(self.shell_sides))

        # Center the polygon at the room center
        cx = ox
        cy = oy + nl / 2

        # Determine portal segment
        portal_segment = self._find_portal_segment(self.shell_sides, "SOUTH") if self.has_entrance else -1

        # Compute vestibule clip zone to avoid floor/ceiling overlap
        vestibule_clip = self._compute_vestibule_clip_zone(
            cx, cy, radius, self.shell_sides, portal_segment,
            PORTAL_WIDTH, t,
            room_origin=(ox, oy, oz), room_length=nl, room_width=hw * 2
        )

        # Generate floor (with clip zone to avoid vestibule overlap)
        brushes.extend(self._generate_polygonal_floor(
            cx, cy, oz, radius, self.shell_sides, t,
            is_ceiling=False,
            vestibule_clip_zone=vestibule_clip
        ))

        # Generate ceiling (with clip zone to avoid vestibule overlap)
        brushes.extend(self._generate_polygonal_floor(
            cx, cy, oz + nh, radius, self.shell_sides, t,
            is_ceiling=True,
            vestibule_clip_zone=vestibule_clip
        ))

        # Set portal target for grid alignment before generating walls
        self._set_portal_target_for_polygonal(cx, cy, radius, self.shell_sides, portal_segment)

        # Generate walls
        brushes.extend(self._generate_polygonal_walls(
            cx, cy, oz, oz + nh, radius, self.shell_sides, t,
            portal_segment=portal_segment,
            portal_width=PORTAL_WIDTH,
            portal_height=PORTAL_HEIGHT,
            portal_target_x=getattr(self, '_portal_target_x', None),
            portal_target_y=getattr(self, '_portal_target_y', None)
        ))

        # Generate connector corridor from portal to grid center
        # Pass room origin and dimensions for accurate footprint edge calculation
        if self.has_entrance:
            brushes.extend(self._generate_polygonal_portal_connector(
                cx, cy, oz, radius, self.shell_sides, portal_segment,
                PORTAL_WIDTH, PORTAL_HEIGHT, t, room_height=nh,
                room_origin=(ox, oy, oz), room_length=nl, room_width=hw * 2
            ))

        return brushes
