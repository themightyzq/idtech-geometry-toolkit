"""
Architectural detail brush generators.

All functions return List[Brush].  All coordinates are integers,
all geometry is axis-aligned or 45-degree angles only.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from quake_levelgenerator.src.conversion.map_writer import Brush, Plane

# Reuse the box helper from the primitives base.  We duplicate a tiny
# standalone version here to avoid coupling the detail system to the
# primitive class hierarchy.

_brush_id_counter = 0


def _next_id() -> int:
    global _brush_id_counter
    _brush_id_counter += 1
    return _brush_id_counter


def _box(x1: float, y1: float, z1: float,
         x2: float, y2: float, z2: float,
         texture: str = "CRATE1_5") -> Optional[Brush]:
    """Axis-aligned box brush (6 planes). Returns None if degenerate."""
    # Guard against zero-volume brushes which produce invalid planes
    if x1 >= x2 or y1 >= y2 or z1 >= z2:
        return None
    planes = [
        Plane((x1, y1, z1), (x1, y2, z1), (x1, y1, z2), texture),
        Plane((x2, y1, z1), (x2, y1, z2), (x2, y2, z1), texture),
        Plane((x1, y1, z1), (x1, y1, z2), (x2, y1, z1), texture),
        Plane((x1, y2, z1), (x2, y2, z1), (x1, y2, z2), texture),
        Plane((x1, y1, z1), (x2, y1, z1), (x1, y2, z1), texture),
        Plane((x1, y1, z2), (x1, y2, z2), (x2, y1, z2), texture),
    ]
    return Brush(planes=planes, brush_id=_next_id())


# ---------------------------------------------------------------------------
# Door frame trim
# ---------------------------------------------------------------------------

def door_frame_trim(
    x1: int, y1: int, x2: int, y2: int,
    floor_z: int, top_z: int,
    axis: str = "x",
    thickness: int = 8,
    depth: int = 8,
    texture: str = "METAL1_1",
) -> List[Brush]:
    """Border geometry around a doorway opening.

    axis: "x" means the opening spans the X-axis (wall faces Y direction).
          "y" means the opening spans the Y-axis (wall faces X direction).

    Returns up to 3 brushes: left jamb, right jamb, lintel.
    """
    brushes: List[Brush] = []
    if axis == "x":
        # Left jamb
        brushes.append(_box(x1 - thickness, y1, floor_z,
                            x1, y1 + depth, top_z, texture))
        # Right jamb
        brushes.append(_box(x2, y1, floor_z,
                            x2 + thickness, y1 + depth, top_z, texture))
        # Lintel (top bar)
        brushes.append(_box(x1 - thickness, y1, top_z,
                            x2 + thickness, y1 + depth, top_z + thickness, texture))
    else:
        # Left jamb
        brushes.append(_box(x1, y1 - thickness, floor_z,
                            x1 + depth, y1, top_z, texture))
        # Right jamb
        brushes.append(_box(x1, y2, floor_z,
                            x1 + depth, y2 + thickness, top_z, texture))
        # Lintel
        brushes.append(_box(x1, y1 - thickness, top_z,
                            x1 + depth, y2 + thickness, top_z + thickness, texture))
    return brushes


# ---------------------------------------------------------------------------
# Floor trim strips
# ---------------------------------------------------------------------------

def floor_trim_strips(
    x1: int, y1: int, x2: int, y2: int,
    floor_z: int,
    height: int = 16,
    width: int = 8,
    texture: str = "METAL1_1",
) -> List[Brush]:
    """Raised trim along floor-wall junctions (4 strips around perimeter)."""
    top = floor_z + height
    return [
        # North wall strip
        _box(x1, y2 - width, floor_z, x2, y2, top, texture),
        # South wall strip
        _box(x1, y1, floor_z, x2, y1 + width, top, texture),
        # West wall strip
        _box(x1, y1 + width, floor_z, x1 + width, y2 - width, top, texture),
        # East wall strip
        _box(x2 - width, y1 + width, floor_z, x2, y2 - width, top, texture),
    ]


# ---------------------------------------------------------------------------
# Ceiling beams
# ---------------------------------------------------------------------------

def ceiling_beams(
    x1: int, y1: int, x2: int, y2: int,
    ceiling_z: int,
    spacing: int = 128,
    beam_width: int = 16,
    beam_depth: int = 24,
    texture: str = "WOOD1_1",
) -> List[Brush]:
    """Horizontal beams across the ceiling (span X-axis)."""
    brushes: List[Brush] = []
    y = y1 + spacing
    while y < y2 - beam_width:
        brushes.append(_box(
            x1, y, ceiling_z - beam_depth,
            x2, y + beam_width, ceiling_z,
            texture,
        ))
        y += spacing
    return brushes


# ---------------------------------------------------------------------------
# Wall buttress
# ---------------------------------------------------------------------------

def wall_buttresses(
    x1: int, y1: int, x2: int, y2: int,
    floor_z: int, ceiling_z: int,
    spacing: int = 192,
    buttress_width: int = 16,
    buttress_depth: int = 32,
    texture: str = "CRATE1_5",
) -> List[Brush]:
    """Triangular buttresses protruding from walls.

    Places buttresses along walls longer than spacing.
    Each buttress is approximated as a rectangular protrusion
    (true triangle would need angled planes; we keep axis-aligned).
    """
    brushes: List[Brush] = []
    room_w = x2 - x1
    room_h = y2 - y1

    # South wall buttresses (protrude in +Y)
    if room_w > spacing:
        x = x1 + spacing
        while x < x2 - buttress_width:
            brushes.append(_box(
                x, y1, floor_z,
                x + buttress_width, y1 + buttress_depth, ceiling_z - 16,
                texture,
            ))
            x += spacing

    # North wall buttresses (protrude in -Y)
    if room_w > spacing:
        x = x1 + spacing
        while x < x2 - buttress_width:
            brushes.append(_box(
                x, y2 - buttress_depth, floor_z,
                x + buttress_width, y2, ceiling_z - 16,
                texture,
            ))
            x += spacing

    # West wall buttresses (protrude in +X)
    if room_h > spacing:
        y = y1 + spacing
        while y < y2 - buttress_width:
            brushes.append(_box(
                x1, y, floor_z,
                x1 + buttress_depth, y + buttress_width, ceiling_z - 16,
                texture,
            ))
            y += spacing

    # East wall buttresses (protrude in -X)
    if room_h > spacing:
        y = y1 + spacing
        while y < y2 - buttress_width:
            brushes.append(_box(
                x2 - buttress_depth, y, floor_z,
                x2, y + buttress_width, ceiling_z - 16,
                texture,
            ))
            y += spacing

    return brushes


# ---------------------------------------------------------------------------
# Pillar
# ---------------------------------------------------------------------------

def pillar(
    cx: int, cy: int,
    floor_z: int, ceiling_z: int,
    width: int = 32,
    texture: str = "METAL1_1",
) -> List[Brush]:
    """Floor-to-ceiling column (square cross-section)."""
    hw = width // 2
    return [_box(cx - hw, cy - hw, floor_z, cx + hw, cy + hw, ceiling_z, texture)]


def pillars_for_room(
    x1: int, y1: int, x2: int, y2: int,
    floor_z: int, ceiling_z: int,
    pillar_width: int = 32,
    min_room_size: int = 256,
    texture: str = "METAL1_1",
) -> List[Brush]:
    """Place pillars in a room if it is large enough.

    Places 2-4 pillars on a grid inset from the walls.
    """
    room_w = x2 - x1
    room_h = y2 - y1
    if room_w < min_room_size and room_h < min_room_size:
        return []

    brushes: List[Brush] = []
    inset = max(64, room_w // 4)
    inset_y = max(64, room_h // 4)

    positions = [
        (x1 + inset, y1 + inset_y),
        (x2 - inset, y1 + inset_y),
        (x1 + inset, y2 - inset_y),
        (x2 - inset, y2 - inset_y),
    ]
    # For smaller rooms, only 2 pillars
    if room_w < min_room_size * 1.5 or room_h < min_room_size * 1.5:
        positions = positions[:2]

    for cx, cy in positions:
        brushes.extend(pillar(cx, cy, floor_z, ceiling_z, pillar_width, texture))
    return brushes


# ---------------------------------------------------------------------------
# Wall niche / alcove
# ---------------------------------------------------------------------------

def wall_niche(
    x1: int, y1: int,
    wall_axis: str,
    floor_z: int,
    niche_width: int = 48,
    niche_height: int = 64,
    niche_depth: int = 24,
    texture: str = "CRATE1_5",
) -> List[Brush]:
    """Recessed alcove in a wall (5 brushes forming the niche cavity).

    wall_axis: "north", "south", "east", "west"
    x1, y1: corner origin of the niche on the wall.
    """
    brushes: List[Brush] = []
    base_z = floor_z + 32  # Niche starts 32 units above floor
    top_z = base_z + niche_height
    d = niche_depth
    w = niche_width

    if wall_axis == "south":
        # Niche recesses into the south wall (-Y)
        # Left side
        brushes.append(_box(x1, y1 - d, base_z, x1 + 8, y1, top_z, texture))
        # Right side
        brushes.append(_box(x1 + w - 8, y1 - d, base_z, x1 + w, y1, top_z, texture))
        # Top
        brushes.append(_box(x1, y1 - d, top_z, x1 + w, y1, top_z + 8, texture))
        # Bottom shelf
        brushes.append(_box(x1, y1 - d, base_z - 8, x1 + w, y1, base_z, texture))
        # Back
        brushes.append(_box(x1, y1 - d - 8, base_z - 8, x1 + w, y1 - d, top_z + 8, texture))
    elif wall_axis == "north":
        brushes.append(_box(x1, y1, base_z, x1 + 8, y1 + d, top_z, texture))
        brushes.append(_box(x1 + w - 8, y1, base_z, x1 + w, y1 + d, top_z, texture))
        brushes.append(_box(x1, y1, top_z, x1 + w, y1 + d, top_z + 8, texture))
        brushes.append(_box(x1, y1, base_z - 8, x1 + w, y1 + d, base_z, texture))
        brushes.append(_box(x1, y1 + d, base_z - 8, x1 + w, y1 + d + 8, top_z + 8, texture))
    elif wall_axis == "west":
        brushes.append(_box(x1 - d, y1, base_z, x1, y1 + 8, top_z, texture))
        brushes.append(_box(x1 - d, y1 + w - 8, base_z, x1, y1 + w, top_z, texture))
        brushes.append(_box(x1 - d, y1, top_z, x1, y1 + w, top_z + 8, texture))
        brushes.append(_box(x1 - d, y1, base_z - 8, x1, y1 + w, base_z, texture))
        brushes.append(_box(x1 - d - 8, y1, base_z - 8, x1 - d, y1 + w, top_z + 8, texture))
    elif wall_axis == "east":
        brushes.append(_box(x1, y1, base_z, x1 + d, y1 + 8, top_z, texture))
        brushes.append(_box(x1, y1 + w - 8, base_z, x1 + d, y1 + w, top_z, texture))
        brushes.append(_box(x1, y1, top_z, x1 + d, y1 + w, top_z + 8, texture))
        brushes.append(_box(x1, y1, base_z - 8, x1 + d, y1 + w, base_z, texture))
        brushes.append(_box(x1 + d, y1, base_z - 8, x1 + d + 8, y1 + w, top_z + 8, texture))

    return brushes
