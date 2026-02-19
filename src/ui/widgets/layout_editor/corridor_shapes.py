"""
Corridor shape definitions for layout editor visualization.

Defines which cells within a primitive's footprint are "corridor" (open space)
versus "wall" (filled/unused). Used by both palette icons and canvas items.
"""

from typing import Dict, Set, Tuple


# Corridor cell definitions for hall primitives.
# Each entry maps primitive_type -> set of (x, y) cell coordinates that are corridor.
# Cells not in this set are drawn as walls/unused space.
CORRIDOR_CELLS: Dict[str, Set[Tuple[int, int]]] = {
    # StraightHall: 1x2, vertical corridor through both cells
    # Portal layout:
    #   Row 1: [X] <- NORTH portal
    #   Row 0: [X] <- SOUTH portal
    'StraightHall': {(0, 0), (0, 1)},

    # SquareCorner: 2x2, L-shaped 90-degree corner
    # Portal layout:
    #   Row 1: [X] [X] <- cell (0,1) is corridor, EAST portal at (1,1)
    #   Row 0: [X] [ ] <- SOUTH portal at (0,0), cell (1,0) is SOLID wall
    # Corridor path: (0,0) -> (0,1) -> (1,1)
    # Forms inverted-L: up from south, then right to east
    'SquareCorner': {(0, 0), (0, 1), (1, 1)},

    # TJunction: 3x2, TRUE T-shaped junction
    # Portal layout:
    #   Row 1: [ ] [X] [ ] <- NORTH portal at (1,1) - stem
    #   Row 0: [X] [X] [X] <- WEST portal at (0,0), EAST portal at (2,0) - crossbar
    # Visual: [W]---+---[E] crossbar with [N] stem pointing up
    #              |
    #             [N]
    'TJunction': {(0, 0), (1, 0), (2, 0), (1, 1)},

    # Crossroads: 3x3, + shaped intersection
    # Portal layout:
    #   Row 2: [ ] [X] [ ] <- NORTH portal at (1,2)
    #   Row 1: [X] [X] [X] <- WEST at (0,1), EAST at (2,1)
    #   Row 0: [ ] [X] [ ] <- SOUTH portal at (1,0)
    'Crossroads': {(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)},

    # VerticalStairHall: 2x4, vertical stairwell connecting two floor levels
    # Portal layout:
    #   Row 3: [X] [X] <- TOP (NORTH) portal at z_level=128
    #   Row 2: [X] [X] <- stair section
    #   Row 1: [X] [X] <- stair section
    #   Row 0: [X] [X] <- BOTTOM (SOUTH) portal at z_level=0
    # All cells are corridor (open stairwell)
    'VerticalStairHall': {(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 3), (1, 3)},

    # SecretHall: 1x2, same as StraightHall but with CLIP side walls
    # Portal layout:
    #   Row 1: [X] <- NORTH portal
    #   Row 0: [X] <- SOUTH portal
    'SecretHall': {(0, 0), (0, 1)},
}


def get_corridor_cells(primitive_type: str, width_cells: int, depth_cells: int) -> Set[Tuple[int, int]]:
    """
    Get the corridor cells for a primitive type.

    For hall primitives, returns the defined corridor shape.
    For rooms and other primitives, returns all cells (full rectangle).

    Args:
        primitive_type: The type of primitive
        width_cells: Width of the footprint in cells
        depth_cells: Depth of the footprint in cells

    Returns:
        Set of (x, y) cell coordinates that are corridor/interior
    """
    if primitive_type in CORRIDOR_CELLS:
        return CORRIDOR_CELLS[primitive_type]

    # For rooms and other primitives, all cells are interior
    return {(x, y) for x in range(width_cells) for y in range(depth_cells)}
