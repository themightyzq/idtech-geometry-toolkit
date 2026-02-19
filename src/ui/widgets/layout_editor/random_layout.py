"""
Random layout generator for the layout editor.

Generates a random DungeonLayout with connected primitives that can be
edited by the user before generating final geometry.

Supports single-floor and multi-floor dungeon generation with
Z-level aware connectivity validation.
"""

import random
from typing import Dict, List, Optional, Set, Tuple

from .data_model import (
    DungeonLayout, PlacedPrimitive, CellCoord, Connection, PortalDirection
)
from .palette_widget import PRIMITIVE_FOOTPRINTS, FLOOR_LEVELS
from .traversability import (
    validate_multi_floor_path,
    check_connection_requires_stair,
    find_isolated_regions,
    MAX_STEP_HEIGHT,
)
from .parameter_randomizer import randomize_primitive_params
from .spatial_validation import check_placement_collision


# Set to True for verbose debug output during generation
DEBUG = False


def _debug(msg: str):
    """Print debug message if DEBUG is enabled."""
    if DEBUG:
        print(msg, flush=True)


# Primitive types by category
# NOTE: These must match the keys in PRIMITIVE_FOOTPRINTS (palette_widget.py)
# and the display names from the primitive catalog (catalog.py)
HALL_TYPES = ['StraightHall', 'TJunction', 'Crossroads', 'SquareCorner']
ROOM_TYPES = [
    'Sanctuary', 'Tomb', 'Tower', 'Chamber', 'Storage', 'GreatHall',
    'Prison', 'Armory', 'Cistern', 'Stronghold', 'Courtyard',
    'Arena', 'Laboratory', 'Vault', 'Barracks', 'Shrine', 'Pit', 'Antechamber'
]
# Rooms that are too tall for multi-floor layouts with 160-unit floor separation.
# Formula: height + floor_thickness + ceiling_thickness must be <= 160
# Rooms with thick walls (t > 16) also need extra separation.
# These rooms exceed the safe limit and would collide with adjacent floor geometry.
TALL_ROOMS = [
    'Tower',      # height=256, t=16 (needs 288 separation)
    'Sanctuary',  # nave_height=192, t=16 (needs 224 separation)
    'GreatHall',  # height=192, t=16 (needs 224 separation)
    'Stronghold', # height=192, t=32 (needs 256 separation)
    'Arena',      # height=160, t=16 (needs 192 separation)
    'Vault',      # height=128, t=24 (needs 176 separation - thick walls)
]
# Secret room type - placed based on secret_room_frequency
SECRET_ROOM_TYPE = 'SecretChamber'


def _connection_exists(connections: List[Connection], conn: Connection) -> bool:
    """
    Check if an equivalent connection already exists in the list.

    Connections are considered equivalent if they connect the same primitives
    through the same portals (regardless of order: A->B equals B->A).
    """
    for existing in connections:
        # Check both orderings since connections are bidirectional
        if (existing.primitive_a_id == conn.primitive_a_id and
            existing.portal_a_id == conn.portal_a_id and
            existing.primitive_b_id == conn.primitive_b_id and
            existing.portal_b_id == conn.portal_b_id):
            return True
        if (existing.primitive_a_id == conn.primitive_b_id and
            existing.portal_a_id == conn.portal_b_id and
            existing.primitive_b_id == conn.primitive_a_id and
            existing.portal_b_id == conn.portal_a_id):
            return True
    return False


def generate_random_layout(
    room_count: int = 5,
    map_width: int = 20,
    map_height: int = 20,
    seed: Optional[int] = None,
    complexity: int = 3,
    preferred_room_types: Optional[List[str]] = None,
    preferred_hall_types: Optional[List[str]] = None,
    room_probability: float = 0.4,
    min_hall_between_rooms: int = 1,
    allow_dead_ends: bool = True,
    secret_room_frequency: int = 0,
    exclude_tall: bool = False,
    require_stair_space: bool = False,
) -> Tuple[DungeonLayout, int]:
    """
    Generate a random dungeon layout with connected halls and rooms.

    Args:
        room_count: Number of rooms to place
        map_width: Grid width in cells
        map_height: Grid height in cells
        seed: Random seed for reproducibility (None = random)
        complexity: 1-5, affects branching and layout density
        preferred_room_types: List of room types to prefer (None = use defaults)
        preferred_hall_types: List of hall types to prefer (None = use defaults)
        room_probability: Base probability of placing a room (0.0-1.0)
        min_hall_between_rooms: Minimum halls between consecutive rooms
        allow_dead_ends: Whether to allow dead-end corridors
        secret_room_frequency: Percentage of rooms that should be SecretChambers (0-100)
        exclude_tall: If True, exclude tall rooms (Tower, Sanctuary, etc.) for multi-floor layouts
        require_stair_space: If True, stop generation early if no stair-friendly portals remain

    Returns:
        Tuple of (DungeonLayout, actual_seed_used)
    """
    _debug(f"\n[RANDOM_LAYOUT] === Starting random layout generation ===")
    _debug(f"[RANDOM_LAYOUT] Parameters: rooms={room_count}, size={map_width}x{map_height}, complexity={complexity}")

    # Initialize random seed
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    random.seed(seed)
    # Create seeded RNG for parameter randomization (reproducibility)
    param_rng = random.Random(seed)
    _debug(f"[RANDOM_LAYOUT] Using seed: {seed}")

    layout = DungeonLayout()
    occupied_cells: Set[Tuple[int, int]] = set()
    open_portals: List[Tuple[str, str, CellCoord, PortalDirection]] = []  # (prim_id, portal_id, cell, direction)

    # Start with a crossroads or T-junction at center
    center_x = map_width // 2
    center_y = map_height // 2

    start_type = random.choice(['Crossroads', 'TJunction'])
    _debug(f"[RANDOM_LAYOUT] Placing start primitive: {start_type} at ({center_x}, {center_y})")
    start_prim = _place_primitive(layout, start_type, CellCoord(center_x, center_y), 0, occupied_cells, param_rng)
    if start_prim:
        _debug(f"[RANDOM_LAYOUT] Start primitive placed with ID: {start_prim.id}")
        _collect_open_portals(start_prim, open_portals, layout)
        _debug(f"[RANDOM_LAYOUT] Open portals after start: {len(open_portals)}")
    else:
        _debug(f"[RANDOM_LAYOUT] ERROR: Failed to place start primitive!")

    # Track placed primitives
    placed_rooms = 0
    placed_halls = 0
    placed_secret_rooms = 0
    halls_since_last_room = 0
    max_halls = room_count * 3  # Limit halls to prevent infinite growth
    # Calculate target number of secret rooms based on frequency
    target_secret_rooms = int(room_count * secret_room_frequency / 100.0)

    # Main generation loop - extend from open portals
    iterations = 0
    max_iterations = room_count * 10
    _debug(f"[RANDOM_LAYOUT] Starting main loop, max_iterations={max_iterations}")

    while open_portals and iterations < max_iterations:
        iterations += 1
        _debug(f"\n[RANDOM_LAYOUT] --- Iteration {iterations} ---")
        _debug(f"[RANDOM_LAYOUT] Open portals: {len(open_portals)}, Placed rooms: {placed_rooms}, Placed halls: {placed_halls}")

        # Handle dead ends: if not allowed and only one portal left, close it
        if not allow_dead_ends and len(open_portals) == 1 and placed_rooms >= room_count:
            _debug(f"[RANDOM_LAYOUT] Closing last portal (dead ends not allowed)")
            open_portals.pop(0)
            continue

        # Pick a random open portal to extend from
        portal_idx = random.randint(0, len(open_portals) - 1)
        prim_id, portal_id, portal_cell, portal_dir = open_portals[portal_idx]
        _debug(f"[RANDOM_LAYOUT] Selected portal: '{portal_id}' from primitive {prim_id[:8]}...")

        # Determine what to place
        # Use template room_probability, adjusted based on progress and min_hall constraint
        effective_room_prob = room_probability
        if halls_since_last_room < min_hall_between_rooms:
            # Force halls until minimum is met
            effective_room_prob = 0.0
        elif placed_halls > max_halls * 0.7:
            # Increase room probability when we have many halls
            effective_room_prob = min(0.8, room_probability + 0.2)

        if placed_rooms < room_count and random.random() < effective_room_prob:
            # Try to place a room - check if we should place a secret room
            prim_type = _select_room_type(
                complexity, preferred_room_types,
                placed_rooms, placed_secret_rooms, target_secret_rooms,
                exclude_tall=exclude_tall,
            )
            _debug(f"[RANDOM_LAYOUT] Trying to place ROOM: {prim_type}")
        else:
            # Place a hall
            # Prefer junctions when few open portals remain (ensures stair placement options)
            prefer_junctions = len(open_portals) < 4
            prim_type = _select_hall_type(complexity, preferred_hall_types, prefer_junctions)
            _debug(f"[RANDOM_LAYOUT] Trying to place HALL: {prim_type}")

        # Find valid position and rotation for the new primitive
        result = _find_placement(
            prim_type, portal_cell, portal_dir, occupied_cells,
            map_width, map_height
        )

        if result is None:
            # Can't place here, remove this portal
            _debug(f"[RANDOM_LAYOUT] FAILED: No valid placement found for {prim_type}")
            open_portals.pop(portal_idx)
            continue

        new_origin, new_rotation, connecting_portal_id = result
        _debug(f"[RANDOM_LAYOUT] Found placement: origin=({new_origin.x}, {new_origin.y}), rotation={new_rotation}")

        # STAIR SPACE PRESERVATION: When require_stair_space=True and we're running
        # low on open portals, check if placing this primitive would leave enough
        # portals for stair placement. If not, try a junction instead.
        MIN_STAIR_FRIENDLY_PORTALS = 3
        if require_stair_space and len(open_portals) <= 5:
            new_prim_footprint = PRIMITIVE_FOOTPRINTS.get(prim_type)
            new_portal_count = len(new_prim_footprint.portals) - 1 if new_prim_footprint else 0
            predicted_open_portals = len(open_portals) - 1 + new_portal_count

            if predicted_open_portals < MIN_STAIR_FRIENDLY_PORTALS and prim_type not in ['Crossroads', 'TJunction']:
                _debug(f"[RANDOM_LAYOUT] Switching {prim_type} -> junction (only {predicted_open_portals} portals predicted)")
                alt_type = random.choice(['Crossroads', 'TJunction'])
                alt_result = _find_placement(
                    alt_type, portal_cell, portal_dir, occupied_cells,
                    map_width, map_height
                )
                if alt_result is not None:
                    prim_type = alt_type
                    result = alt_result
                    new_origin, new_rotation, connecting_portal_id = result

        # Place the primitive
        new_prim = _place_primitive(layout, prim_type, new_origin, new_rotation, occupied_cells, param_rng)
        if new_prim is None:
            _debug(f"[RANDOM_LAYOUT] FAILED: Could not place primitive (cells occupied?)")
            open_portals.pop(portal_idx)
            continue

        _debug(f"[RANDOM_LAYOUT] SUCCESS: Placed {prim_type}")

        # Create connection
        layout.connections.append(Connection(
            primitive_a_id=prim_id,
            portal_a_id=portal_id,
            primitive_b_id=new_prim.id,
            portal_b_id=connecting_portal_id
        ))

        # Remove the used portal
        open_portals.pop(portal_idx)

        # Collect new open portals from the placed primitive
        _collect_open_portals(new_prim, open_portals, layout, exclude_portal=connecting_portal_id)
        _debug(f"[RANDOM_LAYOUT] Open portals now: {len(open_portals)}")

        # Update counts
        if prim_type in ROOM_TYPES or prim_type == SECRET_ROOM_TYPE:
            placed_rooms += 1
            halls_since_last_room = 0
            if prim_type == SECRET_ROOM_TYPE:
                placed_secret_rooms += 1
        else:
            placed_halls += 1
            halls_since_last_room += 1

        # Stop if we've placed enough rooms
        if placed_rooms >= room_count:
            # Continue to maybe add a few more halls for variety
            if placed_halls >= max_halls or random.random() < 0.3:
                _debug(f"[RANDOM_LAYOUT] Stopping: placed_rooms={placed_rooms} >= room_count={room_count}")
                break

        # When require_stair_space=True, stop early if stair-friendly portals are depleting
        if require_stair_space and placed_rooms >= 3:
            stair_portals = _count_stair_friendly_portals(
                open_portals, occupied_cells, map_width, map_height
            )
            # Stop if we only have 1-2 stair-friendly portals left (preserve them)
            if stair_portals <= 2:
                _debug(f"[RANDOM_LAYOUT] Stopping early: only {stair_portals} stair-friendly portals remain")
                break

    _debug(f"\n[RANDOM_LAYOUT] === Generation complete ===")
    _debug(f"[RANDOM_LAYOUT] Total primitives: {len(layout.primitives)}")
    _debug(f"[RANDOM_LAYOUT] Total connections: {len(layout.connections)}")
    _debug(f"[RANDOM_LAYOUT] Iterations used: {iterations}")
    return layout, seed


def _select_room_type(
    complexity: int,
    preferred: Optional[List[str]] = None,
    placed_rooms: int = 0,
    placed_secret_rooms: int = 0,
    target_secret_rooms: int = 0,
    exclude_tall: bool = False,
) -> str:
    """Select a room type based on complexity, preferences, and secret room targets.

    Args:
        complexity: Layout complexity (1-5)
        preferred: Preferred room types to use
        placed_rooms: Number of rooms already placed
        placed_secret_rooms: Number of secret rooms already placed
        target_secret_rooms: Target number of secret rooms to place
        exclude_tall: If True, exclude tall rooms (for multi-floor layouts)

    Returns:
        Selected room type name
    """
    # Check if we should place a secret room
    # Place one if we haven't reached target and probability check passes
    if placed_secret_rooms < target_secret_rooms:
        # Calculate probability: increase as we approach room target with few secrets
        rooms_remaining = max(1, 10 - placed_rooms)  # Assume ~10 total rooms
        secrets_needed = target_secret_rooms - placed_secret_rooms
        # Higher probability if we're behind on secret rooms
        secret_prob = min(0.8, secrets_needed / rooms_remaining)
        if random.random() < secret_prob:
            return SECRET_ROOM_TYPE

    # Use preferred types if provided and valid
    if preferred:
        valid_preferred = [r for r in preferred if r in ROOM_TYPES]
        if exclude_tall:
            valid_preferred = [r for r in valid_preferred if r not in TALL_ROOMS]
        if valid_preferred:
            return random.choice(valid_preferred)

    # Fall back to complexity-based selection
    # Complexity 1-2: Simple rooms (small, no complex features)
    # Complexity 3-4: Medium rooms (various types)
    # Complexity 5: All room types including large/complex
    if complexity <= 2:
        choices = ['Tower', 'Storage', 'Chamber']
    elif complexity <= 4:
        choices = ['Tower', 'Chamber', 'Storage', 'Armory', 'Tomb', 'Shrine']
    else:
        choices = ROOM_TYPES

    # Exclude tall rooms for multi-floor layouts
    if exclude_tall:
        choices = [r for r in choices if r not in TALL_ROOMS]

    return random.choice(choices)


def _select_hall_type(complexity: int, preferred: Optional[List[str]] = None,
                       prefer_junctions: bool = False) -> str:
    """Select a hall type based on complexity and preferences.

    Args:
        complexity: Layout complexity (1-5)
        preferred: Preferred hall types to use
        prefer_junctions: If True, prefer TJunction/Crossroads to create more open portals
    """
    # Use preferred types if provided and valid
    if preferred:
        valid_preferred = [h for h in preferred if h in HALL_TYPES]
        if valid_preferred:
            return random.choice(valid_preferred)

    # When prefer_junctions is True, bias toward multi-portal halls
    if prefer_junctions:
        choices = ['TJunction', 'TJunction', 'Crossroads', 'StraightHall', 'SquareCorner']
        return random.choice(choices)

    # Fall back to complexity-based selection
    if complexity <= 2:
        choices = ['StraightHall', 'StraightHall', 'SquareCorner', 'SquareCorner']
    elif complexity <= 4:
        choices = ['StraightHall', 'SquareCorner', 'TJunction', 'SquareCorner']
    else:
        choices = HALL_TYPES
    return random.choice(choices)


def _place_primitive(
    layout: DungeonLayout,
    prim_type: str,
    origin: CellCoord,
    rotation: int,
    occupied: Set[Tuple[int, int]],
    rng: Optional[random.Random] = None,
    z_offset: float = 0.0,
    combined_layout: Optional[DungeonLayout] = None,
) -> Optional[PlacedPrimitive]:
    """Place a primitive and mark cells as occupied.

    Args:
        layout: The dungeon layout to add the primitive to
        prim_type: Type name of the primitive
        origin: Grid cell origin for placement
        rotation: Rotation in degrees (0, 90, 180, 270)
        occupied: Set of occupied cell coordinates
        rng: Optional seeded random.Random for parameter randomization
        z_offset: Z-offset for the primitive (default 0.0)
        combined_layout: Optional combined layout for 3D collision checking

    Returns:
        The placed primitive, or None if placement failed
    """
    footprint = PRIMITIVE_FOOTPRINTS.get(prim_type)
    if footprint is None:
        return None

    # Get cells this primitive would occupy
    cells = _get_primitive_cells(prim_type, origin, rotation)

    # Check if any cells are already occupied
    for cell in cells:
        if cell in occupied:
            return None

    # Generate randomized parameters if RNG provided
    params = {}
    if rng is not None:
        params = randomize_primitive_params(prim_type, rng)

    # Place the primitive
    prim = PlacedPrimitive.create(
        primitive_type=prim_type,
        origin=origin,
        rotation=rotation,
        parameters=params,
    )
    # Set z_offset before collision check
    prim.z_offset = z_offset

    # CRITICAL: Set footprint so portal alignment works correctly
    prim.set_footprint(footprint)

    # Check for 3D spatial collision against all existing primitives
    if combined_layout is not None:
        collision = check_placement_collision(combined_layout, prim, footprint)
        if collision is not None:
            _debug(f"[PLACE_PRIMITIVE] Rejected {prim_type} at ({origin.x}, {origin.y}) z={z_offset}: {collision.message}")
            return None  # Reject placement

    layout.add_primitive(prim)

    # Mark cells as occupied
    occupied.update(cells)

    return prim


def _get_primitive_cells(
    prim_type: str,
    origin: CellCoord,
    rotation: int
) -> List[Tuple[int, int]]:
    """Get all cells occupied by a primitive."""
    footprint = PRIMITIVE_FOOTPRINTS.get(prim_type)
    if footprint is None:
        return [(origin.x, origin.y)]

    width, depth = footprint.rotated_size(rotation)
    cells = []
    for dx in range(width):
        for dy in range(depth):
            cells.append((origin.x + dx, origin.y + dy))
    return cells


def _collect_open_portals(
    prim: PlacedPrimitive,
    open_portals: List[Tuple[str, str, CellCoord, PortalDirection]],
    layout: DungeonLayout,
    exclude_portal: Optional[str] = None
):
    """Collect all open (unconnected) portals from a primitive."""
    footprint = PRIMITIVE_FOOTPRINTS.get(prim.primitive_type)
    if footprint is None:
        return

    # Check which portals are already connected
    connected_portals = set()
    for conn in layout.connections:
        if conn.primitive_a_id == prim.id:
            connected_portals.add(conn.portal_a_id)
        if conn.primitive_b_id == prim.id:
            connected_portals.add(conn.portal_b_id)

    # Add unconnected portals
    for portal in footprint.portals:
        if portal.id == exclude_portal:
            continue
        if portal.id in connected_portals:
            continue

        # Get world position and direction of this portal
        world_cell = portal.world_cell(
            prim.origin_cell, prim.rotation,
            footprint.width_cells, footprint.depth_cells
        )
        world_dir = portal.rotated_direction(prim.rotation)

        open_portals.append((prim.id, portal.id, world_cell, world_dir))


def _find_placement(
    prim_type: str,
    connect_cell: CellCoord,
    connect_dir: PortalDirection,
    occupied: Set[Tuple[int, int]],
    map_width: int,
    map_height: int
) -> Optional[Tuple[CellCoord, int, str]]:
    """
    Find a valid placement for a primitive that connects to the given portal.

    Args:
        prim_type: Type of primitive to place
        connect_cell: Cell where the existing portal is
        connect_dir: Direction the existing portal faces
        occupied: Set of occupied cells
        map_width: Grid width limit
        map_height: Grid height limit

    Returns:
        Tuple of (origin, rotation, portal_id) or None if no valid placement
    """
    footprint = PRIMITIVE_FOOTPRINTS.get(prim_type)
    if footprint is None:
        _debug(f"[FIND_PLACEMENT] ERROR: No footprint for {prim_type}")
        return None

    # The new primitive's portal must face opposite to connect_dir
    required_dir = connect_dir.opposite()
    _debug(f"[FIND_PLACEMENT] Looking for {prim_type} portal facing {required_dir.name}")

    # Try each rotation to find one where a portal faces required_dir
    for rotation in [0, 90, 180, 270]:
        for portal in footprint.portals:
            rotated_dir = portal.rotated_direction(rotation)
            if rotated_dir != required_dir:
                continue

            # This portal faces the right direction
            # Calculate where the primitive origin needs to be
            # so that this portal aligns with connect_cell

            # Get the portal's offset within the primitive (at this rotation)
            portal_offset = portal._rotate_offset(
                portal.cell_offset, rotation,
                footprint.width_cells, footprint.depth_cells
            )

            # The portal cell in the new primitive should be adjacent to connect_cell
            # in the connect_dir direction
            target_cell = connect_cell.neighbor(connect_dir)

            # Origin = target_cell - portal_offset
            origin = CellCoord(
                target_cell.x - portal_offset.x,
                target_cell.y - portal_offset.y
            )

            # Get all cells this placement would occupy
            cells = _get_primitive_cells(prim_type, origin, rotation)

            # Check bounds
            valid = True
            for cx, cy in cells:
                if cx < 0 or cx >= map_width or cy < 0 or cy >= map_height:
                    valid = False
                    break
                if (cx, cy) in occupied:
                    valid = False
                    break

            if valid:
                _debug(f"[FIND_PLACEMENT] Found valid placement at ({origin.x}, {origin.y}) rotation={rotation}")
                return (origin, rotation, portal.id)

    _debug(f"[FIND_PLACEMENT] No valid placement found for {prim_type}")
    return None


def _is_portal_connected(
    layout: DungeonLayout,
    prim_id: str,
    portal_id: str
) -> bool:
    """Check if a portal is already connected."""
    for conn in layout.connections:
        if conn.primitive_a_id == prim_id and conn.portal_a_id == portal_id:
            return True
        if conn.primitive_b_id == prim_id and conn.portal_b_id == portal_id:
            return True
    return False


def _verify_stair_connectivity(
    layout: DungeonLayout,
    stair_prim: PlacedPrimitive
) -> Tuple[bool, bool]:
    """
    Verify that a VerticalStairHall has both portals connected.

    Args:
        layout: The dungeon layout containing connections
        stair_prim: The VerticalStairHall primitive to check

    Returns:
        Tuple of (bottom_connected, top_connected)
    """
    bottom_connected = _is_portal_connected(layout, stair_prim.id, 'bottom')
    top_connected = _is_portal_connected(layout, stair_prim.id, 'top')
    return (bottom_connected, top_connected)


def _attempt_stair_top_recovery(
    combined_layout: DungeonLayout,
    floor_layout: DungeonLayout,
    stair_prim: PlacedPrimitive,
    floor_occupied: Set[Tuple[int, int]],
    map_width: int,
    map_height: int,
    z_offset: float,
    rng: random.Random,
) -> bool:
    """
    Attempt to connect a stair's disconnected top portal.

    Places a StraightHall, TJunction, or Crossroads at the stair top if
    the top portal is not yet connected to the floor network.

    Args:
        combined_layout: The combined multi-floor layout
        floor_layout: The floor layout where the top portal should connect
        stair_prim: The VerticalStairHall with disconnected top
        floor_occupied: Set of occupied cells on this floor
        map_width: Grid width limit
        map_height: Grid height limit
        z_offset: Z-offset for this floor
        rng: Random number generator for parameter randomization

    Returns:
        True if recovery succeeded, False otherwise
    """
    vstair_footprint = PRIMITIVE_FOOTPRINTS.get('VerticalStairHall')
    if vstair_footprint is None:
        return False

    # Find the top portal
    top_portal = None
    for portal in vstair_footprint.portals:
        if portal.id == 'top':
            top_portal = portal
            break

    if top_portal is None:
        return False

    # Get top portal world position and direction
    top_world_cell = top_portal.world_cell(
        stair_prim.origin_cell, stair_prim.rotation,
        vstair_footprint.width_cells, vstair_footprint.depth_cells
    )
    top_world_dir = top_portal.rotated_direction(stair_prim.rotation)

    _debug(f"[STAIR_RECOVERY] Attempting recovery at ({top_world_cell.x}, {top_world_cell.y}) facing {top_world_dir.name}")

    # Try to place a hall at the stair top
    # Prefer junctions to create more connection opportunities
    hall_types = ['Crossroads', 'TJunction', 'StraightHall']

    for hall_type in hall_types:
        result = _find_placement(
            hall_type, top_world_cell, top_world_dir,
            floor_occupied, map_width, map_height
        )

        if result is None:
            continue

        hall_origin, hall_rotation, hall_portal_id = result

        # Place the hall
        hall_footprint = PRIMITIVE_FOOTPRINTS.get(hall_type)
        hall_prim = PlacedPrimitive.create(
            primitive_type=hall_type,
            origin=hall_origin,
            rotation=hall_rotation,
            z_offset=z_offset,
            parameters={}
        )
        hall_prim.set_footprint(hall_footprint)

        # Add to layouts
        combined_layout.add_primitive(hall_prim)
        floor_layout.add_primitive(hall_prim)

        # Update occupied cells
        hall_cells = _get_primitive_cells(hall_type, hall_origin, hall_rotation)
        floor_occupied.update(hall_cells)

        # Connect stair top to the new hall
        combined_layout.connections.append(Connection(
            primitive_a_id=stair_prim.id,
            portal_a_id='top',
            primitive_b_id=hall_prim.id,
            portal_b_id=hall_portal_id
        ))
        floor_layout.connections.append(Connection(
            primitive_a_id=stair_prim.id,
            portal_a_id='top',
            primitive_b_id=hall_prim.id,
            portal_b_id=hall_portal_id
        ))

        _debug(f"[STAIR_RECOVERY] SUCCESS: Placed {hall_type} to connect stair top")
        return True

    _debug(f"[STAIR_RECOVERY] FAILED: Could not place any hall type")
    return False


def _rollback_stair_placement(
    combined_layout: DungeonLayout,
    stair_prim: PlacedPrimitive,
    lower_occupied: Set[Tuple[int, int]],
    upper_occupied: Set[Tuple[int, int]],
):
    """
    Remove a stair and its connections, freeing occupied cells.

    Used when stair placement fails to establish proper connectivity.

    Args:
        combined_layout: The combined multi-floor layout
        stair_prim: The VerticalStairHall to remove
        lower_occupied: Occupied cells on lower floor
        upper_occupied: Occupied cells on upper floor
    """
    stair_id = stair_prim.id

    # Remove from layout
    if stair_id in combined_layout.primitives:
        del combined_layout.primitives[stair_id]

    # Remove connections
    combined_layout.connections = [
        c for c in combined_layout.connections
        if c.primitive_a_id != stair_id and c.primitive_b_id != stair_id
    ]

    # Free occupied cells
    stair_cells = _get_primitive_cells(
        'VerticalStairHall', stair_prim.origin_cell, stair_prim.rotation
    )
    for cell in stair_cells:
        lower_occupied.discard(cell)
        upper_occupied.discard(cell)

    _debug(f"[STAIR_ROLLBACK] Removed stair at ({stair_prim.origin_cell.x}, {stair_prim.origin_cell.y})")


def _find_all_viable_stair_positions(
    floor_layout: DungeonLayout,
    combined_layout: DungeonLayout,
    lower_occupied: Set[Tuple[int, int]],
    upper_occupied: Set[Tuple[int, int]],
    map_width: int,
    map_height: int,
) -> List[Tuple[CellCoord, int, str, str]]:
    """
    Find all viable stair placement positions on a floor.

    Returns a sorted list of (origin, rotation, prim_id, portal_id) tuples,
    sorted by distance from map center (closest first).

    Args:
        floor_layout: Layout of the floor to scan for open portals
        combined_layout: Combined layout for connection checking
        lower_occupied: Occupied cells on lower floor
        upper_occupied: Occupied cells on upper floor (currently empty for new floors)
        map_width: Grid width limit
        map_height: Grid height limit

    Returns:
        List of viable (origin, rotation, prim_id, portal_id) tuples
    """
    vstair_footprint = PRIMITIVE_FOOTPRINTS.get('VerticalStairHall')
    if vstair_footprint is None:
        return []

    center_x, center_y = map_width // 2, map_height // 2
    candidates: List[Tuple[float, CellCoord, int, str, str]] = []

    for prim in floor_layout.primitives.values():
        footprint = PRIMITIVE_FOOTPRINTS.get(prim.primitive_type)
        if footprint is None:
            continue

        # Get connected portals
        connected_portals = set()
        for conn in combined_layout.connections:
            if conn.primitive_a_id == prim.id:
                connected_portals.add(conn.portal_a_id)
            if conn.primitive_b_id == prim.id:
                connected_portals.add(conn.portal_b_id)

        # Check each open portal
        for portal in footprint.portals:
            if portal.id in connected_portals:
                continue

            world_cell = portal.world_cell(
                prim.origin_cell, prim.rotation,
                footprint.width_cells, footprint.depth_cells
            )
            world_dir = portal.rotated_direction(prim.rotation)

            # Check if stair can fit here
            result = _find_vstair_placement_space_only(
                world_cell, world_dir,
                lower_occupied, upper_occupied,
                map_width, map_height
            )

            if result is not None:
                origin, rotation = result
                # Calculate distance from center (for sorting)
                dist = abs(origin.x - center_x) + abs(origin.y - center_y)
                candidates.append((dist, origin, rotation, prim.id, portal.id))

    # Sort by distance (closest to center first)
    candidates.sort(key=lambda x: x[0])

    # Return without distance component
    return [(c[1], c[2], c[3], c[4]) for c in candidates]


def _get_stair_reserved_cells(
    origin: CellCoord,
    rotation: int,
) -> Set[Tuple[int, int]]:
    """
    Get the cells that should be reserved for a stair placement.

    Args:
        origin: Stair origin cell
        rotation: Stair rotation

    Returns:
        Set of (x, y) tuples for cells to reserve
    """
    return set(_get_primitive_cells('VerticalStairHall', origin, rotation))


def _find_stair_reservation_location(
    entry_cell: CellCoord,
    entry_dir: PortalDirection,
    occupied: Set[Tuple[int, int]],
    map_width: int,
    map_height: int,
) -> Optional[Tuple[CellCoord, int]]:
    """
    Find a good location to reserve for the next stair.

    Tries to place the stair on the opposite side of the map from the entry point,
    which encourages good layout flow and ensures stairs don't cluster.

    Args:
        entry_cell: Cell where the current floor's entry (stair top) is
        entry_dir: Direction the entry portal faces
        occupied: Currently occupied cells (including current stair)
        map_width: Grid width
        map_height: Grid height

    Returns:
        Tuple of (origin, rotation) for the stair, or None if no valid location
    """
    vstair_footprint = PRIMITIVE_FOOTPRINTS.get('VerticalStairHall')
    if vstair_footprint is None:
        return None

    # Target the opposite quadrant of the map from the entry point
    center_x, center_y = map_width // 2, map_height // 2

    # Calculate target region (opposite corner from entry)
    if entry_cell.x < center_x:
        target_x_range = range(center_x, map_width - 3)
    else:
        target_x_range = range(3, center_x)

    if entry_cell.y < center_y:
        target_y_range = range(center_y, map_height - 3)
    else:
        target_y_range = range(3, center_y)

    # Try to find a valid stair location in the target region
    vstair_width = vstair_footprint.width_cells
    vstair_depth = vstair_footprint.depth_cells

    best_location = None
    best_distance = 0  # We want maximum distance from entry

    for x in target_x_range:
        for y in target_y_range:
            origin = CellCoord(x, y)

            # Try each rotation
            for rotation in [0, 90, 180, 270]:
                # Get cells for this placement
                cells = _get_primitive_cells('VerticalStairHall', origin, rotation)

                # Check bounds and occupation
                valid = True
                for cx, cy in cells:
                    if cx < 0 or cx >= map_width or cy < 0 or cy >= map_height:
                        valid = False
                        break
                    if (cx, cy) in occupied:
                        valid = False
                        break

                if valid:
                    # Calculate distance from entry
                    dist = abs(x - entry_cell.x) + abs(y - entry_cell.y)
                    if dist > best_distance:
                        best_distance = dist
                        best_location = (origin, rotation)

    if best_location:
        _debug(f"[STAIR_RESERVE] Found reservation at ({best_location[0].x}, {best_location[0].y}) rot={best_location[1]}, dist={best_distance}")
    else:
        _debug(f"[STAIR_RESERVE] Could not find reservation location")

    return best_location


def _count_stair_friendly_portals(
    open_portals: List[Tuple[str, str, CellCoord, PortalDirection]],
    occupied: Set[Tuple[int, int]],
    map_width: int,
    map_height: int,
) -> int:
    """
    Count how many open portals can accommodate a stair placement.

    Args:
        open_portals: List of (prim_id, portal_id, cell, direction) tuples
        occupied: Set of occupied cells
        map_width: Grid width
        map_height: Grid height

    Returns:
        Number of portals that can fit a stair
    """
    count = 0
    for _, _, portal_cell, portal_dir in open_portals:
        result = _find_vstair_placement_space_only(
            portal_cell, portal_dir,
            occupied, set(),  # Empty upper_occupied
            map_width, map_height
        )
        if result is not None:
            count += 1
    return count


def _ensure_stair_friendly_portal(
    layout: DungeonLayout,
    combined_layout: DungeonLayout,
    occupied: Set[Tuple[int, int]],
    map_width: int,
    map_height: int,
    z_offset: float,
    rng: random.Random,
    max_extensions: int = 5,
) -> bool:
    """
    Ensure the layout has at least one open portal that can accommodate a stair.

    If no stair-friendly portal exists, extend the layout with halls/junctions
    to create one near the center of the map.

    Args:
        max_extensions: Maximum number of extension primitives to add (recursion limit)

    Returns:
        True if a stair-friendly portal exists or was created
    """
    if max_extensions <= 0:
        _debug(f"[ENSURE_STAIR] Max extensions reached, giving up")
        return False
    vstair_footprint = PRIMITIVE_FOOTPRINTS.get('VerticalStairHall')
    if vstair_footprint is None:
        return False

    # First, check if any existing open portal can accommodate a stair
    # Sort by distance to center - prefer center-ish portals
    center_x, center_y = map_width // 2, map_height // 2
    candidate_portals = []

    for prim in layout.primitives.values():
        footprint = PRIMITIVE_FOOTPRINTS.get(prim.primitive_type)
        if footprint is None:
            continue

        connected_portals = set()
        for conn in combined_layout.connections:
            if conn.primitive_a_id == prim.id:
                connected_portals.add(conn.portal_a_id)
            if conn.primitive_b_id == prim.id:
                connected_portals.add(conn.portal_b_id)

        for portal in footprint.portals:
            if portal.id in connected_portals:
                continue

            world_cell = portal.world_cell(
                prim.origin_cell, prim.rotation,
                footprint.width_cells, footprint.depth_cells
            )
            world_dir = portal.rotated_direction(prim.rotation)

            # Check if a stair can be placed here - _find_vstair_placement_space_only
            # handles bounds checking internally
            result = _find_vstair_placement_space_only(
                world_cell, world_dir,
                occupied, set(),  # Empty upper_occupied
                map_width, map_height
            )
            if result is not None:
                dist = abs(world_cell.x - center_x) + abs(world_cell.y - center_y)
                candidate_portals.append((dist, world_cell, world_dir))

    # Return the most central stair-friendly portal
    if candidate_portals:
        candidate_portals.sort(key=lambda x: x[0])
        best = candidate_portals[0]
        _debug(f"[ENSURE_STAIR] Found stair-friendly portal at ({best[1].x}, {best[1].y}) facing {best[2].name}")
        return True

    _debug(f"[ENSURE_STAIR] No stair-friendly portals found, attempting to extend layout")

    # No stair-friendly portal exists - try to extend the layout
    # Find an open portal closest to the map center
    center_x, center_y = map_width // 2, map_height // 2
    best_portal = None
    best_distance = float('inf')

    for prim in layout.primitives.values():
        footprint = PRIMITIVE_FOOTPRINTS.get(prim.primitive_type)
        if footprint is None:
            continue

        connected_portals = set()
        for conn in combined_layout.connections:
            if conn.primitive_a_id == prim.id:
                connected_portals.add(conn.portal_a_id)
            if conn.primitive_b_id == prim.id:
                connected_portals.add(conn.portal_b_id)

        for portal in footprint.portals:
            if portal.id in connected_portals:
                continue

            world_cell = portal.world_cell(
                prim.origin_cell, prim.rotation,
                footprint.width_cells, footprint.depth_cells
            )
            world_dir = portal.rotated_direction(prim.rotation)

            # Check if we can place ANY primitive here (junction/hall)
            for ext_type in ['TJunction', 'Crossroads', 'StraightHall']:
                ext_result = _find_placement(
                    ext_type, world_cell, world_dir,
                    occupied, map_width, map_height
                )
                if ext_result is not None:
                    # Calculate distance to center
                    dist = abs(world_cell.x - center_x) + abs(world_cell.y - center_y)
                    if dist < best_distance:
                        best_distance = dist
                        best_portal = (prim.id, portal.id, world_cell, world_dir, ext_type, ext_result)
                    break

    if best_portal is None:
        _debug(f"[ENSURE_STAIR] Cannot extend layout - no placement found")
        return False

    prim_id, portal_id, portal_cell, portal_dir, ext_type, (ext_origin, ext_rotation, ext_portal_id) = best_portal

    # Place the extension primitive with z_offset and collision checking
    ext_prim = _place_primitive(
        layout, ext_type, ext_origin, ext_rotation, occupied, rng,
        z_offset=z_offset, combined_layout=combined_layout
    )
    if ext_prim is None:
        _debug(f"[ENSURE_STAIR] Failed to place extension {ext_type}")
        return False

    # Add to combined_layout (z_offset already set by _place_primitive)
    combined_layout.primitives[ext_prim.id] = ext_prim

    # Connect extension to the source portal
    conn = Connection(
        primitive_a_id=prim_id,
        portal_a_id=portal_id,
        primitive_b_id=ext_prim.id,
        portal_b_id=ext_portal_id
    )
    layout.connections.append(conn)
    combined_layout.connections.append(conn)

    _debug(f"[ENSURE_STAIR] Added extension {ext_type} at ({ext_origin.x}, {ext_origin.y})")

    # Recursively check if we now have a stair-friendly portal
    return _ensure_stair_friendly_portal(
        layout, combined_layout, occupied, map_width, map_height, z_offset, rng,
        max_extensions=max_extensions - 1
    )


def _find_adjacent_network_portal(
    layout: DungeonLayout,
    primitive_id: str,
    open_portal_id: str,
    z_level: float
) -> Optional[Tuple[str, str]]:
    """Find an adjacent primitive's open portal that can connect to the given portal.

    Looks for a primitive in the main floor network that has an unconnected portal
    facing toward the given primitive's open portal.

    Args:
        layout: The combined dungeon layout
        primitive_id: ID of the primitive with the open portal
        open_portal_id: ID of the open portal on that primitive
        z_level: Z-level to search on

    Returns:
        Tuple of (adjacent_prim_id, adjacent_portal_id) or None if not found
    """
    prim = layout.primitives.get(primitive_id)
    if not prim:
        return None

    footprint = PRIMITIVE_FOOTPRINTS.get(prim.primitive_type)
    if not footprint:
        return None

    # Find the open portal
    open_portal = None
    for portal in footprint.portals:
        if portal.id == open_portal_id:
            open_portal = portal
            break

    if not open_portal:
        return None

    # Get world position and direction of the open portal
    portal_cell = prim.get_portal_world_cell(open_portal)
    portal_dir = open_portal.rotated_direction(prim.rotation)

    # Find cell adjacent to this portal (where a connection would come from)
    adjacent_cell = portal_cell.neighbor(portal_dir)

    # Check if any primitive on this z-level has an open portal at that cell
    for other_id, other_prim in layout.primitives.items():
        if other_id == primitive_id:
            continue

        # Must be same z-level (within tolerance)
        if abs(other_prim.z_offset - z_level) > 2:
            continue

        other_footprint = PRIMITIVE_FOOTPRINTS.get(other_prim.primitive_type)
        if not other_footprint:
            continue

        for other_portal in other_footprint.portals:
            other_cell = other_prim.get_portal_world_cell(other_portal)
            other_dir = other_portal.rotated_direction(other_prim.rotation)

            # Check: adjacent cell matches AND directions face each other
            if other_cell == adjacent_cell and other_dir == portal_dir.opposite():
                # Verify this portal isn't already connected
                if not _is_portal_connected(layout, other_id, other_portal.id):
                    _debug(f"[FIND_ADJ_PORTAL] Found: {other_id[:8]}:{other_portal.id} at cell ({adjacent_cell.x}, {adjacent_cell.y})")
                    return (other_id, other_portal.id)

    return None


def generate_multi_floor_layout(
    floor_count: int = 2,
    rooms_per_floor: int = 5,
    map_width: int = 20,
    map_height: int = 20,
    seed: Optional[int] = None,
    complexity: int = 3,
    auto_connect_floors: bool = True,
    preferred_room_types: Optional[List[str]] = None,
    preferred_hall_types: Optional[List[str]] = None,
    room_probability: float = 0.4,
    min_hall_between_rooms: int = 1,
    allow_dead_ends: bool = True,
    secret_room_frequency: int = 0,
    max_retry_attempts: int = 3,
) -> Tuple[DungeonLayout, int]:
    """
    Generate a multi-floor dungeon layout with optional automatic stair connections.

    This improved version includes:
    - Z-level aware connectivity validation
    - Automatic retry with different seeds if floor connections fail
    - Verification that all floors are traversable

    Args:
        floor_count: Number of floors to generate (1-4)
        rooms_per_floor: Number of rooms per floor
        map_width: Grid width in cells
        map_height: Grid height in cells
        seed: Random seed for reproducibility (None = random)
        complexity: 1-5, affects branching and layout density
        auto_connect_floors: If True, automatically place VerticalStairHalls between floors
        preferred_room_types: List of room types to prefer (None = use defaults)
        preferred_hall_types: List of hall types to prefer (None = use defaults)
        room_probability: Base probability of placing a room (0.0-1.0)
        min_hall_between_rooms: Minimum halls between consecutive rooms
        allow_dead_ends: Whether to allow dead-end corridors
        secret_room_frequency: Percentage of rooms that should be SecretChambers (0-100)
        max_retry_attempts: Maximum number of retry attempts if stair placement fails

    Returns:
        Tuple of (DungeonLayout, actual_seed_used)
    """
    _debug(f"\n[MULTI_FLOOR] === Starting multi-floor layout generation ===")
    _debug(f"[MULTI_FLOOR] Parameters: floors={floor_count}, rooms_per_floor={rooms_per_floor}")

    # Validate floor count
    floor_count = max(1, min(4, floor_count))

    # Multi-floor layouts need larger maps to accommodate stair placement
    # Minimum map size scales with floor count to ensure room for multiple stairs
    # Increased to give more room for stair placement at edges
    min_map_size = 30 + (floor_count - 1) * 15  # 30 for 1 floor, 45 for 2, 60 for 3, 75 for 4
    if map_width < min_map_size or map_height < min_map_size:
        _debug(f"[MULTI_FLOOR] Increasing map size from {map_width}x{map_height} to {min_map_size}x{min_map_size} for {floor_count} floors")
        map_width = max(map_width, min_map_size)
        map_height = max(map_height, min_map_size)

    # Initialize random seed
    original_seed = seed
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    random.seed(seed)
    _debug(f"[MULTI_FLOOR] Using seed: {seed}")

    # Get floor level Z-offsets in order (Basement -> Ground -> Upper -> Tower)
    floor_names = ['Basement', 'Ground', 'Upper', 'Tower']
    floor_z_offsets = [FLOOR_LEVELS[name] for name in floor_names[:floor_count]]

    # Start with Ground floor (index 1 if available, else first floor)
    if floor_count >= 2:
        start_floor_idx = 1  # Ground
    else:
        start_floor_idx = 0

    combined_layout = DungeonLayout()
    floor_layouts: Dict[int, DungeonLayout] = {}
    floor_occupied: Dict[int, Set[Tuple[int, int]]] = {}
    stairs_placed_count = 0
    stairs_needed_count = floor_count - 1 if floor_count > 1 else 0

    vstair_footprint = PRIMITIVE_FOOTPRINTS.get('VerticalStairHall')

    # REVISED STRATEGY: Generate floors iteratively with stair-driven connectivity
    # Each floor (after the first) is generated emanating from the stair's top portal
    # to ensure connectivity.

    for floor_idx, z_offset in enumerate(floor_z_offsets):
        floor_name = floor_names[floor_idx]
        _debug(f"\n[MULTI_FLOOR] --- Generating floor {floor_name} (z={z_offset}) ---")

        floor_seed = seed + floor_idx
        floor_occupied[floor_idx] = set()

        if floor_idx == 0:
            # First floor: generate normally, but exclude tall rooms for multi-floor
            # layouts to prevent cross-floor collisions
            safe_room_types = None
            if preferred_room_types:
                safe_room_types = [r for r in preferred_room_types if r not in TALL_ROOMS]
            else:
                # Use filtered ROOM_TYPES if no preferences given
                safe_room_types = [r for r in ROOM_TYPES if r not in TALL_ROOMS]

            # Require stair space on all floors except the top floor
            needs_stair_space = floor_idx < floor_count - 1
            floor_layout, _ = generate_random_layout(
                room_count=rooms_per_floor,
                map_width=map_width,
                map_height=map_height,
                seed=floor_seed,
                complexity=complexity,
                preferred_room_types=safe_room_types,
                preferred_hall_types=preferred_hall_types,
                room_probability=room_probability,
                min_hall_between_rooms=min_hall_between_rooms,
                allow_dead_ends=allow_dead_ends,
                secret_room_frequency=secret_room_frequency,
                exclude_tall=True,  # Multi-floor: exclude tall rooms to prevent collisions
                require_stair_space=needs_stair_space,
            )

            # Track occupied cells
            for prim in floor_layout.primitives.values():
                cells = _get_primitive_cells(prim.primitive_type, prim.origin_cell, prim.rotation)
                floor_occupied[floor_idx].update(cells)

            # Update z_offset
            for prim in floor_layout.primitives.values():
                prim.z_offset = float(z_offset)

            # Merge into combined layout (with deduplication check for connections)
            for prim in floor_layout.primitives.values():
                combined_layout.primitives[prim.id] = prim
            for conn in floor_layout.connections:
                if not _connection_exists(combined_layout.connections, conn):
                    combined_layout.connections.append(conn)

            # If this isn't the last floor and auto_connect is enabled,
            # ensure there's a stair-friendly portal for the next floor
            if auto_connect_floors and floor_idx < floor_count - 1:
                param_rng = random.Random(seed + floor_idx)
                _ensure_stair_friendly_portal(
                    floor_layout, combined_layout, floor_occupied[floor_idx],
                    map_width, map_height, z_offset, param_rng
                )

            floor_layouts[floor_idx] = floor_layout
            _debug(f"[MULTI_FLOOR] Floor {floor_name}: {len(floor_layout.primitives)} primitives")

        else:
            # Subsequent floors: place stair FIRST, then generate floor from stair top
            prev_idx = floor_idx - 1
            prev_z = floor_z_offsets[prev_idx]

            if auto_connect_floors and vstair_footprint:
                _debug(f"[MULTI_FLOOR] Placing stair to connect floor {prev_idx} -> {floor_idx}")

                # Find stair placement from previous floor
                stair_result = _place_stair_and_get_top_portal(
                    combined_layout,
                    floor_layouts[prev_idx],
                    floor_occupied[prev_idx],
                    floor_occupied[floor_idx],
                    prev_z,
                    z_offset,
                    map_width,
                    map_height,
                )

                if stair_result:
                    stair_prim, top_portal_cell, top_portal_dir = stair_result
                    _debug(f"[MULTI_FLOOR] Stair placed, top portal at ({top_portal_cell.x}, {top_portal_cell.y}) facing {top_portal_dir.name}")

                    # If this isn't the last floor, reserve space for the NEXT stair
                    # BEFORE generating this floor, to prevent floor generation from
                    # blocking the stair location
                    reserved_next_stair_cells: Set[Tuple[int, int]] = set()
                    needs_stair_space = floor_idx < floor_count - 1
                    if needs_stair_space:
                        # Find a suitable stair location opposite to current stair
                        # to encourage good layout connectivity
                        reserve_origin = _find_stair_reservation_location(
                            top_portal_cell, top_portal_dir,
                            floor_occupied[floor_idx],
                            map_width, map_height
                        )
                        if reserve_origin:
                            reserved_next_stair_cells = _get_stair_reserved_cells(
                                reserve_origin[0], reserve_origin[1]
                            )
                            floor_occupied[floor_idx].update(reserved_next_stair_cells)
                            _debug(f"[MULTI_FLOOR] Reserved {len(reserved_next_stair_cells)} cells for next stair at ({reserve_origin[0].x}, {reserve_origin[0].y})")

                    # Generate this floor starting from stair's top portal
                    floor_layout = _generate_floor_from_portal(
                        top_portal_cell,
                        top_portal_dir,
                        stair_prim.id,
                        z_offset,
                        rooms_per_floor,
                        map_width,
                        map_height,
                        floor_seed,
                        complexity,
                        preferred_room_types,
                        preferred_hall_types,
                        room_probability,
                        min_hall_between_rooms,
                        allow_dead_ends,
                        secret_room_frequency,
                        floor_occupied[floor_idx],
                        combined_layout=combined_layout,
                        require_stair_space=needs_stair_space,
                    )

                    # VERIFY: Check that both stair portals are connected
                    bottom_conn, top_conn = _verify_stair_connectivity(combined_layout, stair_prim)
                    _debug(f"[MULTI_FLOOR] Stair connectivity: bottom={bottom_conn}, top={top_conn}")

                    if not top_conn:
                        _debug(f"[MULTI_FLOOR] WARNING: Stair top portal not connected, attempting recovery")
                        param_rng = random.Random(seed + floor_idx + 1000)
                        recovery_success = _attempt_stair_top_recovery(
                            combined_layout,
                            floor_layout,
                            stair_prim,
                            floor_occupied[floor_idx],
                            map_width,
                            map_height,
                            z_offset,
                            param_rng,
                        )

                        if not recovery_success:
                            _debug(f"[MULTI_FLOOR] Recovery failed, rolling back stair and trying extension approach")
                            _rollback_stair_placement(
                                combined_layout,
                                stair_prim,
                                floor_occupied[prev_idx],
                                floor_occupied[floor_idx],
                            )
                            stair_result = None  # Fall through to extension approach
                        else:
                            stairs_placed_count += 1
                    else:
                        stairs_placed_count += 1

                    # Release reserved stair cells before checking for stair-friendly portal
                    # The reserved cells may have been blocked by floor generation anyway,
                    # or _ensure_stair_friendly_portal may find a better location
                    if reserved_next_stair_cells:
                        for cell in reserved_next_stair_cells:
                            floor_occupied[floor_idx].discard(cell)
                        _debug(f"[MULTI_FLOOR] Released {len(reserved_next_stair_cells)} reserved stair cells")

                    # If stair was placed successfully and this isn't the last floor,
                    # ensure there's a stair-friendly portal for the NEXT stair
                    if stair_result is not None and floor_idx < floor_count - 1:
                        _debug(f"[MULTI_FLOOR] Ensuring stair-friendly portal for next floor (floor_idx={floor_idx}, floor_count={floor_count})")
                        param_rng = random.Random(seed + floor_idx + 500)
                        ensured = _ensure_stair_friendly_portal(
                            floor_layout, combined_layout, floor_occupied[floor_idx],
                            map_width, map_height, z_offset, param_rng
                        )
                        _debug(f"[MULTI_FLOOR] Ensure stair-friendly result: {ensured}")

                if stair_result is None:
                    _debug(f"[MULTI_FLOOR] Stair-first approach failed, trying extension-based approach")

                    # PHASE 4 FIX: Find viable stair positions BEFORE generating upper floor
                    # Reserve cells for the best stair candidate to prevent blocking
                    potential_stair_positions = _find_all_viable_stair_positions(
                        floor_layouts[prev_idx],
                        combined_layout,
                        floor_occupied[prev_idx],
                        floor_occupied[floor_idx],  # Currently empty for new floor
                        map_width,
                        map_height,
                    )

                    reserved_cells: Set[Tuple[int, int]] = set()
                    if potential_stair_positions:
                        # Reserve cells for best stair position
                        best_origin, best_rotation, _, _ = potential_stair_positions[0]
                        reserved_cells = _get_stair_reserved_cells(best_origin, best_rotation)
                        floor_occupied[floor_idx].update(reserved_cells)
                        _debug(f"[MULTI_FLOOR] Reserved {len(reserved_cells)} cells for stair at ({best_origin.x}, {best_origin.y})")

                    # Generate this floor with reserved cells blocked
                    # Require stair space on all floors except the top floor
                    needs_stair_space = floor_idx < floor_count - 1
                    floor_layout, _ = generate_random_layout(
                        room_count=rooms_per_floor,
                        map_width=map_width,
                        map_height=map_height,
                        seed=floor_seed,
                        complexity=complexity,
                        preferred_room_types=preferred_room_types,
                        preferred_hall_types=preferred_hall_types,
                        room_probability=room_probability,
                        min_hall_between_rooms=min_hall_between_rooms,
                        allow_dead_ends=allow_dead_ends,
                        secret_room_frequency=secret_room_frequency,
                        exclude_tall=True,  # Multi-floor: exclude tall rooms
                        require_stair_space=needs_stair_space,
                    )

                    for prim in floor_layout.primitives.values():
                        cells = _get_primitive_cells(prim.primitive_type, prim.origin_cell, prim.rotation)
                        floor_occupied[floor_idx].update(cells)

                    # Release reserved cells before stair placement (they'll be re-claimed by stair)
                    for cell in reserved_cells:
                        floor_occupied[floor_idx].discard(cell)

                    # Now try extension-based stair placement between the two floors
                    if _place_vertical_stair_with_extension(
                        combined_layout,
                        floor_layouts[prev_idx],
                        floor_layout,
                        floor_occupied[prev_idx],
                        floor_occupied[floor_idx],
                        prev_z,
                        z_offset,
                        map_width,
                        map_height,
                    ):
                        stairs_placed_count += 1
                        _debug(f"[MULTI_FLOOR] Extension-based stair placed successfully")

                        # Ensure stair-friendly portal for next floor if not last floor
                        if floor_idx < floor_count - 1:
                            param_rng = random.Random(seed + floor_idx + 600)
                            _ensure_stair_friendly_portal(
                                floor_layout, combined_layout, floor_occupied[floor_idx],
                                map_width, map_height, z_offset, param_rng
                            )
                    else:
                        _debug(f"[MULTI_FLOOR] WARNING: Could not connect floors - layout will be isolated")
            else:
                # No auto-connect: generate floor normally
                # Still require stair space in case user manually adds stairs later
                needs_stair_space = floor_idx < floor_count - 1
                floor_layout, _ = generate_random_layout(
                    room_count=rooms_per_floor,
                    map_width=map_width,
                    map_height=map_height,
                    seed=floor_seed,
                    complexity=complexity,
                    preferred_room_types=preferred_room_types,
                    preferred_hall_types=preferred_hall_types,
                    room_probability=room_probability,
                    min_hall_between_rooms=min_hall_between_rooms,
                    allow_dead_ends=allow_dead_ends,
                    secret_room_frequency=secret_room_frequency,
                    exclude_tall=True,  # Multi-floor: exclude tall rooms
                    require_stair_space=needs_stair_space,
                )

                for prim in floor_layout.primitives.values():
                    cells = _get_primitive_cells(prim.primitive_type, prim.origin_cell, prim.rotation)
                    floor_occupied[floor_idx].update(cells)

            # Update z_offset
            for prim in floor_layout.primitives.values():
                prim.z_offset = float(z_offset)

            # Merge into combined layout (with deduplication check for connections)
            for prim in floor_layout.primitives.values():
                combined_layout.primitives[prim.id] = prim
            for conn in floor_layout.connections:
                if not _connection_exists(combined_layout.connections, conn):
                    combined_layout.connections.append(conn)

            floor_layouts[floor_idx] = floor_layout
            _debug(f"[MULTI_FLOOR] Floor {floor_name}: {len(floor_layout.primitives)} primitives")

    # Run Z-aware connectivity validation
    trav_result = validate_multi_floor_path(combined_layout)

    _debug(f"\n[MULTI_FLOOR] === Multi-floor generation complete ===")
    _debug(f"[MULTI_FLOOR] Total primitives: {len(combined_layout.primitives)}")
    _debug(f"[MULTI_FLOOR] Total connections: {len(combined_layout.connections)}")
    _debug(f"[MULTI_FLOOR] Stairs placed: {stairs_placed_count}/{stairs_needed_count}")
    _debug(f"[MULTI_FLOOR] Traversability: {trav_result.traversable_connections} OK, "
           f"{trav_result.non_traversable_connections} blocked")

    if trav_result.isolated_region_count > 0:
        _debug(f"[MULTI_FLOOR] WARNING: {trav_result.isolated_region_count} isolated region(s) detected!")
        for issue in trav_result.issues:
            if issue.is_error:
                _debug(f"[MULTI_FLOOR]   - {issue.message}")

    return combined_layout, seed


def _place_vertical_stair(
    combined_layout: DungeonLayout,
    lower_floor: DungeonLayout,
    upper_floor: DungeonLayout,
    lower_occupied: Set[Tuple[int, int]],
    upper_occupied: Set[Tuple[int, int]],
    lower_z: float,
    map_width: int,
    map_height: int,
) -> bool:
    """
    Place a VerticalStairHall connecting two floors.

    Finds open portals on the lower floor that have space on both floors
    for the stair footprint, AND connects the top portal to the upper floor.

    Returns:
        True if stair was placed successfully with BOTH portals connected
    """
    vstair_footprint = PRIMITIVE_FOOTPRINTS.get('VerticalStairHall')
    if vstair_footprint is None:
        return False

    vstair_width = vstair_footprint.width_cells
    vstair_depth = vstair_footprint.depth_cells

    # Collect open portals from lower floor
    lower_open_portals: List[Tuple[str, str, CellCoord, PortalDirection]] = []
    for prim in lower_floor.primitives.values():
        footprint = PRIMITIVE_FOOTPRINTS.get(prim.primitive_type)
        if footprint is None:
            continue

        # Check which portals are connected
        connected_portals = set()
        for conn in combined_layout.connections:
            if conn.primitive_a_id == prim.id:
                connected_portals.add(conn.portal_a_id)
            if conn.primitive_b_id == prim.id:
                connected_portals.add(conn.portal_b_id)

        # Collect unconnected portals
        for portal in footprint.portals:
            if portal.id in connected_portals:
                continue

            world_cell = portal.world_cell(
                prim.origin_cell, prim.rotation,
                footprint.width_cells, footprint.depth_cells
            )
            world_dir = portal.rotated_direction(prim.rotation)
            lower_open_portals.append((prim.id, portal.id, world_cell, world_dir))

    # Collect open portals from upper floor for connecting the top portal
    upper_open_portals: Dict[Tuple[int, int, str], Tuple[str, str]] = {}
    for prim in upper_floor.primitives.values():
        footprint = PRIMITIVE_FOOTPRINTS.get(prim.primitive_type)
        if footprint is None:
            continue

        # Check which portals are connected
        connected_portals = set()
        for conn in combined_layout.connections:
            if conn.primitive_a_id == prim.id:
                connected_portals.add(conn.portal_a_id)
            if conn.primitive_b_id == prim.id:
                connected_portals.add(conn.portal_b_id)

        # Collect unconnected portals with their world positions and directions
        for portal in footprint.portals:
            if portal.id in connected_portals:
                continue

            world_cell = portal.world_cell(
                prim.origin_cell, prim.rotation,
                footprint.width_cells, footprint.depth_cells
            )
            world_dir = portal.rotated_direction(prim.rotation)
            # Key by (cell_x, cell_y, direction_needed) where direction_needed is what the stair needs
            # The stair's top portal faces some direction, upper floor portal must face opposite
            key = (world_cell.x, world_cell.y, world_dir.name)
            upper_open_portals[key] = (prim.id, portal.id)

    # Shuffle portals for randomness
    random.shuffle(lower_open_portals)

    # Try to place stair at each open portal
    for prim_id, portal_id, portal_cell, portal_dir in lower_open_portals:
        # VerticalStairHall's bottom portal faces SOUTH
        # We need to find a rotation where the bottom portal aligns

        result = _find_vstair_placement(
            portal_cell, portal_dir,
            lower_occupied, upper_occupied,
            map_width, map_height
        )

        if result is None:
            continue

        origin, rotation, connecting_portal_id = result

        # CRITICAL: Before placing, check if we can connect the TOP portal to upper floor
        # Find where the top portal would be located
        top_portal = None
        for portal in vstair_footprint.portals:
            if portal.id == 'top':
                top_portal = portal
                break

        if top_portal is None:
            _debug("[PLACE_VSTAIR] ERROR: VerticalStairHall has no 'top' portal!")
            continue

        # Calculate the top portal's world position and direction
        top_world_cell = top_portal.world_cell(
            origin, rotation,
            vstair_footprint.width_cells, vstair_footprint.depth_cells
        )
        top_world_dir = top_portal.rotated_direction(rotation)

        # The upper floor portal must be adjacent to the stair's top portal
        # and face OPPOSITE to the stair's top portal direction
        adjacent_cell = top_world_cell.neighbor(top_world_dir)
        required_upper_dir = top_world_dir.opposite()

        # Look for matching upper floor portal
        upper_key = (adjacent_cell.x, adjacent_cell.y, required_upper_dir.name)
        upper_match = upper_open_portals.get(upper_key)

        if upper_match is None:
            _debug(f"[PLACE_VSTAIR] No upper floor portal at ({adjacent_cell.x}, {adjacent_cell.y}) facing {required_upper_dir.name}")
            continue

        upper_prim_id, upper_portal_id = upper_match

        # SUCCESS: We can connect both bottom AND top portals!
        _debug(f"[PLACE_VSTAIR] Found complete stair connection:")
        _debug(f"[PLACE_VSTAIR]   Lower: {prim_id[:8]}:{portal_id} -> stair:bottom")
        _debug(f"[PLACE_VSTAIR]   Upper: stair:top -> {upper_prim_id[:8]}:{upper_portal_id}")

        # Place the stair
        stair_prim = PlacedPrimitive.create(
            primitive_type='VerticalStairHall',
            origin=origin,
            rotation=rotation,
            z_offset=lower_z,
            parameters={}
        )
        # CRITICAL: Set footprint so portal alignment works correctly
        stair_prim.set_footprint(vstair_footprint)

        # NOTE: We skip 3D collision check for stairs because:
        # 1. Stairs intentionally span two floors (lower_z to lower_z + 320)
        # 2. The 2D cell checking (lower_occupied, upper_occupied) already ensures
        #    proper XY placement on both floors
        # 3. The portals create open passageways, so the "collision" is expected
        #    and doesn't represent actual sealed geometry overlap

        combined_layout.add_primitive(stair_prim)

        # Mark cells as occupied on BOTH floors
        stair_cells = _get_primitive_cells('VerticalStairHall', origin, rotation)
        lower_occupied.update(stair_cells)
        upper_occupied.update(stair_cells)

        # Create connection from lower floor primitive to stair's bottom portal
        combined_layout.connections.append(Connection(
            primitive_a_id=prim_id,
            portal_a_id=portal_id,
            primitive_b_id=stair_prim.id,
            portal_b_id='bottom'  # Always use 'bottom' for lower connection
        ))

        # CRITICAL: Create connection from stair's top portal to upper floor primitive
        combined_layout.connections.append(Connection(
            primitive_a_id=stair_prim.id,
            portal_a_id='top',
            primitive_b_id=upper_prim_id,
            portal_b_id=upper_portal_id
        ))

        return True

    return False


def _ensure_floor_connection(
    combined_layout: DungeonLayout,
    lower_floor: DungeonLayout,
    upper_floor: DungeonLayout,
    floor_occupied: Dict[int, Set[Tuple[int, int]]],
    lower_idx: int,
    upper_idx: int,
    lower_z: float,
    map_width: int,
    map_height: int,
) -> bool:
    """
    Ensure there is a traversable connection between two floors.

    This function uses _place_vertical_stair() which now properly connects
    BOTH the bottom portal (to lower floor) AND the top portal (to upper floor).

    Args:
        combined_layout: The combined dungeon layout
        lower_floor: Layout for the lower floor
        upper_floor: Layout for the upper floor
        floor_occupied: Dict of occupied cells per floor index
        lower_idx: Index of lower floor
        upper_idx: Index of upper floor
        lower_z: Z-offset of lower floor
        map_width: Grid width limit
        map_height: Grid height limit

    Returns:
        True if a connection was established
    """
    lower_occupied = floor_occupied.get(lower_idx, set())
    upper_occupied = floor_occupied.get(upper_idx, set())

    # _place_vertical_stair now handles both bottom AND top portal connections
    if _place_vertical_stair(
        combined_layout, lower_floor, upper_floor,
        lower_occupied, upper_occupied, lower_z, map_width, map_height
    ):
        return True

    _debug(f"[ENSURE_CONN] Could not establish floor connection - no matching portal pairs found")
    return False


def _place_vertical_stair_with_extension(
    combined_layout: DungeonLayout,
    lower_floor: DungeonLayout,
    upper_floor: DungeonLayout,
    lower_occupied: Set[Tuple[int, int]],
    upper_occupied: Set[Tuple[int, int]],
    lower_z: float,
    upper_z: float,
    map_width: int,
    map_height: int,
) -> bool:
    """
    Place a VerticalStairHall and extend the upper floor if needed.

    This is a fallback when no natural portal alignment exists. It:
    1. Finds where a stair CAN be placed (lower floor has open portal + space)
    2. Extends the upper floor with a hall to create a matching portal

    Returns:
        True if stair was placed with both portals connected
    """
    vstair_footprint = PRIMITIVE_FOOTPRINTS.get('VerticalStairHall')
    if vstair_footprint is None:
        return False

    # Collect open portals from lower floor
    lower_open_portals: List[Tuple[str, str, CellCoord, PortalDirection]] = []
    for prim in lower_floor.primitives.values():
        footprint = PRIMITIVE_FOOTPRINTS.get(prim.primitive_type)
        if footprint is None:
            continue

        connected_portals = set()
        for conn in combined_layout.connections:
            if conn.primitive_a_id == prim.id:
                connected_portals.add(conn.portal_a_id)
            if conn.primitive_b_id == prim.id:
                connected_portals.add(conn.portal_b_id)

        for portal in footprint.portals:
            if portal.id in connected_portals:
                continue

            world_cell = portal.world_cell(
                prim.origin_cell, prim.rotation,
                footprint.width_cells, footprint.depth_cells
            )
            world_dir = portal.rotated_direction(prim.rotation)
            lower_open_portals.append((prim.id, portal.id, world_cell, world_dir))

    random.shuffle(lower_open_portals)

    # Find top portal info from vstair footprint
    top_portal = None
    for portal in vstair_footprint.portals:
        if portal.id == 'top':
            top_portal = portal
            break

    if top_portal is None:
        return False

    # Collect all upper floor primitive cells for distance calculation
    upper_floor_cells: Set[Tuple[int, int]] = set()
    for prim in upper_floor.primitives.values():
        cells = _get_primitive_cells(prim.primitive_type, prim.origin_cell, prim.rotation)
        upper_floor_cells.update(cells)

    def _distance_to_upper_floor(cell: CellCoord) -> float:
        """Calculate minimum Manhattan distance from cell to any upper floor cell."""
        if not upper_floor_cells:
            return float('inf')
        return min(abs(cell.x - c[0]) + abs(cell.y - c[1]) for c in upper_floor_cells)

    # Collect all valid stair placement candidates with distance scores
    stair_candidates: List[Tuple[float, str, str, CellCoord, PortalDirection, CellCoord, int, CellCoord, PortalDirection]] = []

    for prim_id, portal_id, portal_cell, portal_dir in lower_open_portals:
        # Find valid stair placement (just space check, not upper floor portal)
        result = _find_vstair_placement_space_only(
            portal_cell, portal_dir,
            lower_occupied, upper_occupied,
            map_width, map_height
        )

        if result is None:
            continue

        origin, rotation = result

        # Calculate where the top portal exits
        top_world_cell = top_portal.world_cell(
            origin, rotation,
            vstair_footprint.width_cells, vstair_footprint.depth_cells
        )
        top_world_dir = top_portal.rotated_direction(rotation)
        adjacent_cell = top_world_cell.neighbor(top_world_dir)

        # Check if this cell is free on upper floor (we'll place a hall there)
        if (adjacent_cell.x, adjacent_cell.y) in upper_occupied:
            continue

        # Check bounds
        if adjacent_cell.x < 0 or adjacent_cell.x >= map_width:
            continue
        if adjacent_cell.y < 0 or adjacent_cell.y >= map_height:
            continue

        # Calculate distance to upper floor network
        distance = _distance_to_upper_floor(adjacent_cell)

        stair_candidates.append((
            distance, prim_id, portal_id, portal_cell, portal_dir,
            origin, rotation, adjacent_cell, top_world_dir
        ))

    # Sort by distance (closest to upper floor network first)
    stair_candidates.sort(key=lambda x: x[0])

    _debug(f"[PLACE_VSTAIR_EXT] Found {len(stair_candidates)} stair candidates, distances: {[c[0] for c in stair_candidates[:5]]}")

    hall_footprint = PRIMITIVE_FOOTPRINTS.get('StraightHall')
    if hall_footprint is None:
        return False

    # Try each candidate (closest first)
    for distance, prim_id, portal_id, portal_cell, portal_dir, origin, rotation, adjacent_cell, top_world_dir in stair_candidates:
        required_hall_dir = top_world_dir.opposite()

        # Find rotation where StraightHall has portal facing required direction
        # Pass upper_floor so we can prefer placements adjacent to main network
        hall_result = _find_hall_placement_for_stair(
            adjacent_cell, required_hall_dir,
            upper_occupied, map_width, map_height,
            upper_floor_layout=upper_floor,
            upper_z=upper_z
        )

        if hall_result is None:
            continue

        hall_origin, hall_rotation, hall_portal_id = hall_result

        # SUCCESS: We can place both stair and extension hall

        # Place the stair on lower floor
        stair_prim = PlacedPrimitive.create(
            primitive_type='VerticalStairHall',
            origin=origin,
            rotation=rotation,
            z_offset=lower_z,
            parameters={}
        )
        # CRITICAL: Set footprint so portal alignment works correctly
        stair_prim.set_footprint(vstair_footprint)
        combined_layout.add_primitive(stair_prim)

        stair_cells = _get_primitive_cells('VerticalStairHall', origin, rotation)
        lower_occupied.update(stair_cells)
        upper_occupied.update(stair_cells)

        # Place the extension hall on upper floor
        hall_prim = PlacedPrimitive.create(
            primitive_type='StraightHall',
            origin=hall_origin,
            rotation=hall_rotation,
            z_offset=upper_z,
            parameters={}
        )
        # CRITICAL: Set footprint so portal alignment works correctly
        hall_prim.set_footprint(hall_footprint)
        combined_layout.add_primitive(hall_prim)
        upper_floor.add_primitive(hall_prim)

        hall_cells = _get_primitive_cells('StraightHall', hall_origin, hall_rotation)
        upper_occupied.update(hall_cells)

        # Connect lower floor to stair bottom
        combined_layout.connections.append(Connection(
            primitive_a_id=prim_id,
            portal_a_id=portal_id,
            primitive_b_id=stair_prim.id,
            portal_b_id='bottom'
        ))

        # Connect stair top to extension hall
        combined_layout.connections.append(Connection(
            primitive_a_id=stair_prim.id,
            portal_a_id='top',
            primitive_b_id=hall_prim.id,
            portal_b_id=hall_portal_id
        ))

        _debug(f"[PLACE_VSTAIR_EXT] Placed stair with extension hall at upper floor")

        # CRITICAL FIX: Bridge extension hall to main upper floor network
        # Keep extending with halls until we reach the main network (max 5 extensions)
        current_prim = hall_prim
        current_open_portal = 'front' if hall_portal_id == 'back' else 'back'
        max_extensions = 5
        bridged = False

        for ext_attempt in range(max_extensions):
            # Try to connect current extension to main upper floor network
            bridge_result = _find_adjacent_network_portal(
                combined_layout,
                current_prim.id,
                current_open_portal,
                upper_z
            )

            if bridge_result:
                adjacent_prim_id, adjacent_portal_id = bridge_result
                combined_layout.connections.append(Connection(
                    primitive_a_id=current_prim.id,
                    portal_a_id=current_open_portal,
                    primitive_b_id=adjacent_prim_id,
                    portal_b_id=adjacent_portal_id
                ))
                _debug(f"[PLACE_VSTAIR_EXT] SUCCESS: Bridged to main network via {adjacent_prim_id[:8]}:{adjacent_portal_id} (after {ext_attempt} extra extensions)")
                bridged = True
                break

            # Not adjacent yet - try to extend further with another hall
            # Get the open portal's world position and direction
            hall_fp = PRIMITIVE_FOOTPRINTS.get(current_prim.primitive_type)
            if not hall_fp:
                break

            open_portal_obj = None
            for p in hall_fp.portals:
                if p.id == current_open_portal:
                    open_portal_obj = p
                    break
            if not open_portal_obj:
                break

            portal_cell = current_prim.get_portal_world_cell(open_portal_obj)
            portal_dir = open_portal_obj.rotated_direction(current_prim.rotation)
            next_cell = portal_cell.neighbor(portal_dir)

            # Check if we can place another hall
            if (next_cell.x, next_cell.y) in upper_occupied:
                _debug(f"[PLACE_VSTAIR_EXT] Cannot extend further - cell occupied")
                break

            # Find placement for next extension hall
            next_hall_result = _find_hall_placement_for_stair(
                next_cell, portal_dir.opposite(),
                upper_occupied, map_width, map_height,
                upper_floor_layout=upper_floor,
                upper_z=upper_z
            )

            if not next_hall_result:
                _debug(f"[PLACE_VSTAIR_EXT] Cannot find placement for extension #{ext_attempt + 1}")
                break

            next_origin, next_rotation, next_portal_id = next_hall_result
            next_hall_fp = PRIMITIVE_FOOTPRINTS.get('StraightHall')

            # Place next extension hall
            next_hall = PlacedPrimitive.create(
                primitive_type='StraightHall',
                origin=next_origin,
                rotation=next_rotation,
                z_offset=upper_z,
                parameters={}
            )
            next_hall.set_footprint(next_hall_fp)
            combined_layout.add_primitive(next_hall)
            upper_floor.add_primitive(next_hall)

            next_cells = _get_primitive_cells('StraightHall', next_origin, next_rotation)
            upper_occupied.update(next_cells)

            # Connect current extension to next
            combined_layout.connections.append(Connection(
                primitive_a_id=current_prim.id,
                portal_a_id=current_open_portal,
                primitive_b_id=next_hall.id,
                portal_b_id=next_portal_id
            ))

            _debug(f"[PLACE_VSTAIR_EXT] Extended with additional hall #{ext_attempt + 1}")

            # Update current for next iteration
            current_prim = next_hall
            current_open_portal = 'front' if next_portal_id == 'back' else 'back'

        if not bridged:
            _debug(f"[PLACE_VSTAIR_EXT] WARNING: Could not bridge to main network after {max_extensions} attempts")
            # FIX: Return False when bridging fails - the stair+extension is isolated
            return False

        return True

    return False


def _find_vstair_placement_space_only(
    connect_cell: CellCoord,
    connect_dir: PortalDirection,
    lower_occupied: Set[Tuple[int, int]],
    upper_occupied: Set[Tuple[int, int]],
    map_width: int,
    map_height: int
) -> Optional[Tuple[CellCoord, int]]:
    """Find stair placement checking only space, not upper floor portal."""
    footprint = PRIMITIVE_FOOTPRINTS.get('VerticalStairHall')
    if footprint is None:
        return None

    required_dir = connect_dir.opposite()
    rejection_reasons = []

    for rotation in [0, 90, 180, 270]:
        for portal in footprint.portals:
            if portal.id != 'bottom':
                continue

            rotated_dir = portal.rotated_direction(rotation)
            if rotated_dir != required_dir:
                continue

            portal_offset = portal._rotate_offset(
                portal.cell_offset, rotation,
                footprint.width_cells, footprint.depth_cells
            )

            target_cell = connect_cell.neighbor(connect_dir)
            origin = CellCoord(
                target_cell.x - portal_offset.x,
                target_cell.y - portal_offset.y
            )

            cells = _get_primitive_cells('VerticalStairHall', origin, rotation)

            valid = True
            rejection_reason = None
            for cx, cy in cells:
                if cx < 0 or cx >= map_width or cy < 0 or cy >= map_height:
                    rejection_reason = f"out_of_bounds({cx},{cy})"
                    valid = False
                    break
                if (cx, cy) in lower_occupied:
                    rejection_reason = f"lower_occupied({cx},{cy})"
                    valid = False
                    break
                if (cx, cy) in upper_occupied:
                    rejection_reason = f"upper_occupied({cx},{cy})"
                    valid = False
                    break

            if valid:
                return (origin, rotation)
            elif rejection_reason:
                rejection_reasons.append(f"rot{rotation}:{rejection_reason}")

    # Log first few rejection reasons for debugging
    if rejection_reasons:
        _debug(f"[VSTAIR_SPACE] Rejected: {rejection_reasons[0]}")
    return None


def _find_hall_placement_for_stair(
    target_cell: CellCoord,
    required_portal_dir: PortalDirection,
    occupied: Set[Tuple[int, int]],
    map_width: int,
    map_height: int,
    upper_floor_layout: Optional[DungeonLayout] = None,
    upper_z: float = 0.0
) -> Optional[Tuple[CellCoord, int, str]]:
    """Find hall placement where one portal faces the required direction at target cell.

    Enhanced to prefer placements where the OTHER portal (not connecting to stair)
    is adjacent to the main floor network, enabling connectivity bridging.
    """
    footprint = PRIMITIVE_FOOTPRINTS.get('StraightHall')
    if footprint is None:
        return None

    # Collect all valid candidates: (origin, rotation, portal_id, has_network_adjacency)
    candidates: List[Tuple[CellCoord, int, str, bool]] = []

    for rotation in [0, 90, 180, 270]:
        for portal in footprint.portals:
            rotated_dir = portal.rotated_direction(rotation)
            if rotated_dir != required_portal_dir:
                continue

            # Calculate origin so this portal is at target_cell
            portal_offset = portal._rotate_offset(
                portal.cell_offset, rotation,
                footprint.width_cells, footprint.depth_cells
            )

            origin = CellCoord(
                target_cell.x - portal_offset.x,
                target_cell.y - portal_offset.y
            )

            cells = _get_primitive_cells('StraightHall', origin, rotation)

            valid = True
            for cx, cy in cells:
                if cx < 0 or cx >= map_width or cy < 0 or cy >= map_height:
                    valid = False
                    break
                if (cx, cy) in occupied:
                    valid = False
                    break

            if valid:
                # Check if the OTHER portal would be adjacent to main network
                has_adjacency = False
                if upper_floor_layout is not None:
                    other_portal_id = 'front' if portal.id == 'back' else 'back'
                    other_portal = None
                    for p in footprint.portals:
                        if p.id == other_portal_id:
                            other_portal = p
                            break

                    if other_portal:
                        # Get the world cell of the other portal
                        other_offset = other_portal._rotate_offset(
                            other_portal.cell_offset, rotation,
                            footprint.width_cells, footprint.depth_cells
                        )
                        other_cell = CellCoord(
                            origin.x + other_offset.x,
                            origin.y + other_offset.y
                        )
                        other_dir = other_portal.rotated_direction(rotation)
                        adjacent_cell = other_cell.neighbor(other_dir)

                        # Check if any upper floor primitive has a portal at adjacent cell
                        for prim_id, prim in upper_floor_layout.primitives.items():
                            if abs(prim.z_offset - upper_z) > 2:
                                continue
                            prim_footprint = PRIMITIVE_FOOTPRINTS.get(prim.primitive_type)
                            if not prim_footprint:
                                continue
                            for prim_portal in prim_footprint.portals:
                                prim_portal_cell = prim.get_portal_world_cell(prim_portal)
                                prim_portal_dir = prim_portal.rotated_direction(prim.rotation)
                                if (prim_portal_cell == adjacent_cell and
                                        prim_portal_dir == other_dir.opposite()):
                                    has_adjacency = True
                                    break
                            if has_adjacency:
                                break

                candidates.append((origin, rotation, portal.id, has_adjacency))

    if not candidates:
        return None

    # Prefer candidates with network adjacency
    candidates_with_adjacency = [c for c in candidates if c[3]]
    if candidates_with_adjacency:
        chosen = candidates_with_adjacency[0]
        _debug(f"[FIND_HALL_FOR_STAIR] Found placement with network adjacency")
        return (chosen[0], chosen[1], chosen[2])

    # Fall back to any valid placement
    chosen = candidates[0]
    _debug(f"[FIND_HALL_FOR_STAIR] Using fallback placement (no adjacency found)")
    return (chosen[0], chosen[1], chosen[2])


def _find_vstair_placement(
    connect_cell: CellCoord,
    connect_dir: PortalDirection,
    lower_occupied: Set[Tuple[int, int]],
    upper_occupied: Set[Tuple[int, int]],
    map_width: int,
    map_height: int
) -> Optional[Tuple[CellCoord, int, str]]:
    """
    Find a valid placement for VerticalStairHall connecting to the given portal.

    The stair must have space on both floors.

    Returns:
        Tuple of (origin, rotation, portal_id) or None if no valid placement
    """
    footprint = PRIMITIVE_FOOTPRINTS.get('VerticalStairHall')
    if footprint is None:
        return None

    # The stair's bottom portal must face opposite to connect_dir
    required_dir = connect_dir.opposite()

    # Try each rotation
    for rotation in [0, 90, 180, 270]:
        for portal in footprint.portals:
            # Only consider bottom portal for lower floor connection
            if portal.id != 'bottom':
                continue

            rotated_dir = portal.rotated_direction(rotation)
            if rotated_dir != required_dir:
                continue

            # Calculate origin position
            portal_offset = portal._rotate_offset(
                portal.cell_offset, rotation,
                footprint.width_cells, footprint.depth_cells
            )

            target_cell = connect_cell.neighbor(connect_dir)
            origin = CellCoord(
                target_cell.x - portal_offset.x,
                target_cell.y - portal_offset.y
            )

            # Get all cells this stair would occupy
            cells = _get_primitive_cells('VerticalStairHall', origin, rotation)

            # Check bounds and occupation on BOTH floors
            valid = True
            for cx, cy in cells:
                if cx < 0 or cx >= map_width or cy < 0 or cy >= map_height:
                    valid = False
                    break
                if (cx, cy) in lower_occupied or (cx, cy) in upper_occupied:
                    valid = False
                    break

            if valid:
                return (origin, rotation, portal.id)

    return None


def _place_stair_and_get_top_portal(
    combined_layout: DungeonLayout,
    lower_floor: DungeonLayout,
    lower_occupied: Set[Tuple[int, int]],
    upper_occupied: Set[Tuple[int, int]],
    lower_z: float,
    upper_z: float,
    map_width: int,
    map_height: int,
) -> Optional[Tuple[PlacedPrimitive, CellCoord, PortalDirection]]:
    """
    Place a VerticalStairHall at the lower floor and return the top portal info.

    This is used by the revised multi-floor generation to place a stair first,
    then generate the upper floor emanating from the stair's top portal.

    Returns:
        Tuple of (stair_primitive, top_portal_cell, top_portal_direction) or None
    """
    vstair_footprint = PRIMITIVE_FOOTPRINTS.get('VerticalStairHall')
    if vstair_footprint is None:
        return None

    # Find top portal in footprint
    top_portal = None
    for portal in vstair_footprint.portals:
        if portal.id == 'top':
            top_portal = portal
            break
    if top_portal is None:
        return None

    # Collect open portals from lower floor
    lower_open_portals: List[Tuple[str, str, CellCoord, PortalDirection]] = []
    for prim in lower_floor.primitives.values():
        footprint = PRIMITIVE_FOOTPRINTS.get(prim.primitive_type)
        if footprint is None:
            continue

        connected_portals = set()
        for conn in combined_layout.connections:
            if conn.primitive_a_id == prim.id:
                connected_portals.add(conn.portal_a_id)
            if conn.primitive_b_id == prim.id:
                connected_portals.add(conn.portal_b_id)

        for portal in footprint.portals:
            if portal.id in connected_portals:
                continue
            world_cell = portal.world_cell(
                prim.origin_cell, prim.rotation,
                footprint.width_cells, footprint.depth_cells
            )
            world_dir = portal.rotated_direction(prim.rotation)
            lower_open_portals.append((prim.id, portal.id, world_cell, world_dir))

    _debug(f"[PLACE_STAIR] Found {len(lower_open_portals)} open portals on lower floor")
    random.shuffle(lower_open_portals)

    # Collect candidates with expansion potential scoring
    stair_candidates: List[Tuple[int, str, str, CellCoord, PortalDirection, CellCoord, int, CellCoord, PortalDirection]] = []

    for prim_id, portal_id, portal_cell, portal_dir in lower_open_portals:
        result = _find_vstair_placement_space_only(
            portal_cell, portal_dir,
            lower_occupied, upper_occupied,
            map_width, map_height
        )

        if result is None:
            _debug(f"[PLACE_STAIR] Portal {portal_id} at ({portal_cell.x}, {portal_cell.y}) facing {portal_dir.name} - no space for stair")
            continue

        origin, rotation = result

        # Calculate top portal world position and direction
        top_world_cell = top_portal.world_cell(
            origin, rotation,
            vstair_footprint.width_cells, vstair_footprint.depth_cells
        )
        top_world_dir = top_portal.rotated_direction(rotation)

        # Check if there's room for expansion from the top portal
        # Calculate how many cells are available in the expansion direction
        expansion_cell = top_world_cell.neighbor(top_world_dir)

        # Check bounds and occupation - adaptive threshold for edge cases
        space_score = 0
        valid_cells_in_range = 0
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                check_x = expansion_cell.x + dx
                check_y = expansion_cell.y + dy
                if 0 <= check_x < map_width and 0 <= check_y < map_height:
                    valid_cells_in_range += 1
                    if (check_x, check_y) not in upper_occupied:
                        space_score += 1

        # Calculate centrality score - prefer stairs closer to center
        # This ensures the next floor has room for the NEXT stair
        center_x, center_y = map_width // 2, map_height // 2
        # Distance from center (lower is better)
        center_dist = abs(top_world_cell.x - center_x) + abs(top_world_cell.y - center_y)
        # Invert and scale - higher score = more central
        # Max distance is roughly map_width/2 + map_height/2
        max_dist = map_width // 2 + map_height // 2
        centrality_score = max(0, max_dist - center_dist)

        # Combined score: expansion room + centrality bonus
        combined_score = space_score * 2 + centrality_score

        # Score the candidate - higher score = more expansion room
        # We collect ALL valid spatial candidates and sort by score
        stair_candidates.append((
            combined_score, space_score, centrality_score, prim_id, portal_id, portal_cell, portal_dir,
            origin, rotation, top_world_cell, top_world_dir
        ))

    if not stair_candidates:
        _debug(f"[PLACE_STAIR] No valid stair placements found on lower floor")
        return None

    # Sort by combined score (descending - prefer expansion room + centrality)
    stair_candidates.sort(key=lambda x: -x[0])

    _debug(f"[PLACE_STAIR] Found {len(stair_candidates)} stair candidates, scores: {[(c[0], c[1], c[2]) for c in stair_candidates[:5]]}")

    # Try each candidate (sorted by combined score, best first)
    for (combined_score, space_score, centrality_score, prim_id, portal_id, portal_cell, portal_dir,
         origin, rotation, top_world_cell, top_world_dir) in stair_candidates:

        # Need at least 5 free cells for expansion to place a junction (3x3 footprint)
        # at the top portal. Low space_score often means the top portal faces a map edge
        # where no floor can be generated.
        MIN_EXPANSION_CELLS = 5
        if space_score < MIN_EXPANSION_CELLS:
            _debug(f"[PLACE_STAIR] Skipping ({origin.x}, {origin.y}) - insufficient expansion space ({space_score}<{MIN_EXPANSION_CELLS})")
            continue

        _debug(f"[PLACE_STAIR] Trying placement at ({origin.x}, {origin.y}) with score {combined_score} (space={space_score}, central={centrality_score})")

        # Place the stair
        stair_prim = PlacedPrimitive.create(
            primitive_type='VerticalStairHall',
            origin=origin,
            rotation=rotation,
            z_offset=lower_z,
            parameters={}
        )
        stair_prim.set_footprint(vstair_footprint)
        combined_layout.add_primitive(stair_prim)

        # Mark cells as occupied on both floors
        stair_cells = _get_primitive_cells('VerticalStairHall', origin, rotation)
        lower_occupied.update(stair_cells)
        upper_occupied.update(stair_cells)

        # Connect stair bottom to lower floor
        combined_layout.connections.append(Connection(
            primitive_a_id=prim_id,
            portal_a_id=portal_id,
            primitive_b_id=stair_prim.id,
            portal_b_id='bottom'
        ))

        return (stair_prim, top_world_cell, top_world_dir)

    return None


def _generate_floor_from_portal(
    start_portal_cell: CellCoord,
    start_portal_dir: PortalDirection,
    start_prim_id: str,
    z_offset: float,
    room_count: int,
    map_width: int,
    map_height: int,
    seed: int,
    complexity: int,
    preferred_room_types: Optional[List[str]],
    preferred_hall_types: Optional[List[str]],
    room_probability: float,
    min_hall_between_rooms: int,
    allow_dead_ends: bool,
    secret_room_frequency: int,
    occupied: Set[Tuple[int, int]],
    combined_layout: Optional[DungeonLayout] = None,
    require_stair_space: bool = False,
) -> DungeonLayout:
    """
    Generate a floor layout starting from a specific portal (stair top).

    Instead of starting from a random center position, this generates the floor
    emanating from the stair's top portal, ensuring connectivity.

    Args:
        start_portal_cell: Cell where the starting portal is located
        start_portal_dir: Direction the starting portal faces
        start_prim_id: ID of the primitive with the starting portal (the stair)
        z_offset: Z-level for this floor
        ... (other generation parameters)
        occupied: Set of already-occupied cells (stair footprint)

    Returns:
        DungeonLayout with connected primitives
    """
    _debug(f"[GEN_FROM_PORTAL] Starting floor generation from ({start_portal_cell.x}, {start_portal_cell.y}) facing {start_portal_dir.name}")

    random.seed(seed)
    # Create seeded RNG for parameter randomization (reproducibility)
    param_rng = random.Random(seed)
    layout = DungeonLayout()

    # The starting portal faces start_portal_dir, so we place a hall/room
    # in the adjacent cell facing opposite
    adjacent_cell = start_portal_cell.neighbor(start_portal_dir)
    required_dir = start_portal_dir.opposite()

    # Start with a junction (Crossroads/TJunction) to maximize open portals
    # This ensures the floor can branch out from the stair
    start_type = random.choice(['Crossroads', 'TJunction', 'Crossroads'])

    # Find placement for starting primitive
    result = _find_placement(
        start_type, start_portal_cell, start_portal_dir,
        occupied, map_width, map_height
    )

    if result is None:
        # Try other hall types in order of portal count (prefer more portals)
        for alt_type in ['Crossroads', 'TJunction', 'SquareCorner', 'StraightHall']:
            result = _find_placement(
                alt_type, start_portal_cell, start_portal_dir,
                occupied, map_width, map_height
            )
            if result:
                start_type = alt_type
                break

    if result is None:
        _debug(f"[GEN_FROM_PORTAL] ERROR: Could not place starting primitive!")
        return layout

    new_origin, new_rotation, connecting_portal_id = result

    # Place the starting primitive with z_offset and collision checking
    start_prim = _place_primitive(
        layout, start_type, new_origin, new_rotation, occupied, param_rng,
        z_offset=z_offset, combined_layout=combined_layout
    )
    if start_prim is None:
        _debug(f"[GEN_FROM_PORTAL] ERROR: Failed to place starting primitive!")
        return layout

    # Add to combined_layout for subsequent collision checks
    if combined_layout is not None:
        combined_layout.primitives[start_prim.id] = start_prim

    _debug(f"[GEN_FROM_PORTAL] Placed starting {start_type} at ({new_origin.x}, {new_origin.y})")

    # Create connection from stair top to this primitive
    stair_connection = Connection(
        primitive_a_id=start_prim_id,
        portal_a_id='top',
        primitive_b_id=start_prim.id,
        portal_b_id=connecting_portal_id
    )
    layout.connections.append(stair_connection)

    # CRITICAL: Also add to combined_layout so connectivity check sees it
    if combined_layout is not None:
        combined_layout.connections.append(stair_connection)

    # Collect open portals from starting primitive
    open_portals: List[Tuple[str, str, CellCoord, PortalDirection]] = []
    _collect_open_portals(start_prim, open_portals, layout, exclude_portal=connecting_portal_id)

    # Now generate the rest of the floor using similar logic to generate_random_layout
    placed_rooms = 0
    placed_halls = 1  # Count the starting hall
    halls_since_last_room = 1
    max_halls = room_count * 3
    target_secret_rooms = int(room_count * secret_room_frequency / 100.0)
    placed_secret_rooms = 0

    iterations = 0
    max_iterations = room_count * 10

    # Calculate map center for center-biased growth
    center_x, center_y = map_width // 2, map_height // 2

    while open_portals and iterations < max_iterations:
        iterations += 1

        if not allow_dead_ends and len(open_portals) == 1 and placed_rooms >= room_count:
            open_portals.pop(0)
            continue

        # Bias towards portals that would grow towards center (85% of time)
        # This helps ensure subsequent floors have room for stairs
        if random.random() < 0.85:
            # Bias towards portals that would grow towards center
            best_portal_idx = 0
            best_center_score = float('-inf')
            for idx, (pid, poid, pcell, pdir) in enumerate(open_portals):
                # Direction vector
                dx, dy = {
                    PortalDirection.NORTH: (0, 1),
                    PortalDirection.SOUTH: (0, -1),
                    PortalDirection.EAST: (1, 0),
                    PortalDirection.WEST: (-1, 0),
                }[pdir]
                # Score: positive if direction points towards center
                to_center_x = center_x - pcell.x
                to_center_y = center_y - pcell.y
                score = dx * to_center_x + dy * to_center_y
                if score > best_center_score:
                    best_center_score = score
                    best_portal_idx = idx
            portal_idx = best_portal_idx
        else:
            portal_idx = random.randint(0, len(open_portals) - 1)

        prim_id, portal_id, portal_cell, portal_dir = open_portals[portal_idx]

        # Determine what to place
        effective_room_prob = room_probability
        if halls_since_last_room < min_hall_between_rooms:
            effective_room_prob = 0.0
        elif placed_halls > max_halls * 0.7:
            effective_room_prob = min(0.8, room_probability + 0.2)

        if placed_rooms < room_count and random.random() < effective_room_prob:
            # Exclude tall rooms in multi-floor layouts to prevent cross-floor collisions
            prim_type = _select_room_type(
                complexity, preferred_room_types,
                placed_rooms, placed_secret_rooms, target_secret_rooms,
                exclude_tall=True,
            )
        else:
            # For small floors, prefer junctions to create more open portals
            if len(layout.primitives) < 4:
                prim_type = random.choice(['Crossroads', 'TJunction', 'TJunction'])
            else:
                prim_type = _select_hall_type(complexity, preferred_hall_types)

        # Find placement
        result = _find_placement(
            prim_type, portal_cell, portal_dir,
            occupied, map_width, map_height
        )

        if result is None:
            open_portals.pop(portal_idx)
            continue

        new_origin, new_rotation, connecting_portal_id = result

        # STAIR SPACE PRESERVATION: When require_stair_space=True and we're running
        # low on open portals, check if placing this primitive would leave enough
        # stair-friendly portals. If not, try a junction instead.
        MIN_STAIR_FRIENDLY_PORTALS = 3
        if require_stair_space and len(open_portals) <= 5:
            # Predict open portals after placement
            new_prim_footprint = PRIMITIVE_FOOTPRINTS.get(prim_type)
            new_portal_count = len(new_prim_footprint.portals) - 1 if new_prim_footprint else 0
            predicted_open_portals = len(open_portals) - 1 + new_portal_count

            # If we'd drop below minimum, try a junction instead (creates more portals)
            if predicted_open_portals < MIN_STAIR_FRIENDLY_PORTALS and prim_type not in ['Crossroads', 'TJunction']:
                _debug(f"[GEN_FROM_PORTAL] Switching {prim_type} -> junction (only {predicted_open_portals} portals predicted)")
                alt_type = random.choice(['Crossroads', 'TJunction'])
                alt_result = _find_placement(
                    alt_type, portal_cell, portal_dir,
                    occupied, map_width, map_height
                )
                if alt_result is not None:
                    prim_type = alt_type
                    result = alt_result
                    new_origin, new_rotation, connecting_portal_id = result
                # If junction doesn't fit either, proceed with original choice

        # Place the primitive with z_offset and collision checking
        new_prim = _place_primitive(
            layout, prim_type, new_origin, new_rotation, occupied, param_rng,
            z_offset=z_offset, combined_layout=combined_layout
        )
        if new_prim is None:
            open_portals.pop(portal_idx)
            continue

        # Add to combined_layout for subsequent collision checks
        if combined_layout is not None:
            combined_layout.primitives[new_prim.id] = new_prim

        # Create connection
        layout.connections.append(Connection(
            primitive_a_id=prim_id,
            portal_a_id=portal_id,
            primitive_b_id=new_prim.id,
            portal_b_id=connecting_portal_id
        ))

        open_portals.pop(portal_idx)
        _collect_open_portals(new_prim, open_portals, layout, exclude_portal=connecting_portal_id)

        # Update counts
        if prim_type in ROOM_TYPES or prim_type == SECRET_ROOM_TYPE:
            placed_rooms += 1
            halls_since_last_room = 0
            if prim_type == SECRET_ROOM_TYPE:
                placed_secret_rooms += 1
        else:
            placed_halls += 1
            halls_since_last_room += 1

        # Keep generating until we have enough rooms AND primitives
        # Minimum 5 primitives to ensure enough open portals for next stair
        min_primitives = 5
        if placed_rooms >= room_count and len(layout.primitives) >= min_primitives:
            if require_stair_space:
                # Check if we have at least one stair-friendly portal before stopping
                stair_portals = _count_stair_friendly_portals(
                    open_portals, occupied, map_width, map_height
                )
                if stair_portals >= 1:
                    # We have stair space, can stop
                    if placed_halls >= max_halls or random.random() < 0.3:
                        _debug(f"[GEN_FROM_PORTAL] Stopping: have {stair_portals} stair-friendly portals")
                        break
                else:
                    # No stair-friendly portals! Keep generating to try to create one
                    _debug(f"[GEN_FROM_PORTAL] Need stair space, continuing generation...")
            else:
                # No stair space required, can stop normally
                if placed_halls >= max_halls or random.random() < 0.3:
                    break

        # Safety limit: don't generate forever
        if len(layout.primitives) >= min_primitives * 3:
            _debug(f"[GEN_FROM_PORTAL] Safety limit reached ({len(layout.primitives)} primitives)")
            break

    # If we need stair space but don't have any, try to extend the layout
    if require_stair_space:
        stair_portals = _count_stair_friendly_portals(
            open_portals, occupied, map_width, map_height
        )
        if stair_portals == 0:
            _debug(f"[GEN_FROM_PORTAL] No stair-friendly portals! Attempting forced extension...")
            # Try to extend each remaining open portal until we find one that works
            for prim_id, portal_id, portal_cell, portal_dir in list(open_portals):
                for ext_type in ['StraightHall', 'TJunction', 'Crossroads']:
                    result = _find_placement(
                        ext_type, portal_cell, portal_dir,
                        occupied, map_width, map_height
                    )
                    if result is not None:
                        ext_origin, ext_rotation, ext_portal_id = result
                        ext_prim = _place_primitive(
                            layout, ext_type, ext_origin, ext_rotation, occupied, param_rng,
                            z_offset=z_offset, combined_layout=combined_layout
                        )
                        if ext_prim is not None:
                            if combined_layout is not None:
                                combined_layout.primitives[ext_prim.id] = ext_prim
                            layout.connections.append(Connection(
                                primitive_a_id=prim_id,
                                portal_a_id=portal_id,
                                primitive_b_id=ext_prim.id,
                                portal_b_id=ext_portal_id
                            ))
                            _debug(f"[GEN_FROM_PORTAL] Added extension {ext_type} at ({ext_origin.x}, {ext_origin.y})")
                            # Check if this gives us a stair-friendly portal
                            new_open_portals: List[Tuple[str, str, CellCoord, PortalDirection]] = []
                            _collect_open_portals(ext_prim, new_open_portals, layout, exclude_portal=ext_portal_id)
                            if _count_stair_friendly_portals(new_open_portals, occupied, map_width, map_height) > 0:
                                _debug(f"[GEN_FROM_PORTAL] Found stair-friendly portal after extension!")
                                open_portals.extend(new_open_portals)
                                break
                else:
                    continue
                break

    _debug(f"[GEN_FROM_PORTAL] Generated floor with {len(layout.primitives)} primitives, {placed_rooms} rooms")
    return layout
