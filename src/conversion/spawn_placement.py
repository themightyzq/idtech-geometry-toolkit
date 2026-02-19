"""
Minimal spawn point placement for generated maps.

Places an info_player_start entity in the first suitable room.
"""

from typing import Dict, List, Optional, Tuple

from quake_levelgenerator.src.conversion.map_writer import MapWriter


def place_player_spawn(
    map_writer: MapWriter,
    rooms: List[Dict],
    floor_z: float = 0.0,
) -> bool:
    """Place an info_player_start in the best available room.

    Prefers entrance-type rooms, falls back to the largest room.

    Args:
        map_writer: MapWriter instance to add the entity to.
        rooms: List of room dicts from layout export (must have 'bounds' key).
        floor_z: Z level of the floor.

    Returns:
        True if a spawn was placed.
    """
    if not rooms:
        return False

    # Pick candidate: entrance first, then largest
    candidate: Optional[Dict] = None
    for r in rooms:
        if r.get("type") == "ENTRANCE":
            candidate = r
            break
    if candidate is None:
        candidate = max(rooms, key=lambda r: r["bounds"]["width"] * r["bounds"]["height"])

    bounds = candidate["bounds"]
    sx = bounds["x"] + bounds["width"] // 2
    sy = bounds["y"] + bounds["height"] // 2
    sz = floor_z + 32  # 32 units above floor

    map_writer.add_player_start((sx, sy, sz))
    return True
