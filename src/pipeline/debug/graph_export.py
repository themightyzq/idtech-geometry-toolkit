"""
Graph export utilities for layout debugging.

Provides export functions to visualize dungeon layouts in:
- DOT format (Graphviz) for visual graph inspection
- JSON format for programmatic analysis and reproducibility tracking
"""

from typing import Dict, Any
import json


def export_layout_dot(bsp_export: Dict[str, Any]) -> str:
    """Export BSP layout as Graphviz DOT format.

    Args:
        bsp_export: Result of BSPGenerator.export_layout()

    Returns:
        DOT format string for visualization with Graphviz or online viewers
    """
    lines = ['digraph DungeonLayout {']
    lines.append('  rankdir=LR;')
    lines.append('  node [shape=box, style=filled];')
    lines.append('')

    # Nodes (rooms)
    rooms = bsp_export.get('rooms', [])
    for room in rooms:
        room_id = room.get('id', 0)
        room_type = room.get('type', 'standard')
        bounds = room.get('bounds', {})
        center_x = bounds.get('x', 0) + bounds.get('width', 0) // 2
        center_y = bounds.get('y', 0) + bounds.get('height', 0) // 2
        z_offset = room.get('z_offset', 0)

        # Build label with room info
        label_lines = [
            f"{room_type.upper()}",
            f"id: {room_id}",
            f"pos: ({center_x}, {center_y})",
        ]
        if z_offset != 0:
            label_lines.append(f"z: {z_offset}")

        label = '\\n'.join(label_lines)

        # Color by room type
        colors = {
            'entrance': '#90EE90',   # Light green
            'exit': '#FFB6C1',       # Light pink
            'arena': '#FFD700',      # Gold
            'junction': '#87CEEB',   # Sky blue
            'vertical': '#DDA0DD',   # Plum
            'standard': '#D3D3D3',   # Light gray
        }
        color = colors.get(room_type.lower(), '#D3D3D3')

        lines.append(f'  room_{room_id} [label="{label}" fillcolor="{color}"];')

    lines.append('')

    # Edges (corridors)
    corridors = bsp_export.get('corridors', [])
    for corr in corridors:
        start_id = corr.get('start_room_id', corr.get('room_a_id', 0))
        end_id = corr.get('end_room_id', corr.get('room_b_id', 0))
        is_vertical = corr.get('is_vertical', False)

        style = 'dashed' if is_vertical else 'solid'
        lines.append(f'  room_{start_id} -> room_{end_id} [dir=none, style={style}];')

    lines.append('}')
    return '\n'.join(lines)


def export_layout_json(bsp_export: Dict[str, Any], seed: int) -> str:
    """Export BSP layout as JSON with metadata.

    Args:
        bsp_export: Result of BSPGenerator.export_layout()
        seed: The seed used for generation

    Returns:
        JSON string with layout and debug metadata
    """
    # Compute some statistics
    rooms = bsp_export.get('rooms', [])
    corridors = bsp_export.get('corridors', [])

    room_types = {}
    for room in rooms:
        rtype = room.get('type', 'standard')
        room_types[rtype] = room_types.get(rtype, 0) + 1

    # Find entrance and exit rooms
    entrance_room = None
    exit_room = None
    for room in rooms:
        rtype = room.get('type', '').lower()
        if rtype == 'entrance':
            entrance_room = room.get('id')
        elif rtype == 'exit':
            exit_room = room.get('id')

    output = {
        'metadata': {
            'seed': seed,
            'version': '1.0',
            'generator': 'idtech-geometry-toolkit',
        },
        'statistics': {
            'room_count': len(rooms),
            'corridor_count': len(corridors),
            'room_types': room_types,
            'entrance_room_id': entrance_room,
            'exit_room_id': exit_room,
        },
        'layout': bsp_export
    }
    return json.dumps(output, indent=2)


def derive_primitive_seed(global_seed: int, primitive_index: int) -> int:
    """Derive deterministic per-primitive seed from global seed.

    Args:
        global_seed: The global pipeline seed
        primitive_index: Index of the primitive in generation order

    Returns:
        Deterministic seed for this specific primitive
    """
    return (global_seed * 31 + primitive_index) % (2**31 - 1)
