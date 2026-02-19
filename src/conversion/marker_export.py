"""
Marker export utilities.

Export markers in various formats for:
- idTech entity placement (info_notnull with targetname)
- JSON for external tools
- CSV for analysis/spreadsheets
"""

import json
import csv
import io
from typing import List, Dict, Any

from quake_levelgenerator.src.pipeline.layout_state import Marker, MarkerType


def export_markers_to_json(markers: List[Marker], include_tags: bool = True) -> str:
    """
    Export markers as JSON.

    Args:
        markers: List of markers to export
        include_tags: Whether to include tag metadata

    Returns:
        JSON string with marker data
    """
    output = []

    for marker in markers:
        entry = {
            'name': marker.name,
            'type': marker.marker_type.name,
            'position': {
                'x': marker.position[0],
                'y': marker.position[1],
                'z': marker.position[2],
            },
            'room_id': marker.room_id,
        }
        if include_tags and marker.tags:
            entry['tags'] = marker.tags

        output.append(entry)

    return json.dumps(output, indent=2)


def export_markers_to_entities(markers: List[Marker]) -> str:
    """
    Export markers as idTech entity definitions.

    Uses info_notnull entities with targetname for marker identification.
    These can be loaded in TrenchBroom for visual reference.

    Args:
        markers: List of markers to export

    Returns:
        idTech entity format string
    """
    lines = []

    for marker in markers:
        # Map marker types to appropriate entity classnames
        classname = _marker_type_to_classname(marker.marker_type)

        lines.append('{')
        lines.append(f'"classname" "{classname}"')
        lines.append(f'"targetname" "{marker.name}"')
        lines.append(f'"origin" "{int(marker.position[0])} {int(marker.position[1])} {int(marker.position[2])}"')

        # Add marker type as a key
        lines.append(f'"marker_type" "{marker.marker_type.name}"')

        # Add room_id if present
        if marker.room_id is not None:
            lines.append(f'"room_id" "{marker.room_id}"')

        # Add selected tags as entity keys
        for key, value in marker.tags.items():
            if isinstance(value, (str, int, float, bool)):
                lines.append(f'"{key}" "{value}"')

        lines.append('}')
        lines.append('')

    return '\n'.join(lines)


def export_markers_to_csv(markers: List[Marker]) -> str:
    """
    Export markers as CSV for analysis.

    Args:
        markers: List of markers to export

    Returns:
        CSV format string
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        'name', 'type', 'x', 'y', 'z', 'room_id', 'tags'
    ])

    for marker in markers:
        writer.writerow([
            marker.name,
            marker.marker_type.name,
            marker.position[0],
            marker.position[1],
            marker.position[2],
            marker.room_id if marker.room_id is not None else '',
            json.dumps(marker.tags) if marker.tags else ''
        ])

    return output.getvalue()


def export_markers_for_trenchbroom(markers: List[Marker]) -> str:
    """
    Export markers as commented entity definitions for TrenchBroom.

    This format includes helpful comments explaining each marker
    and uses point entities that are visible in the editor.

    Args:
        markers: List of markers to export

    Returns:
        idTech entity format string with comments
    """
    lines = []

    # Group markers by type for organized output
    by_type: Dict[MarkerType, List[Marker]] = {}
    for marker in markers:
        if marker.marker_type not in by_type:
            by_type[marker.marker_type] = []
        by_type[marker.marker_type].append(marker)

    for marker_type in MarkerType:
        type_markers = by_type.get(marker_type, [])
        if not type_markers:
            continue

        lines.append(f'// === {marker_type.name} MARKERS ({len(type_markers)}) ===')
        lines.append('')

        for marker in type_markers:
            classname = _marker_type_to_classname(marker.marker_type)

            lines.append(f'// Marker: {marker.name}')
            if marker.room_id is not None:
                lines.append(f'// Room: {marker.room_id}')
            if marker.tags:
                for key, value in marker.tags.items():
                    lines.append(f'// {key}: {value}')

            lines.append('{')
            lines.append(f'"classname" "{classname}"')
            lines.append(f'"targetname" "{marker.name}"')
            lines.append(f'"origin" "{int(marker.position[0])} {int(marker.position[1])} {int(marker.position[2])}"')
            lines.append('}')
            lines.append('')

    return '\n'.join(lines)


def _marker_type_to_classname(marker_type: MarkerType) -> str:
    """
    Map marker types to appropriate idTech entity classnames.

    Uses standard idTech entities where appropriate, falls back
    to info_notnull for custom markers.
    """
    mapping = {
        MarkerType.SPAWN_POINT: 'info_player_start',
        MarkerType.LEVEL_GOAL: 'trigger_changelevel',
        MarkerType.KEY: 'item_key1',
        MarkerType.LOCK: 'func_door',
        MarkerType.ITEM: 'item_health',
        MarkerType.ENEMY: 'monster_army',
        MarkerType.SECRET: 'trigger_secret',
        MarkerType.CHECKPOINT: 'info_notnull',
        MarkerType.CUSTOM: 'info_notnull',
    }
    return mapping.get(marker_type, 'info_notnull')


def count_markers_by_type(markers: List[Marker]) -> Dict[str, int]:
    """
    Count markers by type for statistics.

    Args:
        markers: List of markers

    Returns:
        Dict mapping marker type names to counts
    """
    counts: Dict[str, int] = {}
    for marker in markers:
        type_name = marker.marker_type.name
        counts[type_name] = counts.get(type_name, 0) + 1
    return counts
