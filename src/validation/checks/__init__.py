"""
Validation check modules.

Each module provides specific validation checks:
- geometry_checks: Integer coords, collinear points, brush dimensions
- layout_checks: Layout structure, connections
- spatial_checks: Spatial overlap, footprint bounds
- format_checks: idTech 1/4 format compliance
- module_checks: Module registration and contract compliance
- filler_gap_checks: Filler brush detection, hull gap detection
"""

from .geometry_checks import (
    validate_brushes,
    validate_plane_geometry,
    check_integer_coordinates,
    check_collinear_points,
    check_brush_dimensions,
)

from .module_checks import (
    validate_module_registration,
    validate_module_contract,
)

from .format_checks import (
    validate_export_format,
    validate_map_structure,
)

from .filler_gap_checks import (
    detect_filler_brushes,
    detect_hull_gaps,
    scan_module_filler_gaps,
    compute_brush_aabb,
    is_axis_aligned_box,
    FILLER_MIN_VOL,
)

__all__ = [
    # Geometry
    'validate_brushes',
    'validate_plane_geometry',
    'check_integer_coordinates',
    'check_collinear_points',
    'check_brush_dimensions',
    # Module
    'validate_module_registration',
    'validate_module_contract',
    # Format
    'validate_export_format',
    'validate_map_structure',
    # Filler/Gap
    'detect_filler_brushes',
    'detect_hull_gaps',
    'scan_module_filler_gaps',
    'compute_brush_aabb',
    'is_axis_aligned_box',
    'FILLER_MIN_VOL',
]
