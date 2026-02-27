"""
Connective geometry primitives: Bridge, Platform, Rampart, Gallery.

These are OPEN ELEMENTS - they connect between sealed spaces or extend from walls.
They do NOT need to be sealed themselves.

See CLAUDE.md for the distinction between sealed (room) and open (connective) primitives.
"""

from __future__ import annotations
from typing import Any, Dict, List

from quake_levelgenerator.src.conversion.map_writer import Brush
from .base import GeometricPrimitive


class Bridge(GeometricPrimitive):
    """A bridge span with optional side walls / railings."""

    span: float = 256.0
    bridge_width: float = 96.0
    thickness: float = 16.0
    railing_height: float = 48.0
    railing: bool = True
    has_supports: bool = True

    @classmethod
    def get_display_name(cls) -> str:
        return "Bridge"

    @classmethod
    def get_category(cls) -> str:
        return "Connective"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "span": {
                "type": "float", "default": 256.0, "min": 64, "max": 1024, "label": "Span Length",
                "description": "Total length of the bridge from one end to the other"
            },
            "bridge_width": {
                "type": "float", "default": 96.0, "min": 32, "max": 256, "label": "Width",
                "description": "Width of the walkable deck surface (player is 32 units wide)"
            },
            "thickness": {
                "type": "float", "default": 16.0, "min": 8, "max": 64, "label": "Floor Thickness",
                "description": "Vertical thickness of the bridge deck"
            },
            "railing_height": {
                "type": "float", "default": 48.0, "min": 16, "max": 96, "label": "Railing Height",
                "description": "Height of side railings above the deck (player is 56 units tall)"
            },
            "railing": {
                "type": "bool", "default": True, "label": "Add Railings",
                "description": "Add protective side railings to prevent falling"
            },
            "has_supports": {
                "type": "bool", "default": True, "label": "Support Beams",
                "description": "Add under-deck support beams running length of bridge"
            },
        }

    def generate(self) -> List[Brush]:
        ox, oy, oz = self.params.origin
        hw = self.bridge_width / 2
        brushes: List[Brush] = []

        # Deck (uses floor texture - it's a walking surface)
        brushes.append(self._floor_box(
            ox - hw, oy, oz - self.thickness,
            ox + hw, oy + self.span, oz,
        ))

        if self.railing:
            rw = 8.0
            # Left railing (uses structural texture)
            brushes.append(self._structural_box(
                ox - hw - rw, oy, oz,
                ox - hw, oy + self.span, oz + self.railing_height,
            ))
            # Right railing (uses structural texture)
            brushes.append(self._structural_box(
                ox + hw, oy, oz,
                ox + hw + rw, oy + self.span, oz + self.railing_height,
            ))

        if self.has_supports:
            beam_w = 8.0
            beam_h = 16.0
            beam_inset = hw * 0.3  # Beams at 30% from each edge
            # Left beam
            brushes.append(self._structural_box(
                ox - beam_inset - beam_w / 2, oy, oz - self.thickness - beam_h,
                ox - beam_inset + beam_w / 2, oy + self.span, oz - self.thickness,
            ))
            # Right beam
            brushes.append(self._structural_box(
                ox + beam_inset - beam_w / 2, oy, oz - self.thickness - beam_h,
                ox + beam_inset + beam_w / 2, oy + self.span, oz - self.thickness,
            ))

        return brushes


class Platform(GeometricPrimitive):
    """A platform extending from a wall with railing.

    Can represent a balcony, landing, ledge, or any elevated platform.
    """

    platform_width: float = 128.0
    platform_depth: float = 64.0
    thickness: float = 16.0
    railing_height: float = 48.0
    has_brackets: bool = True

    @classmethod
    def get_display_name(cls) -> str:
        return "Platform"

    @classmethod
    def get_category(cls) -> str:
        return "Connective"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "platform_width": {
                "type": "float", "default": 128.0, "min": 64, "max": 512, "label": "Width",
                "description": "Width of the platform (side to side)"
            },
            "platform_depth": {
                "type": "float", "default": 64.0, "min": 32, "max": 256, "label": "Depth",
                "description": "Depth of the platform (how far it extends from the wall)"
            },
            "thickness": {
                "type": "float", "default": 16.0, "min": 8, "max": 64, "label": "Floor Thickness",
                "description": "Vertical thickness of the platform deck"
            },
            "railing_height": {
                "type": "float", "default": 48.0, "min": 16, "max": 96, "label": "Railing Height",
                "description": "Height of protective railings on three sides"
            },
            "has_brackets": {
                "type": "bool", "default": True, "label": "Support Brackets",
                "description": "Add wedge support brackets underneath at front corners"
            },
        }

    def generate(self) -> List[Brush]:
        ox, oy, oz = self.params.origin
        hw = self.platform_width / 2
        brushes: List[Brush] = []

        # Platform floor (uses floor texture)
        brushes.append(self._floor_box(
            ox - hw, oy, oz - self.thickness,
            ox + hw, oy + self.platform_depth, oz,
        ))

        rw = 8.0
        # Front railing (uses structural texture)
        brushes.append(self._structural_box(
            ox - hw, oy + self.platform_depth, oz,
            ox + hw, oy + self.platform_depth + rw, oz + self.railing_height,
        ))
        # Left railing (uses structural texture)
        brushes.append(self._structural_box(
            ox - hw - rw, oy, oz,
            ox - hw, oy + self.platform_depth + rw, oz + self.railing_height,
        ))
        # Right railing (uses structural texture)
        brushes.append(self._structural_box(
            ox + hw, oy, oz,
            ox + hw + rw, oy + self.platform_depth + rw, oz + self.railing_height,
        ))

        if self.has_brackets:
            bracket_size = 16.0
            bracket_depth = min(32.0, self.platform_depth / 2)
            # Left bracket (wedge underneath)
            brushes.append(self._wedge(
                ox - hw, oy + self.platform_depth - bracket_depth, oz - self.thickness - bracket_size,
                ox - hw + bracket_size, oy + self.platform_depth, oz - self.thickness,
                ramp_axis="y",
            ))
            # Right bracket (wedge underneath)
            brushes.append(self._wedge(
                ox + hw - bracket_size, oy + self.platform_depth - bracket_depth, oz - self.thickness - bracket_size,
                ox + hw, oy + self.platform_depth, oz - self.thickness,
                ramp_axis="y",
            ))

        return brushes


class Rampart(GeometricPrimitive):
    """Wall-top defensive walkway.

    Ramparts are elevated walkways that run along the top of castle walls,
    allowing defenders to patrol and engage attackers. Often paired with
    battlements (crenellated walls) on the outer parapet.

    Features:
    - Elevated walkway surface
    - Optional inner wall (castle-interior side)
    - Optional outer parapet (for battlement placement)
    - Configurable dimensions
    """

    length: float = 256.0           # Total length
    walkway_width: float = 64.0     # Width of walkway surface
    height: float = 96.0            # Height above ground
    thickness: float = 16.0         # Floor thickness
    inner_wall: bool = True         # Low wall on castle-interior side
    inner_wall_height: float = 32.0 # Inner wall height
    outer_parapet: bool = True      # Outer defensive wall
    parapet_height: float = 48.0    # Outer parapet height
    has_buttresses: bool = False
    buttress_spacing: float = 64.0

    @classmethod
    def get_display_name(cls) -> str:
        return "Rampart"

    @classmethod
    def get_category(cls) -> str:
        return "Connective"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "length": {
                "type": "float", "default": 256.0, "min": 64, "max": 1024, "label": "Length",
                "description": "Total length of the rampart walkway"
            },
            "walkway_width": {
                "type": "float", "default": 64.0, "min": 32, "max": 128, "label": "Walkway Width",
                "description": "Width of the patrol walkway surface (player is 32 units wide)"
            },
            "height": {
                "type": "float", "default": 96.0, "min": 32, "max": 256, "label": "Height",
                "description": "Height of the rampart above ground level"
            },
            "thickness": {
                "type": "float", "default": 16.0, "min": 8, "max": 32, "label": "Floor Thickness",
                "description": "Vertical thickness of the walkway surface"
            },
            "inner_wall": {
                "type": "bool", "default": True, "label": "Inner Wall",
                "description": "Add low wall on the castle-interior side"
            },
            "inner_wall_height": {
                "type": "float", "default": 32.0, "min": 16, "max": 64, "label": "Inner Wall Height",
                "description": "Height of the inner safety wall"
            },
            "outer_parapet": {
                "type": "bool", "default": True, "label": "Outer Parapet",
                "description": "Add defensive wall on the outer edge (for battlements)"
            },
            "parapet_height": {
                "type": "float", "default": 48.0, "min": 32, "max": 96, "label": "Parapet Height",
                "description": "Height of the outer defensive parapet"
            },
            "has_buttresses": {
                "type": "bool", "default": False, "label": "Buttress Supports",
                "description": "Add buttress supports projecting outward at regular intervals"
            },
            "buttress_spacing": {
                "type": "float", "default": 64.0, "min": 32, "max": 128, "label": "Buttress Spacing",
                "description": "Distance between buttress supports"
            },
        }

    def generate(self) -> List[Brush]:
        ox, oy, oz = self.params.origin
        brushes: List[Brush] = []

        hw = self.walkway_width / 2
        wall_t = 8.0  # Wall thickness

        # Support structure (wall underneath the walkway - uses wall texture)
        brushes.append(self._wall_box(
            ox - hw - wall_t, oy, oz,
            ox + hw + wall_t, oy + self.length, oz + self.height - self.thickness
        ))

        # Walkway floor (uses floor texture)
        brushes.append(self._floor_box(
            ox - hw - wall_t, oy, oz + self.height - self.thickness,
            ox + hw + wall_t, oy + self.length, oz + self.height
        ))

        # Inner wall (castle interior side, on -X - uses structural texture)
        if self.inner_wall:
            brushes.append(self._structural_box(
                ox - hw - wall_t, oy, oz + self.height,
                ox - hw, oy + self.length, oz + self.height + self.inner_wall_height
            ))

        # Outer parapet (for battlement, on +X - uses structural texture)
        if self.outer_parapet:
            brushes.append(self._structural_box(
                ox + hw, oy, oz + self.height,
                ox + hw + wall_t, oy + self.length, oz + self.height + self.parapet_height
            ))

        if self.has_buttresses:
            butt_w = 16.0
            butt_depth = 16.0
            num_buttresses = max(1, int(self.length / self.buttress_spacing))
            for i in range(num_buttresses + 1):
                by = oy + i * self.buttress_spacing
                if by + butt_w > oy + self.length:
                    break
                # Buttress projecting outward on +X side (beyond parapet)
                brushes.append(self._structural_box(
                    ox + hw + wall_t, by, oz,
                    ox + hw + wall_t + butt_depth, by + butt_w, oz + self.height,
                ))

        return brushes


class Gallery(GeometricPrimitive):
    """Covered walkway with arches on one side (cloister/arcade).

    Galleries are covered walkways found in monasteries, castles, and
    courtyards. One side has repeated arches opening to the outside,
    while the other side is typically a solid wall.

    Features:
    - Repeated arch openings along length
    - Optional solid back wall
    - Optional ceiling (can be open-air)
    - Configurable column/arch dimensions
    """

    length: float = 256.0       # Total length
    gallery_width: float = 96.0 # Width of covered walkway
    height: float = 128.0       # Ceiling height
    arch_count: int = 4         # Number of arches (2-8)
    arch_height: float = 96.0   # Height of arch openings
    column_width: float = 16.0  # Width of columns between arches
    back_wall: bool = True      # Solid wall on back side
    ceiling: bool = True        # Add ceiling (can disable for open-air)
    has_column_details: bool = True

    @classmethod
    def get_display_name(cls) -> str:
        return "Gallery"

    @classmethod
    def get_category(cls) -> str:
        return "Connective"

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "length": {
                "type": "float", "default": 256.0, "min": 128, "max": 1024, "label": "Length",
                "description": "Total length of the gallery walkway"
            },
            "gallery_width": {
                "type": "float", "default": 96.0, "min": 64, "max": 192, "label": "Width",
                "description": "Width of the covered walkway"
            },
            "height": {
                "type": "float", "default": 128.0, "min": 96, "max": 256, "label": "Height",
                "description": "Ceiling height of the gallery"
            },
            "arch_count": {
                "type": "int", "default": 4, "min": 2, "max": 8, "label": "Arch Count",
                "description": "Number of arch openings along the open side"
            },
            "arch_height": {
                "type": "float", "default": 96.0, "min": 64, "max": 192, "label": "Arch Height",
                "description": "Height of each arch opening (lintel fills above)"
            },
            "column_width": {
                "type": "float", "default": 16.0, "min": 8, "max": 32, "label": "Column Width",
                "description": "Width of columns between arch openings"
            },
            "back_wall": {
                "type": "bool", "default": True, "label": "Back Wall",
                "description": "Add solid wall on the back side of the gallery"
            },
            "ceiling": {
                "type": "bool", "default": True, "label": "Ceiling",
                "description": "Add ceiling (disable for open-air colonnade)"
            },
            "has_column_details": {
                "type": "bool", "default": True, "label": "Column Details",
                "description": "Add base and capital detail to each column"
            },
        }

    def generate(self) -> List[Brush]:
        ox, oy, oz = self.params.origin
        brushes: List[Brush] = []

        wall_t = 8.0
        col_w = self.column_width

        # Calculate arch spacing
        arch_spacing = self.length / self.arch_count
        arch_width = arch_spacing - col_w

        # Floor (uses floor texture)
        brushes.append(self._floor_box(
            ox, oy, oz - wall_t,
            ox + self.gallery_width, oy + self.length, oz
        ))

        # Ceiling (uses ceiling texture)
        if self.ceiling:
            brushes.append(self._ceiling_box(
                ox, oy, oz + self.height,
                ox + self.gallery_width, oy + self.length, oz + self.height + wall_t
            ))

        # Back wall (solid, on +X side - uses wall texture)
        if self.back_wall:
            brushes.append(self._wall_box(
                ox + self.gallery_width, oy, oz,
                ox + self.gallery_width + wall_t, oy + self.length, oz + self.height
            ))

        # Front columns and lintels (open side on -X / ox side - uses structural texture)
        # First column at start
        brushes.append(self._structural_box(
            ox - col_w, oy, oz,
            ox, oy + col_w, oz + self.height
        ))

        for i in range(self.arch_count):
            arch_y_start = oy + i * arch_spacing
            arch_y_end = arch_y_start + arch_spacing

            # Column at end of this arch bay (uses structural texture)
            brushes.append(self._structural_box(
                ox - col_w, arch_y_end - col_w, oz,
                ox, arch_y_end, oz + self.height
            ))

            # Lintel above arch opening (uses structural texture)
            if self.arch_height < self.height:
                brushes.append(self._structural_box(
                    ox - col_w, arch_y_start + col_w, oz + self.arch_height,
                    ox, arch_y_end - col_w, oz + self.height
                ))

        # End walls to close the gallery (uses wall texture)
        # Front end wall (at oy)
        brushes.append(self._wall_box(
            ox - col_w, oy - wall_t, oz,
            ox + self.gallery_width + wall_t, oy, oz + self.height
        ))

        # Back end wall (at oy + length)
        brushes.append(self._wall_box(
            ox - col_w, oy + self.length, oz,
            ox + self.gallery_width + wall_t, oy + self.length + wall_t, oz + self.height
        ))

        if self.has_column_details:
            detail_h = 8.0
            detail_extra = 4.0
            # Collect column Y positions
            col_positions = [oy]
            for i in range(self.arch_count):
                arch_y_end = oy + (i + 1) * arch_spacing
                col_positions.append(arch_y_end - col_w)

            for col_y in col_positions:
                # Column base (wider, 8u tall)
                brushes.append(self._structural_box(
                    ox - col_w - detail_extra, col_y - detail_extra, oz,
                    ox + detail_extra, col_y + col_w + detail_extra, oz + detail_h,
                ))
                # Column capital (wider, 8u tall at top)
                brushes.append(self._structural_box(
                    ox - col_w - detail_extra, col_y - detail_extra, oz + self.height - detail_h,
                    ox + detail_extra, col_y + col_w + detail_extra, oz + self.height,
                ))

        return brushes
