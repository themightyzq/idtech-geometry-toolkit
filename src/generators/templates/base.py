"""
GenerationTemplate dataclass — defines preset parameter bundles.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class GenerationTemplate:
    """
    A preset bundle of parameters for dungeon generation.

    Templates provide quick starting points for different gameplay styles,
    from arena combat to atmospheric exploration.
    """

    # Identity
    name: str
    description: str
    category: str  # "Combat", "Exploration", "Atmospheric"

    # Layout parameters (match LayoutPanel fields)
    map_width: int = 30
    map_height: int = 30
    room_count: int = 15
    complexity: int = 5
    corridor_width: int = 96

    # Generation hints (influence random generation)
    preferred_room_types: List[str] = field(default_factory=list)
    preferred_hall_types: List[str] = field(default_factory=list)
    room_probability: float = 0.4
    min_hall_between_rooms: int = 1
    allow_dead_ends: bool = True

    def to_layout_params(self) -> Dict[str, Any]:
        """
        Convert template to layout panel parameters.

        Returns:
            Dictionary compatible with LayoutPanel.set_parameters()
        """
        return {
            "map_width": self.map_width,
            "map_height": self.map_height,
            "room_count": self.room_count,
            "complexity": self.complexity,
            "corridor_width": self.corridor_width,
        }

    def get_generation_hints(self) -> Dict[str, Any]:
        """
        Get generation hints for the random layout generator.

        Returns:
            Dictionary of hints for generate_random_layout()
        """
        return {
            "preferred_room_types": self.preferred_room_types.copy() if self.preferred_room_types else None,
            "preferred_hall_types": self.preferred_hall_types.copy() if self.preferred_hall_types else None,
            "room_probability": self.room_probability,
            "min_hall_between_rooms": self.min_hall_between_rooms,
            "allow_dead_ends": self.allow_dead_ends,
        }
