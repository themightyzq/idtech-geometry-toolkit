"""
Parameter randomization for random layout generation.

This module defines which parameters are SAFE to randomize for each primitive type.
Safe parameters are interior/decorative and do NOT affect:
- Footprint size (width, length, height)
- Portal positioning or alignment
- Wall thickness

NEVER RANDOMIZE:
- width, length, height (footprint-critical)
- nave_width, nave_length, hall_width (structural)
- wall_thickness (sealed geometry)
- has_entrance, has_exit, has_side (portal control - set by layout system)
- All portal_* parameters
"""

import random
from typing import Any, Dict, Optional


# Safe parameters to randomize per primitive type.
# Each entry maps parameter name -> randomization config
#
# Config types:
#   'int':    {'type': 'int', 'min': x, 'max': y}
#   'float':  {'type': 'float', 'min': x, 'max': y}
#   'choice': {'type': 'choice', 'choices': [...]}
#   'bool':   {'type': 'bool'}
#   'bool_weighted': {'type': 'bool_weighted', 'true_prob': 0.3}

SAFE_RANDOMIZABLE_PARAMS: Dict[str, Dict[str, Dict[str, Any]]] = {
    # ==========================================================================
    # ROOMS
    # ==========================================================================
    'Sanctuary': {
        'random_seed': {'type': 'int', 'min': 1, 'max': 999999},
        # sanctuary_type: FORBIDDEN — basilica/cruciform/hall_church add aisles/transepts
        # that extend geometry beyond the footprint, causing BSP leaks.
        # apse: FORBIDDEN — extends geometry beyond the back wall boundary.
        'shell_sides': {
            'type': 'choice',
            'choices': [3, 4, 5, 6, 8, 10],
        },
        'pillar_style': {
            'type': 'choice',
            'choices': ['square', 'hexagonal', 'octagonal', 'round']
        },
        'pillar_capital': {'type': 'bool_weighted', 'true_prob': 0.3},
    },

    'Tomb': {
        'random_seed': {'type': 'int', 'min': 1, 'max': 999999},
        # shell_sides: EXCLUDED — multi-portal room (entrance + exit).
        # Polygonal shell system only supports single-portal rooms.
        'coffin_count': {'type': 'int', 'min': 0, 'max': 8},
        'coffin_layout': {
            'type': 'choice',
            'choices': ['rows', 'alcoves', 'perimeter']
        },
        'central_platform': {'type': 'bool_weighted', 'true_prob': 0.7},
    },

    'Tower': {
        'random_seed': {'type': 'int', 'min': 1, 'max': 999999},
        'shell_sides': {
            'type': 'choice',
            'choices': [3, 4, 5, 6, 8, 10],
        },
    },

    'Chamber': {
        'random_seed': {'type': 'int', 'min': 1, 'max': 999999},
        'shell_sides': {
            'type': 'choice',
            'choices': [3, 4, 5, 6, 8, 10],
        },
        'pillar_style': {
            'type': 'choice',
            'choices': ['square', 'hexagonal', 'octagonal', 'round']
        },
        'pillar_capital': {'type': 'bool_weighted', 'true_prob': 0.3},
    },

    'Storage': {
        'random_seed': {'type': 'int', 'min': 1, 'max': 999999},
        # 2×2 footprint — conservative set (high shell_sides may exceed footprint)
        'shell_sides': {
            'type': 'choice',
            'choices': [3, 4, 5, 6],
        },
    },

    'GreatHall': {
        'random_seed': {'type': 'int', 'min': 1, 'max': 999999},
        # shell_sides: EXCLUDED — asymmetric 4×6 footprint with off-center entrance
        # portal (cell 1,0) causes vestibule X misalignment in polygonal mode.
        'pillar_style': {
            'type': 'choice',
            'choices': ['square', 'hexagonal', 'octagonal', 'round']
        },
        'pillar_capital': {'type': 'bool_weighted', 'true_prob': 0.4},
    },

    'Prison': {
        'random_seed': {'type': 'int', 'min': 1, 'max': 999999},
        'shell_sides': {
            'type': 'choice',
            'choices': [3, 4, 5, 6, 8, 10],
        },
    },

    'Armory': {
        'random_seed': {'type': 'int', 'min': 1, 'max': 999999},
        # 2×3 footprint — slightly conservative set
        'shell_sides': {
            'type': 'choice',
            'choices': [3, 4, 5, 6, 8],
        },
    },

    'Cistern': {
        'random_seed': {'type': 'int', 'min': 1, 'max': 999999},
        'shell_sides': {
            'type': 'choice',
            'choices': [3, 4, 5, 6, 8, 10],
        },
        'pillar_style': {
            'type': 'choice',
            'choices': ['square', 'hexagonal', 'octagonal', 'round']
        },
        'pillar_capital': {'type': 'bool_weighted', 'true_prob': 0.3},
    },

    'Stronghold': {
        'random_seed': {'type': 'int', 'min': 1, 'max': 999999},
        'shell_sides': {
            'type': 'choice',
            'choices': [3, 4, 5, 6, 8, 10],
        },
    },

    'Courtyard': {
        'random_seed': {'type': 'int', 'min': 1, 'max': 999999},
        'shell_sides': {
            'type': 'choice',
            'choices': [3, 4, 5, 6, 8, 10],
        },
    },

    'Arena': {
        'random_seed': {'type': 'int', 'min': 1, 'max': 999999},
        'shell_sides': {
            'type': 'choice',
            'choices': [3, 4, 5, 6, 8, 10],
        },
    },

    'Laboratory': {
        'random_seed': {'type': 'int', 'min': 1, 'max': 999999},
        'shell_sides': {
            'type': 'choice',
            'choices': [3, 4, 5, 6, 8, 10],
        },
    },

    'Vault': {
        'random_seed': {'type': 'int', 'min': 1, 'max': 999999},
        'shell_sides': {
            'type': 'choice',
            'choices': [3, 4, 5, 6, 8, 10],
        },
    },

    'Barracks': {
        'random_seed': {'type': 'int', 'min': 1, 'max': 999999},
        'shell_sides': {
            'type': 'choice',
            'choices': [3, 4, 5, 6, 8, 10],
        },
    },

    'Shrine': {
        'random_seed': {'type': 'int', 'min': 1, 'max': 999999},
        'shell_sides': {
            'type': 'choice',
            'choices': [3, 4, 5, 6, 8, 10],
        },
    },

    'Pit': {
        'random_seed': {'type': 'int', 'min': 1, 'max': 999999},
        'shell_sides': {
            'type': 'choice',
            'choices': [3, 4, 5, 6, 8, 10],
        },
    },

    'Antechamber': {
        'random_seed': {'type': 'int', 'min': 1, 'max': 999999},
        # shell_sides: EXCLUDED — multi-portal room (entrance, exit, side_east, side_west).
        # Polygonal shell system only supports single-portal rooms.
    },

    'Hub': {
        'random_seed': {'type': 'int', 'min': 1, 'max': 999999},
        'center_feature': {'type': 'choice', 'choices': ['none', 'pillars', 'platform', 'pit']},
        'pillar_style': {
            'type': 'choice',
            'choices': ['square', 'hexagonal', 'octagonal', 'round']
        },
    },

    # ==========================================================================
    # TEMPLATE-SPECIFIC ROOMS
    # ==========================================================================
    'Gatehouse': {
        'random_seed': {'type': 'int', 'min': 1, 'max': 999999},
        'passage_width': {'type': 'float', 'min': 32, 'max': 96},
    },

    'Sewer': {
        'random_seed': {'type': 'int', 'min': 1, 'max': 999999},
        'channel_depth': {'type': 'float', 'min': 24, 'max': 96},
        'has_bridge': {'type': 'bool'},
    },

    'Ossuary': {
        'random_seed': {'type': 'int', 'min': 1, 'max': 999999},
        'niche_count': {'type': 'int', 'min': 2, 'max': 8},
        'niche_rows': {'type': 'int', 'min': 1, 'max': 3},
    },

    'Cloister': {
        'random_seed': {'type': 'int', 'min': 1, 'max': 999999},
        'pillar_count': {'type': 'int', 'min': 4, 'max': 16},
        'pillar_style': {
            'type': 'choice',
            'choices': ['square', 'hexagonal', 'octagonal', 'round']
        },
        'pillar_capital': {'type': 'bool_weighted', 'true_prob': 0.3},
    },

    'Colosseum': {
        'random_seed': {'type': 'int', 'min': 1, 'max': 999999},
        'tier_count': {'type': 'int', 'min': 1, 'max': 3},
        'tier_height': {'type': 'float', 'min': 8, 'max': 32},
    },

    # ==========================================================================
    # MULTI-FLOOR ROOMS - Vertical connectivity rooms
    # Only random_seed is safe (upper_portal params are critical for alignment)
    # ==========================================================================
    'Amphitheater': {
        'random_seed': {'type': 'int', 'min': 1, 'max': 999999},
    },

    'CatwalkChamber': {
        'random_seed': {'type': 'int', 'min': 1, 'max': 999999},
    },

    'BalconyRoom': {
        'random_seed': {'type': 'int', 'min': 1, 'max': 999999},
    },

    'SunkenChamber': {
        'random_seed': {'type': 'int', 'min': 1, 'max': 999999},
    },

    'LibraryArchive': {
        'random_seed': {'type': 'int', 'min': 1, 'max': 999999},
    },

    'Grotto': {
        'random_seed': {'type': 'int', 'min': 1, 'max': 999999},
    },

    'RadialShrine': {
        'random_seed': {'type': 'int', 'min': 1, 'max': 999999},
    },

    'Forge': {
        'random_seed': {'type': 'int', 'min': 1, 'max': 999999},
    },

    # ==========================================================================
    # NEW ARCHETYPE ROOMS
    # ==========================================================================
    'ThroneRoom': {
        'random_seed': {'type': 'int', 'min': 1, 'max': 999999},
        'dais_height': {'type': 'float', 'min': 8, 'max': 24},
        'pillar_style': {
            'type': 'choice',
            'choices': ['square', 'hexagonal', 'octagonal', 'round']
        },
        'pillar_capital': {'type': 'bool'},
        'has_pillars': {'type': 'bool'},
    },

    'Vestibule': {
        'random_seed': {'type': 'int', 'min': 1, 'max': 999999},
    },

    'Processional': {
        'random_seed': {'type': 'int', 'min': 1, 'max': 999999},
        'pillar_count': {'type': 'int', 'min': 2, 'max': 4},
        'pillar_style': {
            'type': 'choice',
            'choices': ['square', 'hexagonal', 'octagonal', 'round']
        },
        'pillar_capital': {'type': 'bool'},
    },

    'DoglegRoom': {
        'random_seed': {'type': 'int', 'min': 1, 'max': 999999},
    },

    # ==========================================================================
    # HALLS - Minimal randomization (portal alignment critical)
    # Only random_seed is safe for halls
    # ==========================================================================
    'StraightHall': {
        'random_seed': {'type': 'int', 'min': 1, 'max': 999999},
    },

    'TJunction': {
        'random_seed': {'type': 'int', 'min': 1, 'max': 999999},
    },

    'Crossroads': {
        'random_seed': {'type': 'int', 'min': 1, 'max': 999999},
    },

    'SquareCorner': {
        'random_seed': {'type': 'int', 'min': 1, 'max': 999999},
    },

    'WideCorner': {
        'random_seed': {'type': 'int', 'min': 1, 'max': 999999},
    },

    'VerticalStairHall': {
        'random_seed': {'type': 'int', 'min': 1, 'max': 999999},
    },

    'NarrowPassage': {
        'random_seed': {'type': 'int', 'min': 1, 'max': 999999},
    },

    'WideCorridor': {
        'random_seed': {'type': 'int', 'min': 1, 'max': 999999},
    },
}

# Parameters that must NEVER be randomized (safety check)
FORBIDDEN_PARAMS = frozenset([
    'width', 'length', 'height',
    'nave_width', 'nave_length', 'hall_width',
    'wall_thickness',
    'has_entrance', 'has_exit', 'has_side',
    'portal_width', 'portal_height',
    # Sanctuary: type/apse affect geometry envelope (aisles, transepts, apse extend beyond footprint)
    'sanctuary_type', 'apse',
])


def randomize_primitive_params(
    primitive_type: str,
    rng: random.Random,
    override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate randomized safe parameters for a primitive.

    Args:
        primitive_type: The type name of the primitive (e.g., 'Sanctuary', 'Tower')
        rng: Seeded random.Random instance for reproducibility
        override: Optional dict of params to override (not randomized)

    Returns:
        Dict of parameter name -> randomized value
    """
    params: Dict[str, Any] = {}
    safe_params = SAFE_RANDOMIZABLE_PARAMS.get(primitive_type, {})

    for param_name, config in safe_params.items():
        # Skip if overridden
        if override and param_name in override:
            params[param_name] = override[param_name]
            continue

        param_type = config.get('type')

        if param_type == 'int':
            params[param_name] = rng.randint(config['min'], config['max'])

        elif param_type == 'float':
            params[param_name] = rng.uniform(config['min'], config['max'])

        elif param_type == 'choice':
            params[param_name] = rng.choice(config['choices'])

        elif param_type == 'bool':
            params[param_name] = rng.choice([True, False])

        elif param_type == 'bool_weighted':
            true_prob = config.get('true_prob', 0.5)
            params[param_name] = rng.random() < true_prob

    return params


def validate_safe_params() -> bool:
    """Verify no forbidden parameters are in the safe list.

    Returns:
        True if validation passes, raises ValueError otherwise.
    """
    for prim_type, safe_params in SAFE_RANDOMIZABLE_PARAMS.items():
        for param_name in safe_params:
            if param_name in FORBIDDEN_PARAMS:
                raise ValueError(
                    f"SAFETY VIOLATION: Forbidden param '{param_name}' "
                    f"found in SAFE_RANDOMIZABLE_PARAMS for '{prim_type}'"
                )
    return True


def get_all_primitive_types() -> list:
    """Return list of all primitive types with randomization defined."""
    return list(SAFE_RANDOMIZABLE_PARAMS.keys())
