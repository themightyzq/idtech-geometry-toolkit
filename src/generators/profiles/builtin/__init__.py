"""
Built-in game profiles for idTech engines.

This module exports the built-in profiles that ship with the toolkit:
- QUAKE1_PROFILE: Original Quake (idTech 1)
- DOOM3_PROFILE: Doom 3 (idTech 4)
"""

from .quake1 import QUAKE1_PROFILE
from .doom3 import DOOM3_PROFILE

__all__ = ['QUAKE1_PROFILE', 'DOOM3_PROFILE']
