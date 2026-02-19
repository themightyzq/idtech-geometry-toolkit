"""
Primitive catalog — registry of all available geometric primitives.

Per CLAUDE.md Section 6 (Agent System [BINDING]):
Module registration includes optional validation via the unified validation package.
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Type

from .base import GeometricPrimitive
from .structural import StraightStaircase, Arch, Pillar, Buttress, Battlement
from .rooms import (
    Sanctuary, Tomb, Tower, Chamber, Storage, GreatHall, Prison, Armory, Cistern, Stronghold, Courtyard,
    Arena, Laboratory, Vault, Barracks, Shrine, Pit, Antechamber, SecretChamber
)
from .connective import Bridge, Platform, Rampart, Gallery
from .halls import StraightHall, TJunction, Crossroads, SquareCorner, VerticalStairHall, SecretHall

logger = logging.getLogger(__name__)


class PrimitiveCatalog:
    """Registry mapping names to primitive classes.

    Supports optional validation on registration per CLAUDE.md Section 3 (Quality Gates).
    When validate_on_register=True, modules are checked for basic contract compliance.
    """

    def __init__(self, validate_on_register: bool = True):
        """Initialize the catalog.

        Args:
            validate_on_register: If True, validate modules during registration.
                                  Enabled by default per CLAUDE.md Section 3 (Quality Gates).
                                  Note: Validation logs FAIL issues but still registers
                                  modules to avoid blocking app startup (soft-fail).
        """
        self._primitives: Dict[str, Type[GeometricPrimitive]] = {}
        self._validation_enabled = validate_on_register
        self._validation_results: Dict[str, 'ValidationResult'] = {}

    def register(self, cls: Type[GeometricPrimitive]) -> bool:
        """Register a primitive class in the catalog.

        Args:
            cls: GeometricPrimitive subclass to register

        Returns:
            True if registration succeeded, False if validation failed
        """
        name = cls.get_display_name()

        # Optional validation during registration
        if self._validation_enabled:
            try:
                from quake_levelgenerator.src.validation import get_validator
                validator = get_validator()
                result = validator.validate_module_registration(
                    cls, check_footprint=False, check_catalog=False
                )
                self._validation_results[name] = result
                if result.failed:
                    for issue in result.errors:
                        logger.error(
                            f"[FAIL] {issue.code} module={name} :: "
                            f"{issue.message} :: fix={issue.remediation or 'N/A'}"
                        )
            except ImportError:
                # Validation package not available
                pass

        self._primitives[name] = cls
        return True

    def enable_validation(self, enabled: bool = True) -> None:
        """Enable or disable validation on registration.

        Args:
            enabled: Whether to validate modules during registration
        """
        self._validation_enabled = enabled

    def get_validation_result(self, name: str) -> Optional['ValidationResult']:
        """Get validation result for a registered module.

        Args:
            name: Module display name

        Returns:
            ValidationResult if available, None otherwise
        """
        return self._validation_results.get(name)

    def list_primitives(self, category: Optional[str] = None) -> List[str]:
        if category is None:
            return sorted(self._primitives.keys())
        return sorted(
            name for name, cls in self._primitives.items()
            if cls.get_category().lower() == category.lower()
        )

    def list_categories(self) -> List[str]:
        return sorted({cls.get_category() for cls in self._primitives.values()})

    def get_primitive(self, name: str) -> Optional[Type[GeometricPrimitive]]:
        # Try exact match first
        if name in self._primitives:
            return self._primitives[name]
        # Try with spaces inserted before capitals (StraightHall -> Straight Hall)
        # Also handle single-letter prefixes (LJunction -> L Junction, TJunction -> T Junction)
        import re
        spaced_name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
        spaced_name = re.sub(r'([A-Z])([A-Z][a-z])', r'\1 \2', spaced_name)
        if spaced_name in self._primitives:
            return self._primitives[spaced_name]
        return None


# Global singleton
PRIMITIVE_CATALOG = PrimitiveCatalog()

for _cls in [
    StraightStaircase, Arch, Pillar, Buttress, Battlement,
    Sanctuary, Tomb, Tower, Chamber, Storage, GreatHall, Prison, Armory, Cistern, Stronghold, Courtyard,
    Arena, Laboratory, Vault, Barracks, Shrine, Pit, Antechamber, SecretChamber,
    Bridge, Platform, Rampart, Gallery,
    StraightHall, TJunction, Crossroads, SquareCorner, VerticalStairHall, SecretHall,
]:
    PRIMITIVE_CATALOG.register(_cls)
