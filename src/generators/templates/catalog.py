"""
Template catalog — registry of all available generation templates.
"""

from __future__ import annotations
from typing import Dict, List, Optional

from .base import GenerationTemplate


class TemplateCatalog:
    """Registry mapping names to generation templates."""

    def __init__(self):
        self._templates: Dict[str, GenerationTemplate] = {}

    def register(self, template: GenerationTemplate):
        """Register a template in the catalog."""
        self._templates[template.name] = template

    def get_template(self, name: str) -> Optional[GenerationTemplate]:
        """Get a template by name."""
        return self._templates.get(name)

    def list_templates(self, category: Optional[str] = None) -> List[str]:
        """
        List all template names, optionally filtered by category.

        Args:
            category: Optional category filter (case-insensitive)

        Returns:
            Sorted list of template names
        """
        if category is None:
            return sorted(self._templates.keys())
        return sorted(
            name for name, template in self._templates.items()
            if template.category.lower() == category.lower()
        )

    def list_categories(self) -> List[str]:
        """Get all unique categories."""
        return sorted({t.category for t in self._templates.values()})


# Global singleton
TEMPLATE_CATALOG = TemplateCatalog()

# Import and register built-in templates
from .builtin import register_builtin_templates
register_builtin_templates(TEMPLATE_CATALOG)
