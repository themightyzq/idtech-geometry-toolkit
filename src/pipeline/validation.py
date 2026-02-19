"""
Map validation utilities.

Provides validation for generated maps before and after compilation.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import math

from ..conversion.map_writer import Entity, Brush, Plane


class ValidationLevel(Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a validation issue found in a map."""
    level: ValidationLevel
    category: str
    message: str
    location: Optional[str] = None
    suggestion: Optional[str] = None


class MapValidator:
    """
    Validates idTech maps for common issues and errors.
    
    Performs various checks on map geometry, entities, and structure
    to identify problems before compilation or gameplay.
    """
    
    def __init__(self):
        """Initialize the map validator."""
        self.issues: List[ValidationIssue] = []
        self.validation_rules: Dict[str, bool] = {
            "check_player_start": True,
            "check_brush_validity": True,
            "check_texture_alignment": True,
            "check_entity_properties": True,
            "check_map_bounds": True,
            "check_sealed_map": True,
            "check_brush_intersections": False,  # Expensive check
        }
    
    def validate_map(self, entities: List[Entity]) -> List[ValidationIssue]:
        """
        Validate a complete map at a basic level.
        
        Args:
            entities: List of entities in the map
            
        Returns:
            List of validation issues found
        """
        self.issues.clear()
        if not entities:
            self._add_issue(ValidationLevel.ERROR, "structure", "No entities present in map")
            return self.issues
        
        # Run checks
        if self.validation_rules.get("check_player_start", True):
            self.check_player_start(entities)
        if self.validation_rules.get("check_brush_validity", True):
            for ent in entities:
                for brush in ent.brushes:
                    self.validate_brush(brush)
        if self.validation_rules.get("check_map_bounds", True):
            self.check_map_bounds(entities)
        if self.validation_rules.get("check_sealed_map", False):
            self.check_sealed_map(entities)
        
        return self.issues
    
    def check_player_start(self, entities: List[Entity]) -> None:
        """Check that the map has at least one player start."""
        has_sp = any(e.classname in ("info_player_start", "info_player_deathmatch") for e in entities)
        if not has_sp:
            self._add_issue(
                ValidationLevel.ERROR,
                "entities",
                "No player start found (info_player_start or info_player_deathmatch)"
            )
    
    def check_worldspawn(self, entities: List[Entity]) -> None:
        """Check that worldspawn entity exists and has brushes or properties."""
        has_ws = any(e.classname == "worldspawn" for e in entities)
        if not has_ws:
            self._add_issue(ValidationLevel.ERROR, "entities", "Missing worldspawn entity")
    
    def validate_brush(self, brush: Brush) -> None:
        """Validate a single brush for basic geometric correctness."""
        if len(brush.planes) < 4:
            self._add_issue(ValidationLevel.ERROR, "geometry", f"Brush {brush.brush_id} has fewer than 4 planes")
        # Check for obviously degenerate planes (identical points)
        for p in brush.planes:
            if p.p1 == p.p2 or p.p2 == p.p3 or p.p1 == p.p3:
                self._add_issue(ValidationLevel.ERROR, "geometry", f"Brush {brush.brush_id} has degenerate plane with duplicate points")
    
    def validate_plane(self, plane: Plane) -> None:
        """
        Validate a single plane definition.
        
        Args:
            plane: Plane to validate
        """
        # TODO: Check that three points are not collinear
        # TODO: Validate texture name format
        # TODO: Check texture offset and scale values
        # TODO: Verify rotation angle is valid
        pass
    
    def check_brush_intersections(self, brushes: List[Brush]) -> None:
        """
        Check for problematic brush intersections.
        
        Args:
            brushes: List of brushes to check
        """
        # TODO: Compare all brush pairs
        # TODO: Detect overlapping brushes
        # TODO: Check for micro-brushes
        # TODO: Identify floating brushes
        pass
    
    def check_map_bounds(self, entities: List[Entity]) -> None:
        """Check that map stays within reasonable bounds (basic heuristic)."""
        try:
            min_x = min_y = min_z = float("inf")
            max_x = max_y = max_z = float("-inf")
            for e in entities:
                for b in e.brushes:
                    for pl in b.planes:
                        for (x, y, z) in (pl.p1, pl.p2, pl.p3):
                            min_x, max_x = min(min_x, x), max(max_x, x)
                            min_y, max_y = min(min_y, y), max(max_y, y)
                            min_z, max_z = min(min_z, z), max(max_z, z)
            # Rough idTech limits; warn if extreme
            limit = 32768
            if any(abs(v) > limit for v in (min_x, max_x, min_y, max_y, min_z, max_z)):
                self._add_issue(
                    ValidationLevel.WARNING,
                    "bounds",
                    f"Map extents exceed ±{limit} units; consider reducing size"
                )
        except Exception:
            # Non-fatal
            pass
    
    def check_sealed_map(self, entities: List[Entity]) -> None:
        """
        Check if the map is properly sealed (no leaks).
        
        Args:
            entities: List of entities to check
        """
        # TODO: This is a complex check that requires
        # TODO: analyzing brush connectivity
        # TODO: For now, add placeholder warning
        # TODO: Could integrate with qbsp leak detection
        pass
    
    def validate_entity_properties(self, entity: Entity) -> None:
        """
        Validate entity properties and values.
        
        Args:
            entity: Entity to validate
        """
        # TODO: Check required properties for entity class
        # TODO: Validate property value formats
        # TODO: Check for unknown properties
        # TODO: Validate entity-specific requirements
        pass
    
    def check_texture_usage(self, entities: List[Entity]) -> None:
        """
        Check texture usage and alignment.
        
        Args:
            entities: List of entities to check
        """
        # TODO: Collect all texture names used
        # TODO: Check for missing or invalid textures
        # TODO: Analyze texture alignment issues
        # TODO: Check texture scale values
        pass
    
    def get_issues_by_level(self, level: ValidationLevel) -> List[ValidationIssue]:
        """
        Get all issues of a specific severity level.
        
        Args:
            level: Severity level to filter by
            
        Returns:
            List of issues at the specified level
        """
        return [issue for issue in self.issues if issue.level == level]
    
    def has_critical_issues(self) -> bool:
        """
        Check if any critical issues were found.
        
        Returns:
            True if critical issues exist
        """
        return any(issue.level == ValidationLevel.CRITICAL for issue in self.issues)
    
    def has_errors(self) -> bool:
        """
        Check if any errors were found.
        
        Returns:
            True if errors exist
        """
        return any(issue.level == ValidationLevel.ERROR for issue in self.issues)
    
    def generate_report(self) -> str:
        """Generate a formatted validation report."""
        if not self.issues:
            return "No issues found"
        lines = []
        for issue in self.issues:
            loc = f" @ {issue.location}" if issue.location else ""
            sug = f" | Suggestion: {issue.suggestion}" if issue.suggestion else ""
            lines.append(f"[{issue.level.value.upper()}] {issue.category}: {issue.message}{loc}{sug}")
        return "\n".join(lines)
    
    def _add_issue(self, level: ValidationLevel, category: str, 
                   message: str, location: str = None, 
                   suggestion: str = None) -> None:
        """
        Add a validation issue to the results.
        
        Args:
            level: Severity level
            category: Issue category
            message: Issue description
            location: Optional location information
            suggestion: Optional fix suggestion
        """
        issue = ValidationIssue(
            level=level,
            category=category,
            message=message,
            location=location,
            suggestion=suggestion
        )
        self.issues.append(issue)
