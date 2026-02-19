"""
idTech MAP format writer.

Handles conversion of 3D geometry to standard idTech MAP format files.
Generates valid MAP files compatible with ericw-tools (qbsp/vis/light).
"""

from typing import List, TextIO, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
import math
from datetime import datetime

if TYPE_CHECKING:
    from quake_levelgenerator.src.generators.profiles import GameProfile


@dataclass
class Plane:
    """
    Represents a plane defined by three points.
    
    In idTech MAP format, planes are defined by three non-collinear points
    that form a plane. The points should be in counter-clockwise order
    when viewed from outside the brush (the side you want to be solid).
    """
    p1: Tuple[float, float, float]  # First point (x, y, z)
    p2: Tuple[float, float, float]  # Second point (x, y, z)
    p3: Tuple[float, float, float]  # Third point (x, y, z)
    texture: str = "CRATE1_5"
    x_offset: float = 0.0
    y_offset: float = 0.0
    rotation: float = 0.0
    x_scale: float = 1.0
    y_scale: float = 1.0


@dataclass
class Brush:
    """
    Represents a brush (convex solid) in idTech.
    
    A brush is a convex polyhedron defined by the intersection of
    multiple half-spaces (planes). Each plane faces outward from
    the solid interior of the brush.
    """
    planes: List[Plane] = field(default_factory=list)
    brush_id: int = 0
    
    def validate(self) -> bool:
        """Validate brush has minimum required planes."""
        return len(self.planes) >= 4  # Minimum for a tetrahedron


@dataclass
class Entity:
    """
    Represents an entity in the map.
    
    Entities can be point entities (like lights, spawns) or
    brush entities (like doors, triggers, worldspawn).
    """
    classname: str
    properties: dict = field(default_factory=dict)
    brushes: List[Brush] = field(default_factory=list)
    
    def set_property(self, key: str, value: any) -> None:
        """Set an entity property with proper type conversion."""
        if isinstance(value, (list, tuple)) and len(value) == 3:
            # Convert 3D coordinates to space-separated string
            self.properties[key] = f"{value[0]} {value[1]} {value[2]}"
        elif isinstance(value, bool):
            self.properties[key] = "1" if value else "0"
        else:
            self.properties[key] = str(value)


class MapWriter:
    """
    Writes idTech MAP format files.
    
    Converts internal geometry representation to the standard MAP format
    that can be compiled by qbsp and related tools (ericw-tools).
    
    Coordinate System:
    - X: Right/Left
    - Y: Forward/Back  
    - Z: Up/Down
    
    MAP Format Notes:
    - Brushes are convex solids defined by plane intersections
    - Planes are defined by three points in counter-clockwise order
    - Texture alignment uses offset, rotation, and scale parameters
    """
    
    # Standard idTech textures
    TEXTURES = {
        'wall': ['CRATE1_5', 'BRICKA2_4', 'CITY2_1', 'METAL1_1', 'ROCK1_1'],
        'floor': ['GROUND1_6', 'FLOOR01_5', 'WIZMET1_2', 'METAL2_1'],
        'ceiling': ['CEIL1_1', 'CEIL3_5', 'SKY1'],
        'trim': ['METAL1_1', 'WBRICK1_5'],
        'door': ['DOOR02_1', 'DOOR03_1'],
        'liquid': ['*WATER1', '*LAVA1', '*SLIME1'],
        'trigger': ['TRIGGER', 'CLIP', 'SKIP']
    }
    
    def __init__(self, export_format: str = "idtech1"):
        """Initialize the MAP writer.

        Args:
            export_format: "idtech1" (default) or "idtech4".
        """
        self.entities: List[Entity] = []
        self.brush_counter = 0
        self.grid_snap = 1.0  # Snap coordinates to grid
        self.export_format = export_format
    
    def add_entity(self, entity: Entity) -> None:
        """
        Add an entity to the map.
        
        Args:
            entity: Entity to add
        """
        self.entities.append(entity)
    
    def add_brush(self, brush: Brush, entity_index: int = 0) -> None:
        """
        Add a brush to an entity.
        
        Args:
            brush: Brush to add
            entity_index: Index of entity to add brush to
        """
        if entity_index < len(self.entities):
            brush.brush_id = self.brush_counter
            self.brush_counter += 1
            if brush.validate():
                self.entities[entity_index].brushes.append(brush)
            else:
                raise ValueError(f"Invalid brush geometry: needs at least 4 planes")
    
    def create_box_brush(self, min_point: Tuple[float, float, float], 
                        max_point: Tuple[float, float, float], 
                        texture: str = "CRATE1_5") -> Brush:
        """
        Create a box-shaped brush with 6 faces.
        
        Args:
            min_point: Minimum corner (x, y, z)
            max_point: Maximum corner (x, y, z)
            texture: Texture name to use
            
        Returns:
            Box brush with 6 planes
        """
        x1, y1, z1 = min_point
        x2, y2, z2 = max_point
        
        # Snap to grid
        x1, y1, z1 = self._snap_to_grid(x1, y1, z1)
        x2, y2, z2 = self._snap_to_grid(x2, y2, z2)
        
        # Create the 6 faces of the box using proper idTech plane format
        # Each plane is defined by 3 non-colinear points
        # Use actual box boundary coordinates to ensure non-colinear points
        planes = [
            # Left face (X = x1 plane)
            Plane((x1, y1, z1), (x1, y2, z1), (x1, y1, z2), texture),
            
            # Right face (X = x2 plane)
            Plane((x2, y1, z1), (x2, y1, z2), (x2, y2, z1), texture),
            
            # Front face (Y = y1 plane)
            Plane((x1, y1, z1), (x1, y1, z2), (x2, y1, z1), texture),
            
            # Back face (Y = y2 plane)
            Plane((x1, y2, z1), (x2, y2, z1), (x1, y2, z2), texture),
            
            # Bottom face (Z = z1 plane)
            Plane((x1, y1, z1), (x2, y1, z1), (x1, y2, z1), texture),
            
            # Top face (Z = z2 plane)
            Plane((x1, y1, z2), (x1, y2, z2), (x2, y1, z2), texture)
        ]
        
        return Brush(planes=planes, brush_id=self.brush_counter)
    
    def create_room_brushes(self, room_bounds: Tuple[float, float, float, float], 
                          height: float = 128.0,
                          wall_thickness: float = 8.0,
                          wall_texture: str = "CRATE1_5", 
                          floor_texture: str = "GROUND1_6",
                          ceiling_texture: str = "CEIL1_1") -> List[Brush]:
        """
        Create brushes for a hollow room (walls, floor, ceiling).
        
        Args:
            room_bounds: (min_x, min_y, max_x, max_y) in world units
            height: Room height in units
            wall_thickness: Thickness of walls
            wall_texture: Texture for walls
            floor_texture: Texture for floor
            ceiling_texture: Texture for ceiling
            
        Returns:
            List of brushes forming the room
        """
        min_x, min_y, max_x, max_y = room_bounds
        brushes = []
        
        # Floor brush (extends below room)
        floor_brush = self.create_box_brush(
            (min_x - wall_thickness, min_y - wall_thickness, -wall_thickness),
            (max_x + wall_thickness, max_y + wall_thickness, 0),
            floor_texture
        )
        brushes.append(floor_brush)
        
        # Ceiling brush (extends above room)
        ceiling_brush = self.create_box_brush(
            (min_x - wall_thickness, min_y - wall_thickness, height),
            (max_x + wall_thickness, max_y + wall_thickness, height + wall_thickness),
            ceiling_texture
        )
        brushes.append(ceiling_brush)
        
        # Wall brushes (4 walls)
        # North wall (+Y)
        north_wall = self.create_box_brush(
            (min_x - wall_thickness, max_y, 0),
            (max_x + wall_thickness, max_y + wall_thickness, height),
            wall_texture
        )
        brushes.append(north_wall)
        
        # South wall (-Y)
        south_wall = self.create_box_brush(
            (min_x - wall_thickness, min_y - wall_thickness, 0),
            (max_x + wall_thickness, min_y, height),
            wall_texture
        )
        brushes.append(south_wall)
        
        # East wall (+X)
        east_wall = self.create_box_brush(
            (max_x, min_y, 0),
            (max_x + wall_thickness, max_y, height),
            wall_texture
        )
        brushes.append(east_wall)
        
        # West wall (-X)
        west_wall = self.create_box_brush(
            (min_x - wall_thickness, min_y, 0),
            (min_x, max_y, height),
            wall_texture
        )
        brushes.append(west_wall)
        
        return brushes
    
    def write_to_file(self, filename: str, validate: bool = True) -> None:
        """
        Write the map to a MAP format file.

        Uses the format writer selected by ``self.export_format``
        (``"idtech1"`` or ``"idtech4"``).

        Per CLAUDE.md Section 3 (Quality Gates [BINDING]):
        - Validates map structure and format compliance before export
        - Raises ValidationError if critical issues found (when validate=True)

        Args:
            filename: Output filename (should end with .map)
            validate: If True, run validation gate before export (default True).
                      WARNING: Setting validate=False bypasses the export validation
                      gate and should ONLY be used for debugging/testing purposes.
                      Production exports should always use validate=True.

        Raises:
            ValueError: If no entities to write
            ValidationError: If validation fails with FAIL severity issues
        """
        if not self.entities:
            raise ValueError("No entities to write. Add at least a worldspawn entity.")

        # Run export validation gate per CLAUDE.md Section 3
        if validate:
            self._run_export_validation()

        from quake_levelgenerator.src.conversion.format_writers import get_writer
        from quake_levelgenerator.src.conversion.plane_math import PlaneGeometry

        writer = get_writer(self.export_format)

        with open(filename, 'w') as file:
            is_idtech4 = self.export_format.lower() == "idtech4"

            if is_idtech4:
                # Doom 3 format requires Version 2 header
                file.write("Version 2\n")
            else:
                file.write(f"// Generated by idTech Geometry Toolkit\n")
                file.write(f"// Format: {self.export_format}\n")
                file.write(f"// Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                total_brushes = sum(len(e.brushes) for e in self.entities)
                file.write(f"// Total entities: {len(self.entities)}\n")
                file.write(f"// Total brushes: {total_brushes}\n\n")

            for entity_idx, entity in enumerate(self.entities):
                # Convert Brush/Plane objects to PlaneGeometry lists
                brush_plane_lists = []
                for brush in entity.brushes:
                    planes = []
                    for p in brush.planes:
                        planes.append(PlaneGeometry.from_three_points(
                            p.p1, p.p2, p.p3,
                            texture=p.texture,
                            offset_x=p.x_offset,
                            offset_y=p.y_offset,
                            rotation=p.rotation,
                            scale_x=p.x_scale,
                            scale_y=p.y_scale,
                        ))
                    brush_plane_lists.append(planes)

                file.write(writer.write_entity(
                    entity.classname,
                    entity.properties,
                    brush_plane_lists,
                    entity_index=entity_idx,
                ))
    
    def _write_entity(self, entity: Entity, file: TextIO) -> None:
        """
        Write a single entity to the file in MAP format.
        
        Args:
            entity: Entity to write
            file: File handle to write to
        """
        # Opening brace for entity
        file.write("{\n")
        
        # Write entity properties (key-value pairs)
        file.write(f'"classname" "{entity.classname}"\n')
        for key, value in entity.properties.items():
            # Escape quotes in values
            value_str = str(value).replace('"', '\\"')
            file.write(f'"{key}" "{value_str}"\n')
        
        # Write entity brushes (if any)
        for brush in entity.brushes:
            self._write_brush(brush, file)
        
        # Closing brace for entity
        file.write("}\n")
    
    def _write_brush(self, brush: Brush, file: TextIO) -> None:
        """
        Write a single brush to the file in MAP format.
        
        Args:
            brush: Brush to write
            file: File handle to write to
        """
        # Brush comment (optional but helpful for debugging)
        file.write(f"// brush {brush.brush_id}\n")
        
        # Opening brace for brush
        file.write("{\n")
        
        # Write each plane in the brush
        for plane in brush.planes:
            self._write_plane(plane, file)
        
        # Closing brace for brush
        file.write("}\n")
    
    def _write_plane(self, plane: Plane, file: TextIO) -> None:
        """
        Write a single plane to the file in MAP format.
        
        Format: ( x1 y1 z1 ) ( x2 y2 z2 ) ( x3 y3 z3 ) texture x_off y_off rot x_scale y_scale
        
        Args:
            plane: Plane to write
            file: File handle to write to
        """
        # Format the three points
        p1_str = f"( {plane.p1[0]:.0f} {plane.p1[1]:.0f} {plane.p1[2]:.0f} )"
        p2_str = f"( {plane.p2[0]:.0f} {plane.p2[1]:.0f} {plane.p2[2]:.0f} )"
        p3_str = f"( {plane.p3[0]:.0f} {plane.p3[1]:.0f} {plane.p3[2]:.0f} )"
        
        # Write the complete plane line
        file.write(f"{p1_str} {p2_str} {p3_str} {plane.texture} ")
        file.write(f"{plane.x_offset:.0f} {plane.y_offset:.0f} ")
        file.write(f"{plane.rotation:.0f} {plane.x_scale:.2f} {plane.y_scale:.2f}\n")
    
    def create_worldspawn(
        self,
        wad_file: str = "id1.wad",
        profile: Optional['GameProfile'] = None
    ) -> Entity:
        """
        Create the worldspawn entity (required for all maps).

        Args:
            wad_file: WAD file containing textures (used if no profile specified)
            profile: Optional GameProfile with engine-specific worldspawn properties

        Returns:
            Worldspawn entity with default properties
        """
        worldspawn = Entity("worldspawn")

        if profile:
            # Use profile-specific worldspawn properties
            for key, value in profile.get_worldspawn_properties().items():
                worldspawn.set_property(key, value)
        elif self.export_format.lower() == "idtech4":
            # Doom 3 worldspawn — no wad or sunlight keys
            pass
        else:
            # Default Quake 1 worldspawn
            worldspawn.set_property("wad", wad_file)
            worldspawn.set_property("light", "50")
            worldspawn.set_property("_sunlight", "150")
            worldspawn.set_property("_sunlight_pitch", "-45")
            worldspawn.set_property("_sunlight_angle", "315")
        return worldspawn
    
    def add_player_start(self, position: Tuple[float, float, float],
                        angle: float = 0.0) -> None:
        """
        Add a player start point entity.

        Args:
            position: Start position (x, y, z)
            angle: Start angle in degrees (0-360)
        """
        player_start = Entity("info_player_start")
        player_start.set_property("origin", position)
        player_start.set_property("angle", angle)
        self.add_entity(player_start)

    def _snap_to_grid(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        Snap coordinates to grid for cleaner geometry.
        
        Args:
            x, y, z: Coordinates to snap
            
        Returns:
            Snapped coordinates
        """
        if self.grid_snap <= 0:
            return (x, y, z)
        
        return (
            round(x / self.grid_snap) * self.grid_snap,
            round(y / self.grid_snap) * self.grid_snap,
            round(z / self.grid_snap) * self.grid_snap
        )
    
    def validate_map(self) -> List[str]:
        """
        Validate the map for common issues.
        
        Returns:
            List of validation warnings/errors
        """
        issues = []
        
        # Check for worldspawn
        has_worldspawn = any(e.classname == "worldspawn" for e in self.entities)
        if not has_worldspawn:
            issues.append("ERROR: Missing worldspawn entity")
        
        # Check for player start
        has_player_start = any(e.classname in ["info_player_start", "info_player_deathmatch"] 
                              for e in self.entities)
        if not has_player_start:
            issues.append("WARNING: No player start point found")
        
        # Check brush validity
        for entity in self.entities:
            for i, brush in enumerate(entity.brushes):
                if not brush.validate():
                    issues.append(f"ERROR: Brush {i} in entity '{entity.classname}' has < 4 planes")
        
        # Check for sealed world (basic check)
        worldspawn = next((e for e in self.entities if e.classname == "worldspawn"), None)
        if worldspawn and len(worldspawn.brushes) < 6:
            issues.append("WARNING: World may not be sealed (too few brushes)")

        return issues

    def _run_export_validation(self) -> None:
        """Run export validation gate.

        Per CLAUDE.md Section 3 (Quality Gates [BINDING]):
        - MAP-001: Worldspawn entity exists
        - MAP-002: info_player_start exists
        - Format-specific validation

        Raises:
            ValidationError: If validation fails with FAIL severity issues
        """
        try:
            from quake_levelgenerator.src.validation import get_validator, ValidationError

            validator = get_validator()
            result = validator.validate_export(
                self.entities,
                self.export_format,
                check_structure=True,
                check_format=True,
            )

            # Raise on FAIL issues
            if result.failed:
                raise ValidationError(result)

        except ImportError:
            # Validation package not available - use legacy validation
            issues = self.validate_map()
            errors = [i for i in issues if i.startswith("ERROR")]
            if errors:
                raise ValueError("\n".join(errors))