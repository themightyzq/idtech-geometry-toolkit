"""
idTech Map Generator - Parameter Panel Widget

This widget provides controls for configuring map generation parameters,
including map size, room count, complexity, and other generation settings.
"""

from typing import Dict, Any, Optional
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QSlider, QPushButton,
    QScrollArea, QFrame
)
from quake_levelgenerator.src.ui.safe_checkbox import SafeCheckBox
from quake_levelgenerator.src.ui.safe_spinbox import SafeSpinBox
from quake_levelgenerator.src.ui.safe_combobox import SafeComboBox

# Alias for compatibility - use safe versions to avoid macOS crashes
QSpinBox = SafeSpinBox
QComboBox = SafeComboBox  # Click to cycle, no dropdown
from PyQt5.QtCore import Qt, pyqtSignal, QSettings
from PyQt5.QtGui import QFont

from quake_levelgenerator.src.ui import style_constants as sc
from quake_levelgenerator.src.ui.style_constants import set_accessible
from quake_levelgenerator.src.generators.templates import TEMPLATE_CATALOG, GenerationTemplate
from quake_levelgenerator.src.generators.profiles import PROFILE_CATALOG, GameProfile


class LayoutPanel(QWidget):
    """Panel for controlling map generation parameters."""

    # Signal emitted when parameters change
    parameters_changed = pyqtSignal(dict)
    # Signal emitted when a template is selected
    template_selected = pyqtSignal(object)  # GenerationTemplate or None
    
    # Parameter constraints
    MAP_SIZE_MIN = 10
    MAP_SIZE_MAX = 100
    ROOM_COUNT_MIN = 5
    ROOM_COUNT_MAX = 50
    COMPLEXITY_MIN = 3
    COMPLEXITY_MAX = 10
    CORRIDOR_WIDTH_MIN = 64
    CORRIDOR_WIDTH_MAX = 128
    CORRIDOR_WIDTH_STEP = 16
    
    def __init__(self, parent=None):
        """Initialize the parameter panel."""
        super().__init__(parent)
        
        # Settings for persistent parameter values
        self.settings = QSettings("idTechMapGenerator", "Parameters")
        
        # Parameter storage
        self.parameters = {}

        # Current template (None = custom settings)
        self._current_template: Optional[GenerationTemplate] = None

        # Initialize UI
        self._init_ui()
        
        # Load saved parameters
        self._load_parameters()
        
    def _init_ui(self):
        """Initialize the user interface."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        
        # Create scroll area for parameters
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        main_layout.addWidget(scroll_area)
        
        # Container widget for scroll area
        container = QWidget()
        scroll_area.setWidget(container)
        
        # Container layout
        layout = QVBoxLayout(container)
        layout.setSpacing(12)

        # Templates Group (above dimensions for quick access)
        templates_group = self._create_templates_group()
        layout.addWidget(templates_group)

        # Map Dimensions Group
        dimensions_group = self._create_dimensions_group()
        layout.addWidget(dimensions_group)
        
        # Generation Settings Group
        generation_group = self._create_generation_group()
        layout.addWidget(generation_group)
        
        # Advanced Settings Group
        advanced_group = self._create_advanced_group()
        layout.addWidget(advanced_group)
        
        # Presets Group
        presets_group = self._create_presets_group()
        layout.addWidget(presets_group)
        
        # Add stretch to push everything to the top
        layout.addStretch()
        
        # Apply styling
        self.setStyleSheet(f"""
            QGroupBox {{
                font-weight: bold;
                border: 1px solid {sc.BORDER_DARK};
                border-radius: {sc.BORDER_RADIUS_LG};
                margin-top: {sc.SPACING_SM}px;
                padding-top: {sc.SPACING_SM}px;
                background-color: {sc.BG_MEDIUM};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: {sc.TEXT_PRIMARY};
            }}
            QLabel {{
                color: {sc.TEXT_PRIMARY};
                font-weight: normal;
                font-size: {sc.FONT_SIZE_SM};
            }}
            QSlider::groove:horizontal {{
                height: 6px;
                background: {sc.BORDER_MEDIUM};
                border-radius: {sc.BORDER_RADIUS_SM};
            }}
            QSlider::handle:horizontal {{
                background: {sc.PRIMARY_ACTION};
                width: 14px;
                height: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }}
            QSlider::handle:horizontal:hover {{
                background: #5CBF60;
            }}
            QSpinBox, QComboBox {{
                background-color: {sc.BG_LIGHT};
                border: 1px solid {sc.BORDER_MEDIUM};
                border-radius: {sc.BORDER_RADIUS_MD};
                padding: {sc.SPACING_XS}px;
                color: {sc.TEXT_PRIMARY};
            }}
            QSpinBox:focus, QComboBox:focus {{
                border: 1px solid {sc.FOCUS_COLOR};
            }}
            QPushButton {{
                background-color: {sc.BG_LIGHT};
                border: 1px solid {sc.BORDER_MEDIUM};
                border-radius: {sc.BORDER_RADIUS_MD};
                padding: {sc.SPACING_SM}px {sc.SPACING_MD}px;
                color: {sc.TEXT_PRIMARY};
                font-size: {sc.FONT_SIZE_SM};
            }}
            QPushButton:hover {{
                background-color: {sc.BG_HIGHLIGHT};
            }}
            QPushButton:pressed {{
                background-color: {sc.BG_PRESSED};
            }}
            QPushButton:focus {{
                outline: 2px solid {sc.FOCUS_COLOR};
                outline-offset: {sc.FOCUS_OUTLINE_OFFSET};
            }}
        """)
        
    def _create_dimensions_group(self):
        """Create the map dimensions group."""
        group = QGroupBox("Map Dimensions")
        layout = QVBoxLayout()
        
        # Map Width
        width_layout = QHBoxLayout()
        width_label = QLabel("Width (tiles):")
        width_label.setMinimumWidth(70)
        width_layout.addWidget(width_label)
        
        self.width_slider = QSlider(Qt.Horizontal)
        self.width_slider.setRange(self.MAP_SIZE_MIN, self.MAP_SIZE_MAX)
        self.width_slider.setValue(30)
        self.width_slider.setTickPosition(QSlider.TicksBelow)
        self.width_slider.setTickInterval(10)
        width_layout.addWidget(self.width_slider)
        
        self.width_spinbox = QSpinBox()
        self.width_spinbox.setRange(self.MAP_SIZE_MIN, self.MAP_SIZE_MAX)
        self.width_spinbox.setValue(30)
        self.width_spinbox.setMinimumWidth(60)
        set_accessible(self.width_spinbox, "Map Width",
                      f"Map width in tiles, range {self.MAP_SIZE_MIN} to {self.MAP_SIZE_MAX}")
        width_layout.addWidget(self.width_spinbox)
        
        layout.addLayout(width_layout)
        
        # Map Height
        height_layout = QHBoxLayout()
        height_label = QLabel("Height (tiles):")
        height_label.setMinimumWidth(70)
        height_layout.addWidget(height_label)
        
        self.height_slider = QSlider(Qt.Horizontal)
        self.height_slider.setRange(self.MAP_SIZE_MIN, self.MAP_SIZE_MAX)
        self.height_slider.setValue(30)
        self.height_slider.setTickPosition(QSlider.TicksBelow)
        self.height_slider.setTickInterval(10)
        height_layout.addWidget(self.height_slider)
        
        self.height_spinbox = QSpinBox()
        self.height_spinbox.setRange(self.MAP_SIZE_MIN, self.MAP_SIZE_MAX)
        self.height_spinbox.setValue(30)
        self.height_spinbox.setMinimumWidth(60)
        set_accessible(self.height_spinbox, "Map Height",
                      f"Map height in tiles, range {self.MAP_SIZE_MIN} to {self.MAP_SIZE_MAX}")
        height_layout.addWidget(self.height_spinbox)
        
        layout.addLayout(height_layout)
        
        # Connect signals
        self.width_slider.valueChanged.connect(self.width_spinbox.setValue)
        self.width_spinbox.valueChanged.connect(self.width_slider.setValue)
        self.width_slider.valueChanged.connect(self._on_parameter_changed)
        
        self.height_slider.valueChanged.connect(self.height_spinbox.setValue)
        self.height_spinbox.valueChanged.connect(self.height_slider.setValue)
        self.height_slider.valueChanged.connect(self._on_parameter_changed)
        
        group.setLayout(layout)
        return group
        
    def _create_generation_group(self):
        """Create the generation settings group."""
        group = QGroupBox("Generation Settings")
        layout = QVBoxLayout()
        
        # Room Count
        room_layout = QHBoxLayout()
        room_label = QLabel("Room Count:")
        room_label.setMinimumWidth(70)
        room_layout.addWidget(room_label)
        
        self.room_slider = QSlider(Qt.Horizontal)
        self.room_slider.setRange(self.ROOM_COUNT_MIN, self.ROOM_COUNT_MAX)
        self.room_slider.setValue(15)
        self.room_slider.setTickPosition(QSlider.TicksBelow)
        self.room_slider.setTickInterval(5)
        room_layout.addWidget(self.room_slider)
        
        self.room_spinbox = QSpinBox()
        self.room_spinbox.setRange(self.ROOM_COUNT_MIN, self.ROOM_COUNT_MAX)
        self.room_spinbox.setValue(15)
        self.room_spinbox.setMinimumWidth(60)
        set_accessible(self.room_spinbox, "Room Count",
                      f"Number of rooms to generate, range {self.ROOM_COUNT_MIN} to {self.ROOM_COUNT_MAX}")
        room_layout.addWidget(self.room_spinbox)
        
        layout.addLayout(room_layout)
        
        # Complexity
        complexity_layout = QHBoxLayout()
        complexity_label = QLabel("Complexity:")
        complexity_label.setMinimumWidth(70)
        complexity_layout.addWidget(complexity_label)
        
        self.complexity_slider = QSlider(Qt.Horizontal)
        self.complexity_slider.setRange(self.COMPLEXITY_MIN, self.COMPLEXITY_MAX)
        self.complexity_slider.setValue(5)
        self.complexity_slider.setTickPosition(QSlider.TicksBelow)
        self.complexity_slider.setTickInterval(1)
        complexity_layout.addWidget(self.complexity_slider)
        
        self.complexity_spinbox = QSpinBox()
        self.complexity_spinbox.setRange(self.COMPLEXITY_MIN, self.COMPLEXITY_MAX)
        self.complexity_spinbox.setValue(5)
        self.complexity_spinbox.setMinimumWidth(60)
        set_accessible(self.complexity_spinbox, "Complexity",
                      f"Map complexity level, range {self.COMPLEXITY_MIN} to {self.COMPLEXITY_MAX}")
        complexity_layout.addWidget(self.complexity_spinbox)
        
        layout.addLayout(complexity_layout)

        # Floor Count (for multi-floor generation)
        floor_layout = QHBoxLayout()
        floor_label = QLabel("Floors:")
        floor_label.setMinimumWidth(70)
        floor_layout.addWidget(floor_label)

        self.floor_spinbox = QSpinBox()
        self.floor_spinbox.setRange(1, 4)
        self.floor_spinbox.setValue(1)
        self.floor_spinbox.setMinimumWidth(60)
        self.floor_spinbox.setToolTip("Number of dungeon floors (1=single floor, 2-4=multi-floor)")
        set_accessible(self.floor_spinbox, "Floor Count",
                      "Number of dungeon floors to generate (1 to 4)")
        floor_layout.addWidget(self.floor_spinbox)

        # Floor level indicator
        self.floor_level_label = QLabel("(Ground only)")
        self.floor_level_label.setStyleSheet("color: #888; font-size: 10pt;")
        floor_layout.addWidget(self.floor_level_label)
        floor_layout.addStretch()

        layout.addLayout(floor_layout)

        # Auto-connect floors checkbox
        self.auto_connect_floors_check = SafeCheckBox("Auto-connect floors with stairs")
        self.auto_connect_floors_check.setChecked(True)
        self.auto_connect_floors_check.setToolTip(
            "Automatically place VerticalStairHalls between floors.\n"
            "Uncheck to manually place stairs after generation."
        )
        set_accessible(self.auto_connect_floors_check, "Auto-connect floors",
                      "Automatically place staircases between floors")
        self.auto_connect_floors_check.setEnabled(False)  # Only enabled when floors > 1
        layout.addWidget(self.auto_connect_floors_check)

        # Connect floor count change to update label and checkbox
        self.floor_spinbox.valueChanged.connect(self._on_floor_count_changed)

        # Corridor Width
        corridor_layout = QHBoxLayout()
        corridor_label = QLabel("Corridor Width:")
        corridor_label.setMinimumWidth(70)
        corridor_layout.addWidget(corridor_label)
        
        self.corridor_slider = QSlider(Qt.Horizontal)
        self.corridor_slider.setRange(
            self.CORRIDOR_WIDTH_MIN // self.CORRIDOR_WIDTH_STEP,
            self.CORRIDOR_WIDTH_MAX // self.CORRIDOR_WIDTH_STEP
        )
        self.corridor_slider.setValue(96 // self.CORRIDOR_WIDTH_STEP)
        self.corridor_slider.setTickPosition(QSlider.TicksBelow)
        self.corridor_slider.setTickInterval(1)
        corridor_layout.addWidget(self.corridor_slider)
        
        self.corridor_spinbox = QSpinBox()
        self.corridor_spinbox.setRange(self.CORRIDOR_WIDTH_MIN, self.CORRIDOR_WIDTH_MAX)
        self.corridor_spinbox.setSingleStep(self.CORRIDOR_WIDTH_STEP)
        self.corridor_spinbox.setValue(96)
        self.corridor_spinbox.setSuffix(" units")
        self.corridor_spinbox.setMinimumWidth(80)
        set_accessible(self.corridor_spinbox, "Corridor Width",
                      f"Corridor width in units, range {self.CORRIDOR_WIDTH_MIN} to {self.CORRIDOR_WIDTH_MAX}")
        corridor_layout.addWidget(self.corridor_spinbox)
        
        layout.addLayout(corridor_layout)

        # Secret Room Frequency
        secret_layout = QHBoxLayout()
        secret_label = QLabel("Secret Rooms:")
        secret_label.setMinimumWidth(70)
        secret_layout.addWidget(secret_label)

        self.secret_slider = QSlider(Qt.Horizontal)
        self.secret_slider.setRange(0, 100)
        self.secret_slider.setValue(0)  # Default: no secret rooms
        self.secret_slider.setTickPosition(QSlider.TicksBelow)
        self.secret_slider.setTickInterval(10)
        secret_layout.addWidget(self.secret_slider)

        self.secret_spinbox = QSpinBox()
        self.secret_spinbox.setRange(0, 100)
        self.secret_spinbox.setValue(0)
        self.secret_spinbox.setSuffix("%")
        self.secret_spinbox.setMinimumWidth(60)
        set_accessible(self.secret_spinbox, "Secret Room Frequency",
                      "Percentage of rooms that will be SecretChambers (0-100%)")
        secret_layout.addWidget(self.secret_spinbox)

        layout.addLayout(secret_layout)

        # Connect signals
        self.room_slider.valueChanged.connect(self.room_spinbox.setValue)
        self.room_spinbox.valueChanged.connect(self.room_slider.setValue)
        self.room_slider.valueChanged.connect(self._on_parameter_changed)
        
        self.complexity_slider.valueChanged.connect(self.complexity_spinbox.setValue)
        self.complexity_spinbox.valueChanged.connect(self.complexity_slider.setValue)
        self.complexity_slider.valueChanged.connect(self._on_parameter_changed)
        
        self.corridor_slider.valueChanged.connect(
            lambda v: self.corridor_spinbox.setValue(v * self.CORRIDOR_WIDTH_STEP)
        )
        self.corridor_spinbox.valueChanged.connect(
            lambda v: self.corridor_slider.setValue(v // self.CORRIDOR_WIDTH_STEP)
        )
        self.corridor_slider.valueChanged.connect(self._on_parameter_changed)

        self.secret_slider.valueChanged.connect(self.secret_spinbox.setValue)
        self.secret_spinbox.valueChanged.connect(self.secret_slider.setValue)
        self.secret_slider.valueChanged.connect(self._on_parameter_changed)

        group.setLayout(layout)
        return group

    def _create_advanced_group(self):
        """Create the advanced settings group."""
        group = QGroupBox("Advanced Settings")
        layout = QVBoxLayout()

        # Export options
        self.export_obj_check = SafeCheckBox("Export OBJ (mesh preview)")
        self.export_obj_check.setChecked(False)
        set_accessible(self.export_obj_check, "Export OBJ",
                      "Also export OBJ mesh format for preview")
        layout.addWidget(self.export_obj_check)

        self.export_graph_check = SafeCheckBox("Export debug graph")
        self.export_graph_check.setChecked(False)
        set_accessible(self.export_graph_check, "Export Debug Graph",
                      "Export debug graph visualization")
        layout.addWidget(self.export_graph_check)

        # Seed control
        seed_layout = QHBoxLayout()
        seed_label = QLabel("Seed:")
        seed_label.setMinimumWidth(70)
        seed_layout.addWidget(seed_label)

        self.seed_spinbox = QSpinBox()
        self.seed_spinbox.setRange(0, 2147483647)
        self.seed_spinbox.setSpecialValueText("Random")
        self.seed_spinbox.setValue(0)  # 0 = random
        self.seed_spinbox.setMinimumWidth(120)
        self.seed_spinbox.setToolTip("Set to 0 for random seed, or enter a value for reproducible generation")
        set_accessible(self.seed_spinbox, "Random Seed",
                      "Set to 0 for random seed, or enter a value for reproducible generation")
        seed_layout.addWidget(self.seed_spinbox)
        seed_layout.addStretch()

        layout.addLayout(seed_layout)

        # Connect signals
        self.export_obj_check.toggled.connect(self._on_parameter_changed)
        self.export_graph_check.toggled.connect(self._on_parameter_changed)
        self.seed_spinbox.valueChanged.connect(self._on_parameter_changed)

        group.setLayout(layout)
        return group

    def get_selected_profile(self) -> Optional[GameProfile]:
        """Get the game profile (returns default Doom 3 profile).

        Returns:
            The default GameProfile (Doom 3 with minimal worldspawn).
        """
        return PROFILE_CATALOG.get_default_profile()
        
    def _create_presets_group(self):
        """Create the presets group."""
        group = QGroupBox("Presets")
        layout = QVBoxLayout()
        
        # Preset buttons
        preset_layout = QHBoxLayout()
        
        small_btn = QPushButton("Small")
        small_btn.clicked.connect(lambda: self._apply_preset("small"))
        preset_layout.addWidget(small_btn)
        
        medium_btn = QPushButton("Medium")
        medium_btn.clicked.connect(lambda: self._apply_preset("medium"))
        preset_layout.addWidget(medium_btn)
        
        large_btn = QPushButton("Large")
        large_btn.clicked.connect(lambda: self._apply_preset("large"))
        preset_layout.addWidget(large_btn)
        
        layout.addLayout(preset_layout)
        
        # Reset button
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self.reset_parameters)
        layout.addWidget(reset_btn)
        
        group.setLayout(layout)
        return group

    def _create_templates_group(self):
        """Create the generation templates group."""
        group = QGroupBox("Generation Templates")
        layout = QVBoxLayout()

        # Template selector dropdown
        template_layout = QHBoxLayout()
        template_label = QLabel("Template:")
        template_label.setMinimumWidth(70)
        template_layout.addWidget(template_label)

        self.template_combo = QComboBox()
        self.template_combo.addItem("Custom", None)  # First item for custom settings
        for template_name in TEMPLATE_CATALOG.list_templates():
            template = TEMPLATE_CATALOG.get_template(template_name)
            if template:
                self.template_combo.addItem(template_name, template)
        set_accessible(self.template_combo, "Generation Template",
                      "Select a preset template for dungeon generation style")
        template_layout.addWidget(self.template_combo)
        template_layout.addStretch()

        layout.addLayout(template_layout)

        # Template description label
        self.template_description = QLabel("Customize parameters below")
        self.template_description.setWordWrap(True)
        self.template_description.setStyleSheet(f"""
            QLabel {{
                color: {sc.TEXT_SECONDARY};
                font-style: italic;
                padding: 4px;
            }}
        """)
        layout.addWidget(self.template_description)

        # Connect signal
        self.template_combo.currentIndexChanged.connect(self._on_template_changed)

        group.setLayout(layout)
        return group

    def _on_template_changed(self, index: int):
        """Handle template selection change."""
        template = self.template_combo.currentData()
        self._current_template = template

        if template is None:
            self.template_description.setText("Customize parameters below")
        else:
            self.template_description.setText(template.description)
            # Apply template parameters to UI
            self.set_parameters(template.to_layout_params())

        # Emit signal for parent to handle
        self.template_selected.emit(template)

    def get_current_template(self) -> Optional[GenerationTemplate]:
        """Get the currently selected template, or None if custom."""
        return self._current_template

    def get_generation_hints(self) -> Dict[str, Any]:
        """Get generation hints from the current template, or defaults if custom."""
        if self._current_template:
            return self._current_template.get_generation_hints()
        # Default hints for custom mode
        return {
            "preferred_room_types": None,
            "preferred_hall_types": None,
            "room_probability": 0.4,
            "min_hall_between_rooms": 1,
            "allow_dead_ends": True,
        }

    def _apply_preset(self, preset_name: str):
        """Apply a preset configuration."""
        presets = {
            "small": {
                "width": 20,
                "height": 20,
                "rooms": 8,
                "complexity": 3,
                "corridor_width": 64
            },
            "medium": {
                "width": 40,
                "height": 40,
                "rooms": 20,
                "complexity": 5,
                "corridor_width": 96
            },
            "large": {
                "width": 80,
                "height": 80,
                "rooms": 40,
                "complexity": 8,
                "corridor_width": 128
            }
        }
        
        if preset_name in presets:
            preset = presets[preset_name]
            self.width_slider.setValue(preset["width"])
            self.height_slider.setValue(preset["height"])
            self.room_slider.setValue(preset["rooms"])
            self.complexity_slider.setValue(preset["complexity"])
            self.corridor_spinbox.setValue(preset["corridor_width"])
            
    def reset_parameters(self):
        """Reset all parameters to default values."""
        self.width_slider.setValue(30)
        self.height_slider.setValue(30)
        self.room_slider.setValue(15)
        self.complexity_slider.setValue(5)
        self.floor_spinbox.setValue(1)
        self.auto_connect_floors_check.setChecked(True)
        self.corridor_spinbox.setValue(96)
        self.secret_spinbox.setValue(0)
        self.export_obj_check.setChecked(False)
        self.export_graph_check.setChecked(False)
        self.seed_spinbox.setValue(0)
        
    def get_parameters(self) -> Dict[str, Any]:
        """Get the current parameter values."""
        # Seed: 0 means random (None), otherwise use the value
        seed_value = self.seed_spinbox.value()
        seed = None if seed_value == 0 else seed_value

        return {
            "map_width": self.width_slider.value(),
            "map_height": self.height_slider.value(),
            "room_count": self.room_slider.value(),
            "complexity": self.complexity_slider.value(),
            "floor_count": self.floor_spinbox.value(),
            "auto_connect_floors": self.auto_connect_floors_check.isChecked(),
            "corridor_width": self.corridor_spinbox.value(),
            "secret_room_frequency": self.secret_spinbox.value(),
            "export_obj": self.export_obj_check.isChecked(),
            "export_graph": self.export_graph_check.isChecked(),
            "seed": seed
        }
        
    def set_parameters(self, parameters: Dict[str, Any]):
        """Set parameter values from a dictionary."""
        if "map_width" in parameters:
            self.width_slider.setValue(parameters["map_width"])
        if "map_height" in parameters:
            self.height_slider.setValue(parameters["map_height"])
        if "room_count" in parameters:
            self.room_slider.setValue(parameters["room_count"])
        if "complexity" in parameters:
            self.complexity_slider.setValue(parameters["complexity"])
        if "floor_count" in parameters:
            self.floor_spinbox.setValue(parameters["floor_count"])
        if "auto_connect_floors" in parameters:
            self.auto_connect_floors_check.setChecked(parameters["auto_connect_floors"])
        if "corridor_width" in parameters:
            self.corridor_spinbox.setValue(parameters["corridor_width"])
        if "secret_room_frequency" in parameters:
            self.secret_spinbox.setValue(parameters["secret_room_frequency"])
        if "export_obj" in parameters:
            self.export_obj_check.setChecked(parameters["export_obj"])
        if "export_graph" in parameters:
            self.export_graph_check.setChecked(parameters["export_graph"])
        if "seed" in parameters:
            # Convert None to 0 for the spinbox
            seed_val = parameters["seed"]
            self.seed_spinbox.setValue(0 if seed_val is None else seed_val)

    def validate_parameters(self) -> bool:
        """Validate the current parameters."""
        params = self.get_parameters()
        
        # Check map dimensions
        if params["map_width"] < self.MAP_SIZE_MIN or params["map_width"] > self.MAP_SIZE_MAX:
            return False
        if params["map_height"] < self.MAP_SIZE_MIN or params["map_height"] > self.MAP_SIZE_MAX:
            return False
            
        # Check room count doesn't exceed map capacity
        max_possible_rooms = (params["map_width"] * params["map_height"]) // 100  # Rough estimate
        if params["room_count"] > max_possible_rooms:
            return False
            
        # Check corridor width
        if params["corridor_width"] < self.CORRIDOR_WIDTH_MIN or params["corridor_width"] > self.CORRIDOR_WIDTH_MAX:
            return False
            
        return True

    def _on_floor_count_changed(self, floor_count: int):
        """Update UI based on floor count selection."""
        # Update floor level label
        floor_names = {
            1: "(Ground only)",
            2: "(Basement + Ground)",
            3: "(Basement + Ground + Upper)",
            4: "(Basement + Ground + Upper + Tower)",
        }
        self.floor_level_label.setText(floor_names.get(floor_count, ""))

        # Enable/disable auto-connect checkbox
        self.auto_connect_floors_check.setEnabled(floor_count > 1)
        if floor_count == 1:
            self.auto_connect_floors_check.setChecked(False)
        else:
            self.auto_connect_floors_check.setChecked(True)

        # Trigger parameter change
        self._on_parameter_changed()

    def _on_parameter_changed(self):
        """Handle parameter change events."""
        self.parameters = self.get_parameters()
        self.parameters_changed.emit(self.parameters)
        self._save_parameters()
        
    def _save_parameters(self):
        """Save current parameters to settings."""
        params = self.get_parameters()
        for key, value in params.items():
            self.settings.setValue(key, value)
            
    def _load_parameters(self):
        """Load parameters from settings."""
        params = {}
        for key in ["map_width", "map_height", "room_count", "complexity",
                   "floor_count", "auto_connect_floors",
                   "corridor_width", "secret_room_frequency",
                   "export_obj", "export_graph", "seed"]:
            value = self.settings.value(key)
            if value is not None:
                # Convert string boolean values
                if key.startswith("export_"):
                    params[key] = value.lower() == "true" if isinstance(value, str) else bool(value)
                elif key == "seed":
                    # Handle seed: convert to int or None
                    if value is None or value == "" or value == "None":
                        params[key] = None
                    else:
                        try:
                            params[key] = int(value)
                        except (ValueError, TypeError):
                            params[key] = None
                else:
                    params[key] = value
                    
        if params:
            self.set_parameters(params)
