"""
Main application window for the Dark Fantasy Geometry Toolkit.

Two modes: Layout (BSP dungeon generation) and Primitive (individual geometry).
Exports .map files in idTech 1 or idTech 4 format.

Features real-time 3D preview with orbit camera controls.
"""

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QPushButton,
    QTextEdit, QProgressBar, QMessageBox, QLineEdit,
    QSplitter, QStackedWidget, QScrollArea,
    QShortcut, QTabWidget, QMenu, QAction, QMenuBar,
    QFileDialog, QApplication,
)
from quake_levelgenerator.src.ui.safe_combobox import SafeComboBox

# Alias for compatibility - click to cycle, no dropdown
QComboBox = SafeComboBox
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings, QTimer
from PyQt5.QtGui import QKeySequence
from typing import Optional, List
import json
import logging
import os
import traceback
from pathlib import Path

from quake_levelgenerator.src.pipeline.automated_pipeline import (
    AutomatedPipeline, PipelineSettings, PipelineMode,
    PipelineProgress, PipelineResult,
)
from quake_levelgenerator.src.ui.widgets.mode_selector import ModeSelector
from quake_levelgenerator.src.ui.widgets.primitive_panel import PrimitivePanel
from quake_levelgenerator.src.ui.widgets.layout_panel import LayoutPanel
from quake_levelgenerator.src.ui.preview import PreviewWidget, QuadViewWidget
from quake_levelgenerator.src.ui.widgets.layout_editor import LayoutEditorWidget
from quake_levelgenerator.src.ui.widgets.layout_editor.layout_generator import generate_from_layout
from quake_levelgenerator.src.ui.widgets.layout_editor.validation_panel import ValidationPanel
from quake_levelgenerator.src.ui.widgets.layout_editor.random_layout import (
    generate_random_layout, generate_multi_floor_layout
)
from quake_levelgenerator.src.conversion.map_writer import MapWriter
from quake_levelgenerator.src.ui import style_constants as sc
from quake_levelgenerator.src.generators.textures import TEXTURE_SETTINGS
from quake_levelgenerator.src.ui.style_constants import set_accessible
from quake_levelgenerator.src.ui.safe_menu_action import SafeMenuAction


# ---------------------------------------------------------------------------
# Generation thread
# ---------------------------------------------------------------------------

class GenerationThread(QThread):
    progress_updated = pyqtSignal(object)
    generation_complete = pyqtSignal(object)
    generation_failed = pyqtSignal(str)

    def __init__(self, settings: PipelineSettings):
        super().__init__()
        self._settings = settings
        self.pipeline: Optional[AutomatedPipeline] = None
        self.is_cancelled = False

    def run(self):
        try:
            self.pipeline = AutomatedPipeline(self._settings)
            self.pipeline.set_progress_callback(
                lambda p: (not self.is_cancelled) and self.progress_updated.emit(p)
            )
            result = self.pipeline.generate()
            if result.success:
                self.generation_complete.emit(result)
            else:
                self.generation_failed.emit("\n".join(result.errors))
        except Exception as e:
            full_trace = traceback.format_exc()
            logger = logging.getLogger(__name__)
            logger.error("Generation failed:\n%s", full_trace)
            # Emit both the user-friendly summary and full trace separated by marker
            self.generation_failed.emit(f"{e}\n---TRACEBACK---\n{full_trace}")

    def cancel_generation(self):
        self.is_cancelled = True
        if self.pipeline:
            self.pipeline.cancel()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    # Settings keys
    SETTINGS_ORG = "idTechGeometryToolkit"
    SETTINGS_APP = "MainWindow"
    MAX_RECENT_FILES = 5

    def __init__(self):
        super().__init__()
        self.generation_thread: Optional[GenerationThread] = None
        self.current_result: Optional[PipelineResult] = None
        self._generated_brushes = []  # Store generated brushes for export
        self._spawn_position = (0, 0, 56)  # Default spawn position

        # Initialize settings
        self._settings = QSettings(self.SETTINGS_ORG, self.SETTINGS_APP)
        self._recent_files: List[str] = []
        self._recent_files_menu: Optional[QMenu] = None
        self._pending_quad_view_sync = False
        self._load_settings()

        self._setup_ui()
        self._setup_menu_bar()
        self._connect_signals()

        # Restore window geometry and splitter sizes
        self._restore_geometry()

        # Auto-save timer (every 5 minutes when dirty)
        self._autosave_timer = QTimer(self)
        self._autosave_timer.timeout.connect(self._auto_save_layout)
        self._autosave_timer.start(5 * 60 * 1000)  # 5 minutes

        # Check for crash recovery on startup
        self._check_autosave_recovery()

        # Start in primitive mode with preview
        self.mode_selector.set_mode("primitive")
        self._on_mode_changed("primitive")
        # NOTE: Do NOT trigger preview here - OpenGL context isn't ready yet.
        # Preview will update automatically on first parameter change.

        # Now that mode is initialized, sync quad view if it was restored from settings
        if hasattr(self, '_pending_quad_view_sync') and self._pending_quad_view_sync:
            self._pending_quad_view_sync = False
            # Set the center stack to quad view and sync appropriate content
            self._center_stack.setCurrentIndex(2)
            self._sync_mesh_to_quad_view()

        # Show first-run onboarding if needed
        self._check_first_run()

    # ---------------------------------------------------------------
    # UI setup
    # ---------------------------------------------------------------

    def _setup_ui(self):
        self.setWindowTitle("idTech Geometry Toolkit")
        self.setMinimumSize(1100, 650)  # Ensure menu bar visibility
        self.resize(1400, 850)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)

        # Main horizontal splitter (3 panels)
        self.main_splitter = QSplitter(Qt.Horizontal)
        root.addWidget(self.main_splitter)

        # === LEFT PANEL (Controls) ===
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)

        # Mode selector
        self.mode_selector = ModeSelector()
        left_layout.addWidget(self.mode_selector)

        # Stacked panels
        self.panel_stack = QStackedWidget()

        # Index 0: layout panel (BSP generation parameters)
        self.layout_panel = LayoutPanel()
        scroll_layout = QScrollArea()
        scroll_layout.setWidgetResizable(True)
        scroll_layout.setWidget(self.layout_panel)
        self.panel_stack.addWidget(scroll_layout)

        # Index 1: primitive panel
        self.primitive_panel = PrimitivePanel()
        scroll_prim = QScrollArea()
        scroll_prim.setWidgetResizable(True)
        scroll_prim.setWidget(self.primitive_panel)
        self.panel_stack.addWidget(scroll_prim)

        # Panel stack gets all the stretch - expands to fill available space
        left_layout.addWidget(self.panel_stack, stretch=1)

        # Compact generation controls (fixed at bottom)
        gen_group = QGroupBox("Generate")
        gen_layout = QVBoxLayout(gen_group)
        gen_layout.setContentsMargins(6, 8, 6, 6)
        gen_layout.setSpacing(4)

        # Random Dungeon button (Layout mode only)
        self.random_gen_btn = QPushButton("Random Dungeon")
        self.random_gen_btn.setToolTip("Generate random BSP dungeon (Ctrl+R)")
        set_accessible(self.random_gen_btn, "Random Dungeon",
                      "Generate a random BSP dungeon layout")
        self.random_gen_btn.setStyleSheet(f"""
            QPushButton {{ font-weight:bold; padding:4px 8px;
                          background:{sc.SPECIAL_ACTION}; color:white; border:none; border-radius:{sc.BORDER_RADIUS_MD}; }}
            QPushButton:hover {{ background:{sc.SPECIAL_ACTION_HOVER}; }}
            QPushButton:disabled {{ background:{sc.TEXT_SECONDARY}; color:{sc.BORDER_LIGHT}; }}
        """)
        gen_layout.addWidget(self.random_gen_btn)

        # Build + Cancel row
        gen_btn_row = QHBoxLayout()
        gen_btn_row.setSpacing(4)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setToolTip("Cancel generation (Escape)")
        set_accessible(self.cancel_btn, "Cancel", "Cancel the current generation")
        self.cancel_btn.setStyleSheet(f"""
            QPushButton {{ font-weight:bold; padding:4px 8px; background:{sc.DANGER_COLOR};
                          color:white; border:none; border-radius:{sc.BORDER_RADIUS_MD}; }}
            QPushButton:hover {{ background:{sc.DANGER_HOVER}; }}
            QPushButton:disabled {{ background:{sc.BORDER_MEDIUM}; color:{sc.TEXT_DISABLED}; }}
        """)

        self.generate_btn = QPushButton("Build")
        self.generate_btn.setToolTip("Build geometry (Ctrl+G)")
        set_accessible(self.generate_btn, "Build Geometry",
                      "Generate brushes from current layout or module")
        self.generate_btn.setStyleSheet(f"""
            QPushButton {{ font-weight:bold; padding:4px 12px;
                          background:{sc.PRIMARY_ACTION}; color:white; border:none; border-radius:{sc.BORDER_RADIUS_MD}; }}
            QPushButton:hover {{ background:{sc.PRIMARY_ACTION_HOVER}; }}
            QPushButton:disabled {{ background:{sc.BORDER_MEDIUM}; color:{sc.TEXT_DISABLED}; }}
        """)

        gen_btn_row.addWidget(self.cancel_btn)
        gen_btn_row.addWidget(self.generate_btn)
        gen_layout.addLayout(gen_btn_row)

        # Compact progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumHeight(16)
        set_accessible(self.progress_bar, "Progress", "Generation progress")
        gen_layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("Ready")
        self.progress_label.setStyleSheet(f"font-size: {sc.FONT_SIZE_SM}; color: {sc.TEXT_SECONDARY};")
        set_accessible(self.progress_label, "Status", "Generation status")
        gen_layout.addWidget(self.progress_label)

        left_layout.addWidget(gen_group)

        # Compact export controls
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)
        export_layout.setContentsMargins(6, 8, 6, 6)
        export_layout.setSpacing(4)

        # Filename + Format on one row each
        name_row = QHBoxLayout()
        name_row.setSpacing(4)
        name_label = QLabel("File:")
        name_label.setFixedWidth(40)
        name_row.addWidget(name_label)
        self.map_name_edit = QLineEdit("generated_map")
        self.map_name_edit.setToolTip("Output filename")
        set_accessible(self.map_name_edit, "Filename", "Output filename")
        name_label.setBuddy(self.map_name_edit)
        name_row.addWidget(self.map_name_edit)
        export_layout.addLayout(name_row)

        # Export buttons row
        export_btn_row = QHBoxLayout()
        export_btn_row.setSpacing(4)

        self.export_map_btn = QPushButton(".map")
        self.export_map_btn.setEnabled(False)
        self.export_map_btn.setToolTip("Generate a map first (click Build above)")
        set_accessible(self.export_map_btn, "Export MAP", "Export to MAP format")
        self.export_map_btn.setStyleSheet(f"""
            QPushButton {{ font-weight:bold; padding:4px 8px;
                          background:{sc.SECONDARY_ACTION}; color:white; border:none; border-radius:{sc.BORDER_RADIUS_MD}; }}
            QPushButton:hover {{ background:{sc.SECONDARY_ACTION_HOVER}; }}
            QPushButton:disabled {{ background:{sc.TEXT_SECONDARY}; color:{sc.BORDER_LIGHT}; }}
        """)
        export_btn_row.addWidget(self.export_map_btn)

        self.export_obj_btn = QPushButton(".obj")
        self.export_obj_btn.setEnabled(False)
        self.export_obj_btn.setToolTip("Generate a map first (click Build above)")
        set_accessible(self.export_obj_btn, "Export OBJ", "Export to OBJ format")
        self.export_obj_btn.setStyleSheet(f"""
            QPushButton {{ font-weight:bold; padding:4px 8px;
                          background:{sc.WARNING_COLOR}; color:white; border:none; border-radius:{sc.BORDER_RADIUS_MD}; }}
            QPushButton:hover {{ background:{sc.WARNING_HOVER}; }}
            QPushButton:disabled {{ background:{sc.TEXT_SECONDARY}; color:{sc.BORDER_LIGHT}; }}
        """)
        export_btn_row.addWidget(self.export_obj_btn)

        self.open_output_btn = QPushButton("Open")
        self.open_output_btn.setEnabled(False)
        self.open_output_btn.setToolTip("Export a map first, then open its folder")
        set_accessible(self.open_output_btn, "Open Folder", "Open output folder")
        export_btn_row.addWidget(self.open_output_btn)

        export_layout.addLayout(export_btn_row)
        left_layout.addWidget(export_group)

        self.main_splitter.addWidget(left)

        # === CENTER PANEL (Layout Editor or Preview) ===
        self._center_stack = QStackedWidget()

        # Index 0: Layout editor (for Layout mode)
        self.layout_editor = LayoutEditorWidget()
        self._center_stack.addWidget(self.layout_editor)

        # Index 1: Preview widget (for Primitive mode - single view)
        try:
            self.preview_widget = PreviewWidget()
            self._opengl_available = True
        except Exception as e:
            logging.getLogger(__name__).warning("OpenGL preview unavailable: %s", e)
            self.preview_widget = QWidget()
            fallback = QLabel("3D Preview unavailable\n(OpenGL not supported on this system)")
            fallback.setAlignment(Qt.AlignCenter)
            fallback.setStyleSheet("color: #888; font-size: 14pt;")
            fl = QVBoxLayout(self.preview_widget)
            fl.addWidget(fallback)
            self._opengl_available = False
        self._center_stack.addWidget(self.preview_widget)

        # Index 2: Quad-view widget (for Primitive mode - quad view)
        try:
            self._quad_view_widget = QuadViewWidget()
        except Exception as e:
            logging.getLogger(__name__).warning("Quad view unavailable: %s", e)
            self._quad_view_widget = QWidget()
            self._opengl_available = False
        self._center_stack.addWidget(self._quad_view_widget)

        # Track quad-view state
        self._quad_view_enabled = False

        # Start in primitive mode (index 1 = single preview)
        self._center_stack.setCurrentIndex(1)

        self.main_splitter.addWidget(self._center_stack)

        # === RIGHT PANEL (Log + Properties) ===
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)

        # Tab widget for log and properties
        self.right_tabs = QTabWidget()

        # Quick help button
        help_btn = QPushButton("?")
        help_btn.setFixedSize(24, 24)
        help_btn.setToolTip("Show help and keyboard shortcuts (F1)")
        set_accessible(help_btn, "Help", "Show keyboard shortcuts and tips")
        help_btn.setStyleSheet(f"""
            QPushButton {{
                font-weight: bold;
                font-size: {sc.FONT_SIZE_LG};
                background: {sc.PRIMARY_ACTION};
                color: white;
                border: none;
                border-radius: {sc.BORDER_RADIUS_XL};
            }}
            QPushButton:hover {{
                background: {sc.PRIMARY_ACTION_HOVER};
            }}
            QPushButton:focus {{
                outline: 2px solid {sc.PRIMARY_ACTION};
                outline-offset: {sc.FOCUS_OUTLINE_OFFSET};
            }}
        """)
        help_btn.clicked.connect(self._show_help)
        self.right_tabs.setCornerWidget(help_btn, Qt.TopRightCorner)

        # Log tab
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        log_layout.setContentsMargins(4, 4, 4, 4)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setPlaceholderText("Generation output will appear here...")
        set_accessible(self.log_text, "Generation Log",
                      "Output log showing generation progress and results")
        self.log_text.setStyleSheet(f"""
            QTextEdit {{
                font-family: Menlo, Monaco, 'Courier New', monospace;
                font-size: {sc.FONT_SIZE_SM};
                background-color: {sc.BG_DARKEST};
                color: {sc.TEXT_PRIMARY};
            }}
        """)
        # Copy log button
        log_btn_row = QHBoxLayout()
        log_btn_row.addStretch()
        copy_log_btn = QPushButton("Copy Log")
        copy_log_btn.setToolTip("Copy log contents to clipboard")
        copy_log_btn.setFixedWidth(80)
        copy_log_btn.clicked.connect(self._on_copy_log)
        log_btn_row.addWidget(copy_log_btn)
        clear_log_btn = QPushButton("Clear")
        clear_log_btn.setToolTip("Clear log contents")
        clear_log_btn.setFixedWidth(60)
        clear_log_btn.clicked.connect(self.log_text.clear)
        log_btn_row.addWidget(clear_log_btn)

        log_layout.addWidget(self.log_text)
        log_layout.addLayout(log_btn_row)
        self.right_tabs.addTab(log_widget, "Log")

        # Help tab with camera controls
        help_widget = QWidget()
        help_layout = QVBoxLayout(help_widget)
        help_layout.setContentsMargins(8, 8, 8, 8)
        help_text = QLabel("""
<h3>Camera Controls (FPS Style)</h3>
<table>
<tr><td><b>Right-drag</b></td><td>Mouselook (look around)</td></tr>
<tr><td><b>W/S</b></td><td>Fly forward/back</td></tr>
<tr><td><b>A/D</b></td><td>Strafe left/right</td></tr>
<tr><td><b>Q/E</b></td><td>Fly down/up</td></tr>
<tr><td><b>Middle-drag</b></td><td>Pan</td></tr>
<tr><td><b>Scroll</b></td><td>Zoom (move forward/back)</td></tr>
</table>

<h3>View Presets</h3>
<table>
<tr><td><b>1</b></td><td>Front</td></tr>
<tr><td><b>2</b></td><td>Back</td></tr>
<tr><td><b>3</b></td><td>Left</td></tr>
<tr><td><b>4</b></td><td>Right</td></tr>
<tr><td><b>5</b></td><td>Top</td></tr>
<tr><td><b>6</b></td><td>Bottom</td></tr>
<tr><td><b>F</b></td><td>Fit to bounds</td></tr>
</table>

<h3>Shortcuts</h3>
<table>
<tr><td><b>Ctrl+G</b></td><td>Build geometry from layout/module</td></tr>
<tr><td><b>Ctrl+R</b></td><td>Generate random dungeon (Layout mode)</td></tr>
<tr><td><b>Ctrl+E</b></td><td>Export .map file</td></tr>
<tr><td><b>Ctrl+O</b></td><td>Open output folder</td></tr>
<tr><td><b>Ctrl+1</b></td><td>Switch to Layout mode</td></tr>
<tr><td><b>Ctrl+2</b></td><td>Switch to Module mode</td></tr>
<tr><td><b>Escape</b></td><td>Cancel generation</td></tr>
</table>

<h3>Quad View (Both Modes)</h3>
<table>
<tr><td><b>Shift+Q</b></td><td>Toggle quad-view</td></tr>
<tr><td><b>Alt+7</b></td><td>Maximize Top (XY) view</td></tr>
<tr><td><b>Alt+1</b></td><td>Maximize Front (XZ) view</td></tr>
<tr><td><b>Alt+3</b></td><td>Maximize Side (YZ) view</td></tr>
<tr><td><b>Alt+0</b></td><td>Maximize 3D view</td></tr>
<tr><td><b>Escape</b></td><td>Restore quad-view</td></tr>
</table>
        """)
        help_text.setWordWrap(True)
        help_text.setStyleSheet(f"font-size: {sc.FONT_SIZE_SM};")
        help_layout.addWidget(help_text)
        help_layout.addStretch()
        self.right_tabs.addTab(help_widget, "Help")

        # Validation tab (for layout mode)
        self._validation_panel = ValidationPanel()
        self._validation_panel.issue_clicked.connect(self._on_validation_issue_clicked)
        self.right_tabs.addTab(self._validation_panel, "Validation")

        # Track tab notification state
        self._tab_has_new_content = {0: False, 2: False}  # Log=0, Validation=2
        self.right_tabs.currentChanged.connect(self._on_right_tab_changed)

        right_layout.addWidget(self.right_tabs)
        self.main_splitter.addWidget(right)

        # Set splitter sizes and make all panels resizable
        # Left panel - responsive sizing
        left.setMinimumWidth(280)  # Allow shrinking at low res
        # No maximum - let user expand as needed

        # Right panel for log - responsive sizing
        right.setMinimumWidth(220)  # Allow shrinking at low res
        # No maximum - let user expand as needed

        # Set initial sizes - give left panel more room (380), decent log area (300)
        self.main_splitter.setSizes([380, 720, 300])
        self.main_splitter.setStretchFactor(0, 0)  # Left: prefer fixed
        self.main_splitter.setStretchFactor(1, 1)  # Center: stretches
        self.main_splitter.setStretchFactor(2, 0)  # Right: prefer fixed

        # Handle style for splitter handles - make them clearly visible and draggable
        self.main_splitter.setHandleWidth(8)
        self.main_splitter.setStyleSheet("""
            QSplitter::handle {
                background: #555;
                border-radius: 2px;
            }
            QSplitter::handle:hover {
                background: #777;
            }
            QSplitter::handle:pressed {
                background: #999;
            }
        """)

        # Style status bar for better visibility
        self.statusBar().setStyleSheet(f"QStatusBar {{ font-size: {sc.FONT_SIZE_SM}; color: {sc.TEXT_SECONDARY}; }}")
        self.statusBar().showMessage("Ready - Right-drag to look, WASD to fly, scroll to zoom")

    def _setup_menu_bar(self):
        """Create the application menu bar with File menu and Recent Files."""
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("&File")

        # New Layout
        new_action = QAction("&New Layout", self)
        new_action.setShortcut("Ctrl+N")
        new_action.setToolTip("Create a new layout")
        new_action.triggered.connect(self._on_new_layout)
        file_menu.addAction(new_action)

        # Open Layout
        open_action = QAction("&Open Layout...", self)
        open_action.setShortcut("Ctrl+Shift+O")
        open_action.setToolTip("Open an existing layout file")
        open_action.triggered.connect(self._on_open_layout)
        file_menu.addAction(open_action)

        # Save Layout
        save_action = QAction("&Save Layout...", self)
        save_action.setShortcut("Ctrl+S")
        save_action.setToolTip("Save the current layout")
        save_action.triggered.connect(self._on_save_layout)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        # Recent Files submenu
        self._recent_files_menu = file_menu.addMenu("Recent Files")
        self._update_recent_files_menu()

        file_menu.addSeparator()

        # Export .map
        export_map_action = QAction("Export .&map...", self)
        export_map_action.setShortcut("Ctrl+E")
        export_map_action.setToolTip("Export to idTech .map format")
        export_map_action.triggered.connect(self._on_export_map)
        file_menu.addAction(export_map_action)

        # Export .obj
        export_obj_action = QAction("Export .&obj...", self)
        export_obj_action.setToolTip("Export to OBJ mesh format")
        export_obj_action.triggered.connect(self._on_export_obj)
        file_menu.addAction(export_obj_action)

        file_menu.addSeparator()

        # Quit
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.setToolTip("Quit the application")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Edit menu
        edit_menu = menu_bar.addMenu("&Edit")

        # Undo (Layout mode)
        self._undo_action = QAction("&Undo", self)
        self._undo_action.setShortcut("Ctrl+Z")
        self._undo_action.setToolTip("Undo last action (Ctrl+Z)")
        self._undo_action.setEnabled(False)
        self._undo_action.triggered.connect(self._on_undo)
        edit_menu.addAction(self._undo_action)

        # Redo (Layout mode)
        self._redo_action = QAction("&Redo", self)
        self._redo_action.setShortcut("Ctrl+Shift+Z")
        self._redo_action.setToolTip("Redo last action (Ctrl+Shift+Z)")
        self._redo_action.setEnabled(False)
        self._redo_action.triggered.connect(self._on_redo)
        edit_menu.addAction(self._redo_action)

        edit_menu.addSeparator()

        # Duplicate (Layout mode)
        self._duplicate_action = QAction("&Duplicate", self)
        self._duplicate_action.setShortcut("Ctrl+D")
        self._duplicate_action.setToolTip("Duplicate selected primitive (Ctrl+D)")
        self._duplicate_action.setEnabled(False)
        self._duplicate_action.triggered.connect(self._on_duplicate)
        edit_menu.addAction(self._duplicate_action)

        # Delete (Layout mode)
        self._delete_action = QAction("De&lete", self)
        self._delete_action.setShortcut("Delete")
        self._delete_action.setToolTip("Delete selected primitive (Delete)")
        self._delete_action.setEnabled(False)
        self._delete_action.triggered.connect(self._on_delete)
        edit_menu.addAction(self._delete_action)

        edit_menu.addSeparator()

        # Texture Settings
        texture_settings_action = QAction("&Texture Settings...", self)
        texture_settings_action.setToolTip("Configure placeholder texture names")
        texture_settings_action.triggered.connect(self._on_texture_settings)
        edit_menu.addAction(texture_settings_action)

        # View menu
        view_menu = menu_bar.addMenu("&View")

        # Flow Visualization toggle (Layout mode)
        self._flow_action = SafeMenuAction("&Flow Visualization", self)
        self._flow_action.setShortcut("Ctrl+F")
        self._flow_action.setToolTip("Show flow paths between rooms (Ctrl+F)")
        self._flow_action.setEnabled(False)
        self._flow_action.toggled.connect(self._on_flow_toggled)
        view_menu.addAction(self._flow_action)

        # 3D Preview toggle (Layout mode)
        self._preview_3d_action = SafeMenuAction("&3D Preview", self)
        self._preview_3d_action.setShortcut("Ctrl+3")
        self._preview_3d_action.setToolTip("Toggle 3D preview of layout (Ctrl+3)")
        self._preview_3d_action.setEnabled(False)
        self._preview_3d_action.toggled.connect(self._on_3d_preview_toggled)
        view_menu.addAction(self._preview_3d_action)

        # Floor Filter submenu (Layout mode)
        self._floor_filter_menu = view_menu.addMenu("Floor &Filter")
        self._floor_filter_menu.setToolTip("Highlight primitives on a specific floor level")
        self._floor_filter_menu.setEnabled(False)

        # Create floor filter actions with action group for exclusive selection
        from PyQt5.QtWidgets import QActionGroup
        self._floor_filter_group = QActionGroup(self)
        self._floor_filter_group.setExclusive(True)

        # Show All (default)
        self._floor_all_action = SafeMenuAction("Show &All Floors", self)
        self._floor_all_action.setChecked(True)
        self._floor_all_action.triggered.connect(lambda: self._on_floor_filter_changed(None))
        self._floor_filter_group.addAction(self._floor_all_action)
        self._floor_filter_menu.addAction(self._floor_all_action)

        self._floor_filter_menu.addSeparator()

        # Standard floor levels
        self._floor_basement_action = SafeMenuAction("&Basement (Z:-128)", self)
        self._floor_basement_action.triggered.connect(lambda: self._on_floor_filter_changed(-128.0))
        self._floor_filter_group.addAction(self._floor_basement_action)
        self._floor_filter_menu.addAction(self._floor_basement_action)

        self._floor_ground_action = SafeMenuAction("&Ground (Z:0)", self)
        self._floor_ground_action.triggered.connect(lambda: self._on_floor_filter_changed(0.0))
        self._floor_filter_group.addAction(self._floor_ground_action)
        self._floor_filter_menu.addAction(self._floor_ground_action)

        self._floor_upper_action = SafeMenuAction("&Upper (Z:160)", self)
        self._floor_upper_action.triggered.connect(lambda: self._on_floor_filter_changed(160.0))
        self._floor_filter_group.addAction(self._floor_upper_action)
        self._floor_filter_menu.addAction(self._floor_upper_action)

        self._floor_tower_action = SafeMenuAction("&Tower (Z:320)", self)
        self._floor_tower_action.triggered.connect(lambda: self._on_floor_filter_changed(320.0))
        self._floor_filter_group.addAction(self._floor_tower_action)
        self._floor_filter_menu.addAction(self._floor_tower_action)

        view_menu.addSeparator()

        # Zoom In (Layout mode)
        self._zoom_in_action = QAction("Zoom &In", self)
        self._zoom_in_action.setShortcut("Ctrl++")
        self._zoom_in_action.setToolTip("Zoom in (Ctrl++)")
        self._zoom_in_action.setEnabled(False)
        self._zoom_in_action.triggered.connect(self._on_zoom_in)
        view_menu.addAction(self._zoom_in_action)

        # Zoom Out (Layout mode)
        self._zoom_out_action = QAction("Zoom &Out", self)
        self._zoom_out_action.setShortcut("Ctrl+-")
        self._zoom_out_action.setToolTip("Zoom out (Ctrl+-)")
        self._zoom_out_action.setEnabled(False)
        self._zoom_out_action.triggered.connect(self._on_zoom_out)
        view_menu.addAction(self._zoom_out_action)

        # Fit to Content (Layout mode)
        self._fit_action = QAction("&Fit to Content", self)
        self._fit_action.setShortcut("F")
        self._fit_action.setToolTip("Fit view to show all content (F)")
        self._fit_action.setEnabled(False)
        self._fit_action.triggered.connect(self._on_fit)
        view_menu.addAction(self._fit_action)

        # Reset View (Layout mode)
        self._reset_view_action = QAction("&Reset View", self)
        self._reset_view_action.setToolTip("Reset view to origin")
        self._reset_view_action.setEnabled(False)
        self._reset_view_action.triggered.connect(self._on_reset_view)
        view_menu.addAction(self._reset_view_action)

        view_menu.addSeparator()

        # Quad View toggle (Module mode)
        self._quad_view_action = SafeMenuAction("&Quad View", self)
        self._quad_view_action.setShortcut("Shift+Q")
        self._quad_view_action.setToolTip("Toggle quad-view with orthographic views to see Z-levels (Shift+Q)")
        self._quad_view_action.setEnabled(True)  # Available in both Layout and Module modes
        self._quad_view_action.toggled.connect(self._on_quad_view_toggled)
        view_menu.addAction(self._quad_view_action)

        # Maximize Pane submenu (for quad-view)
        # Note: Menu starts enabled because shortcuts are application-wide via QShortcut
        self._maximize_pane_menu = view_menu.addMenu("&Maximize Pane")
        self._maximize_pane_menu.setToolTip("Maximize a single pane in quad-view")
        self._maximize_pane_menu.setEnabled(False)

        # Use Alt+number to avoid conflicts with 3D view presets (1-6)
        self._maximize_top_action = QAction("&Top (XY) - Alt+7", self)
        self._maximize_top_action.triggered.connect(lambda: self._on_maximize_pane("top"))
        self._maximize_pane_menu.addAction(self._maximize_top_action)

        self._maximize_front_action = QAction("&Front (XZ) - Alt+1", self)
        self._maximize_front_action.triggered.connect(lambda: self._on_maximize_pane("front"))
        self._maximize_pane_menu.addAction(self._maximize_front_action)

        self._maximize_side_action = QAction("&Side (YZ) - Alt+3", self)
        self._maximize_side_action.triggered.connect(lambda: self._on_maximize_pane("side"))
        self._maximize_pane_menu.addAction(self._maximize_side_action)

        self._maximize_3d_action = QAction("&3D Perspective - Alt+0", self)
        self._maximize_3d_action.triggered.connect(lambda: self._on_maximize_pane("3d"))
        self._maximize_pane_menu.addAction(self._maximize_3d_action)

        self._maximize_pane_menu.addSeparator()

        self._restore_quad_action = QAction("&Restore Quad View - Escape", self)
        self._restore_quad_action.triggered.connect(self._on_restore_quad_view)
        self._maximize_pane_menu.addAction(self._restore_quad_action)

        view_menu.addSeparator()

        # High contrast toggle - uses SafeMenuAction to avoid macOS accessibility crash
        self._high_contrast_action = SafeMenuAction("&High Contrast Mode", self)
        self._high_contrast_action.setChecked(sc.HIGH_CONTRAST_MODE)
        self._high_contrast_action.setToolTip(
            "Enable high-contrast colors for accessibility (requires restart)"
        )
        self._high_contrast_action.toggled.connect(self._on_high_contrast_toggled)
        view_menu.addAction(self._high_contrast_action)

        # Help menu
        help_menu = menu_bar.addMenu("&Help")

        # Quick Start
        self._quick_start_action = QAction("&Quick Start...", self)
        self._quick_start_action.setToolTip("Show quick start tutorial")
        self._quick_start_action.triggered.connect(self._show_quick_start)
        help_menu.addAction(self._quick_start_action)

        # Keyboard Shortcuts
        self._shortcuts_action = QAction("&Keyboard Shortcuts...", self)
        self._shortcuts_action.setShortcut("F1")
        self._shortcuts_action.setToolTip("Show keyboard shortcuts help (F1)")
        self._shortcuts_action.triggered.connect(self._on_show_shortcuts)
        help_menu.addAction(self._shortcuts_action)

    def _update_recent_files_menu(self):
        """Update the Recent Files submenu with current recent files."""
        if not self._recent_files_menu:
            return

        self._recent_files_menu.clear()

        # Filter out non-existent files
        valid_files = [f for f in self._recent_files if os.path.exists(f)]
        if valid_files != self._recent_files:
            self._recent_files = valid_files

        if not self._recent_files:
            no_recent = QAction("(No recent files)", self)
            no_recent.setEnabled(False)
            self._recent_files_menu.addAction(no_recent)
            return

        # Add numbered entries
        for i, file_path in enumerate(self._recent_files):
            file_name = Path(file_path).name
            action = QAction(f"&{i + 1}. {file_name}", self)
            action.setToolTip(file_path)
            action.setData(file_path)
            action.triggered.connect(self._on_open_recent_file)
            self._recent_files_menu.addAction(action)

        # Add Clear Recent Files
        self._recent_files_menu.addSeparator()
        clear_action = QAction("Clear Recent Files", self)
        clear_action.triggered.connect(self._on_clear_recent_files)
        self._recent_files_menu.addAction(clear_action)

    def _on_open_recent_file(self):
        """Handle opening a recent file from the menu."""
        action = self.sender()
        if action:
            file_path = action.data()
            if file_path and os.path.exists(file_path):
                if self.layout_editor.load_file(file_path):
                    self.statusBar().showMessage(f"Loaded: {Path(file_path).name}")
            else:
                QMessageBox.warning(self, "File Not Found",
                                   f"The file no longer exists:\n{file_path}")
                # Remove from recent files
                if file_path in self._recent_files:
                    self._recent_files.remove(file_path)
                self._update_recent_files_menu()

    def _on_clear_recent_files(self):
        """Clear the recent files list."""
        self._recent_files = []
        self._update_recent_files_menu()
        self.statusBar().showMessage("Recent files cleared")

    def _on_high_contrast_toggled(self, checked: bool):
        """Handle high-contrast mode toggle (requires restart)."""
        settings = QSettings('idTechGeometryToolkit', 'MainWindow')
        settings.setValue('high_contrast', checked)

        QMessageBox.information(
            self,
            "Restart Required",
            "High-contrast mode will be applied after restarting the application."
        )

    def _on_texture_settings(self):
        """Open the texture settings dialog."""
        from quake_levelgenerator.src.ui.texture_settings_dialog import show_texture_settings
        if show_texture_settings(self):
            self.statusBar().showMessage("Texture settings updated", 3000)

    def _on_new_layout(self):
        """Create a new layout via File menu."""
        self.layout_editor.new_layout()

    def _on_open_layout(self):
        """Open a layout via File menu."""
        self.layout_editor.open_layout()

    def _on_save_layout(self):
        """Save the layout via File menu."""
        self.layout_editor.save_layout()

    def _on_undo(self):
        """Undo last layout action."""
        self.layout_editor.undo()

    def _on_redo(self):
        """Redo last undone layout action."""
        self.layout_editor.redo()

    def _on_duplicate(self):
        """Duplicate selected primitive in layout."""
        self.layout_editor.duplicate_selected()

    def _on_delete(self):
        """Delete selected primitive in layout."""
        self.layout_editor.delete_selected()

    def _on_flow_toggled(self, checked: bool):
        """Toggle flow visualization in layout editor."""
        self.layout_editor.set_show_flow(checked)

    def _on_floor_filter_changed(self, z_offset):
        """Handle floor filter selection change.

        Args:
            z_offset: Z-offset to filter by, or None to show all floors
        """
        self.layout_editor.set_floor_filter(z_offset)
        if z_offset is None:
            self.statusBar().showMessage("Floor filter: Showing all floors", 3000)
        else:
            floor_name = {
                -128.0: "Basement",
                0.0: "Ground",
                160.0: "Upper",
                320.0: "Tower",
            }.get(z_offset, f"Z:{int(z_offset)}")
            self.statusBar().showMessage(f"Floor filter: {floor_name} (other floors dimmed)", 3000)

    def _on_3d_preview_toggled(self, checked: bool):
        """Toggle 3D preview in layout editor."""
        self.layout_editor.set_3d_preview(checked)

    def _on_zoom_in(self):
        """Zoom in on layout canvas."""
        self.layout_editor.zoom_in()

    def _on_zoom_out(self):
        """Zoom out on layout canvas."""
        self.layout_editor.zoom_out()

    def _on_fit(self):
        """Fit layout view to content."""
        self.layout_editor.fit_to_content()

    def _on_reset_view(self):
        """Reset layout view to origin."""
        self.layout_editor.reset_view()

    def _on_show_shortcuts(self):
        """Show keyboard shortcuts help dialog."""
        self.layout_editor.show_shortcuts_help()

    def _on_quad_view_toggled(self, checked: bool):
        """Toggle quad-view mode in both Layout and Module modes."""
        self._quad_view_enabled = checked
        self._maximize_pane_menu.setEnabled(checked)

        current_mode = self.mode_selector.current_mode()

        if current_mode == "layout":
            if checked:
                # Switch to quad-view (index 2)
                self._center_stack.setCurrentIndex(2)
                # Enable interactive mode: GridCanvas + flow views + 3D
                layout = self.layout_editor.get_layout()
                self._quad_view_widget.set_interactive_mode(True)
                if layout:
                    self._quad_view_widget.set_layout(layout)
                # Generate 3D preview for the perspective pane
                self._sync_layout_to_quad_view()
                self.statusBar().showMessage(
                    "Quad View (Layout): Edit layout top-left, see Z-levels in Front/Side, 3D bottom-right", 5000
                )
            else:
                # Switch back to layout editor (index 0)
                self._quad_view_widget.set_interactive_mode(False)
                self._center_stack.setCurrentIndex(0)
                self.statusBar().showMessage("Layout Editor mode", 3000)
        elif current_mode == "primitive":
            if checked:
                # Switch to quad-view (index 2)
                self._center_stack.setCurrentIndex(2)
                # Module mode: mesh-based ortho views
                self._quad_view_widget.set_interactive_mode(False)
                self._sync_mesh_to_quad_view()
                self.statusBar().showMessage(
                    "Quad View: Pan with middle-drag, zoom with scroll, F to fit", 5000
                )
            else:
                # Switch back to single preview (index 1)
                self._center_stack.setCurrentIndex(1)
                self.statusBar().showMessage("Single View mode", 3000)

    def _on_maximize_pane(self, pane: str):
        """Maximize a pane in quad-view."""
        if not self._quad_view_enabled:
            return
        self._quad_view_widget.toggle_maximize(pane)
        if self._quad_view_widget.is_maximized():
            self.statusBar().showMessage(f"Maximized {pane.upper()} view - press Escape or 0 to restore", 3000)
        else:
            self.statusBar().showMessage("Quad View restored", 2000)

    def _on_restore_quad_view(self):
        """Restore quad-view from maximized state."""
        if self._quad_view_enabled:
            self._quad_view_widget.restore_quad_view()
            self.statusBar().showMessage("Quad View restored", 2000)

    # --- Quad canvas command routing ---

    def _on_quad_canvas_command(self, command):
        """Route a command from the quad GridCanvas through LayoutEditorWidget."""
        from quake_levelgenerator.src.ui.widgets.layout_editor.commands import PlacePrimitiveCommand

        # Execute via layout editor's command pipeline (undo/redo, modified state)
        self.layout_editor.execute_external_command(command)

        # Determine selection ID for quad canvas refresh
        select_id = None
        if isinstance(command, PlacePrimitiveCommand) and command.placed_id:
            select_id = command.placed_id

        # Refresh quad canvas to reflect the change
        self._quad_view_widget.refresh_layout_canvas(select_id)

        # Request debounced geometry regeneration for 3D panes
        self._quad_view_widget.request_regen()

    def _on_quad_regen_requested(self):
        """Handle debounced regen request from quad view."""
        self._sync_layout_to_quad_view()

    def _on_quad_canvas_selection(self, prim_id: str):
        """Handle primitive selection in quad canvas - sync to layout editor."""
        self.layout_editor.select_primitive(prim_id)

    def _on_quad_canvas_selection_cleared(self):
        """Handle selection cleared in quad canvas."""
        self.layout_editor.clear_selection()

    def _sync_quad_canvas_from_editor(self):
        """Sync quad canvas and flow views when layout editor modifies the layout."""
        if (self._quad_view_enabled and
                self._quad_view_widget.is_interactive() and
                self.mode_selector.current_mode() == "layout"):
            self._quad_view_widget.refresh_layout_canvas()
            # Debounced regen updates the 3D pane
            self._quad_view_widget.request_regen()

    def _sync_mesh_to_quad_view(self):
        """Sync current mesh data from single preview to quad-view."""
        # Get mesh from single preview's internal state and send to quad-view
        # This is called when switching to quad-view mode
        prim_name = self.primitive_panel.get_primitive_name()
        params = self._inject_texture_settings(self.primitive_panel.get_parameters())
        if prim_name:
            from quake_levelgenerator.src.generators.primitives.catalog import PRIMITIVE_CATALOG
            from quake_levelgenerator.src.generators.textures import TEXTURE_SETTINGS
            from quake_levelgenerator.src.ui.preview.mesh_builder import (
                MeshBuilder, build_wireframe_mesh, build_surface_meshes
            )

            cls = PRIMITIVE_CATALOG.get_primitive(prim_name)
            if cls:
                try:
                    instance = cls()
                    instance.apply_params(params)
                    instance._texture_wall = TEXTURE_SETTINGS.get_texture("wall")
                    instance._texture_floor = TEXTURE_SETTINGS.get_texture("floor")
                    instance._texture_ceiling = TEXTURE_SETTINGS.get_texture("ceiling")
                    instance._texture_trim = TEXTURE_SETTINGS.get_texture("trim")
                    instance._texture_structural = TEXTURE_SETTINGS.get_texture("structural")

                    brushes = instance.generate()

                    # Build meshes
                    surface_meshes = build_surface_meshes(brushes)
                    builder = MeshBuilder()
                    builder.add_brushes(brushes)
                    mesh = builder.build()
                    wire_verts, wire_indices = build_wireframe_mesh(brushes)

                    # Update quad-view
                    self._quad_view_widget.set_mesh(mesh, (wire_verts, wire_indices), surface_meshes)
                    self._quad_view_widget.fit_all_to_bounds()
                except Exception as e:
                    print(f"Failed to sync mesh to quad-view: {e}")

    def _sync_layout_to_quad_view(self):
        """Sync current layout geometry to quad-view.

        Generates brushes from the layout editor and displays them in quad-view
        so users can visualize Z-levels of multi-floor dungeons.
        """
        try:
            from quake_levelgenerator.src.ui.preview.mesh_builder import (
                MeshBuilder, build_wireframe_mesh, build_surface_meshes
            )

            # Get layout from editor
            layout = self.layout_editor.get_layout()

            if not layout or not layout.primitives:
                # No layout yet - show empty view
                self.statusBar().showMessage(
                    "No layout to display. Place modules first, then enable Quad View.", 5000
                )
                return

            # Generate brushes from layout
            result = generate_from_layout(layout)

            if not result or not result.brushes:
                self.statusBar().showMessage(
                    "Could not generate geometry from layout.", 3000
                )
                return

            brushes = result.brushes

            # Build meshes
            surface_meshes = build_surface_meshes(brushes)
            builder = MeshBuilder()
            builder.add_brushes(brushes)
            mesh = builder.build()
            wire_verts, wire_indices = build_wireframe_mesh(brushes)

            # Update quad-view (in interactive mode, only 3D pane gets mesh)
            self._quad_view_widget.set_mesh(mesh, (wire_verts, wire_indices), surface_meshes)
            if self._quad_view_widget.is_interactive():
                # Only fit the 3D perspective view; flow views keep their
                # current pan/zoom so the camera doesn't jump after edits.
                self._quad_view_widget._perspective_view.fit_to_bounds()
                self._quad_view_widget._refresh_flow_views()
            else:
                self._quad_view_widget.fit_all_to_bounds()

            # Show helpful message
            num_prims = len(layout.primitives)
            self.statusBar().showMessage(
                f"Quad View: {num_prims} modules, {len(brushes)} brushes. Use Front/Side views to see Z-levels.", 5000
            )

        except Exception as e:
            self.statusBar().showMessage(f"Failed to generate quad-view: {e}", 5000)

    def _on_undo_state_changed(self, can_undo: bool, can_redo: bool,
                                undo_desc: str, redo_desc: str):
        """Update undo/redo menu items based on layout editor state."""
        if self.mode_selector.current_mode() == "layout":
            self._undo_action.setEnabled(can_undo)
            self._redo_action.setEnabled(can_redo)
            if undo_desc:
                self._undo_action.setText(f"&Undo {undo_desc}")
            else:
                self._undo_action.setText("&Undo")
            if redo_desc:
                self._redo_action.setText(f"&Redo {redo_desc}")
            else:
                self._redo_action.setText("&Redo")

    def _on_file_operation(self, file_path: str):
        """Handle a file save or load operation to update recent files."""
        self._add_recent_file(file_path)
        self._update_recent_files_menu()

    def _update_validation_panel(self):
        """Update the validation panel with the current layout."""
        layout = self.layout_editor.get_layout()
        self._validation_panel.set_layout(layout)
        self._notify_tab(2)

    def _on_validation_issue_clicked(self, primitive_id: str):
        """Handle clicking on a validation issue - select and center on the primitive."""
        # Ensure layout editor is visible (not quad view)
        if self.mode_selector.current_mode() == "layout" and not self._quad_view_enabled:
            self._center_stack.setCurrentIndex(0)
        self.layout_editor.select_primitive(primitive_id)

    def _on_right_tab_changed(self, index: int):
        """Clear notification badge when tab is activated."""
        if index in self._tab_has_new_content and self._tab_has_new_content[index]:
            self._tab_has_new_content[index] = False
            # Restore original tab title (remove dot prefix)
            title = self.right_tabs.tabText(index)
            if title.startswith("\u2022 "):
                self.right_tabs.setTabText(index, title[2:])

    def _notify_tab(self, tab_index: int):
        """Show a notification dot on a tab if it's not currently active."""
        if self.right_tabs.currentIndex() != tab_index:
            if not self._tab_has_new_content.get(tab_index):
                self._tab_has_new_content[tab_index] = True
                title = self.right_tabs.tabText(tab_index)
                if not title.startswith("\u2022 "):
                    self.right_tabs.setTabText(tab_index, f"\u2022 {title}")

    # ---------------------------------------------------------------
    # Signals
    # ---------------------------------------------------------------

    def _connect_signals(self):
        self.mode_selector.mode_changed.connect(self._on_mode_changed)
        self.generate_btn.clicked.connect(self._on_generate)
        self.random_gen_btn.clicked.connect(self._on_random_generate)
        self.cancel_btn.clicked.connect(self._on_cancel)
        self.export_map_btn.clicked.connect(self._on_export_map)
        self.export_obj_btn.clicked.connect(self._on_export_obj)
        self.open_output_btn.clicked.connect(self._on_open_output)

        # Connect primitive panel to preview and validation
        self.primitive_panel.parameters_changed.connect(self._on_primitive_params_changed)
        self.primitive_panel.validation_changed.connect(self._on_validation_status)

        # Connect layout editor file signals for recent files and status bar
        self.layout_editor.file_saved.connect(self._on_file_operation)
        self.layout_editor.file_loaded.connect(self._on_file_operation)
        self.layout_editor.file_saved.connect(
            lambda p: self.statusBar().showMessage(f"Layout saved: {Path(p).name}", 5000)
        )
        self.layout_editor.file_loaded.connect(
            lambda p: self.statusBar().showMessage(f"Layout loaded: {Path(p).name}", 5000)
        )

        # Connect layout editor to validation panel
        self.layout_editor.layout_changed.connect(self._update_validation_panel)

        # Tab notification badges
        self.log_text.textChanged.connect(lambda: self._notify_tab(0))

        # Connect layout editor undo/redo state changes to menu
        self.layout_editor.undo_state_changed.connect(self._on_undo_state_changed)

        # Connect quad view interactive signals
        self._quad_view_widget.command_requested.connect(self._on_quad_canvas_command)
        self._quad_view_widget.regen_requested.connect(self._on_quad_regen_requested)
        self._quad_view_widget.primitive_selected.connect(self._on_quad_canvas_selection)
        self._quad_view_widget.selection_cleared.connect(self._on_quad_canvas_selection_cleared)

        # Sync quad canvas when layout editor changes
        self.layout_editor.layout_changed.connect(self._sync_quad_canvas_from_editor)

        # Keyboard shortcuts
        self._shortcut_generate = QShortcut(QKeySequence("Ctrl+G"), self)
        self._shortcut_generate.activated.connect(self._on_generate)

        self._shortcut_random = QShortcut(QKeySequence("Ctrl+R"), self)
        self._shortcut_random.activated.connect(self._on_random_generate)

        self._shortcut_cancel = QShortcut(QKeySequence("Escape"), self)
        self._shortcut_cancel.activated.connect(self._on_cancel)

        self._shortcut_open = QShortcut(QKeySequence("Ctrl+O"), self)
        self._shortcut_open.activated.connect(self._on_open_output)

        self._shortcut_export_map = QShortcut(QKeySequence("Ctrl+E"), self)
        self._shortcut_export_map.activated.connect(self._on_export_map)

        self._shortcut_help = QShortcut(QKeySequence("F1"), self)
        self._shortcut_help.activated.connect(self._show_help)

        # Quad-view maximize pane shortcuts (application-wide)
        # Note: Shift+Q is already handled by _quad_view_action shortcut
        self._shortcut_maximize_top = QShortcut(QKeySequence("Alt+7"), self)
        self._shortcut_maximize_top.activated.connect(lambda: self._on_maximize_pane("top"))

        self._shortcut_maximize_front = QShortcut(QKeySequence("Alt+1"), self)
        self._shortcut_maximize_front.activated.connect(lambda: self._on_maximize_pane("front"))

        self._shortcut_maximize_side = QShortcut(QKeySequence("Alt+3"), self)
        self._shortcut_maximize_side.activated.connect(lambda: self._on_maximize_pane("side"))

        self._shortcut_maximize_3d = QShortcut(QKeySequence("Alt+0"), self)
        self._shortcut_maximize_3d.activated.connect(lambda: self._on_maximize_pane("3d"))

        # Mode switching shortcuts
        self._shortcut_layout_mode = QShortcut(QKeySequence("Ctrl+1"), self)
        self._shortcut_layout_mode.activated.connect(lambda: self.mode_selector.set_mode("layout"))

        self._shortcut_module_mode = QShortcut(QKeySequence("Ctrl+2"), self)
        self._shortcut_module_mode.activated.connect(lambda: self.mode_selector.set_mode("primitive"))

    def _inject_texture_settings(self, params: dict) -> dict:
        """Inject global texture settings into primitive parameters.

        Ensures textures from Edit > Texture Settings are applied to primitives.
        Maintains parameter parity between Module Mode and Layout Mode.
        """
        result = params.copy()
        for surface in ['wall', 'floor', 'ceiling', 'trim', 'structural']:
            param_key = f'texture_{surface}'
            if param_key not in result or not result.get(param_key):
                result[param_key] = TEXTURE_SETTINGS.get_texture(surface)
        if 'texture' not in result or not result.get('texture'):
            result['texture'] = TEXTURE_SETTINGS.get_texture('wall')
        return result

    def _on_mode_changed(self, mode: str):
        self.panel_stack.setCurrentIndex(0 if mode == "layout" else 1)

        # Show/hide Random Dungeon button based on mode
        self.random_gen_btn.setVisible(mode == "layout")

        # Enable/disable layout-specific menu items with tooltip explanations
        is_layout = (mode == "layout")
        layout_only_actions = [
            self._undo_action, self._redo_action,
            self._duplicate_action, self._delete_action,
            self._flow_action, self._preview_3d_action,
            self._zoom_in_action, self._zoom_out_action,
            self._fit_action, self._reset_view_action,
        ]
        for action in layout_only_actions:
            action.setEnabled(is_layout)
            if not is_layout:
                action.setStatusTip("Available in Layout mode only")
            else:
                action.setStatusTip("")
        self._floor_filter_menu.setEnabled(is_layout)

        # Quad-view is available in both modes
        self._quad_view_action.setEnabled(True)
        self._maximize_pane_menu.setEnabled(self._quad_view_enabled)

        # Switch center content based on mode
        if mode == "layout":
            if self._quad_view_enabled:
                self._center_stack.setCurrentIndex(2)  # Quad-view
                # Enable interactive mode with GridCanvas + flow views
                layout = self.layout_editor.get_layout()
                self._quad_view_widget.set_interactive_mode(True)
                if layout:
                    self._quad_view_widget.set_layout(layout)
                self._sync_layout_to_quad_view()
                self.statusBar().showMessage("Layout mode (Quad View) - Edit layout, see Z-levels, 3D preview")
            else:
                self._center_stack.setCurrentIndex(0)  # Layout editor
                self.statusBar().showMessage("Layout mode - Click to place modules, R to rotate, or use Random Dungeon")
        else:
            # Module mode
            if self._quad_view_enabled:
                self._center_stack.setCurrentIndex(2)  # Quad-view
                # Module mode: mesh-based ortho views
                self._quad_view_widget.set_interactive_mode(False)
                self._quad_view_widget.clear()
                self._sync_mesh_to_quad_view()
                self.statusBar().showMessage("Module mode (Quad View) - Pan with middle-drag, zoom with scroll")
            else:
                self._center_stack.setCurrentIndex(1)  # Single preview
                self.statusBar().showMessage("Module mode - Use mouse to orbit, scroll to zoom")
            # Update preview with texture settings
            prim_name = self.primitive_panel.get_primitive_name()
            params = self._inject_texture_settings(self.primitive_panel.get_parameters())
            if prim_name:
                self.preview_widget.update_primitive(prim_name, params)

    def _on_primitive_params_changed(self, params: dict):
        """Update preview when primitive parameters change."""
        if self.mode_selector.current_mode() == "primitive":
            prim_name = self.primitive_panel.get_primitive_name()
            if prim_name:
                params_with_textures = self._inject_texture_settings(params)
                self.preview_widget.update_primitive(prim_name, params_with_textures)

                # Also update quad-view if enabled
                if self._quad_view_enabled:
                    self._sync_mesh_to_quad_view()

    def _on_validation_status(self, message: str, severity: str):
        """Show validation status in status bar."""
        if severity == "ok":
            self.statusBar().setStyleSheet(f"QStatusBar {{ color: {sc.TEXT_SUCCESS}; }}")
            self.statusBar().showMessage(f"✓ {message}", 3000)
        elif severity == "warning":
            self.statusBar().setStyleSheet(f"QStatusBar {{ color: {sc.WARNING_COLOR}; }}")
            self.statusBar().showMessage(f"⚠ {message}", 5000)
        elif severity == "error":
            self.statusBar().setStyleSheet(f"QStatusBar {{ color: {sc.DANGER_COLOR}; }}")
            self.statusBar().showMessage(f"✗ {message}", 5000)
        else:
            self.statusBar().setStyleSheet(f"QStatusBar {{ color: {sc.TEXT_SECONDARY}; }}")
            self.statusBar().showMessage(message, 3000)

    # ---------------------------------------------------------------
    # Generation
    # ---------------------------------------------------------------

    def _get_saved_export_format(self) -> str:
        """Get the saved export format preference from settings."""
        return self._settings.value('export_format', 'idtech4', type=str)

    def _save_export_format(self, fmt: str):
        """Save the export format preference to settings."""
        self._settings.setValue('export_format', fmt)

    def _show_export_format_dialog(self, force_show: bool = False) -> str | None:
        """Show dialog to select export format.

        If the user previously chose "Always use this format", the dialog is
        skipped and the saved format is returned directly (unless *force_show*
        is True).

        Returns format string or None if cancelled.
        """
        saved_fmt = self._get_saved_export_format()

        # Skip dialog if user chose "always use" and we're not forced
        if not force_show and self._settings.value("always_use_export_format", False, type=bool):
            return saved_fmt

        dialog = QMessageBox(self)
        dialog.setWindowTitle("Export Format")
        dialog.setText("Select the export format:")
        dialog.setInformativeText(
            "idTech 1: Quake, Half-Life (3-point plane format)\n"
            "idTech 4: Doom 3, Quake 4 (brushDef3 format)\n\n"
            "Tip: Hold Shift when clicking Export to change format later."
        )

        # Add custom buttons
        idtech1_btn = dialog.addButton("idTech 1", QMessageBox.AcceptRole)
        idtech4_btn = dialog.addButton("idTech 4", QMessageBox.AcceptRole)
        cancel_btn = dialog.addButton(QMessageBox.Cancel)

        # Highlight the saved preference
        if saved_fmt == "idtech1":
            dialog.setDefaultButton(idtech1_btn)
        else:
            dialog.setDefaultButton(idtech4_btn)

        dialog.exec_()

        clicked = dialog.clickedButton()
        if clicked == idtech1_btn:
            self._save_export_format("idtech1")
            self._settings.setValue("always_use_export_format", True)
            return "idtech1"
        elif clicked == idtech4_btn:
            self._save_export_format("idtech4")
            self._settings.setValue("always_use_export_format", True)
            return "idtech4"
        else:
            return None

    def _on_generate(self):
        """Generate geometry and update preview without exporting to files."""
        if self.generation_thread and self.generation_thread.isRunning():
            return

        mode = self.mode_selector.current_mode()

        if mode == "layout":
            # Generate from layout editor (manual layout)
            self._generate_from_layout()
        else:
            # Generate from primitive panel
            self._generate_from_primitive()

    def _validate_before_generation(self, params: dict) -> bool:
        """Validate generation parameters and show actionable warnings.

        Returns True if generation should proceed, False otherwise.
        """
        issues = []

        map_area = params['map_width'] * params['map_height']
        room_count = params['room_count']

        # Each room needs roughly 9-16 tiles (3x3 to 4x4 footprint)
        min_area_needed = room_count * 12
        if min_area_needed > map_area:
            min_side = max(10, int((min_area_needed ** 0.5) + 1))
            issues.append(
                f"Room count {room_count} requires at least ~{min_side}x{min_side} "
                f"map size (current: {params['map_width']}x{params['map_height']})."
            )

        floor_count = params.get('floor_count', 1)
        if floor_count > 1 and room_count < 3:
            issues.append(
                f"Multi-floor layouts ({floor_count} floors) need at least 3 rooms per floor "
                f"to allow stair placement."
            )

        if issues:
            msg = "Potential issues detected:\n\n" + "\n\n".join(f"- {i}" for i in issues)
            msg += "\n\nGeneration may fail or produce poor results. Continue anyway?"
            reply = QMessageBox.warning(
                self, "Parameter Warning", msg,
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            return reply == QMessageBox.Yes
        return True

    def _on_random_generate(self):
        """Generate random dungeon layout and load it into the layout editor."""
        # Only works in layout mode
        if self.mode_selector.current_mode() != "layout":
            QMessageBox.warning(self, "Wrong Mode",
                               "Random dungeon generation is only available in Layout mode.")
            return

        # Get parameters from layout panel
        params = self.layout_panel.get_parameters()

        # Pre-validate parameters
        if not self._validate_before_generation(params):
            return

        self.log_text.clear()
        self.log_text.append("Generating random layout...")
        self.log_text.append(f"Parameters: {params['room_count']} rooms, {params['map_width']}x{params['map_height']} tiles")
        if params.get('seed'):
            self.log_text.append(f"Seed: {params['seed']}")
        else:
            self.log_text.append("Seed: random")

        try:
            # Check if multi-floor generation is requested
            floor_count = params.get('floor_count', 1)
            auto_connect = params.get('auto_connect_floors', True)
            hints = self.layout_panel.get_generation_hints()

            if floor_count > 1:
                # Multi-floor generation
                self.log_text.append(f"Generating {floor_count} floors...")
                layout, actual_seed = generate_multi_floor_layout(
                    floor_count=floor_count,
                    rooms_per_floor=params['room_count'],
                    map_width=params['map_width'],
                    map_height=params['map_height'],
                    seed=params.get('seed'),
                    complexity=params['complexity'],
                    auto_connect_floors=auto_connect,
                    preferred_room_types=hints.get('preferred_room_types'),
                    preferred_hall_types=hints.get('preferred_hall_types'),
                    room_probability=hints.get('room_probability', 0.4),
                    min_hall_between_rooms=hints.get('min_hall_between_rooms', 1),
                    allow_dead_ends=hints.get('allow_dead_ends', True),
                    corridor_width=hints.get('corridor_width'),
                    max_shortcuts=hints.get('max_shortcuts', 0),
                )
            else:
                # Single floor generation (original behavior)
                layout, actual_seed = generate_random_layout(
                    room_count=params['room_count'],
                    map_width=params['map_width'],
                    map_height=params['map_height'],
                    seed=params.get('seed'),
                    complexity=params['complexity'],
                    preferred_room_types=hints.get('preferred_room_types'),
                    preferred_hall_types=hints.get('preferred_hall_types'),
                    room_probability=hints.get('room_probability', 0.4),
                    min_hall_between_rooms=hints.get('min_hall_between_rooms', 1),
                    allow_dead_ends=hints.get('allow_dead_ends', True),
                    corridor_width=hints.get('corridor_width'),
                    max_shortcuts=hints.get('max_shortcuts', 0),
                )

            # Load the layout into the editor
            self.layout_editor.set_layout(layout)

            # Log results
            prim_count = len(layout.primitives)
            conn_count = len(layout.connections)
            self.log_text.append(f"=== RANDOM LAYOUT GENERATED ===")
            self.log_text.append(f"Seed used: {actual_seed}")
            if floor_count > 1:
                self.log_text.append(f"Floors: {floor_count}")
                if auto_connect:
                    self.log_text.append("Stairs: auto-connected")
                else:
                    self.log_text.append("Stairs: manual placement required")
            self.log_text.append(f"Modules: {prim_count}")
            self.log_text.append(f"Connections: {conn_count}")
            self.log_text.append("")
            self.log_text.append("Layout loaded into editor. You can now:")
            self.log_text.append("  - Edit the layout (move, rotate, delete modules)")
            self.log_text.append("  - Add more modules manually")
            self.log_text.append("  - Click 'Generate' to create geometry")
            self.log_text.append("  - Export to .map file")

            self.statusBar().showMessage(
                f"Random layout generated with {prim_count} modules (seed: {actual_seed})"
            )

            # If using random seed, uncheck the fixed seed toggle so next
            # generation gets a fresh random seed
            if not self.layout_panel.fixed_seed_check.isChecked():
                self.layout_panel.seed_spinbox.setValue(0)

            # Center the 2D canvas view on the generated layout
            self.layout_editor.fit_to_content()

            # Update quad view 2D panes if quad view is active in layout mode
            if self._quad_view_enabled and self._quad_view_widget.is_interactive():
                self._quad_view_widget.set_layout(layout)

            # Update default filename from template + seed
            self._update_default_filename(actual_seed)

            # Auto-generate brushes so the user sees geometry immediately
            self._generate_from_layout()

        except Exception as e:
            self.log_text.append(f"ERROR: {e}")
            QMessageBox.warning(self, "Generation Failed", f"Failed to generate layout: {e}")

    def _generate_from_layout(self):
        """Generate brushes from layout editor (without exporting)."""
        layout = self.layout_editor.get_layout()

        if not layout.primitives:
            QMessageBox.warning(self, "Empty Layout",
                               "The layout is empty. Add some modules first.")
            return

        # Run validation and check for critical errors
        validation_result = self._validation_panel.result
        if validation_result is None:
            # Force a validation refresh if not yet run
            self._update_validation_panel()
            validation_result = self._validation_panel.result

        if validation_result and validation_result.error_count > 0:
            # Build error summary
            error_messages = []
            for issue in validation_result.issues:
                if issue.is_error:
                    error_messages.append(f"• {issue.message}")

            error_text = "\n".join(error_messages[:10])  # Show up to 10 errors
            if len(error_messages) > 10:
                error_text += f"\n... and {len(error_messages) - 10} more errors"

            reply = QMessageBox.warning(
                self, "Validation Errors",
                f"The layout has {validation_result.error_count} validation error(s):\n\n"
                f"{error_text}\n\n"
                "The exported map may be unplayable. Continue anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                self.log_text.append("Generation cancelled due to validation errors.")
                self.log_text.append("Fix the errors in the Validation tab and try again.")
                return

        # Get profile settings from layout panel
        params = self.layout_panel.get_parameters()
        random_seed = params.get('seed', None)
        game_profile = self.layout_panel.get_selected_profile()

        self.log_text.clear()
        self.log_text.append("Mode: layout")
        self.log_text.append(f"Profile: {game_profile.name}")

        # Log validation status
        if validation_result:
            if validation_result.error_count > 0:
                self.log_text.append(f"⚠ WARNING: {validation_result.error_count} validation errors (proceeding anyway)")
            elif validation_result.warning_count > 0:
                self.log_text.append(f"Note: {validation_result.warning_count} validation warnings")
            else:
                self.log_text.append("✓ Layout validation passed")

        self.progress_bar.setValue(10)
        self.progress_label.setText("Generating brushes...")

        try:
            # Generate brushes from layout with profile
            result = generate_from_layout(
                layout,
                random_seed=random_seed,
                game_profile=game_profile,
            )

            # Store generated brushes for later export
            self._generated_brushes = result.brushes
            self._spawn_position = result.spawn_position

            self.log_text.append(f"Generated {result.brush_count} brushes from {result.primitive_count} modules")

            for warning in result.warnings:
                self.log_text.append(f"Warning: {warning}")

            self.progress_bar.setValue(100)
            self.progress_label.setText("Ready to export")

            self.log_text.append("=== GENERATION COMPLETE ===")
            self.log_text.append("Use Export buttons to save to file.")

            # Enable export buttons and update tooltips
            self.export_map_btn.setEnabled(True)
            self.export_map_btn.setToolTip("Export to idTech .map format for TrenchBroom (Ctrl+E)")
            self.export_obj_btn.setEnabled(True)
            self.export_obj_btn.setToolTip("Export to OBJ mesh format for Blender")

            self.statusBar().showMessage(f"Built {result.brush_count} brushes - Ready to export", 5000)

            # Update the 3D preview with generated brushes and fit camera
            if hasattr(self, 'preview_widget') and hasattr(self.preview_widget, 'set_brushes'):
                self.preview_widget.set_brushes(result.brushes)
                self.preview_widget.fit_to_bounds()

            # Also sync to quad-view if enabled
            if self._quad_view_enabled:
                self._sync_layout_to_quad_view()

        except Exception as e:
            self.log_text.append(f"FAILED: {e}")
            self.progress_label.setText("Failed")
            self.export_map_btn.setEnabled(False)
            self.export_obj_btn.setEnabled(False)
            QMessageBox.critical(self, "Error", str(e))

    def _generate_from_primitive(self):
        """Generate brushes from primitive panel (without exporting)."""
        from quake_levelgenerator.src.generators.primitives.catalog import PRIMITIVE_CATALOG

        prim_name = self.primitive_panel.get_primitive_name()
        prim_params = self.primitive_panel.get_parameters()

        if not prim_name:
            QMessageBox.warning(self, "No Module",
                               "Select a module type first.")
            return

        # Get profile for entity/worldspawn settings
        game_profile = self.layout_panel.get_selected_profile()

        # Apply textures from global texture settings
        prim_params = self._inject_texture_settings(prim_params)

        self.log_text.clear()
        self.log_text.append(f"Mode: module | Type: {prim_name}")
        self.log_text.append(f"Profile: {game_profile.name}")
        self.progress_bar.setValue(10)
        self.progress_label.setText("Generating brushes...")

        try:
            # Get primitive class and instantiate
            prim_class = PRIMITIVE_CATALOG.get_primitive(prim_name)
            if prim_class is None:
                raise ValueError(f"Unknown primitive: {prim_name}")

            primitive = prim_class()
            primitive.apply_params(prim_params)

            # Generate brushes
            brushes = primitive.generate()

            # Store for export
            self._generated_brushes = brushes
            self._spawn_position = (0, 0, 56)  # Default spawn in front of primitive

            self.log_text.append(f"Generated {len(brushes)} brushes")

            self.progress_bar.setValue(100)
            self.progress_label.setText("Ready to export")

            self.log_text.append("=== GENERATION COMPLETE ===")
            self.log_text.append("Use Export buttons to save to file.")

            # Enable export buttons and update tooltips
            self.export_map_btn.setEnabled(True)
            self.export_map_btn.setToolTip("Export to idTech .map format for TrenchBroom (Ctrl+E)")
            self.export_obj_btn.setEnabled(True)
            self.export_obj_btn.setToolTip("Export to OBJ mesh format for Blender")

            # Update default filename to module name
            self.map_name_edit.setText(prim_name.lower())

            # Update preview
            self.preview_widget.update_primitive(prim_name, prim_params)

            self.statusBar().showMessage(f"Generated {len(brushes)} brushes - Ready to export", 5000)

        except Exception as e:
            self.log_text.append(f"FAILED: {e}")
            self.progress_label.setText("Failed")
            self.export_map_btn.setEnabled(False)
            self.export_obj_btn.setEnabled(False)
            QMessageBox.critical(self, "Error", str(e))

    # ---------------------------------------------------------------
    # Export
    # ---------------------------------------------------------------

    def _update_default_filename(self, seed=None):
        """Update the filename field based on current template and seed."""
        parts = []
        template = self.layout_panel.get_current_template()
        if template:
            parts.append(template.name.lower().replace(' ', '_'))
        if not parts:
            parts.append('generated_map')
        if seed is not None:
            parts.append(str(seed))
        self.map_name_edit.setText('_'.join(parts))

    def _on_export_map(self):
        """Export generated brushes to .map file."""
        if not self._generated_brushes:
            QMessageBox.warning(self, "No Geometry",
                               "Generate geometry first before exporting.")
            return

        # Check validation if in layout mode
        if self.mode_selector.current_mode() == "layout":
            validation_result = self._validation_panel.result
            if validation_result and validation_result.error_count > 0:
                reply = QMessageBox.warning(
                    self, "Export with Errors",
                    f"The layout has {validation_result.error_count} validation error(s).\n\n"
                    "The exported map may have unreachable areas or other issues.\n"
                    "Check the Validation tab for details.\n\n"
                    "Export anyway?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return

        # Show format selection dialog
        fmt = self._show_export_format_dialog()
        if fmt is None:
            return  # User cancelled

        map_name = self.map_name_edit.text().strip() or "generated_map"

        # Get last export directory or default to output/
        last_dir = self._settings.value("last_export_dir", "")
        if not last_dir or not Path(last_dir).is_dir():
            last_dir = str(Path("output").resolve())
            Path("output").mkdir(exist_ok=True)

        default_path = str(Path(last_dir) / f"{map_name}.map")

        output_path_str, _ = QFileDialog.getSaveFileName(
            self, "Export .map File", default_path,
            "MAP Files (*.map);;All Files (*)"
        )
        if not output_path_str:
            return  # User cancelled

        output_path = Path(output_path_str)

        # Remember the directory for next time
        self._settings.setValue("last_export_dir", str(output_path.parent))

        self.progress_bar.setValue(50)
        self.progress_label.setText("Writing .map file...")

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            writer = MapWriter(export_format=fmt)
            # Select profile matching the export format for correct worldspawn defaults
            from quake_levelgenerator.src.generators.profiles import PROFILE_CATALOG
            if fmt == "idtech1":
                export_profile = PROFILE_CATALOG.get_profile("Quake 1")
            else:
                export_profile = PROFILE_CATALOG.get_profile("Doom 3")
            worldspawn = writer.create_worldspawn(profile=export_profile)
            writer.add_entity(worldspawn)

            for brush in self._generated_brushes:
                writer.add_brush(brush)

            # Add spawn point
            writer.add_player_start(self._spawn_position)

            writer.write_to_file(str(output_path))

            self.progress_bar.setValue(100)
            self.progress_label.setText("Export complete")

            size = output_path.stat().st_size
            self.log_text.append(f"Exported: {output_path} ({size:,} bytes)")

            # Store result for "Open Output Folder"
            self.current_result = type('Result', (), {
                'primary_output': str(output_path),
                'success': True
            })()
            self.open_output_btn.setEnabled(True)

            self.statusBar().showMessage(f"Exported to {output_path}", 5000)

        except Exception as e:
            self.log_text.append(f"Export FAILED: {e}")
            self.progress_label.setText("Export failed")
            QMessageBox.critical(self, "Export Error", str(e))

    def _on_export_obj(self):
        """Export generated brushes to .obj file."""
        if not self._generated_brushes:
            QMessageBox.warning(self, "No Geometry",
                               "Generate geometry first before exporting.")
            return

        from quake_levelgenerator.src.conversion.obj_writer import ObjWriter

        map_name = self.map_name_edit.text().strip() or "generated_map"

        # Get last export directory or default to output/
        last_dir = self._settings.value("last_export_dir", "")
        if not last_dir or not Path(last_dir).is_dir():
            last_dir = str(Path("output").resolve())
            Path("output").mkdir(exist_ok=True)

        default_path = str(Path(last_dir) / f"{map_name}.obj")

        obj_path_str, _ = QFileDialog.getSaveFileName(
            self, "Export .obj File", default_path,
            "OBJ Files (*.obj);;All Files (*)"
        )
        if not obj_path_str:
            return  # User cancelled

        obj_path = Path(obj_path_str)
        mtl_path = obj_path.with_suffix(".mtl")

        # Remember the directory for next time
        self._settings.setValue("last_export_dir", str(obj_path.parent))

        self.progress_bar.setValue(50)
        self.progress_label.setText("Writing .obj file...")

        try:
            obj_path.parent.mkdir(parents=True, exist_ok=True)

            writer = ObjWriter()
            writer.add_brushes(self._generated_brushes)

            writer.write(str(obj_path))

            self.progress_bar.setValue(100)
            self.progress_label.setText("Export complete")

            obj_size = obj_path.stat().st_size
            mtl_size = mtl_path.stat().st_size if mtl_path.exists() else 0
            self.log_text.append(f"Exported: {obj_path} ({obj_size:,} bytes)")
            if mtl_size:
                self.log_text.append(f"Exported: {mtl_path} ({mtl_size:,} bytes)")

            # Store result for "Open Output Folder"
            self.current_result = type('Result', (), {
                'primary_output': str(obj_path),
                'success': True
            })()
            self.open_output_btn.setEnabled(True)

            self.statusBar().showMessage(f"Exported to {obj_path}", 5000)

        except Exception as e:
            self.log_text.append(f"Export FAILED: {e}")
            self.progress_label.setText("Export failed")
            QMessageBox.critical(self, "Export Error", str(e))

    def _on_cancel(self):
        """Handle Escape key - cancel generation or restore quad-view."""
        # First, try to cancel any running generation
        if self.generation_thread and self.generation_thread.isRunning():
            self.generation_thread.cancel_generation()
            self.generation_thread.quit()
            self.generation_thread.wait(5000)
            self._reset_ui()
            self.log_text.append("Cancelled.")
            return

        # If quad-view is maximized, restore it
        if self._quad_view_enabled and self._quad_view_widget.is_maximized():
            self._on_restore_quad_view()

    def _on_progress(self, progress: PipelineProgress):
        self.progress_bar.setValue(progress.percentage)
        self.progress_label.setText(f"[{progress.stage.value}] {progress.message}")

    def _on_complete(self, result: PipelineResult):
        self.current_result = result
        self.log_text.append("=== COMPLETE ===")
        self.log_text.append(f"Time: {result.total_time:.1f}s")
        if result.primary_output:
            size = Path(result.primary_output).stat().st_size
            self.log_text.append(f"Output: {result.primary_output} ({size:,} bytes)")
        self.progress_bar.setValue(100)
        self.progress_label.setText("Done")
        self.open_output_btn.setEnabled(True)
        QMessageBox.information(self, "Done", f"Map exported to:\n{result.primary_output}")

    def _on_failed(self, error: str):
        # Split user-friendly message from full stack trace
        if "---TRACEBACK---" in error:
            summary, full_trace = error.split("---TRACEBACK---", 1)
            summary = summary.strip()
            # Log the full trace for debugging
            self.log_text.append(f"FAILED: {summary}")
            self.log_text.append(f"Stack trace:\n{full_trace.strip()}")
            # Show only the summary in the dialog
            QMessageBox.critical(self, "Error", summary)
        else:
            self.log_text.append(f"FAILED: {error}")
            QMessageBox.critical(self, "Error", error)
        self.progress_label.setText("Failed")

    def _on_thread_done(self):
        self._reset_ui()

    def _reset_ui(self):
        self.generate_btn.setEnabled(True)
        self.random_gen_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def _show_help(self):
        """Show help dialog with keyboard shortcuts and tips."""
        help_text = """
<h2>idTech Geometry Toolkit Help</h2>

<h3>Keyboard Shortcuts</h3>
<table style="border-collapse: collapse;">
<tr><td><b>Ctrl+G</b></td><td>Build geometry from layout/module</td></tr>
<tr><td><b>Ctrl+R</b></td><td>Generate random dungeon (Layout mode)</td></tr>
<tr><td><b>Ctrl+E</b></td><td>Export .map file</td></tr>
<tr><td><b>Ctrl+O</b></td><td>Open output folder</td></tr>
<tr><td><b>Ctrl+1</b></td><td>Switch to Layout mode</td></tr>
<tr><td><b>Ctrl+2</b></td><td>Switch to Module mode</td></tr>
<tr><td><b>Escape</b></td><td>Cancel generation</td></tr>
<tr><td><b>F1</b></td><td>Show this help</td></tr>
</table>

<h3>Camera Controls (3D Preview)</h3>
<table style="border-collapse: collapse;">
<tr><td><b>Right-drag</b></td><td>Mouselook (look around)</td></tr>
<tr><td><b>W/S</b></td><td>Fly forward/backward</td></tr>
<tr><td><b>A/D</b></td><td>Strafe left/right</td></tr>
<tr><td><b>Q/E</b></td><td>Fly down/up</td></tr>
<tr><td><b>Shift</b></td><td>Sprint (2x camera speed)</td></tr>
<tr><td><b>Middle-drag</b></td><td>Pan</td></tr>
<tr><td><b>Scroll</b></td><td>Zoom</td></tr>
<tr><td><b>RMB+Scroll</b></td><td>Adjust camera speed</td></tr>
<tr><td><b>Ctrl+Drag</b></td><td>Mouselook (trackpad alternative)</td></tr>
<tr><td><b>F</b></td><td>Fit to bounds</td></tr>
<tr><td><b>Home</b></td><td>Reset view and speed</td></tr>
<tr><td><b>H</b></td><td>Toggle control hints</td></tr>
<tr><td><b>1-6</b></td><td>View presets</td></tr>
</table>

<h3>Quad View (Both Modes)</h3>
<table style="border-collapse: collapse;">
<tr><td><b>G</b></td><td>Focus on selected module</td></tr>
<tr><td><b>Shift+Q</b></td><td>Toggle quad-view (4-pane view)</td></tr>
<tr><td><b>Alt+7</b></td><td>Maximize Top (XY) view</td></tr>
<tr><td><b>Alt+1</b></td><td>Maximize Front (XZ) view</td></tr>
<tr><td><b>Alt+3</b></td><td>Maximize Side (YZ) view</td></tr>
<tr><td><b>Alt+0</b></td><td>Maximize 3D view</td></tr>
<tr><td><b>Escape</b></td><td>Restore quad-view</td></tr>
</table>

<h3>Quick Tips</h3>
<ul>
<li><b>Layout Mode:</b> Design dungeons by placing and connecting modules on a 2D grid</li>
<li><b>Module Mode:</b> Preview individual geometric pieces with real-time parameter editing</li>
<li><b>Quad View:</b> Use View > Quad View to see orthographic views showing Z-height</li>
<li>Use <b>Build Geometry</b> to prepare for export, then <b>Export .map</b> to save</li>
<li>Exported maps can be opened in TrenchBroom or other idTech editors</li>
</ul>
"""
        QMessageBox.information(self, "Help", help_text)

    def _on_copy_log(self):
        """Copy log contents to clipboard."""
        text = self.log_text.toPlainText()
        if text:
            QApplication.clipboard().setText(text)
            self.statusBar().showMessage("Log copied to clipboard", 3000)

    def _on_open_output(self):
        if self.current_result and self.current_result.primary_output:
            folder = str(Path(self.current_result.primary_output).parent)
            if os.path.isdir(folder):
                import subprocess
                subprocess.Popen(["open", folder])

    def closeEvent(self, event):
        if self.generation_thread and self.generation_thread.isRunning():
            reply = QMessageBox.question(
                self, "Running", "Generation running. Cancel and exit?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self.generation_thread.cancel_generation()
                self.generation_thread.quit()
                self.generation_thread.wait(5000)
            else:
                event.ignore()
                return

        # Check for unsaved layout changes
        if self.layout_editor.has_unsaved_changes():
            reply = QMessageBox.warning(
                self, "Unsaved Changes",
                "You have unsaved changes to the layout.\n\n"
                "Save before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save,
            )
            if reply == QMessageBox.Save:
                self.layout_editor.save_layout()
                # If still dirty after save attempt, user may have cancelled the dialog
                if self.layout_editor.has_unsaved_changes():
                    event.ignore()
                    return
            elif reply == QMessageBox.Cancel:
                event.ignore()
                return
            # Discard falls through to close

        # Save settings before closing
        self._save_settings()

        # Clean up preview widget OpenGL resources
        if self._opengl_available:
            self.preview_widget.cleanup()
            self._quad_view_widget.cleanup()
        event.accept()

    # ---------------------------------------------------------------
    # Settings persistence
    # ---------------------------------------------------------------

    def _load_settings(self):
        """Load application settings from QSettings."""
        # Check UI version - if old, clear splitter settings to apply new defaults
        ui_version = self._settings.value("ui_version", 0)
        try:
            ui_version = int(ui_version)
        except (TypeError, ValueError):
            ui_version = 0

        # Version 2: Improved panel sizing (2026-02-05)
        if ui_version < 2:
            self._settings.remove("splitter_sizes")
            self._settings.setValue("ui_version", 2)

        # Load recent files
        recent = self._settings.value("recent_files", [])
        if isinstance(recent, list):
            self._recent_files = recent[:self.MAX_RECENT_FILES]
        elif isinstance(recent, str) and recent:
            self._recent_files = [recent]
        else:
            self._recent_files = []

        # Load quad-view state (will be applied after UI setup)
        self._saved_quad_view_enabled = self._settings.value("quad_view_enabled", False, type=bool)

    def _save_settings(self):
        """Save application settings to QSettings."""
        # Save window geometry
        self._settings.setValue("window_geometry", self.saveGeometry())
        self._settings.setValue("window_state", self.saveState())

        # Save splitter sizes
        if hasattr(self, 'main_splitter'):
            self._settings.setValue("splitter_sizes", self.main_splitter.sizes())

        # Save recent files
        self._settings.setValue("recent_files", self._recent_files)

        # Save quad-view state
        self._settings.setValue("quad_view_enabled", self._quad_view_enabled)
        if hasattr(self, '_quad_view_widget'):
            self._quad_view_widget.save_splitter_state(self._settings)

        # Mark that we've run at least once
        self._settings.setValue("first_run_complete", True)

        # Clean up autosave on clean exit
        self._delete_autosave()

    # ---------------------------------------------------------------
    # Auto-save and crash recovery
    # ---------------------------------------------------------------

    _AUTOSAVE_DIR = Path.home() / ".config" / "quake_levelgenerator"
    _AUTOSAVE_FILE = _AUTOSAVE_DIR / "autosave.json"

    def _auto_save_layout(self):
        """Periodically save the layout if it has unsaved changes."""
        if not self.layout_editor.has_unsaved_changes():
            return
        try:
            self._AUTOSAVE_DIR.mkdir(parents=True, exist_ok=True)
            layout = self.layout_editor.get_layout()
            if layout and layout.primitives:
                data = layout.to_dict()
                with open(self._AUTOSAVE_FILE, 'w') as f:
                    json.dump(data, f, indent=2)
                logging.getLogger(__name__).debug("Auto-saved layout (%d modules)", len(layout.primitives))
        except Exception as e:
            logging.getLogger(__name__).warning("Auto-save failed: %s", e)

    def _delete_autosave(self):
        """Remove the autosave file."""
        try:
            if self._AUTOSAVE_FILE.exists():
                self._AUTOSAVE_FILE.unlink()
        except OSError:
            pass

    def _check_autosave_recovery(self):
        """On startup, offer to restore from autosave if it exists."""
        if not self._AUTOSAVE_FILE.exists():
            return
        try:
            reply = QMessageBox.question(
                self, "Recover Unsaved Work",
                "An auto-saved layout was found from a previous session.\n\n"
                "Would you like to restore it?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                if self.layout_editor.load_file(str(self._AUTOSAVE_FILE)):
                    self.mode_selector.set_mode("layout")
                    self.statusBar().showMessage("Restored auto-saved layout", 5000)
            # Always clean up after offering recovery
            self._delete_autosave()
        except Exception as e:
            logging.getLogger(__name__).warning("Auto-save recovery failed: %s", e)
            self._delete_autosave()

    def _restore_geometry(self):
        """Restore window geometry and splitter sizes from settings."""
        # Restore window geometry
        geometry = self._settings.value("window_geometry")
        if geometry:
            self.restoreGeometry(geometry)

        state = self._settings.value("window_state")
        if state:
            self.restoreState(state)

        # Restore splitter sizes only if they're reasonable
        # Skip restoring if any panel would be too small (respects our new minimums)
        splitter_sizes = self._settings.value("splitter_sizes")
        if splitter_sizes and hasattr(self, 'main_splitter'):
            try:
                sizes = [int(s) for s in splitter_sizes]
                # Only restore if left panel is at least 320 and right at least 250
                if len(sizes) >= 3 and sizes[0] >= 320 and sizes[2] >= 250:
                    self.main_splitter.setSizes(sizes)
                # Otherwise use defaults set in _setup_ui
            except (TypeError, ValueError):
                pass  # Use default sizes

        # Always start in standard (single-pane) view for a clean launch.
        # Quad view can be toggled via View > Quad View during the session.

    def _add_recent_file(self, file_path: str):
        """Add a file to the recent files list."""
        # Remove if already in list
        if file_path in self._recent_files:
            self._recent_files.remove(file_path)

        # Add to front
        self._recent_files.insert(0, file_path)

        # Trim to max size
        self._recent_files = self._recent_files[:self.MAX_RECENT_FILES]

    def _check_first_run(self):
        """Show onboarding dialog on first run."""
        first_run_complete = self._settings.value("first_run_complete", False)

        # Convert string "true"/"false" to bool if needed (QSettings quirk)
        if isinstance(first_run_complete, str):
            first_run_complete = first_run_complete.lower() == "true"

        if not first_run_complete:
            self._show_onboarding()

    def _show_onboarding(self):
        """Show first-run onboarding dialog."""
        self._show_quick_start()

    def _show_quick_start(self):
        """Show quick-start tutorial dialog (accessible from Help menu too)."""
        quick_start_text = """
<h2>Quick Start Guide</h2>

<table cellpadding="6" style="margin: 8px 0;">
<tr>
<td style="vertical-align:top;"><b>Step 1</b></td>
<td>Switch to <b>Layout Mode</b> (Ctrl+1) and pick a <b>Template</b> from the
left panel (Arena, Fortress, Cathedral, Maze), or stay in <b>Module Mode</b>
(Ctrl+2) to preview individual pieces.</td>
</tr>
<tr>
<td style="vertical-align:top;"><b>Step 2</b></td>
<td>Click <b>Random Dungeon</b> to auto-generate a layout, or click cells on
the grid to place modules manually. Press <b>R</b> to rotate before placing.</td>
</tr>
<tr>
<td style="vertical-align:top;"><b>Step 3</b></td>
<td>Click <b>Build</b> (Ctrl+G) to generate sealed brush geometry.
Preview it in 3D &mdash; use <b>WASD</b> + right-drag to fly around.</td>
</tr>
<tr>
<td style="vertical-align:top;"><b>Step 4</b></td>
<td>Click <b>.map</b> (Ctrl+E) to export. Choose idTech 1 (Quake) or
idTech 4 (Doom 3) format, then open the file in TrenchBroom or Radiant.</td>
</tr>
</table>

<h3>Useful Shortcuts</h3>
<table cellpadding="2">
<tr><td><b>Ctrl+G</b></td><td>Build geometry</td>
<td>&nbsp;&nbsp;</td>
<td><b>Ctrl+R</b></td><td>Random dungeon</td></tr>
<tr><td><b>Ctrl+E</b></td><td>Export .map</td>
<td>&nbsp;&nbsp;</td>
<td><b>Shift+Q</b></td><td>Quad view</td></tr>
<tr><td><b>F1</b></td><td>Full help</td>
<td>&nbsp;&nbsp;</td>
<td><b>F</b></td><td>Fit to bounds</td></tr>
</table>

<p><i>Access this guide anytime via Help &gt; Quick Start.</i></p>
"""
        msg = QMessageBox(self)
        msg.setWindowTitle("Quick Start")
        msg.setTextFormat(Qt.RichText)
        msg.setText(quick_start_text)
        msg.setIcon(QMessageBox.Information)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
