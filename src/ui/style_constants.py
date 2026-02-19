"""
Centralized style constants for the idTech Geometry Toolkit UI.

This module defines the design system tokens for consistent styling across
all widgets. Use these constants instead of hardcoded values.

Supports high-contrast mode for accessibility (requires app restart to apply).
"""

from PyQt5.QtCore import QSettings

# =============================================================================
# THEME SELECTION (persisted via QSettings)
# =============================================================================

_settings = QSettings('IdTechGeometryToolkit', 'QuakeLevelGenerator')
HIGH_CONTRAST_MODE = _settings.value('high_contrast', False, type=bool)

# =============================================================================
# COLOR PALETTE - Semantic roles
# =============================================================================

# Primary action colors
PRIMARY_ACTION = "#4CAF50"        # Green - Generate, Build, Primary buttons
PRIMARY_ACTION_HOVER = "#45a049"  # Darker green for hover
PRIMARY_ACTION_DARK = "#388E3C"   # Even darker for pressed states

# Secondary action colors
SECONDARY_ACTION = "#2196F3"      # Blue - Export, navigation
SECONDARY_ACTION_HOVER = "#1976D2"

# Warning/Caution colors
WARNING_COLOR = "#FF9800"         # Orange - Warnings, Export OBJ
WARNING_HOVER = "#F57C00"

# Danger/Error colors
DANGER_COLOR = "#f44336"          # Red - Cancel, errors, delete
DANGER_HOVER = "#da190b"

# Special action colors
SPECIAL_ACTION = "#9C27B0"        # Purple - Random dungeon
SPECIAL_ACTION_HOVER = "#7B1FA2"

# =============================================================================
# STATE COLORS
# =============================================================================

# Selected state (DISTINCT from primary action - blue indicates "currently active")
SELECTED_STATE = "#2196F3"        # Blue - "currently active/selected"
SELECTED_STATE_HOVER = "#1976D2"  # Darker blue for hover on selected

# Focus indicator (DISTINCT from selection - lighter blue for keyboard focus)
FOCUS_COLOR = "#64B5F6"           # Light blue - keyboard focus rings

# Valid/Success
SUCCESS_COLOR = "#4caf50"

# Invalid/Error
ERROR_COLOR = "#f44336"

# =============================================================================
# BACKGROUND COLORS
# =============================================================================

# Surface colors (dark theme)
BG_DARKEST = "#1e1e1e"            # Deepest background (text areas, lists)
BG_DARK = "#2d2d2d"               # Standard dark background
BG_MEDIUM = "#353535"             # Slightly lighter (group boxes)
BG_LIGHT = "#404040"              # Interactive elements (spinboxes, combos)
BG_LIGHTER = "#444444"            # Buttons, hover states
BG_HIGHLIGHT = "#4a4a4a"          # Hover highlight
BG_PRESSED = "#353535"            # Pressed state

# =============================================================================
# TEXT COLORS
# =============================================================================

# High contrast text (primary content)
TEXT_PRIMARY = "#e0e0e0"          # Main text - 5.3:1 on #2d2d2d
TEXT_SECONDARY = "#c0c0c0"        # Secondary text - 4.5:1+ on #2d2d2d
TEXT_TERTIARY = "#a0a0a0"         # Muted text - 4.5:1 on #2d2d2d (WCAG AA minimum)
TEXT_DISABLED = "#888888"         # Disabled state - lower contrast is acceptable

# Colored text for status
TEXT_SUCCESS = "#88ff88"          # Bright green for success messages
TEXT_WARNING = "#ffff88"          # Yellow for warnings
TEXT_ERROR = "#ff8888"            # Bright red for errors

# =============================================================================
# BORDER COLORS
# =============================================================================

BORDER_DARK = "#444444"           # Dark borders
BORDER_MEDIUM = "#555555"         # Standard borders
BORDER_LIGHT = "#666666"          # Lighter borders

# =============================================================================
# TYPOGRAPHY SCALE
# =============================================================================

# Base font size - use 'pt' for better DPI awareness on different displays
# PyQt5 on macOS renders pt sizes more consistently across Retina/non-Retina
FONT_SIZE_BASE = 12  # Base size in points

# Font sizes using pt for better cross-platform rendering
# pt units are DPI-aware and scale better than px
FONT_SIZE_XS = "10pt"             # Minimum size, small labels only
FONT_SIZE_SM = "11pt"             # Standard body text
FONT_SIZE_MD = "12pt"             # Emphasized text, default controls
FONT_SIZE_LG = "13pt"             # Larger buttons, section headers
FONT_SIZE_XL = "15pt"             # Large headers, titles

# Legacy px values for compatibility (deprecated, use pt versions above)
FONT_SIZE_XS_PX = "11px"
FONT_SIZE_SM_PX = "12px"
FONT_SIZE_MD_PX = "13px"
FONT_SIZE_LG_PX = "14px"
FONT_SIZE_XL_PX = "16px"

# =============================================================================
# SPACING SCALE
# =============================================================================

SPACING_XS = 4                    # Tight spacing
SPACING_SM = 8                    # Standard margin/padding
SPACING_MD = 12                   # Group spacing
SPACING_LG = 16                   # Section spacing
SPACING_XL = 24                   # Large section spacing

# =============================================================================
# COMPONENT CONSTANTS
# =============================================================================

# Border radius
BORDER_RADIUS_SM = "3px"
BORDER_RADIUS_MD = "4px"
BORDER_RADIUS_LG = "6px"
BORDER_RADIUS_XL = "12px"         # Pills, circular buttons

# Focus outline
FOCUS_OUTLINE = f"2px solid {FOCUS_COLOR}"
FOCUS_OUTLINE_OFFSET = "2px"

# =============================================================================
# HIT TARGET SIZES (Accessibility Guidelines)
# =============================================================================
# WCAG 2.2 recommends minimum 24x24px, Apple HIG recommends 44x44pt for touch
# For desktop apps, we use 28-32px as a reasonable minimum

HIT_TARGET_MIN = 28               # Minimum clickable size in pixels
HIT_TARGET_COMFORTABLE = 32       # Comfortable click target
HIT_TARGET_LARGE = 36             # Large touch-friendly target

# Button minimum widths
BUTTON_MIN_WIDTH = 32             # Minimum button width (spinbox +/-)
BUTTON_MIN_WIDTH_ICON = 28        # Minimum for icon-only buttons
BUTTON_MIN_WIDTH_TEXT = 64        # Minimum for text buttons

# Input field minimum widths
INPUT_MIN_WIDTH_SM = 60           # Small inputs (numbers)
INPUT_MIN_WIDTH_MD = 100          # Medium inputs
INPUT_MIN_WIDTH_LG = 150          # Large inputs (text fields)

# =============================================================================
# CATEGORY COLORS (for primitives)
# =============================================================================

CATEGORY_HALLS = "#4682B4"        # Steel blue
CATEGORY_ROOMS = "#9467BD"        # Purple
CATEGORY_STRUCTURAL = "#8B7765"   # Brown
CATEGORY_CONNECTIVE = "#2CA02C"   # Green

# =============================================================================
# VALIDATION SEVERITY COLORS
# =============================================================================

SEVERITY_ERROR = "#f44336"        # Red
SEVERITY_WARNING = "#ff9800"      # Orange
SEVERITY_INFO = "#2196F3"         # Blue
SEVERITY_SUCCESS = "#4caf50"      # Green

# =============================================================================
# HIGH-CONTRAST MODE OVERRIDES (WCAG AAA - 7:1+ contrast ratios)
# =============================================================================
# When HIGH_CONTRAST_MODE is enabled, override key colors for maximum visibility.
# This mode uses pure black backgrounds, pure white text, and bright accent colors.

if HIGH_CONTRAST_MODE:
    # Backgrounds - pure black for maximum contrast
    BG_DARKEST = "#000000"
    BG_DARK = "#0a0a0a"
    BG_MEDIUM = "#141414"
    BG_LIGHT = "#1e1e1e"
    BG_LIGHTER = "#282828"
    BG_HIGHLIGHT = "#323232"
    BG_PRESSED = "#1e1e1e"

    # Text - pure white for maximum contrast
    TEXT_PRIMARY = "#ffffff"
    TEXT_SECONDARY = "#f0f0f0"
    TEXT_TERTIARY = "#e0e0e0"
    TEXT_DISABLED = "#a0a0a0"

    # Colored text - brighter for visibility
    TEXT_SUCCESS = "#00ff00"
    TEXT_WARNING = "#ffff00"
    TEXT_ERROR = "#ff4444"

    # Borders - more visible
    BORDER_DARK = "#666666"
    BORDER_MEDIUM = "#888888"
    BORDER_LIGHT = "#aaaaaa"

    # Focus - bright yellow for maximum visibility
    FOCUS_COLOR = "#ffff00"
    FOCUS_OUTLINE = f"2px solid {FOCUS_COLOR}"

    # Selection - bright cyan
    SELECTED_STATE = "#00ffff"
    SELECTED_STATE_HOVER = "#00cccc"

    # Primary action - brighter green
    PRIMARY_ACTION = "#00dd00"
    PRIMARY_ACTION_HOVER = "#00bb00"
    PRIMARY_ACTION_DARK = "#009900"

    # Secondary action - brighter blue
    SECONDARY_ACTION = "#4488ff"
    SECONDARY_ACTION_HOVER = "#2266dd"

    # Warning - brighter orange
    WARNING_COLOR = "#ffaa00"
    WARNING_HOVER = "#dd8800"

    # Danger - brighter red
    DANGER_COLOR = "#ff4444"
    DANGER_HOVER = "#dd2222"

    # Special action - brighter purple
    SPECIAL_ACTION = "#cc44ff"
    SPECIAL_ACTION_HOVER = "#aa22dd"

    # Validation severity - brighter
    SEVERITY_ERROR = "#ff4444"
    SEVERITY_WARNING = "#ffaa00"
    SEVERITY_INFO = "#4488ff"
    SEVERITY_SUCCESS = "#00dd00"


# =============================================================================
# ACCESSIBILITY HELPERS
# =============================================================================

def set_accessible(widget, name: str, description: str = "") -> None:
    """Set accessibility labels for a widget.

    This enables screen readers to announce the widget's purpose and state.

    NOTE: Disabled on macOS due to PyQt5 crash in QAccessible::queryAccessibleInterface
    when widgets with accessibility labels change state. The crash occurs in
    QAbstractButton::setChecked -> QAccessible::updateAccessibility.

    Args:
        widget: The PyQt5 widget to label.
        name: Short accessible name (e.g., "Build Geometry").
        description: Optional longer description of the widget's purpose.
    """
    # DISABLED: Causes crash on macOS with PyQt5 when checkbox state changes
    # widget.setAccessibleName(name)
    # if description:
    #     widget.setAccessibleDescription(description)
    pass
