"""
Orthographic camera for axis-aligned views (Top, Front, Side).

Provides pan/zoom navigation for 2D orthographic projections of 3D geometry.
Used in quad-view layout for visualizing Z-levels and spatial relationships.
"""

import math
import numpy as np
from enum import Enum
from typing import Tuple, Optional


class ViewAxis(Enum):
    """Orthographic view axis alignment."""
    TOP = "top"      # XY plane, looking down -Z
    FRONT = "front"  # XZ plane, looking down -Y
    SIDE = "side"    # YZ plane, looking down -X


class OrthographicCamera:
    """Orthographic camera for axis-locked views.

    Provides pan/zoom navigation without rotation. The view is always
    aligned to one of the three principal axes (TOP, FRONT, SIDE).

    Coordinate conventions:
    - TOP view: X horizontal, Y vertical, Z into screen
    - FRONT view: X horizontal, Z vertical, Y into screen
    - SIDE view: Y horizontal, Z vertical, X into screen
    """

    def __init__(self, view_axis: ViewAxis = ViewAxis.TOP):
        self.view_axis = view_axis

        # Camera center point (look-at target in world coords)
        self.center = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Zoom level (world units visible in half-height of viewport)
        # Higher value = zoomed out, lower = zoomed in
        self.zoom = 500.0

        # Viewport dimensions (updated on resize)
        self.viewport_width = 800
        self.viewport_height = 600

        # Near/far planes (large range to capture all geometry)
        self.near = -10000.0
        self.far = 10000.0

        # Pan sensitivity
        self.pan_sensitivity = 1.0

    def get_view_matrix(self) -> np.ndarray:
        """Calculate the view matrix for this orthographic view.

        The view matrix transforms world coordinates to view coordinates,
        with the camera looking along the appropriate axis.

        Returns:
            4x4 view matrix as numpy array
        """
        # Build look-at matrix based on view axis
        if self.view_axis == ViewAxis.TOP:
            # Looking down -Z, up is +Y
            eye = self.center + np.array([0.0, 0.0, 1000.0], dtype=np.float32)
            target = self.center
            up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        elif self.view_axis == ViewAxis.FRONT:
            # Looking down -Y, up is +Z
            eye = self.center + np.array([0.0, 1000.0, 0.0], dtype=np.float32)
            target = self.center
            up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:  # SIDE
            # Looking down -X, up is +Z
            eye = self.center + np.array([1000.0, 0.0, 0.0], dtype=np.float32)
            target = self.center
            up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        return self._look_at(eye, target, up)

    def _look_at(self, eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
        """Build a look-at view matrix.

        Args:
            eye: Camera position
            target: Point to look at
            up: Up vector

        Returns:
            4x4 view matrix
        """
        forward = target - eye
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            # Degenerate case - pick arbitrary right
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            right = right / right_norm

        cam_up = np.cross(right, forward)

        view = np.eye(4, dtype=np.float32)
        view[0, :3] = right
        view[1, :3] = cam_up
        view[2, :3] = -forward
        view[0, 3] = -np.dot(right, eye)
        view[1, 3] = -np.dot(cam_up, eye)
        view[2, 3] = np.dot(forward, eye)

        return view

    def get_projection_matrix(self) -> np.ndarray:
        """Calculate the orthographic projection matrix.

        Uses glOrtho-style matrix with symmetric viewing volume.

        Returns:
            4x4 projection matrix as numpy array
        """
        aspect = self.viewport_width / max(self.viewport_height, 1)

        # Calculate orthographic bounds
        half_h = self.zoom
        half_w = half_h * aspect

        left = -half_w
        right = half_w
        bottom = -half_h
        top = half_h
        near = self.near
        far = self.far

        # glOrtho matrix
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = 2.0 / (right - left)
        proj[1, 1] = 2.0 / (top - bottom)
        proj[2, 2] = -2.0 / (far - near)
        proj[0, 3] = -(right + left) / (right - left)
        proj[1, 3] = -(top + bottom) / (top - bottom)
        proj[2, 3] = -(far + near) / (far - near)
        proj[3, 3] = 1.0

        return proj

    def pan(self, delta_x: float, delta_y: float):
        """Pan the view by pixel delta.

        Movement is in screen space and converted to world units based
        on current zoom level.

        Args:
            delta_x: Horizontal pixel movement (positive = right)
            delta_y: Vertical pixel movement (positive = down in screen coords)
        """
        # Convert pixel delta to world units
        aspect = self.viewport_width / max(self.viewport_height, 1)
        world_per_pixel_y = (2.0 * self.zoom) / self.viewport_height
        world_per_pixel_x = world_per_pixel_y  # Square pixels

        dx = -delta_x * world_per_pixel_x * self.pan_sensitivity
        dy = delta_y * world_per_pixel_y * self.pan_sensitivity

        # Apply movement based on view axis
        if self.view_axis == ViewAxis.TOP:
            # Screen X -> World X, Screen Y -> World -Y
            self.center[0] += dx
            self.center[1] += dy
        elif self.view_axis == ViewAxis.FRONT:
            # Screen X -> World X, Screen Y -> World Z
            self.center[0] += dx
            self.center[2] += dy
        else:  # SIDE
            # Screen X -> World Y, Screen Y -> World Z
            self.center[1] += dx
            self.center[2] += dy

    def zoom_at(self, factor: float, screen_x: float, screen_y: float):
        """Zoom centered on a screen position.

        Keeps the world point under the cursor stationary during zoom.

        Args:
            factor: Zoom factor (>1 = zoom in, <1 = zoom out)
            screen_x: Screen X coordinate (0 = left edge)
            screen_y: Screen Y coordinate (0 = top edge)
        """
        # Get world position under cursor before zoom
        world_before = self.screen_to_world(screen_x, screen_y)

        # Apply zoom (clamp to reasonable range)
        self.zoom = max(10.0, min(10000.0, self.zoom / factor))

        # Get world position under cursor after zoom
        world_after = self.screen_to_world(screen_x, screen_y)

        # Adjust center to keep point under cursor stationary
        if world_before is not None and world_after is not None:
            delta = world_before - world_after
            self.center += delta

    def zoom_uniform(self, factor: float):
        """Zoom centered on viewport center.

        Args:
            factor: Zoom factor (>1 = zoom in, <1 = zoom out)
        """
        self.zoom_at(factor, self.viewport_width / 2, self.viewport_height / 2)

    def screen_to_world(self, screen_x: float, screen_y: float) -> Optional[np.ndarray]:
        """Convert screen coordinates to world coordinates.

        Returns the 3D world point on the view plane at the given screen position.

        Args:
            screen_x: Screen X coordinate (0 = left edge)
            screen_y: Screen Y coordinate (0 = top edge)

        Returns:
            3D world coordinates as numpy array, or None on error
        """
        # Convert to normalized device coordinates (-1 to 1)
        ndc_x = (2.0 * screen_x / self.viewport_width) - 1.0
        ndc_y = 1.0 - (2.0 * screen_y / self.viewport_height)  # Flip Y

        # Convert to world units relative to center
        aspect = self.viewport_width / max(self.viewport_height, 1)
        half_h = self.zoom
        half_w = half_h * aspect

        world_offset_h = ndc_x * half_w
        world_offset_v = ndc_y * half_h

        # Apply to appropriate axes based on view
        world = self.center.copy()

        if self.view_axis == ViewAxis.TOP:
            world[0] += world_offset_h
            world[1] += world_offset_v
        elif self.view_axis == ViewAxis.FRONT:
            world[0] += world_offset_h
            world[2] += world_offset_v
        else:  # SIDE
            world[1] += world_offset_h
            world[2] += world_offset_v

        return world

    def world_to_screen(self, world_point: np.ndarray) -> Tuple[float, float]:
        """Convert world coordinates to screen coordinates.

        Args:
            world_point: 3D world coordinates

        Returns:
            Tuple of (screen_x, screen_y)
        """
        # Get offset from center in view axes
        aspect = self.viewport_width / max(self.viewport_height, 1)
        half_h = self.zoom
        half_w = half_h * aspect

        if self.view_axis == ViewAxis.TOP:
            offset_h = world_point[0] - self.center[0]
            offset_v = world_point[1] - self.center[1]
        elif self.view_axis == ViewAxis.FRONT:
            offset_h = world_point[0] - self.center[0]
            offset_v = world_point[2] - self.center[2]
        else:  # SIDE
            offset_h = world_point[1] - self.center[1]
            offset_v = world_point[2] - self.center[2]

        # Convert to NDC
        ndc_x = offset_h / half_w
        ndc_y = offset_v / half_h

        # Convert to screen coords
        screen_x = (ndc_x + 1.0) * self.viewport_width / 2.0
        screen_y = (1.0 - ndc_y) * self.viewport_height / 2.0

        return (screen_x, screen_y)

    def set_viewport(self, width: int, height: int):
        """Update viewport dimensions.

        Args:
            width: Viewport width in pixels
            height: Viewport height in pixels
        """
        self.viewport_width = max(1, width)
        self.viewport_height = max(1, height)

    def fit_to_bounds(self, min_pt: Tuple[float, float, float],
                      max_pt: Tuple[float, float, float], padding: float = 1.1):
        """Fit view to show the given bounding box.

        Args:
            min_pt: Minimum corner of bounding box
            max_pt: Maximum corner of bounding box
            padding: Multiplier for padding (1.1 = 10% padding)
        """
        # Calculate center
        center = np.array([
            (min_pt[0] + max_pt[0]) / 2,
            (min_pt[1] + max_pt[1]) / 2,
            (min_pt[2] + max_pt[2]) / 2
        ], dtype=np.float32)

        self.center = center

        # Calculate required zoom based on view axis
        aspect = self.viewport_width / max(self.viewport_height, 1)

        if self.view_axis == ViewAxis.TOP:
            width = max_pt[0] - min_pt[0]
            height = max_pt[1] - min_pt[1]
        elif self.view_axis == ViewAxis.FRONT:
            width = max_pt[0] - min_pt[0]
            height = max_pt[2] - min_pt[2]
        else:  # SIDE
            width = max_pt[1] - min_pt[1]
            height = max_pt[2] - min_pt[2]

        # Calculate zoom to fit both dimensions
        zoom_for_height = (height / 2) * padding
        zoom_for_width = (width / 2 / aspect) * padding

        self.zoom = max(zoom_for_height, zoom_for_width, 10.0)

    def get_pan_axis_value(self, axis: str) -> float:
        """Get the current pan value for a specific world axis.

        Used for synchronizing pan between views that share an axis.

        Args:
            axis: 'x', 'y', or 'z'

        Returns:
            The center coordinate on that axis
        """
        if axis == 'x':
            return self.center[0]
        elif axis == 'y':
            return self.center[1]
        else:
            return self.center[2]

    def set_pan_axis_value(self, axis: str, value: float):
        """Set the pan value for a specific world axis.

        Used for synchronizing pan between views that share an axis.

        Args:
            axis: 'x', 'y', or 'z'
            value: The center coordinate to set
        """
        if axis == 'x':
            self.center[0] = value
        elif axis == 'y':
            self.center[1] = value
        else:
            self.center[2] = value

    def get_view_label(self) -> str:
        """Get a human-readable label for this view.

        Returns:
            View label string (e.g., "Top (XY)")
        """
        labels = {
            ViewAxis.TOP: "Top (XY)",
            ViewAxis.FRONT: "Front (XZ)",
            ViewAxis.SIDE: "Side (YZ)",
        }
        return labels.get(self.view_axis, "Unknown")

    def get_axis_labels(self) -> Tuple[str, str]:
        """Get axis labels for the horizontal and vertical screen axes.

        Returns:
            Tuple of (horizontal_label, vertical_label)
        """
        labels = {
            ViewAxis.TOP: ("X", "Y"),
            ViewAxis.FRONT: ("X", "Z"),
            ViewAxis.SIDE: ("Y", "Z"),
        }
        return labels.get(self.view_axis, ("?", "?"))
