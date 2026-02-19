"""
FPS-style camera for 3D preview navigation.

Provides idTech/TrenchBroom-style fly camera with mouselook and WASD movement.
"""

import math
import numpy as np
from typing import Tuple, Optional


class OrbitCamera:
    """FPS-style fly camera (named OrbitCamera for compatibility).

    This is a true FPS camera where:
    - Position is where the camera is located in space
    - Yaw/Pitch control where you're looking
    - WASD moves you through the scene
    - Mouse rotates your view (mouselook)
    """

    def __init__(self):
        # Camera position in world space
        self.position = np.array([0.0, 500.0, 100.0], dtype=np.float32)

        # Look direction (degrees)
        self.yaw = 180.0      # Horizontal angle (0 = +Y, 90 = +X, 180 = -Y)
        self.pitch = -10.0    # Vertical angle (positive = looking up)

        # Limits
        self.min_pitch = -89.0
        self.max_pitch = 89.0

        # Sensitivity
        self.rotate_sensitivity = 0.3
        self.move_speed = 8.0

        # View/projection matrices
        self.fov = 45.0
        self.aspect = 1.0
        self.near = 1.0
        self.far = 10000.0

        # For compatibility with fit_to_bounds
        self._bounds_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def get_forward(self) -> np.ndarray:
        """Get the direction the camera is looking (unit vector)."""
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)

        # Convert yaw/pitch to direction vector
        # yaw=0 looks toward +Y, yaw=90 looks toward +X
        return np.array([
            math.cos(pitch_rad) * math.sin(yaw_rad),
            math.cos(pitch_rad) * math.cos(yaw_rad),
            math.sin(pitch_rad)
        ], dtype=np.float32)

    def get_right(self) -> np.ndarray:
        """Get the right vector (perpendicular to forward, on XY plane)."""
        yaw_rad = math.radians(self.yaw)
        # Right is 90 degrees clockwise from forward on XY plane
        return np.array([
            math.cos(yaw_rad),
            -math.sin(yaw_rad),
            0.0
        ], dtype=np.float32)

    def get_position(self) -> np.ndarray:
        """Get camera position in world space."""
        return self.position.copy()

    def get_view_matrix(self) -> np.ndarray:
        """Calculate the view matrix."""
        eye = self.position
        forward = self.get_forward()
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # Right vector
        right = np.cross(forward, world_up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            # Looking straight up/down - use yaw to determine right
            right = self.get_right()
        else:
            right = right / right_norm

        # Camera up (perpendicular to forward and right)
        up = np.cross(right, forward)

        # Build view matrix
        view = np.eye(4, dtype=np.float32)
        view[0, :3] = right
        view[1, :3] = up
        view[2, :3] = -forward
        view[0, 3] = -np.dot(right, eye)
        view[1, 3] = -np.dot(up, eye)
        view[2, 3] = np.dot(forward, eye)

        return view

    def get_projection_matrix(self) -> np.ndarray:
        """Calculate the perspective projection matrix."""
        f = 1.0 / math.tan(math.radians(self.fov) / 2.0)
        nf = 1.0 / (self.near - self.far)

        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = f / self.aspect
        proj[1, 1] = f
        proj[2, 2] = (self.far + self.near) * nf
        proj[2, 3] = 2.0 * self.far * self.near * nf
        proj[3, 2] = -1.0

        return proj

    def rotate(self, delta_x: float, delta_y: float):
        """Mouselook - rotate the camera view direction.

        This rotates WHERE YOU'RE LOOKING, not where you are.
        Just like idTech/TrenchBroom mouselook.
        """
        self.yaw -= delta_x * self.rotate_sensitivity
        self.pitch -= delta_y * self.rotate_sensitivity  # Mouse up = look up

        # Clamp pitch to avoid gimbal lock
        self.pitch = max(self.min_pitch, min(self.max_pitch, self.pitch))

        # Wrap yaw
        self.yaw = self.yaw % 360.0

    def pan(self, delta_x: float, delta_y: float):
        """Pan camera (move perpendicular to view direction)."""
        right = self.get_right()

        # Camera up for vertical panning
        forward = self.get_forward()
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        cam_up = np.cross(right, forward)

        # Scale movement
        scale = 2.0
        self.position -= right * delta_x * scale
        self.position += cam_up * delta_y * scale

    def zoom(self, delta: float):
        """Zoom by moving forward/backward."""
        forward = self.get_forward()
        self.position += forward * delta * 50.0

    def fit_to_bounds(self, min_pt: Tuple[float, float, float],
                      max_pt: Tuple[float, float, float]):
        """Position camera to see the given bounding box."""
        # Calculate center
        center = np.array([
            (min_pt[0] + max_pt[0]) / 2,
            (min_pt[1] + max_pt[1]) / 2,
            (min_pt[2] + max_pt[2]) / 2
        ], dtype=np.float32)

        self._bounds_center = center

        # Calculate bounding sphere radius
        size = np.array([
            max_pt[0] - min_pt[0],
            max_pt[1] - min_pt[1],
            max_pt[2] - min_pt[2]
        ])
        radius = np.linalg.norm(size) / 2

        # Calculate distance to fit sphere in view
        fov_rad = math.radians(self.fov)
        distance = radius / math.sin(fov_rad / 2) * 1.5

        # Position camera behind and above center, looking at center
        # Place camera at +Y from center (idTech convention)
        self.position = center + np.array([0.0, distance, distance * 0.3], dtype=np.float32)

        # Look toward center
        to_center = center - self.position
        dist_xy = math.sqrt(to_center[0]**2 + to_center[1]**2)

        self.yaw = math.degrees(math.atan2(to_center[0], to_center[1]))
        self.pitch = math.degrees(math.atan2(to_center[2], dist_xy))

    def set_preset_view(self, preset: str):
        """Set camera to a preset viewpoint relative to bounds center."""
        center = self._bounds_center
        dist = 500.0  # Default distance

        presets = {
            # (position offset, yaw, pitch)
            'front': (np.array([0, dist, 0]), 180.0, 0.0),      # Looking from +Y toward -Y
            'back': (np.array([0, -dist, 0]), 0.0, 0.0),        # Looking from -Y toward +Y
            'left': (np.array([-dist, 0, 0]), 90.0, 0.0),       # Looking from -X toward +X
            'right': (np.array([dist, 0, 0]), -90.0, 0.0),      # Looking from +X toward -X
            'top': (np.array([0, 0, dist]), 0.0, -89.0),        # Looking from +Z down
            'bottom': (np.array([0, 0, -dist]), 0.0, 89.0),     # Looking from -Z up
            'iso': (np.array([dist*0.7, dist*0.7, dist*0.5]), -135.0, -20.0),
        }

        if preset in presets:
            offset, yaw, pitch = presets[preset]
            self.position = center + offset.astype(np.float32)
            self.yaw = yaw
            self.pitch = pitch

    def set_aspect(self, width: int, height: int):
        """Update aspect ratio for projection matrix."""
        if height > 0:
            self.aspect = width / height

    def move_continuous(self, forward_amount: float, right_amount: float, up_amount: float = 0.0):
        """Move camera in FPS/noclip style.

        Movement is relative to where you're looking:
        - W/S: Move in the direction you're facing (including up/down tilt)
        - A/D: Strafe left/right (always horizontal)
        - Q/E: Move down/up (world vertical)

        Args:
            forward_amount: Movement in look direction (-1 to 1)
            right_amount: Strafe movement (-1 to 1, positive = right)
            up_amount: Vertical movement (-1 to 1, positive = up)
        """
        forward = self.get_forward()
        right = self.get_right()
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # Calculate movement
        movement = np.zeros(3, dtype=np.float32)
        movement += forward * forward_amount * self.move_speed
        movement += right * right_amount * self.move_speed
        movement += world_up * up_amount * self.move_speed

        self.position += movement
