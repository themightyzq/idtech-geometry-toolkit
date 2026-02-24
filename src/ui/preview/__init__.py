"""
Preview module for real-time 3D visualization of idTech geometry.

Provides a QOpenGLWidget-based preview with orbit camera controls
and real-time brush-to-mesh conversion.

Includes quad-view support for multi-floor visualization with
orthographic Top/Front/Side views.
"""

from .preview_widget import PreviewWidget
from .camera import OrbitCamera
from .mesh_builder import MeshBuilder
from .renderer import PreviewRenderer, RenderMode
from .texture_manager import TextureManager
from .orthographic_camera import OrthographicCamera, ViewAxis
from .ortho_view_widget import OrthoViewWidget
from .quad_view_widget import QuadViewWidget, ViewSynchronizer

__all__ = [
    'PreviewWidget',
    'OrbitCamera',
    'MeshBuilder',
    'PreviewRenderer',
    'RenderMode',
    'TextureManager',
    'OrthographicCamera',
    'ViewAxis',
    'OrthoViewWidget',
    'QuadViewWidget',
    'ViewSynchronizer',
]
