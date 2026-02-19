"""
Preview module for real-time 3D visualization of idTech geometry.

Provides a QOpenGLWidget-based preview with orbit camera controls
and real-time brush-to-mesh conversion.
"""

from .preview_widget import PreviewWidget
from .camera import OrbitCamera
from .mesh_builder import MeshBuilder
from .renderer import PreviewRenderer, RenderMode
from .texture_manager import TextureManager

__all__ = ['PreviewWidget', 'OrbitCamera', 'MeshBuilder', 'PreviewRenderer', 'RenderMode', 'TextureManager']
