"""
Texture manager for loading, caching, and binding textures in the 3D preview.

Uses PIL/Pillow for image loading and supports TGA, PNG, JPG, and BMP formats.
"""

import os
from typing import Dict, Optional, Set

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from OpenGL.GL import *
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False

import sys as _sys
DEBUG_TEXTURE = False  # Set to True to enable debug output

def _debug_tex(msg):
    if DEBUG_TEXTURE:
        print(f"[TEXTURE] {msg}")
        _sys.stdout.flush()


class TextureManager:
    """Load, cache, and bind textures for OpenGL rendering."""

    def __init__(self):
        self._texture_cache: Dict[str, int] = {}  # path -> GL texture ID
        self._fallback_texture: Optional[int] = None
        self._warned_textures: Set[str] = set()  # Track textures we've warned about
        self._initialized = False
        self._use_legacy = False  # True for OpenGL 2.1 (no glGenerateMipmap)

    def initialize(self, use_legacy: bool = False) -> bool:
        """Initialize the texture manager. Call after GL context is current.

        Args:
            use_legacy: True if using legacy OpenGL 2.1 (no glGenerateMipmap)
        """
        _debug_tex(f"initialize() called, use_legacy={use_legacy}")

        if not OPENGL_AVAILABLE:
            print("TextureManager: OpenGL not available")
            return False

        if not PIL_AVAILABLE:
            print("TextureManager: PIL/Pillow not available - textures disabled")
            return False

        if self._initialized:
            _debug_tex("Already initialized")
            return True

        self._use_legacy = use_legacy
        _debug_tex(f"Legacy mode: {self._use_legacy}")

        # Create fallback texture (1x1 white)
        self._fallback_texture = self._create_fallback_texture()
        _debug_tex(f"Fallback texture created: ID={self._fallback_texture}")

        self._initialized = True
        _debug_tex("Initialization complete")
        return True

    def _create_fallback_texture(self) -> int:
        """Create a 1x1 white fallback texture for missing textures."""
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)

        # 1x1 white pixel
        pixel_data = bytes([255, 255, 255, 255])

        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA,
            1, 1, 0,
            GL_RGBA, GL_UNSIGNED_BYTE, pixel_data
        )

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

        glBindTexture(GL_TEXTURE_2D, 0)
        return texture_id

    def load_texture(self, file_path: str) -> Optional[int]:
        """Load a texture from file and upload to GPU.

        Args:
            file_path: Path to the texture file (TGA, PNG, JPG, BMP)

        Returns:
            OpenGL texture ID, or None if loading failed
        """
        _debug_tex(f"load_texture({file_path})")

        if not self._initialized:
            _debug_tex("ERROR: Not initialized")
            return None

        if not os.path.isfile(file_path):
            if file_path not in self._warned_textures:
                print(f"TextureManager: File not found: {file_path}")
                self._warned_textures.add(file_path)
            return None

        try:
            # Load image with PIL
            image = Image.open(file_path)
            _debug_tex(f"Loaded image: {image.size} mode={image.mode}")

            # Convert to RGBA
            if image.mode != 'RGBA':
                image = image.convert('RGBA')

            # Flip vertically for OpenGL (bottom-left origin)
            image = image.transpose(Image.FLIP_TOP_BOTTOM)

            width, height = image.size
            pixel_data = image.tobytes()
            _debug_tex(f"Converted to RGBA: {width}x{height}, {len(pixel_data)} bytes")

            # Create GL texture
            texture_id = glGenTextures(1)
            _debug_tex(f"Created texture ID: {texture_id}")

            glBindTexture(GL_TEXTURE_2D, texture_id)

            glTexImage2D(
                GL_TEXTURE_2D, 0, GL_RGBA,
                width, height, 0,
                GL_RGBA, GL_UNSIGNED_BYTE, pixel_data
            )
            _debug_tex("Uploaded texture data")

            # Handle mipmaps based on OpenGL version
            if self._use_legacy:
                # Legacy OpenGL 2.1: no glGenerateMipmap, use simpler filtering
                _debug_tex("Using legacy mode - no mipmaps")
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            else:
                # Modern OpenGL: generate mipmaps for better quality
                try:
                    glGenerateMipmap(GL_TEXTURE_2D)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
                    _debug_tex("Generated mipmaps")
                except Exception as mip_err:
                    _debug_tex(f"Mipmap generation failed: {mip_err}, using linear filter")
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

            glBindTexture(GL_TEXTURE_2D, 0)

            _debug_tex(f"Texture loaded successfully: ID={texture_id}")
            return texture_id

        except Exception as e:
            if file_path not in self._warned_textures:
                print(f"TextureManager: Failed to load {file_path}: {e}")
                self._warned_textures.add(file_path)
            _debug_tex(f"EXCEPTION: {e}")
            return None

    def get_or_load(self, texture_path: str) -> int:
        """Get a cached texture or load it if not cached.

        Args:
            texture_path: Path to the texture file

        Returns:
            OpenGL texture ID (fallback texture if loading failed)
        """
        _debug_tex(f"get_or_load({texture_path})")

        if not self._initialized:
            _debug_tex("ERROR: Not initialized, returning 0")
            return 0

        # Check cache first
        abs_path = os.path.abspath(texture_path)
        if abs_path in self._texture_cache:
            cached_id = self._texture_cache[abs_path]
            _debug_tex(f"Cache hit: ID={cached_id}")
            return cached_id

        # Try to load
        _debug_tex("Cache miss, loading...")
        texture_id = self.load_texture(abs_path)
        if texture_id is not None:
            self._texture_cache[abs_path] = texture_id
            _debug_tex(f"Loaded and cached: ID={texture_id}")
            return texture_id

        # Return fallback
        fallback = self._fallback_texture if self._fallback_texture else 0
        _debug_tex(f"Load failed, returning fallback: ID={fallback}")
        return fallback

    def bind_texture(self, texture_id: int, texture_unit: int = 0):
        """Bind a texture to a texture unit.

        Args:
            texture_id: OpenGL texture ID
            texture_unit: Texture unit (default 0)
        """
        if not OPENGL_AVAILABLE:
            return

        glActiveTexture(GL_TEXTURE0 + texture_unit)
        glBindTexture(GL_TEXTURE_2D, texture_id)

    def unbind_texture(self, texture_unit: int = 0):
        """Unbind texture from a texture unit.

        Args:
            texture_unit: Texture unit (default 0)
        """
        if not OPENGL_AVAILABLE:
            return

        glActiveTexture(GL_TEXTURE0 + texture_unit)
        glBindTexture(GL_TEXTURE_2D, 0)

    def get_fallback_texture(self) -> int:
        """Get the fallback texture ID.

        Returns:
            OpenGL texture ID for 1x1 white fallback
        """
        return self._fallback_texture if self._fallback_texture else 0

    def is_valid_texture(self, texture_id: int) -> bool:
        """Check if a texture ID is valid and not the fallback.

        Args:
            texture_id: OpenGL texture ID to check

        Returns:
            True if valid non-fallback texture
        """
        return (
            texture_id != 0 and
            texture_id != self._fallback_texture and
            glIsTexture(texture_id)
        )

    def clear_cache(self):
        """Clear all cached textures from GPU memory."""
        if not OPENGL_AVAILABLE:
            return

        for texture_id in self._texture_cache.values():
            try:
                if texture_id and glIsTexture(texture_id):
                    glDeleteTextures(1, [texture_id])
            except Exception:
                pass  # GL context may be destroyed

        self._texture_cache.clear()
        self._warned_textures.clear()

    def cleanup(self):
        """Clean up all OpenGL resources."""
        if not OPENGL_AVAILABLE:
            return

        self.clear_cache()

        try:
            if self._fallback_texture and glIsTexture(self._fallback_texture):
                glDeleteTextures(1, [self._fallback_texture])
        except Exception:
            pass  # GL context may be destroyed
        self._fallback_texture = None

        self._initialized = False
