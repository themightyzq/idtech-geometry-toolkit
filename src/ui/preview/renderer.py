"""
OpenGL renderer for the preview widget.

Handles shader compilation, buffer management, and draw calls.
"""

import ctypes
import numpy as np
from typing import Optional, Tuple, Dict
from enum import Enum

import sys as _sys
DEBUG_RENDERER = False  # Set to True to enable debug output

def _debug_render(msg):
    if DEBUG_RENDERER:
        print(f"[RENDERER] {msg}")
        _sys.stdout.flush()

try:
    from OpenGL.GL import *
    from OpenGL.GL import shaders as gl_shaders
    OPENGL_AVAILABLE = True
    _debug_render("OpenGL import successful")
except ImportError as e:
    OPENGL_AVAILABLE = False
    _debug_render(f"OpenGL import FAILED: {e}")

from .shaders import (
    VERTEX_SHADER, FRAGMENT_SHADER,
    VERTEX_SHADER_LEGACY, FRAGMENT_SHADER_LEGACY,
    WIREFRAME_VERTEX, WIREFRAME_FRAGMENT,
    WIREFRAME_VERTEX_LEGACY, WIREFRAME_FRAGMENT_LEGACY,
)
from .mesh_builder import RenderMesh, SurfaceMeshes, SurfaceType
from .texture_manager import TextureManager


class RenderMode(Enum):
    """Rendering mode for the preview."""
    SOLID = "solid"
    WIREFRAME = "wireframe"
    SOLID_WIREFRAME = "solid_wireframe"
    TEXTURED = "textured"
    TEXTURED_WIREFRAME = "textured_wireframe"


class PreviewRenderer:
    """OpenGL renderer for idTech geometry preview."""

    def __init__(self):
        self._initialized = False
        self._use_legacy = False

        # Shader programs
        self._solid_program: Optional[int] = None
        self._wireframe_program: Optional[int] = None

        # VAO/VBO for solid mesh
        self._solid_vao: Optional[int] = None
        self._solid_vbo: Optional[int] = None
        self._solid_ebo: Optional[int] = None
        self._solid_triangle_count = 0

        # VAO/VBO for wireframe
        self._wire_vao: Optional[int] = None
        self._wire_vbo: Optional[int] = None
        self._wire_ebo: Optional[int] = None
        self._wire_line_count = 0

        # Texture management
        self._texture_manager = TextureManager()
        self._current_texture_id: Optional[int] = None
        self._texture_path: Optional[str] = None

        # Per-surface texture support (all 5 surface types)
        self._surface_types = ["floor", "ceiling", "wall", "structural", "trim"]
        self._surface_textures: Dict[str, Optional[int]] = {s: None for s in self._surface_types}
        self._surface_texture_paths: Dict[str, Optional[str]] = {s: None for s in self._surface_types}

        # Per-surface mesh data (VBO/EBO per surface type)
        self._surface_vbos: Dict[str, Optional[int]] = {s: None for s in self._surface_types}
        self._surface_ebos: Dict[str, Optional[int]] = {s: None for s in self._surface_types}
        self._surface_triangle_counts: Dict[str, int] = {s: 0 for s in self._surface_types}
        self._use_surface_meshes = False  # True when using per-surface rendering

        # Render settings
        self.render_mode = RenderMode.SOLID
        self.background_color = (0.75, 0.75, 0.75, 1.0)  # Light gray
        self.wireframe_color = (0.0, 0.0, 0.0)
        self.light_color = (1.0, 1.0, 1.0)
        self.ambient_strength = 0.3
        self.specular_strength = 0.3

    def initialize(self) -> bool:
        """Initialize OpenGL resources. Call after GL context is current."""
        _debug_render("initialize() called")
        if not OPENGL_AVAILABLE:
            _debug_render("ERROR: OpenGL not available")
            print("OpenGL not available")
            return False

        if self._initialized:
            _debug_render("Already initialized")
            return True

        # Check OpenGL version
        try:
            version = glGetString(GL_VERSION)
            vendor = glGetString(GL_VENDOR)
            renderer_name = glGetString(GL_RENDERER)
            _debug_render(f"GL_VERSION: {version}")
            _debug_render(f"GL_VENDOR: {vendor}")
            _debug_render(f"GL_RENDERER: {renderer_name}")
        except Exception as e:
            _debug_render(f"ERROR getting GL strings: {e}")
            version = None

        if version:
            version_str = version.decode('utf-8')
            major = int(version_str.split('.')[0])
            if major < 3:
                self._use_legacy = True
                _debug_render(f"Using legacy OpenGL shaders (version: {version_str})")
                print(f"Using legacy OpenGL shaders (version: {version_str})")
            else:
                _debug_render(f"Using modern OpenGL shaders (version: {version_str})")
                print(f"Using modern OpenGL shaders (version: {version_str})")
        else:
            self._use_legacy = True
            _debug_render("No GL version string, using legacy mode")

        # Compile shaders
        _debug_render("Compiling shaders...")
        if not self._compile_shaders():
            _debug_render("ERROR: Shader compilation failed")
            return False
        _debug_render("Shaders compiled successfully")

        # Create VAOs/VBOs
        _debug_render("Creating buffers...")
        self._create_buffers()
        _debug_render("Buffers created")

        # Initialize texture manager with legacy flag
        _debug_render("Initializing texture manager...")
        self._texture_manager.initialize(use_legacy=self._use_legacy)
        _debug_render("Texture manager initialized")

        # Set up OpenGL state
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        # Don't enable face culling - our geometry may have mixed winding
        glDisable(GL_CULL_FACE)

        self._initialized = True
        _debug_render("Initialization complete")
        return True

    def _compile_shaders(self) -> bool:
        """Compile shader programs."""
        try:
            # Select shader source based on GL version
            if self._use_legacy:
                _debug_render("Using LEGACY shaders")
                vs_solid = VERTEX_SHADER_LEGACY
                fs_solid = FRAGMENT_SHADER_LEGACY
                vs_wire = WIREFRAME_VERTEX_LEGACY
                fs_wire = WIREFRAME_FRAGMENT_LEGACY
            else:
                _debug_render("Using MODERN shaders")
                vs_solid = VERTEX_SHADER
                fs_solid = FRAGMENT_SHADER
                vs_wire = WIREFRAME_VERTEX
                fs_wire = WIREFRAME_FRAGMENT

            # Compile solid shader (with explicit attribute binding)
            _debug_render("Compiling solid shader...")
            self._solid_program = self._compile_program(vs_solid, fs_solid, bind_attribs=True)
            if self._solid_program is None:
                _debug_render("ERROR: Solid shader compilation failed")
                return False
            _debug_render(f"Solid shader program: {self._solid_program}")

            # Compile wireframe shader
            _debug_render("Compiling wireframe shader...")
            self._wireframe_program = self._compile_program(vs_wire, fs_wire)
            if self._wireframe_program is None:
                _debug_render("ERROR: Wireframe shader compilation failed")
                return False
            _debug_render(f"Wireframe shader program: {self._wireframe_program}")

            return True

        except Exception as e:
            _debug_render(f"ERROR in _compile_shaders: {e}")
            print(f"Shader compilation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _compile_program(self, vs_source: str, fs_source: str, bind_attribs: bool = False) -> Optional[int]:
        """Compile and link a shader program."""
        try:
            _debug_render("  Compiling vertex shader...")
            vs = gl_shaders.compileShader(vs_source, GL_VERTEX_SHADER)
            _debug_render(f"  Vertex shader compiled: {vs}")

            _debug_render("  Compiling fragment shader...")
            fs = gl_shaders.compileShader(fs_source, GL_FRAGMENT_SHADER)
            _debug_render(f"  Fragment shader compiled: {fs}")

            # Manually create and link program to control attribute binding
            _debug_render("  Creating program...")
            program = glCreateProgram()
            glAttachShader(program, vs)
            glAttachShader(program, fs)

            # Bind attribute locations BEFORE linking (for GLSL 1.20 compatibility)
            if bind_attribs:
                _debug_render("  Binding attribute locations before linking...")
                glBindAttribLocation(program, 0, b"aPos")       # Position at location 0
                glBindAttribLocation(program, 1, b"aNormal")    # Normal at location 1
                glBindAttribLocation(program, 2, b"aTexCoord")  # UV at location 2
                glBindAttribLocation(program, 3, b"aColor")     # Color at location 3
                _debug_render("    aPos -> 0, aNormal -> 1, aTexCoord -> 2, aColor -> 3")

            _debug_render("  Linking program...")
            glLinkProgram(program)

            # Check link status
            link_status = glGetProgramiv(program, GL_LINK_STATUS)
            if link_status != GL_TRUE:
                log = glGetProgramInfoLog(program)
                _debug_render(f"  ERROR: Link failed: {log}")
                return None

            _debug_render(f"  Program linked: {program}")

            # Clean up shader objects
            glDeleteShader(vs)
            glDeleteShader(fs)

            return program
        except Exception as e:
            _debug_render(f"  ERROR compiling shader: {e}")
            print(f"Failed to compile shader: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _create_buffers(self):
        """Create VAOs and VBOs."""
        if self._use_legacy:
            # Legacy OpenGL doesn't use VAOs
            self._solid_vbo = glGenBuffers(1)
            self._solid_ebo = glGenBuffers(1)
            self._wire_vbo = glGenBuffers(1)
            self._wire_ebo = glGenBuffers(1)
        else:
            # Modern OpenGL with VAOs
            self._solid_vao = glGenVertexArrays(1)
            self._solid_vbo = glGenBuffers(1)
            self._solid_ebo = glGenBuffers(1)

            self._wire_vao = glGenVertexArrays(1)
            self._wire_vbo = glGenBuffers(1)
            self._wire_ebo = glGenBuffers(1)

    def upload_solid_mesh(self, mesh: RenderMesh):
        """Upload solid mesh data to GPU."""
        _debug_render(f"upload_solid_mesh: {mesh.triangle_count} triangles, {mesh.vertex_count} verts")
        if mesh.is_empty:
            _debug_render("upload_solid_mesh: mesh is empty")
            self._solid_triangle_count = 0
            return

        # Debug: print first vertex (now 11 floats: pos + norm + uv + color)
        if len(mesh.vertices) > 0:
            v = mesh.vertices[0]
            _debug_render(f"  First vertex: pos=({v[0]:.1f}, {v[1]:.1f}, {v[2]:.1f}) norm=({v[3]:.2f}, {v[4]:.2f}, {v[5]:.2f}) uv=({v[6]:.2f}, {v[7]:.2f}) color=({v[8]:.2f}, {v[9]:.2f}, {v[10]:.2f})")
            _debug_render(f"  Vertex array shape: {mesh.vertices.shape}, dtype: {mesh.vertices.dtype}")
            _debug_render(f"  Index array shape: {mesh.indices.shape}, dtype: {mesh.indices.dtype}")
            _debug_render(f"  First 3 indices: {mesh.indices[0] if len(mesh.indices) > 0 else 'none'}")

        if not self._use_legacy:
            _debug_render(f"upload_solid_mesh: binding VAO {self._solid_vao}")
            glBindVertexArray(self._solid_vao)

        # Upload vertex data
        glBindBuffer(GL_ARRAY_BUFFER, self._solid_vbo)
        glBufferData(GL_ARRAY_BUFFER, mesh.vertices.nbytes, mesh.vertices, GL_STATIC_DRAW)

        # Upload index data
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._solid_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.indices.nbytes, mesh.indices, GL_STATIC_DRAW)

        if not self._use_legacy:
            # Set up vertex attributes
            # Vertex format: position (3) + normal (3) + uv (2) + color (3) = 11 floats
            stride = 11 * 4  # 11 floats * 4 bytes

            # Position (location 0) - offset 0
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))

            # Normal (location 1) - offset 3*4 = 12
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * 4))

            # UV/TexCoord (location 2) - offset 6*4 = 24
            glEnableVertexAttribArray(2)
            glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(6 * 4))

            # Color (location 3) - offset 8*4 = 32
            glEnableVertexAttribArray(3)
            glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8 * 4))

            glBindVertexArray(0)

        self._solid_triangle_count = mesh.triangle_count
        self._use_surface_meshes = False  # Using single mesh mode

    def upload_surface_meshes(self, meshes: SurfaceMeshes):
        """Upload separate meshes for each surface type (floor, ceiling, wall).

        This enables per-surface texturing where each surface type can have
        a different texture applied.

        Args:
            meshes: SurfaceMeshes containing separate floor, ceiling, wall meshes
        """
        _debug_render(f"upload_surface_meshes: floor={meshes.floor.triangle_count}, "
                     f"ceiling={meshes.ceiling.triangle_count}, wall={meshes.wall.triangle_count}")

        # Create VBOs/EBOs for each surface if not exists
        for surface_type in self._surface_types:
            if self._surface_vbos[surface_type] is None:
                self._surface_vbos[surface_type] = glGenBuffers(1)
                self._surface_ebos[surface_type] = glGenBuffers(1)

        # Upload each surface mesh
        surface_mesh_map = {
            "floor": meshes.floor,
            "ceiling": meshes.ceiling,
            "wall": meshes.wall,
            "structural": meshes.structural,
            "trim": meshes.trim,
        }

        for surface_type, mesh in surface_mesh_map.items():
            if mesh.is_empty:
                self._surface_triangle_counts[surface_type] = 0
                continue

            vbo = self._surface_vbos[surface_type]
            ebo = self._surface_ebos[surface_type]

            # Upload vertex data
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, mesh.vertices.nbytes, mesh.vertices, GL_STATIC_DRAW)

            # Upload index data
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.indices.nbytes, mesh.indices, GL_STATIC_DRAW)

            self._surface_triangle_counts[surface_type] = mesh.triangle_count

        # Also upload to combined mesh for non-textured rendering
        # Build combined mesh from all surfaces
        all_verts = []
        all_indices = []
        offset = 0
        for mesh in [meshes.floor, meshes.ceiling, meshes.wall, meshes.structural, meshes.trim]:
            if not mesh.is_empty:
                all_verts.append(mesh.vertices)
                # Offset indices
                all_indices.append(mesh.indices + offset)
                offset += len(mesh.vertices)

        if all_verts:
            combined_verts = np.vstack(all_verts)
            combined_indices = np.vstack(all_indices)
            combined_mesh = RenderMesh(
                vertices=combined_verts,
                indices=combined_indices,
                bounds_min=meshes.bounds_min,
                bounds_max=meshes.bounds_max
            )
            # Upload to main solid VBO/EBO for non-textured mode
            glBindBuffer(GL_ARRAY_BUFFER, self._solid_vbo)
            glBufferData(GL_ARRAY_BUFFER, combined_verts.nbytes, combined_verts, GL_STATIC_DRAW)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._solid_ebo)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, combined_indices.nbytes, combined_indices, GL_STATIC_DRAW)
            self._solid_triangle_count = combined_mesh.triangle_count

        self._use_surface_meshes = True  # Using per-surface mesh mode

    def upload_wireframe_mesh(self, vertices: np.ndarray, indices: np.ndarray):
        """Upload wireframe mesh data to GPU."""
        if len(vertices) == 0:
            self._wire_line_count = 0
            return

        if not self._use_legacy:
            glBindVertexArray(self._wire_vao)

        # Upload vertex data
        glBindBuffer(GL_ARRAY_BUFFER, self._wire_vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        # Upload index data
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._wire_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        if not self._use_legacy:
            # Position only for wireframe
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))

            glBindVertexArray(0)

        self._wire_line_count = len(indices)

    _render_call_count = 0  # Track render calls to avoid spam

    def render(self, view_matrix: np.ndarray, projection_matrix: np.ndarray,
               camera_position: np.ndarray):
        """Render the scene."""
        PreviewRenderer._render_call_count += 1
        if PreviewRenderer._render_call_count <= 5 or PreviewRenderer._render_call_count % 100 == 0:
            _debug_render(f"render() call #{PreviewRenderer._render_call_count}, initialized={self._initialized}, solid_tris={self._solid_triangle_count}")

        if not self._initialized:
            return

        # Clear - DEBUG: using bright red
        glClearColor(*self.background_color)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if PreviewRenderer._render_call_count <= 3:
            err = glGetError()
            if err != GL_NO_ERROR:
                _debug_render(f"GL ERROR after clear: {err}")
            else:
                _debug_render(f"glClear successful, color={self.background_color}")

        # Test triangle disabled - the test was overwriting the mesh VBO data!
        # The issue is with the camera matrices, not the VBO/shader pipeline.

        # Model matrix (identity for now)
        model = np.eye(4, dtype=np.float32)

        # Light position (follows camera)
        light_pos = camera_position + np.array([100.0, 100.0, 200.0], dtype=np.float32)

        # Determine if we should use textures
        use_texture = self.render_mode in (RenderMode.TEXTURED, RenderMode.TEXTURED_WIREFRAME)

        if self.render_mode in (RenderMode.SOLID, RenderMode.SOLID_WIREFRAME,
                                RenderMode.TEXTURED, RenderMode.TEXTURED_WIREFRAME):
            self._render_solid(model, view_matrix, projection_matrix, camera_position, light_pos, use_texture)

        if self.render_mode in (RenderMode.WIREFRAME, RenderMode.SOLID_WIREFRAME, RenderMode.TEXTURED_WIREFRAME):
            self._render_wireframe(model, view_matrix, projection_matrix)

    _solid_render_count = 0

    def _render_solid(self, model: np.ndarray, view: np.ndarray, projection: np.ndarray,
                      view_pos: np.ndarray, light_pos: np.ndarray, use_texture: bool = False):
        """Render solid geometry."""
        PreviewRenderer._solid_render_count += 1
        if PreviewRenderer._solid_render_count <= 3:
            _debug_render(f"_render_solid: {self._solid_triangle_count} triangles, use_texture={use_texture}")

        if self._solid_triangle_count == 0:
            return

        if self._use_legacy:
            # Use pure fixed-function pipeline on macOS (shaders don't work with Metal GL 2.1)
            self._render_solid_fixed_function(model, view, projection, light_pos, use_texture)
        else:
            # Modern OpenGL with shaders
            self._render_solid_shader(model, view, projection, view_pos, light_pos, use_texture)

    def _render_solid_shader(self, model: np.ndarray, view: np.ndarray, projection: np.ndarray,
                             view_pos: np.ndarray, light_pos: np.ndarray, use_texture: bool = False):
        """Render solid geometry using shaders (modern OpenGL 3.3+)."""
        glUseProgram(self._solid_program)

        # Set uniforms
        self._set_uniform_mat4(self._solid_program, "model", model)
        self._set_uniform_mat4(self._solid_program, "view", view)
        self._set_uniform_mat4(self._solid_program, "projection", projection)
        self._set_uniform_vec3(self._solid_program, "lightPos", light_pos)
        self._set_uniform_vec3(self._solid_program, "viewPos", view_pos)
        self._set_uniform_vec3(self._solid_program, "lightColor", np.array(self.light_color, dtype=np.float32))
        self._set_uniform_float(self._solid_program, "ambientStrength", self.ambient_strength)
        self._set_uniform_float(self._solid_program, "specularStrength", self.specular_strength)

        # Handle texture binding
        should_use_texture = use_texture and self._current_texture_id is not None
        self._set_uniform_bool(self._solid_program, "useTexture", should_use_texture)

        if should_use_texture:
            self._texture_manager.bind_texture(self._current_texture_id, 0)
            self._set_uniform_int(self._solid_program, "textureSampler", 0)

        glBindVertexArray(self._solid_vao)
        glDrawElements(GL_TRIANGLES, self._solid_triangle_count * 3, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

        if should_use_texture:
            self._texture_manager.unbind_texture(0)

    def _render_solid_fixed_function(self, model: np.ndarray, view: np.ndarray,
                                     projection: np.ndarray, light_pos: np.ndarray,
                                     use_texture: bool = False):
        """Render solid geometry using fixed-function pipeline (legacy OpenGL 2.1)."""
        if PreviewRenderer._solid_render_count <= 3:
            _debug_render(f"Using fixed-function pipeline, use_texture={use_texture}")

        # No shaders for legacy mode
        glUseProgram(0)

        # Set up matrices
        glMatrixMode(GL_PROJECTION)
        glLoadMatrixf(np.ascontiguousarray(projection.T, dtype=np.float32))

        glMatrixMode(GL_MODELVIEW)
        mv = view @ model
        glLoadMatrixf(np.ascontiguousarray(mv.T, dtype=np.float32))

        # Set up lighting
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        # Light position in eye coordinates
        glLightfv(GL_LIGHT0, GL_POSITION, [light_pos[0], light_pos[1], light_pos[2], 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [self.ambient_strength, self.ambient_strength, self.ambient_strength, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [self.specular_strength, self.specular_strength, self.specular_strength, 1.0])

        # Check if we should use per-surface texturing
        has_surface_textures = use_texture and self._use_surface_meshes and any(
            self._surface_textures.get(s) is not None for s in self._surface_types
        )

        if has_surface_textures:
            # Render each surface type with its own texture
            self._render_surfaces_with_textures()
        else:
            # Single texture or no texture mode
            self._render_single_surface(use_texture)

        glDisable(GL_COLOR_MATERIAL)
        glDisable(GL_LIGHT0)
        glDisable(GL_LIGHTING)

    def _render_surfaces_with_textures(self):
        """Render each surface type with its own texture (legacy fixed-function)."""
        stride = 11 * 4  # 11 floats * 4 bytes

        for surface_type in self._surface_types:
            tri_count = self._surface_triangle_counts.get(surface_type, 0)
            if tri_count == 0:
                continue

            vbo = self._surface_vbos.get(surface_type)
            ebo = self._surface_ebos.get(surface_type)
            texture_id = self._surface_textures.get(surface_type)

            if vbo is None or ebo is None:
                continue

            # Bind buffers for this surface
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)

            # Set up vertex arrays
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, stride, ctypes.c_void_p(0))

            glEnableClientState(GL_NORMAL_ARRAY)
            glNormalPointer(GL_FLOAT, stride, ctypes.c_void_p(12))

            if texture_id is not None:
                glEnable(GL_TEXTURE_2D)
                self._texture_manager.bind_texture(texture_id, 0)
                glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
                glColor3f(1.0, 1.0, 1.0)

                glEnableClientState(GL_TEXTURE_COORD_ARRAY)
                glTexCoordPointer(2, GL_FLOAT, stride, ctypes.c_void_p(24))
            else:
                # No texture for this surface - use vertex colors
                glDisable(GL_TEXTURE_2D)
                glEnableClientState(GL_COLOR_ARRAY)
                glColorPointer(3, GL_FLOAT, stride, ctypes.c_void_p(32))

            # Draw this surface
            glDrawElements(GL_TRIANGLES, tri_count * 3, GL_UNSIGNED_INT, None)

            # Cleanup for this surface
            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_NORMAL_ARRAY)
            if texture_id is not None:
                glDisableClientState(GL_TEXTURE_COORD_ARRAY)
                self._texture_manager.unbind_texture(0)
                glDisable(GL_TEXTURE_2D)
            else:
                glDisableClientState(GL_COLOR_ARRAY)

    def _render_single_surface(self, use_texture: bool):
        """Render all geometry with a single texture or no texture (legacy fixed-function)."""
        stride = 11 * 4  # 11 floats * 4 bytes

        should_use_texture = use_texture and self._current_texture_id is not None
        if PreviewRenderer._solid_render_count <= 3:
            _debug_render(f"Texture: use_texture={use_texture}, texture_id={self._current_texture_id}, should_use={should_use_texture}")

        if should_use_texture:
            glEnable(GL_TEXTURE_2D)
            self._texture_manager.bind_texture(self._current_texture_id, 0)
            glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
            glColor3f(1.0, 1.0, 1.0)
            if PreviewRenderer._solid_render_count <= 3:
                _debug_render(f"Texture bound: ID={self._current_texture_id}")

        # Bind buffers
        glBindBuffer(GL_ARRAY_BUFFER, self._solid_vbo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._solid_ebo)

        # Set up vertex arrays
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, stride, ctypes.c_void_p(0))

        glEnableClientState(GL_NORMAL_ARRAY)
        glNormalPointer(GL_FLOAT, stride, ctypes.c_void_p(12))

        if should_use_texture:
            glEnableClientState(GL_TEXTURE_COORD_ARRAY)
            glTexCoordPointer(2, GL_FLOAT, stride, ctypes.c_void_p(24))
        else:
            glEnableClientState(GL_COLOR_ARRAY)
            glColorPointer(3, GL_FLOAT, stride, ctypes.c_void_p(32))

        # Draw
        glDrawElements(GL_TRIANGLES, self._solid_triangle_count * 3, GL_UNSIGNED_INT, None)

        # Cleanup
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        if should_use_texture:
            glDisableClientState(GL_TEXTURE_COORD_ARRAY)
            self._texture_manager.unbind_texture(0)
            glDisable(GL_TEXTURE_2D)
        else:
            glDisableClientState(GL_COLOR_ARRAY)

    def _render_wireframe(self, model: np.ndarray, view: np.ndarray, projection: np.ndarray):
        """Render wireframe geometry."""
        if self._wire_line_count == 0:
            return

        if self._use_legacy:
            self._render_wireframe_fixed_function(model, view, projection)
        else:
            self._render_wireframe_shader(model, view, projection)

    def _render_wireframe_shader(self, model: np.ndarray, view: np.ndarray, projection: np.ndarray):
        """Render wireframe using shaders (modern OpenGL 3.3+)."""
        glUseProgram(self._wireframe_program)

        # Set uniforms
        self._set_uniform_mat4(self._wireframe_program, "model", model)
        self._set_uniform_mat4(self._wireframe_program, "view", view)
        self._set_uniform_mat4(self._wireframe_program, "projection", projection)
        self._set_uniform_vec3(self._wireframe_program, "wireColor",
                               np.array(self.wireframe_color, dtype=np.float32))

        # Offset for wireframe overlay
        if self.render_mode == RenderMode.SOLID_WIREFRAME:
            glEnable(GL_POLYGON_OFFSET_LINE)
            glPolygonOffset(-1.0, -1.0)

        glBindVertexArray(self._wire_vao)
        glDrawElements(GL_LINES, self._wire_line_count * 2, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

        if self.render_mode == RenderMode.SOLID_WIREFRAME:
            glDisable(GL_POLYGON_OFFSET_LINE)

    def _render_wireframe_fixed_function(self, model: np.ndarray, view: np.ndarray, projection: np.ndarray):
        """Render wireframe using fixed-function pipeline (legacy OpenGL 2.1)."""
        glUseProgram(0)

        # Set up matrices
        glMatrixMode(GL_PROJECTION)
        glLoadMatrixf(np.ascontiguousarray(projection.T, dtype=np.float32))

        glMatrixMode(GL_MODELVIEW)
        mv = view @ model
        glLoadMatrixf(np.ascontiguousarray(mv.T, dtype=np.float32))

        # Disable lighting for wireframe
        glDisable(GL_LIGHTING)

        # Set wireframe color
        glColor3f(*self.wireframe_color)

        # Offset for wireframe overlay
        if self.render_mode == RenderMode.SOLID_WIREFRAME:
            glEnable(GL_POLYGON_OFFSET_LINE)
            glPolygonOffset(-1.0, -1.0)

        # Bind buffers
        glBindBuffer(GL_ARRAY_BUFFER, self._wire_vbo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._wire_ebo)

        # Set up vertex array
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 3 * 4, ctypes.c_void_p(0))

        # Draw
        glDrawElements(GL_LINES, self._wire_line_count * 2, GL_UNSIGNED_INT, None)

        # Cleanup
        glDisableClientState(GL_VERTEX_ARRAY)

        if self.render_mode == RenderMode.SOLID_WIREFRAME:
            glDisable(GL_POLYGON_OFFSET_LINE)

    _uniform_debug_done = False

    def _set_uniform_mat4(self, program: int, name: str, matrix: np.ndarray):
        """Set a mat4 uniform."""
        loc = glGetUniformLocation(program, name)
        if not PreviewRenderer._uniform_debug_done:
            _debug_render(f"  Uniform '{name}' location: {loc}")
        if loc >= 0:
            # Try GL_TRUE: let OpenGL transpose the row-major numpy matrix
            # This is the alternative approach to Python transpose + GL_FALSE
            mat_flat = np.ascontiguousarray(matrix, dtype=np.float32)
            glUniformMatrix4fv(loc, 1, GL_TRUE, mat_flat)
        elif not PreviewRenderer._uniform_debug_done:
            _debug_render(f"  WARNING: Uniform '{name}' not found!")

    def _set_uniform_vec3(self, program: int, name: str, vec: np.ndarray):
        """Set a vec3 uniform."""
        loc = glGetUniformLocation(program, name)
        if loc >= 0:
            glUniform3fv(loc, 1, vec)

    def _set_uniform_float(self, program: int, name: str, value: float):
        """Set a float uniform."""
        loc = glGetUniformLocation(program, name)
        if loc >= 0:
            glUniform1f(loc, value)

    def _set_uniform_int(self, program: int, name: str, value: int):
        """Set an int uniform."""
        loc = glGetUniformLocation(program, name)
        if loc >= 0:
            glUniform1i(loc, value)

    def _set_uniform_bool(self, program: int, name: str, value: bool):
        """Set a bool uniform (as int 0/1)."""
        loc = glGetUniformLocation(program, name)
        if loc >= 0:
            glUniform1i(loc, 1 if value else 0)

    def set_texture(self, texture_path: Optional[str]):
        """Set the texture to use for textured rendering.

        Args:
            texture_path: Path to texture file, or None to disable texturing
        """
        _debug_render(f"set_texture({texture_path})")
        if texture_path is None:
            self._texture_path = None
            self._current_texture_id = None
            _debug_render("Texture cleared")
            return

        self._texture_path = texture_path
        if self._initialized:
            self._current_texture_id = self._texture_manager.get_or_load(texture_path)
            _debug_render(f"Texture loaded: ID={self._current_texture_id}")
        else:
            # Will be loaded when renderer is initialized
            self._current_texture_id = None
            _debug_render("Renderer not initialized, texture will be loaded later")

    def set_surface_textures(self, textures: Dict[str, Optional[str]]):
        """Set textures for each surface type.

        Args:
            textures: Dict mapping surface type to texture file path.
                     Keys: "floor", "ceiling", "wall", "structural", "trim"
                     Values: file path or None
        """
        _debug_render(f"set_surface_textures({textures})")

        for surface_type in self._surface_types:
            path = textures.get(surface_type)
            self._surface_texture_paths[surface_type] = path

            if self._initialized and path:
                self._surface_textures[surface_type] = self._texture_manager.get_or_load(path)
                _debug_render(f"  {surface_type}: ID={self._surface_textures[surface_type]}")
            else:
                self._surface_textures[surface_type] = None

    def get_texture_manager(self) -> TextureManager:
        """Get the texture manager for direct texture loading.

        Returns:
            The TextureManager instance
        """
        return self._texture_manager

    def cleanup(self):
        """Clean up OpenGL resources."""
        if not self._initialized:
            return

        # Clean up texture manager
        self._texture_manager.cleanup()

        if self._solid_program:
            glDeleteProgram(self._solid_program)
        if self._wireframe_program:
            glDeleteProgram(self._wireframe_program)

        if not self._use_legacy:
            if self._solid_vao:
                glDeleteVertexArrays(1, [self._solid_vao])
            if self._wire_vao:
                glDeleteVertexArrays(1, [self._wire_vao])

        if self._solid_vbo:
            glDeleteBuffers(1, [self._solid_vbo])
        if self._solid_ebo:
            glDeleteBuffers(1, [self._solid_ebo])
        if self._wire_vbo:
            glDeleteBuffers(1, [self._wire_vbo])
        if self._wire_ebo:
            glDeleteBuffers(1, [self._wire_ebo])

        self._initialized = False
