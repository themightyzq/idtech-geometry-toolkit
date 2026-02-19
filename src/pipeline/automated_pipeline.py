"""
Automated Pipeline for idTech Map Generation (.map export only).

Orchestrates layout generation, 3D conversion, spawn placement, and .map
file writing.  Compilation, deployment, and engine launch have been removed;
the output is a .map file ready for external compilation.
"""

import os
import sys
import time
import json
import shutil
import tempfile
import logging
import random
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum, auto

from quake_levelgenerator.src.generators.bsp.bsp_generator import (
    BSPGenerator, BSPNode, Room, Corridor
)
from quake_levelgenerator.src.generators.bsp import (
    IDTECH_PLAYER_WIDTH, IDTECH_PLAYER_HEIGHT
)
from quake_levelgenerator.src.conversion.geometry.brush_generator import (
    BrushGenerator, GeometrySettings
)
from quake_levelgenerator.src.conversion.map_writer import MapWriter, Entity, Brush
from quake_levelgenerator.src.conversion.bsp_to_layout import (
    convert_bsp_to_layout2d, convert_bsp_rooms_directly
)
from quake_levelgenerator.src.conversion.spawn_placement import place_player_spawn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PipelineStage(Enum):
    INITIALIZE = "initialize"
    GENERATE_LAYOUT = "generate_layout"
    CONVERT_3D = "convert_3d"
    WRITE_MAP = "write_map"
    COMPLETE = "complete"


class PipelineMode(Enum):
    FAST = "fast"
    PREVIEW = "preview"
    FINAL = "final"
    CUSTOM = "custom"
    PRIMITIVE = "primitive"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class PipelineError(Exception):
    pass


class GenerationCancelledException(PipelineError):
    pass


# ---------------------------------------------------------------------------
# Settings / Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PipelineSettings:
    # BSP Generation
    map_width: int = 2048
    map_height: int = 2048
    room_count: int = 12
    max_depth: int = 4
    min_room_size: int = 128
    max_room_size: int = 384
    corridor_width: int = 96
    enable_vertical_rooms: bool = False
    vertical_room_chance: float = 0.2

    # 3D Conversion
    wall_thickness: float = 8.0
    floor_height: float = 0.0
    ceiling_height: float = 128.0
    grid_size: float = 64.0

    # Texturing
    wall_texture: str = "CRATE1_5"
    floor_texture: str = "GROUND1_6"
    ceiling_texture: str = "CEIL1_1"

    # Texture theme (overrides individual textures when set)
    texture_theme: str = "Base"

    # Export format: "idtech1" or "idtech4"
    export_format: str = "idtech1"

    # OBJ export (alongside .map)
    export_obj: bool = False

    # Pipeline behaviour
    mode: PipelineMode = PipelineMode.PREVIEW
    output_dir: Optional[str] = None
    map_name: str = "generated_map"

    # Primitive mode (Phase 3)
    primitive_name: Optional[str] = None
    primitive_params: Optional[Dict[str, Any]] = None

    # Seeding for reproducible generation
    seed: Optional[int] = None  # None = random seed, otherwise deterministic

    # Debug output
    enable_graph_dump: bool = False
    graph_dump_format: str = "dot"  # "dot" or "json"

    # Misc
    verbose: bool = False


@dataclass
class PipelineProgress:
    stage: PipelineStage
    stage_progress: float
    overall_progress: float
    message: str
    elapsed_time: float
    estimated_remaining: float

    @property
    def percentage(self) -> int:
        return int(self.overall_progress * 100)


@dataclass
class PipelineResult:
    success: bool
    output_files: List[str] = field(default_factory=list)
    primary_output: Optional[str] = None
    stages_completed: List[PipelineStage] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_time(self) -> float:
        return self.metrics.get("total_time", 0.0)

    @property
    def map_file(self) -> Optional[str]:
        return self.primary_output

    def add_error(self, error: str, stage: Optional[PipelineStage] = None):
        if stage:
            error = f"[{stage.value}] {error}"
        self.errors.append(error)

    def add_warning(self, warning: str, stage: Optional[PipelineStage] = None):
        if stage:
            warning = f"[{stage.value}] {warning}"
        self.warnings.append(warning)


# ---------------------------------------------------------------------------
# Progress tracker
# ---------------------------------------------------------------------------

class ProgressTracker:
    STAGE_WEIGHTS = {
        PipelineStage.INITIALIZE: 0.05,
        PipelineStage.GENERATE_LAYOUT: 0.25,
        PipelineStage.CONVERT_3D: 0.40,
        PipelineStage.WRITE_MAP: 0.30,
    }

    def __init__(self):
        self.start_time = time.time()
        self.stage_start_times: Dict[PipelineStage, float] = {}

    def start_stage(self, stage: PipelineStage):
        self.stage_start_times[stage] = time.time()

    def complete_stage(self, stage: PipelineStage):
        pass  # tracked implicitly

    def calculate_progress(self, current_stage: PipelineStage, stage_progress: float) -> PipelineProgress:
        stages = list(self.STAGE_WEIGHTS.keys())
        if current_stage not in stages:
            overall = 1.0
        else:
            idx = stages.index(current_stage)
            completed = sum(self.STAGE_WEIGHTS[s] for s in stages[:idx])
            overall = completed + self.STAGE_WEIGHTS[current_stage] * stage_progress
        elapsed = time.time() - self.start_time
        remaining = (elapsed / overall - elapsed) if overall > 0.01 else 0.0
        return PipelineProgress(
            stage=current_stage,
            stage_progress=stage_progress,
            overall_progress=min(overall, 1.0),
            message="",
            elapsed_time=elapsed,
            estimated_remaining=max(0, remaining),
        )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class AutomatedPipeline:
    """Generates a .map file from BSP layout parameters (or primitives)."""

    def __init__(self, settings: Optional[PipelineSettings] = None):
        self.settings = settings or PipelineSettings()
        self.is_running = False
        self.is_cancelled = False
        self.current_stage = PipelineStage.INITIALIZE
        self.progress_tracker = ProgressTracker()
        self.progress_callback: Optional[Callable[[PipelineProgress], None]] = None

        # Components
        self.bsp_generator: Optional[BSPGenerator] = None
        self.brush_generator: Optional[BrushGenerator] = None
        self.map_writer: Optional[MapWriter] = None

        # Data
        self.layout_data: Optional[Dict] = None
        self.layout_2d = None
        self.brushes: List[Brush] = []

        # Paths
        self.map_file_path: Optional[str] = None
        self._validate_settings()

    # -- helpers --

    def set_progress_callback(self, callback: Callable[[PipelineProgress], None]):
        self.progress_callback = callback

    def cancel(self):
        self.is_cancelled = True

    def _check_cancellation(self):
        if self.is_cancelled:
            raise GenerationCancelledException("Pipeline cancelled by user")

    def _update_progress(self, stage_progress: float, message: str):
        if self.is_cancelled:
            return
        progress = self.progress_tracker.calculate_progress(self.current_stage, stage_progress)
        progress.message = message
        if self.progress_callback:
            try:
                self.progress_callback(progress)
            except Exception:
                pass

    def _validate_settings(self):
        errors = []
        if self.settings.map_width < 512 or self.settings.map_height < 512:
            errors.append("Map dimensions must be at least 512x512")
        if self.settings.room_count < 1 or self.settings.room_count > 50:
            errors.append("Room count must be between 1 and 50")
        if self.settings.min_room_size < 64:
            errors.append("Minimum room size cannot be less than 64 units")
        if self.settings.max_room_size < self.settings.min_room_size:
            errors.append("Maximum room size must be >= minimum room size")
        if errors:
            raise PipelineError(f"Invalid settings: {'; '.join(errors)}")

    # -- stages --

    def _initialize_components(self):
        self.current_stage = PipelineStage.INITIALIZE
        self.progress_tracker.start_stage(PipelineStage.INITIALIZE)
        self._update_progress(0.0, "Initializing components...")

        player_clearance_pad = 16
        min_corridor = max(self.settings.corridor_width, int(IDTECH_PLAYER_WIDTH + 2 * player_clearance_pad))
        required_clear = int(max(min_corridor // 2, IDTECH_PLAYER_WIDTH // 2 + 16)) + 4
        enforced_min_room = max(self.settings.min_room_size, required_clear * 2 + 24)

        self.bsp_generator = BSPGenerator({
            'map_width': self.settings.map_width,
            'map_height': self.settings.map_height,
            'max_depth': self.settings.max_depth,
            'min_room_size': enforced_min_room,
            'max_room_size': self.settings.max_room_size,
            'corridor_width': min_corridor,
            'enable_vertical_rooms': self.settings.enable_vertical_rooms,
            'vertical_room_chance': self.settings.vertical_room_chance,
            'debug': self.settings.verbose,
        })

        geometry_settings = GeometrySettings(
            wall_thickness=self.settings.wall_thickness,
            base_floor_z=self.settings.floor_height,
            standard_height=self.settings.ceiling_height,
            grid_size=self.settings.grid_size,
            corridor_width=float(min_corridor),
            door_width=max(64.0, float(min_corridor)),
        )
        self.brush_generator = BrushGenerator(geometry_settings)
        self.map_writer = MapWriter(export_format=self.settings.export_format)

        self._update_progress(1.0, "Initialization complete")
        self.progress_tracker.complete_stage(PipelineStage.INITIALIZE)

    def _generate_layout(self) -> Dict[str, Any]:
        self.current_stage = PipelineStage.GENERATE_LAYOUT
        self.progress_tracker.start_stage(PipelineStage.GENERATE_LAYOUT)
        self._check_cancellation()
        self._update_progress(0.0, "Generating BSP layout...")

        if not self.bsp_generator.generate(self.settings.room_count):
            raise PipelineError("BSP layout generation failed")

        self.layout_data = self.bsp_generator.export_layout()
        stats = self.bsp_generator.get_layout_stats()
        logger.info("Layout: %d rooms, %d corridors", stats['room_count'], stats['corridor_count'])
        self._update_progress(1.0, f"Layout: {stats['room_count']} rooms, {stats['corridor_count']} corridors")
        self.progress_tracker.complete_stage(PipelineStage.GENERATE_LAYOUT)
        return self.layout_data

    def _convert_to_3d(self) -> List[Brush]:
        self.current_stage = PipelineStage.CONVERT_3D
        self.progress_tracker.start_stage(PipelineStage.CONVERT_3D)
        self._check_cancellation()
        self._update_progress(0.0, "Converting to 3D...")

        if isinstance(self.layout_data, dict):
            self.layout_2d = convert_bsp_to_layout2d(self.layout_data)
            if self.layout_2d is None:
                raise PipelineError("Layout conversion returned None")
        else:
            self.layout_2d = self.layout_data

        self.brushes = self.brush_generator.generate_from_layout(self.layout_2d)
        logger.info("Generated %d brushes", len(self.brushes))
        self._update_progress(1.0, f"3D: {len(self.brushes)} brushes")
        self.progress_tracker.complete_stage(PipelineStage.CONVERT_3D)
        return self.brushes

    def _write_map_file(self) -> str:
        self.current_stage = PipelineStage.WRITE_MAP
        self.progress_tracker.start_stage(PipelineStage.WRITE_MAP)
        self._check_cancellation()
        self._update_progress(0.0, "Writing MAP file...")

        out_dir = Path(self.settings.output_dir) if self.settings.output_dir else Path("output") / "maps"
        out_dir.mkdir(parents=True, exist_ok=True)
        self.map_file_path = str(out_dir / f"{self.settings.map_name}.map")

        # Worldspawn + brushes
        worldspawn = self.map_writer.create_worldspawn()
        self.map_writer.add_entity(worldspawn)
        for brush in self.brushes:
            self.map_writer.add_brush(brush, 0)

        self._update_progress(0.5, "Placing spawn...")

        # Spawn
        if self.layout_data and "rooms" in self.layout_data:
            place_player_spawn(self.map_writer, self.layout_data["rooms"], self.settings.floor_height)

        self._update_progress(0.8, "Writing file...")
        self.map_writer.write_to_file(self.map_file_path)

        if not Path(self.map_file_path).exists():
            raise PipelineError("MAP file was not created")

        size = Path(self.map_file_path).stat().st_size
        logger.info("MAP written: %s (%d bytes)", self.map_file_path, size)

        # Optional OBJ export
        if self.settings.export_obj and self.brushes:
            self._write_obj_file(out_dir)

        self._update_progress(1.0, f"MAP written ({size} bytes)")
        self.progress_tracker.complete_stage(PipelineStage.WRITE_MAP)
        return self.map_file_path

    def _write_obj_file(self, out_dir: Path):
        """Write OBJ mesh alongside the .map file."""
        from quake_levelgenerator.src.conversion.obj_writer import ObjWriter
        obj_path = str(out_dir / f"{self.settings.map_name}.obj")
        writer = ObjWriter()
        writer.add_brushes(self.brushes)
        writer.write(obj_path)
        logger.info("OBJ written: %s (%d verts, %d faces)",
                     obj_path, writer.vertex_count, writer.face_count)

    def _write_graph_dump(self, result: PipelineResult, seed: int):
        """Write debug graph dump (DOT or JSON format)."""
        from quake_levelgenerator.src.pipeline.debug.graph_export import (
            export_layout_dot, export_layout_json
        )

        out_dir = Path(self.settings.output_dir) if self.settings.output_dir else Path("output") / "maps"
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            if self.settings.graph_dump_format == "json":
                content = export_layout_json(self.layout_data, seed)
                graph_path = str(out_dir / f"{self.settings.map_name}_debug.json")
            else:  # Default to DOT
                content = export_layout_dot(self.layout_data)
                graph_path = str(out_dir / f"{self.settings.map_name}_debug.dot")

            with open(graph_path, 'w') as f:
                f.write(content)

            result.output_files.append(graph_path)
            logger.info("Debug graph written: %s", graph_path)
        except Exception as e:
            result.add_warning(f"Failed to write debug graph: {e}")

    # -- main entry --

    def generate(self) -> PipelineResult:
        if self.is_running:
            raise PipelineError("Pipeline is already running")
        self.is_running = True
        self.is_cancelled = False
        result = PipelineResult(success=False)
        start_time = time.time()

        try:
            # Resolve and apply seed for reproducible generation
            if self.settings.seed is not None:
                actual_seed = self.settings.seed
            else:
                actual_seed = random.randint(0, 2**31 - 1)

            random.seed(actual_seed)
            result.metrics['seed'] = actual_seed
            logger.info("Generation seed: %d", actual_seed)

            logger.info("Starting map generation: %d rooms, %dx%d",
                        self.settings.room_count, self.settings.map_width, self.settings.map_height)

            # Primitive mode delegates to primitive pipeline (Phase 3)
            if self.settings.mode == PipelineMode.PRIMITIVE:
                return self._generate_primitive(result, start_time)

            stages = [
                (self._initialize_components, "Initialize"),
                (self._generate_layout, "Generate layout"),
                (self._convert_to_3d, "Convert 3D"),
                (self._write_map_file, "Write MAP"),
            ]
            for stage_fn, desc in stages:
                try:
                    logger.info("Stage: %s", desc)
                    stage_result = stage_fn()
                    result.stages_completed.append(self.current_stage)
                    if self.current_stage == PipelineStage.WRITE_MAP:
                        result.primary_output = stage_result
                        result.output_files.append(stage_result)
                except GenerationCancelledException:
                    result.add_error("Pipeline cancelled by user")
                    return result
                except PipelineError as e:
                    result.add_error(str(e), self.current_stage)
                    return result

            # Export debug graph if enabled
            if self.settings.enable_graph_dump and self.layout_data:
                self._write_graph_dump(result, actual_seed)

            result.success = True
            result.metrics["total_time"] = time.time() - start_time
            logger.info("Pipeline complete in %.2fs", result.metrics["total_time"])
        except Exception as e:
            logger.exception("Unexpected pipeline error")
            result.add_error(f"Unexpected error: {e}")
        finally:
            self.is_running = False
        return result

    def _generate_primitive(self, result: PipelineResult, start_time: float) -> PipelineResult:
        """Generate a single primitive and write to .map."""
        from quake_levelgenerator.src.generators.primitives.catalog import PRIMITIVE_CATALOG

        name = self.settings.primitive_name
        params = self.settings.primitive_params or {}
        if not name:
            result.add_error("No primitive_name specified for PRIMITIVE mode")
            return result

        prim_cls = PRIMITIVE_CATALOG.get_primitive(name)
        if prim_cls is None:
            result.add_error(f"Unknown primitive: {name}")
            return result

        prim = prim_cls()
        prim.apply_params(params)
        brushes = prim.generate()

        writer = MapWriter(export_format=self.settings.export_format)
        ws = writer.create_worldspawn()
        writer.add_entity(ws)
        for b in brushes:
            writer.add_brush(b, 0)

        # Spawn at origin
        writer.add_player_start((0, -128, 32))

        out_dir = Path(self.settings.output_dir) if self.settings.output_dir else Path("output") / "maps"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = str(out_dir / f"{self.settings.map_name}.map")
        writer.write_to_file(path)

        result.success = True
        result.primary_output = path
        result.output_files.append(path)
        result.metrics["total_time"] = time.time() - start_time
        self.is_running = False
        return result
