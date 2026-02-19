"""
idTech Map Generation Pipeline Module.

Provides automated layout generation and .map file export.
"""

from .automated_pipeline import (
    AutomatedPipeline,
    PipelineSettings,
    PipelineResult,
    PipelineProgress,
    PipelineStage,
    PipelineMode,
    PipelineError,
    GenerationCancelledException,
)

from .layout_state import (
    LayoutState,
    PathNetwork,
    Marker,
    MarkerType,
    RoomProgress,
    GateConstraint,
)

__all__ = [
    # Pipeline core
    'AutomatedPipeline',
    'PipelineSettings',
    'PipelineResult',
    'PipelineProgress',
    'PipelineStage',
    'PipelineMode',
    'PipelineError',
    'GenerationCancelledException',
    # Layout state (Phase B & C)
    'LayoutState',
    'PathNetwork',
    'Marker',
    'MarkerType',
    'RoomProgress',
    'GateConstraint',
]
