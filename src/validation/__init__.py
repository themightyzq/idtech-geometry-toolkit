"""
Unified validation package for idTech Geometry Toolkit.

Provides mandatory validation enforcement at all pipeline stages per CLAUDE.md Quality Gates.

Public API:
    - ValidationResult, ValidationIssue, Severity: Core result types
    - ValidationStage: Pipeline stage enumeration
    - UnifiedValidator: Central orchestrator for all validation
    - get_validator(): Get configured validator instance
    - ValidationError: Exception raised on FAIL issues when fail_fast=True
    - validation_gate: Decorator for pipeline stage validation
"""

from .core import (
    Severity,
    ValidationStage,
    ValidationIssue,
    ValidationResult,
    ValidationError,
)
from .unified_validator import UnifiedValidator, get_validator, reset_validator
from .gates import validation_gate

__all__ = [
    # Core types
    'Severity',
    'ValidationStage',
    'ValidationIssue',
    'ValidationResult',
    'ValidationError',
    # Validator
    'UnifiedValidator',
    'get_validator',
    'reset_validator',
    # Decorator
    'validation_gate',
]
