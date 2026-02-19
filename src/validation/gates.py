"""
Validation gate decorator for pipeline stage validation.

Provides the @validation_gate decorator for wrapping functions with
automatic validation at pipeline boundaries.
"""

import functools
import logging
from typing import Callable, Optional, Any

from .core import ValidationStage, ValidationResult, ValidationError

logger = logging.getLogger(__name__)


def validation_gate(
    stage: ValidationStage,
    fail_fast: bool = True,
    log_warnings: bool = True
) -> Callable:
    """Decorator to add validation gate at pipeline boundaries.

    Wraps a function with pre/post validation. If validation fails
    with FAIL severity issues and fail_fast=True, raises ValidationError.

    Args:
        stage: Pipeline stage for this gate
        fail_fast: If True, raise ValidationError on FAIL issues
        log_warnings: If True, log WARN issues

    Returns:
        Decorated function

    Usage:
        @validation_gate(ValidationStage.GENERATION)
        def generate_from_layout(layout: DungeonLayout) -> GenerationResult:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Import here to avoid circular imports
            from .unified_validator import get_validator

            validator = get_validator()

            # Execute the function
            result = func(*args, **kwargs)

            # Post-execution validation based on stage
            validation_result: Optional[ValidationResult] = None

            if stage == ValidationStage.GENERATION:
                # For generation, validate the returned brushes
                if hasattr(result, 'brushes'):
                    validation_result = validator.validate_generation(
                        result.brushes,
                        getattr(result, 'portal_mismatches', [])
                    )

            elif stage == ValidationStage.EXPORT:
                # For export, validate entities
                if hasattr(result, 'entities'):
                    validation_result = validator.validate_export(
                        result.entities,
                        kwargs.get('export_format', 'idtech1')
                    )

            # Handle validation results
            if validation_result:
                # Log warnings
                if log_warnings:
                    for issue in validation_result.warnings:
                        logger.warning(str(issue))

                # Fail fast on errors
                if fail_fast and validation_result.failed:
                    logger.error(f"Validation failed at {stage}: {len(validation_result.errors)} errors")
                    raise ValidationError(validation_result)

            return result

        return wrapper
    return decorator


class ValidationGateContext:
    """Context manager for validation gates.

    Alternative to decorator when more control is needed.

    Usage:
        with ValidationGateContext(ValidationStage.GENERATION) as gate:
            result = generate_geometry()
            gate.validate(result.brushes)
    """

    def __init__(
        self,
        stage: ValidationStage,
        fail_fast: bool = True,
        log_warnings: bool = True
    ):
        self.stage = stage
        self.fail_fast = fail_fast
        self.log_warnings = log_warnings
        self._result: Optional[ValidationResult] = None

    def __enter__(self) -> 'ValidationGateContext':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        # Don't suppress exceptions
        return False

    def validate(self, *args, **kwargs) -> ValidationResult:
        """Run validation appropriate for this gate's stage.

        Args:
            *args: Arguments passed to validator
            **kwargs: Keyword arguments passed to validator

        Returns:
            ValidationResult

        Raises:
            ValidationError: If fail_fast=True and validation fails
        """
        from .unified_validator import get_validator

        validator = get_validator()

        if self.stage == ValidationStage.GENERATION:
            brushes = args[0] if args else kwargs.get('brushes', [])
            mismatches = args[1] if len(args) > 1 else kwargs.get('portal_mismatches', [])
            self._result = validator.validate_generation(brushes, mismatches)

        elif self.stage == ValidationStage.EXPORT:
            entities = args[0] if args else kwargs.get('entities', [])
            export_format = args[1] if len(args) > 1 else kwargs.get('export_format', 'idtech1')
            self._result = validator.validate_export(entities, export_format)

        elif self.stage == ValidationStage.PLACEMENT:
            layout = args[0] if args else kwargs.get('layout')
            self._result = validator.validate_placement(layout)

        elif self.stage == ValidationStage.MODULE_REGISTRATION:
            module_cls = args[0] if args else kwargs.get('module_cls')
            self._result = validator.validate_module_registration(module_cls)

        # Log warnings
        if self._result and self.log_warnings:
            for issue in self._result.warnings:
                logger.warning(str(issue))

        # Fail fast on errors
        if self._result and self.fail_fast and self._result.failed:
            raise ValidationError(self._result)

        return self._result

    @property
    def result(self) -> Optional[ValidationResult]:
        """Get the last validation result."""
        return self._result
