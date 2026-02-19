"""
Base classes for pipeline passes.

A pass is a composable unit of layout processing that takes a LayoutState
and produces a modified LayoutState. Passes can be chained together to
build complex generation pipelines.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from ..layout_state import LayoutState


@dataclass
class PassConfig:
    """
    Configuration for a pipeline pass.

    Attributes:
        enabled: Whether this pass should run
        options: Pass-specific configuration options
    """
    enabled: bool = True
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PassResult:
    """
    Result of executing a pipeline pass.

    Attributes:
        success: Whether the pass completed successfully
        state: The modified layout state
        warnings: Non-fatal issues encountered
        errors: Fatal issues that prevented completion
        metrics: Performance and diagnostic metrics
    """
    success: bool
    state: LayoutState
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)

    def add_error(self, message: str) -> None:
        """Add an error message and mark as failed."""
        self.errors.append(message)
        self.success = False


class LayoutPass(ABC):
    """
    Base class for composable layout passes.

    Each pass implements a specific transformation or enrichment
    of the layout state. Passes should be:
    - Idempotent when possible
    - Independent (minimal coupling to other passes)
    - Validated (check preconditions before running)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this pass."""
        pass

    @property
    def description(self) -> str:
        """Optional description of what this pass does."""
        return ""

    @abstractmethod
    def execute(self, state: LayoutState, config: PassConfig) -> PassResult:
        """
        Execute this pass on the given layout state.

        Args:
            state: The current layout state
            config: Pass configuration

        Returns:
            PassResult with the modified state and any issues
        """
        pass

    def validate_preconditions(self, state: LayoutState) -> List[str]:
        """
        Check if the state meets this pass's requirements.

        Returns a list of error messages if preconditions are not met.
        Override this method to add pass-specific validation.
        """
        return []

    def validate_postconditions(self, state: LayoutState) -> List[str]:
        """
        Verify that the pass produced valid output.

        Returns a list of error messages if postconditions are not met.
        Override this method to add pass-specific validation.
        """
        return []

    def run(self, state: LayoutState, config: Optional[PassConfig] = None) -> PassResult:
        """
        Run this pass with full validation.

        This is the recommended entry point for executing a pass.
        It handles precondition/postcondition validation automatically.
        """
        config = config or PassConfig()

        if not config.enabled:
            return PassResult(success=True, state=state)

        # Check preconditions
        precondition_errors = self.validate_preconditions(state)
        if precondition_errors:
            result = PassResult(success=False, state=state)
            for error in precondition_errors:
                result.add_error(f"Precondition failed: {error}")
            return result

        # Execute the pass
        result = self.execute(state, config)

        # Check postconditions if execution succeeded
        if result.success:
            postcondition_errors = self.validate_postconditions(result.state)
            for error in postcondition_errors:
                result.add_warning(f"Postcondition warning: {error}")

        return result
