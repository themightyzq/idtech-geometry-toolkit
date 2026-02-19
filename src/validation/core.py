"""
Core data structures for the validation system.

Defines the fundamental types used throughout the validation package:
- Severity: Issue severity levels (INFO, WARN, FAIL)
- ValidationStage: Pipeline stages where validation occurs
- ValidationIssue: Individual validation finding
- ValidationResult: Collection of issues with pass/fail status
- ValidationError: Exception raised when validation fails
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional


class Severity(Enum):
    """Validation issue severity levels.

    Per CLAUDE.md Section 3 (Quality Gates [BINDING]):
    - INFO: Informational, logged but doesn't affect pass/fail
    - WARN: Warning, logged but doesn't block generation/export
    - FAIL: Error, blocks generation/export (mandatory gate failure)
    """
    INFO = auto()
    WARN = auto()
    FAIL = auto()

    def __str__(self) -> str:
        return self.name


class ValidationStage(Enum):
    """Pipeline stages where validation occurs.

    Each stage has specific checks appropriate to that point in the pipeline:
    - MODULE_REGISTRATION: When a module is registered in the catalog
    - PLACEMENT: When primitives are placed in a layout
    - GENERATION: When brush geometry is generated from a layout
    - EXPORT: When map is written to file
    """
    MODULE_REGISTRATION = "module_registration"
    PLACEMENT = "placement"
    GENERATION = "generation"
    EXPORT = "export"

    def __str__(self) -> str:
        return self.value


@dataclass
class ValidationIssue:
    """Represents a single validation finding.

    Attributes:
        severity: Issue severity (INFO, WARN, FAIL)
        code: Rule code (e.g., "GEOM-001")
        message: Human-readable description
        rule_reference: CLAUDE.md section reference
        remediation: Optional suggested fix
        location: Optional location info (module name, brush index, etc.)
        module: Optional module name for structured format
        connector: Optional connector/portal ID for structured format
        transform: Optional transform info for structured format
        file_path: Optional file path for structured format
    """
    severity: Severity
    code: str
    message: str
    rule_reference: str
    remediation: Optional[str] = None
    location: Optional[str] = None
    module: Optional[str] = None
    connector: Optional[str] = None
    transform: Optional[str] = None
    file_path: Optional[str] = None

    def format(self) -> str:
        """Format issue for display.

        Returns:
            Formatted string per user specification:
            [SEVERITY] RULE_ID module=M connector=C transform=T file=F :: message :: fix=FIX
        """
        module = self.module or self.location or '-'
        connector = self.connector or '-'
        transform = self.transform or '-'
        file_path = self.file_path or '-'
        fix = self.remediation or 'N/A'

        return (
            f"[{self.severity}] {self.code} "
            f"module={module} connector={connector} "
            f"transform={transform} file={file_path} :: "
            f"{self.message} :: fix={fix}"
        )

    def __str__(self) -> str:
        return self.format()


@dataclass
class ValidationResult:
    """Collection of validation issues with pass/fail determination.

    Attributes:
        issues: List of ValidationIssue objects
        stage: Pipeline stage this result is from

    Properties:
        passed: True if no FAIL severity issues
        failed: True if any FAIL severity issues
        warnings: List of WARN severity issues
        errors: List of FAIL severity issues
        infos: List of INFO severity issues
    """
    issues: List[ValidationIssue] = field(default_factory=list)
    stage: Optional[ValidationStage] = None

    @property
    def passed(self) -> bool:
        """Check if validation passed (no FAIL issues)."""
        return not any(i.severity == Severity.FAIL for i in self.issues)

    @property
    def failed(self) -> bool:
        """Check if validation failed (any FAIL issues)."""
        return any(i.severity == Severity.FAIL for i in self.issues)

    @property
    def warnings(self) -> List[ValidationIssue]:
        """Get all WARN severity issues."""
        return [i for i in self.issues if i.severity == Severity.WARN]

    @property
    def errors(self) -> List[ValidationIssue]:
        """Get all FAIL severity issues."""
        return [i for i in self.issues if i.severity == Severity.FAIL]

    @property
    def infos(self) -> List[ValidationIssue]:
        """Get all INFO severity issues."""
        return [i for i in self.issues if i.severity == Severity.INFO]

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add an issue to the result."""
        self.issues.append(issue)

    def merge(self, other: 'ValidationResult') -> 'ValidationResult':
        """Merge another result into this one.

        Args:
            other: ValidationResult to merge

        Returns:
            Self for chaining
        """
        self.issues.extend(other.issues)
        return self

    def report(self) -> str:
        """Generate a formatted report of all issues.

        Returns:
            Multi-line string with all issues formatted
        """
        if not self.issues:
            return "Validation passed: No issues found"

        lines = []
        stage_str = f" ({self.stage})" if self.stage else ""
        status = "PASSED" if self.passed else "FAILED"
        lines.append(f"Validation {status}{stage_str}: {len(self.issues)} issue(s)")
        lines.append("-" * 60)

        # Group by severity
        for severity in [Severity.FAIL, Severity.WARN, Severity.INFO]:
            severity_issues = [i for i in self.issues if i.severity == severity]
            if severity_issues:
                lines.append(f"\n{severity.name} ({len(severity_issues)}):")
                for issue in severity_issues:
                    lines.append(issue.format())

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'passed': self.passed,
            'stage': str(self.stage) if self.stage else None,
            'issue_count': len(self.issues),
            'fail_count': len(self.errors),
            'warn_count': len(self.warnings),
            'info_count': len(self.infos),
            'issues': [
                {
                    'severity': str(issue.severity),
                    'code': issue.code,
                    'message': issue.message,
                    'rule_reference': issue.rule_reference,
                    'remediation': issue.remediation,
                    'location': issue.location,
                    'module': issue.module,
                    'connector': issue.connector,
                    'transform': issue.transform,
                    'file_path': issue.file_path,
                }
                for issue in self.issues
            ]
        }


class ValidationError(Exception):
    """Exception raised when validation fails with FAIL severity issues.

    Attributes:
        result: The ValidationResult that caused the failure
    """

    def __init__(self, result: ValidationResult):
        self.result = result
        super().__init__(result.report())
