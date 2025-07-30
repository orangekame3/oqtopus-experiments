#!/usr/bin/env python3
"""
Structured analysis result classes for better error handling

Provides comprehensive result objects that include success status,
error information, and recovery suggestions.
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class AnalysisResult:
    """
    Structured result from experiment analysis

    Contains both successful results and comprehensive error information
    to help users understand and resolve issues.
    """

    success: bool
    data: pd.DataFrame | None = None
    errors: list[str] | None = None
    warnings: list[str] | None = None
    suggestions: list[str] | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        """Initialize empty lists if None"""
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.suggestions is None:
            self.suggestions = []
        if self.metadata is None:
            self.metadata = {}

    @classmethod
    def success_result(
        cls,
        data: pd.DataFrame,
        warnings: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "AnalysisResult":
        """Create successful analysis result"""
        return cls(
            success=True,
            data=data,
            warnings=warnings or [],
            metadata=metadata or {},
        )

    @classmethod
    def error_result(
        cls,
        errors: list[str],
        suggestions: list[str] | None = None,
        partial_data: pd.DataFrame | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "AnalysisResult":
        """Create error analysis result"""
        return cls(
            success=False,
            data=partial_data,
            errors=errors,
            suggestions=suggestions or [],
            metadata=metadata or {},
        )

    def add_warning(self, warning: str):
        """Add a warning message"""
        self.warnings.append(warning)

    def add_error(self, error: str):
        """Add an error message"""
        self.errors.append(error)
        self.success = False

    def add_suggestion(self, suggestion: str):
        """Add a recovery suggestion"""
        self.suggestions.append(suggestion)

    def has_errors(self) -> bool:
        """Check if result has any errors"""
        return bool(self.errors)

    def has_warnings(self) -> bool:
        """Check if result has any warnings"""
        return bool(self.warnings)

    def get_summary(self) -> str:
        """Get human-readable summary of the result"""
        if self.success:
            summary = "✅ Analysis completed successfully"
            if self.warnings:
                summary += f" (with {len(self.warnings)} warnings)"
        else:
            summary = f"❌ Analysis failed with {len(self.errors)} errors"
            if self.data is not None:
                summary += " (partial results available)"

        return summary

    def print_report(self):
        """Print comprehensive analysis report"""
        print(self.get_summary())
        print("=" * 50)

        if self.errors:
            print("\n🚨 ERRORS:")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")

        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")

        if self.suggestions:
            print("\n💡 SUGGESTIONS:")
            for i, suggestion in enumerate(self.suggestions, 1):
                print(f"  {i}. {suggestion}")

        if self.data is not None:
            print(
                f"\n📊 DATA: {len(self.data)} rows x {len(self.data.columns)} columns"
            )
        else:
            print("\n📊 DATA: No data available")

    def to_legacy_dataframe(self) -> pd.DataFrame:
        """
        Convert to legacy DataFrame format for backward compatibility

        Returns the data DataFrame if successful, or an error DataFrame if failed.
        This maintains compatibility with existing code that expects pd.DataFrame.
        """
        if self.success and self.data is not None:
            return self.data

        # Create error DataFrame for backward compatibility
        error_info = {
            "error": self.errors[0] if self.errors else "Unknown analysis error",
            "success": self.success,
            "num_errors": len(self.errors),
            "num_warnings": len(self.warnings),
        }

        if self.suggestions:
            error_info["suggestions"] = "; ".join(self.suggestions[:3])  # Limit length

        return pd.DataFrame([error_info])
