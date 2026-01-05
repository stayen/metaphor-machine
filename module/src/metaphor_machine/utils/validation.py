"""
Validation utilities for style components and configuration files.

Provides detailed error messages and suggestions for fixing
invalid configurations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from metaphor_machine.schemas.components import StyleComponents


@dataclass
class ValidationResult:
    """Result of validating a components file."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.valid

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = []

        if self.valid:
            lines.append("✓ Validation passed")
        else:
            lines.append("✗ Validation failed")

        if self.errors:
            lines.append("\nErrors:")
            for err in self.errors:
                lines.append(f"  • {err}")

        if self.warnings:
            lines.append("\nWarnings:")
            for warn in self.warnings:
                lines.append(f"  ⚠ {warn}")

        if self.stats:
            lines.append("\nStatistics:")
            for key, value in self.stats.items():
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)


def validate_components_file(path: str | Path) -> ValidationResult:
    """
    Validate a style_components.yaml file.

    Performs both schema validation and semantic checks.

    Args:
        path: Path to the YAML file

    Returns:
        ValidationResult with errors, warnings, and statistics
    """
    path = Path(path)
    result = ValidationResult(valid=True)

    # Check file exists
    if not path.exists():
        result.valid = False
        result.errors.append(f"File not found: {path}")
        return result

    # Try to parse YAML
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        result.valid = False
        result.errors.append(f"YAML parse error: {e}")
        return result

    if data is None:
        result.valid = False
        result.errors.append("File is empty")
        return result

    # Check required top-level keys
    required_keys = [
        "genre",
        "intimate_gesture",
        "dynamic_tension",
        "sensory_bridge",
        "emotional_anchor",
    ]

    for key in required_keys:
        if key not in data:
            result.valid = False
            result.errors.append(f"Missing required section: {key}")

    if not result.valid:
        return result

    # Try Pydantic validation
    try:
        components = StyleComponents.model_validate(data)
    except ValidationError as e:
        result.valid = False
        for error in e.errors():
            loc = ".".join(str(x) for x in error["loc"])
            result.errors.append(f"{loc}: {error['msg']}")
        return result

    # Semantic checks
    result.stats = components.get_pool_sizes()
    result.stats["combinatorial_space"] = components.estimate_combinatorial_space()

    # Warnings for small pools
    for pool_name, count in result.stats.items():
        if pool_name == "combinatorial_space":
            continue
        if count < 5:
            result.warnings.append(f"Small pool '{pool_name}' has only {count} items")

    # Check for genre family balance
    families = components.genre.genre_families
    empty_families = [name for name, genres in families.items() if not genres]
    if empty_families:
        result.warnings.append(
            f"Empty genre families: {', '.join(empty_families)}"
        )

    return result


def check_duplicate_values(components: StyleComponents) -> list[str]:
    """
    Check for duplicate values across pools.

    Returns list of warnings about duplicates.
    """
    warnings = []

    # Check for duplicates within core genres
    genres = components.genre.all_core_genres
    if len(genres) != len(set(genres)):
        warnings.append("Duplicate values in genre pools")

    # Check for duplicates in motion verbs
    verbs = components.dynamic_tension.motion_verbs
    if len(verbs) != len(set(verbs)):
        warnings.append("Duplicate values in dynamic_tension.motion_verbs")

    # Check for duplicates in environments
    envs = components.sensory_bridge.environments
    if len(envs) != len(set(envs)):
        warnings.append("Duplicate values in sensory_bridge.environments")

    return warnings


def suggest_additions(components: StyleComponents) -> list[str]:
    """
    Suggest additions to improve pool coverage.

    Returns list of suggestions.
    """
    suggestions = []

    # Suggest if certain categories are thin
    genre_count = len(components.genre.all_core_genres)
    if genre_count < 20:
        suggestions.append(
            "Consider adding more genres (currently: "
            f"{genre_count})"
        )

    adjective_count = len(components.intimate_gesture.all_adjectives)
    if adjective_count < 20:
        suggestions.append(
            f"Consider adding more intensity adjectives (currently: {adjective_count})"
        )

    env_count = len(components.sensory_bridge.environments)
    if env_count < 20:
        suggestions.append(
            f"Consider adding more environments (currently: {env_count})"
        )

    return suggestions
