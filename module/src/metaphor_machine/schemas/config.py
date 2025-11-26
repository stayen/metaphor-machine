"""
Configuration schema for the MetaphorGenerator.

This module defines configuration options that control generation behavior,
including seed handling, diversity constraints, and output formatting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OutputFormat(str, Enum):
    """Output format options for generated metaphors."""

    PLAIN = "plain"  # Simple comma-separated string
    SUNO = "suno"  # Formatted for Suno Style of Music field
    PRODUCER = "producer"  # Formatted for Producer.ai Sound Prompt
    JSON = "json"  # JSON dictionary format
    YAML = "yaml"  # YAML format

    def __str__(self) -> str:
        return self.value


class ChainSeparator(str, Enum):
    """Separator styles for 3-act chains."""

    ARROW = " → "
    SEMICOLON = "; "
    NEWLINE = "\n"
    PIPE = " | "

    def __str__(self) -> str:
        return self.value


@dataclass
class DiversityConfig:
    """
    Configuration for diversity constraints in batch generation.

    Attributes:
        min_hamming_distance: Minimum required Hamming distance between
            any two metaphors in a batch (0-5, where 5 = completely different)
        max_retries: Maximum attempts to find a diverse metaphor before giving up
        allow_partial: If True, return partial batches when diversity
            constraints cannot be met
    """

    min_hamming_distance: int = 3
    max_retries: int = 100
    allow_partial: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.min_hamming_distance <= 5:
            raise ValueError(
                f"min_hamming_distance must be 0-5, got {self.min_hamming_distance}"
            )
        if self.max_retries < 1:
            raise ValueError(f"max_retries must be >= 1, got {self.max_retries}")


@dataclass
class SlotBias:
    """
    Bias configuration for a specific slot type.

    Allows constraining or weighting choices for individual slots.

    Attributes:
        allowed_values: If set, only these values can be chosen
        excluded_values: Values to never choose
        weights: Optional weight mapping for probabilistic selection
    """

    allowed_values: list[str] | None = None
    excluded_values: list[str] = field(default_factory=list)
    weights: dict[str, float] | None = None

    def filter_pool(self, pool: list[str]) -> list[str]:
        """Apply bias constraints to a pool of values."""
        if self.allowed_values is not None:
            pool = [v for v in pool if v in self.allowed_values]
        pool = [v for v in pool if v not in self.excluded_values]
        return pool


@dataclass
class GeneratorConfig:
    """
    Complete configuration for MetaphorGenerator.

    Attributes:
        seed: Random seed for reproducibility (None = random)
        diversity: Diversity constraint configuration
        output_format: Default output format
        chain_separator: Separator for 3-act chain formatting
        genre_hint: Optional genre to bias generation toward
        persona: Optional persona name for themed generation
        slot_biases: Per-slot bias configurations

    Example:
        >>> config = GeneratorConfig(
        ...     seed=42,
        ...     diversity=DiversityConfig(min_hamming_distance=3),
        ...     genre_hint="darkwave",
        ... )
        >>> generator = MetaphorGenerator(components, config=config)
    """

    seed: int | None = None
    diversity: DiversityConfig = field(default_factory=DiversityConfig)
    output_format: OutputFormat = OutputFormat.SUNO
    chain_separator: ChainSeparator = ChainSeparator.ARROW
    genre_hint: str | None = None
    persona: str | None = None
    slot_biases: dict[str, SlotBias] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GeneratorConfig:
        """Create config from dictionary (e.g., from YAML)."""
        # Handle nested configs
        if "diversity" in data and isinstance(data["diversity"], dict):
            data["diversity"] = DiversityConfig(**data["diversity"])

        if "slot_biases" in data:
            data["slot_biases"] = {
                k: SlotBias(**v) if isinstance(v, dict) else v
                for k, v in data["slot_biases"].items()
            }

        # Handle enum conversions
        if "output_format" in data and isinstance(data["output_format"], str):
            data["output_format"] = OutputFormat(data["output_format"])

        if "chain_separator" in data and isinstance(data["chain_separator"], str):
            # Try to match by value
            for sep in ChainSeparator:
                if sep.value == data["chain_separator"]:
                    data["chain_separator"] = sep
                    break

        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "seed": self.seed,
            "diversity": {
                "min_hamming_distance": self.diversity.min_hamming_distance,
                "max_retries": self.diversity.max_retries,
                "allow_partial": self.diversity.allow_partial,
            },
            "output_format": str(self.output_format),
            "chain_separator": str(self.chain_separator),
            "genre_hint": self.genre_hint,
            "persona": self.persona,
            "slot_biases": {
                k: {
                    "allowed_values": v.allowed_values,
                    "excluded_values": v.excluded_values,
                    "weights": v.weights,
                }
                for k, v in self.slot_biases.items()
            },
        }

    def with_seed(self, seed: int) -> GeneratorConfig:
        """Return a copy with a different seed."""
        return GeneratorConfig(
            seed=seed,
            diversity=self.diversity,
            output_format=self.output_format,
            chain_separator=self.chain_separator,
            genre_hint=self.genre_hint,
            persona=self.persona,
            slot_biases=self.slot_biases,
        )

    def with_genre_hint(self, genre_hint: str | None) -> GeneratorConfig:
        """Return a copy with a different genre hint."""
        return GeneratorConfig(
            seed=self.seed,
            diversity=self.diversity,
            output_format=self.output_format,
            chain_separator=self.chain_separator,
            genre_hint=genre_hint,
            persona=self.persona,
            slot_biases=self.slot_biases,
        )
