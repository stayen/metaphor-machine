"""
Metaphor Machine - Algorithmic AI Music Prompt Generation and Optimization

A Python package for generating and optimizing structured style descriptions
for AI music platforms like Suno and Producer.ai.

Core features:
- 5-slot metaphor structure (genre, gesture, tension, bridge, anchor)
- Seed-based reproducibility
- Diversity constraints via Hamming distance
- 3-act chain generation (intro → mid → outro)

Optimization features (v1.0):
- Fitness evaluation framework with pluggable evaluators
- Genetic algorithm optimization
- Bayesian optimization for expensive evaluations
- Corpus storage for tracking prompt-outcome pairs

Quick start:
    >>> from metaphor_machine import StyleComponents, MetaphorGenerator
    >>> components = StyleComponents.from_yaml("style_components.yaml")
    >>> generator = MetaphorGenerator(components, seed=42)
    >>> print(generator.generate_single())
    darkwave electro, whispered mantras, spiraling synths, neon-alley reverb, dread crescendo
"""

__version__ = "1.0.0"
__author__ = "stayen"

# Core exports
from metaphor_machine.core.metaphor import (
    SlotType,
    ChainPosition,
    MetaphorSlot,
    Metaphor,
    MetaphorChain,
    batch_min_distance,
)
from metaphor_machine.core.generator import (
    GenerationError,
    MetaphorGenerator,
)

# Schema exports
from metaphor_machine.schemas.components import (
    StyleComponents,
    Components,
)
from metaphor_machine.schemas.config import (
    OutputFormat,
    ChainSeparator,
    DiversityConfig,
    SlotBias,
    GeneratorConfig,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Core types
    "SlotType",
    "ChainPosition",
    "MetaphorSlot",
    "Metaphor",
    "MetaphorChain",
    "batch_min_distance",
    # Generator
    "GenerationError",
    "MetaphorGenerator",
    # Components
    "StyleComponents",
    "Components",
    # Config
    "OutputFormat",
    "ChainSeparator",
    "DiversityConfig",
    "SlotBias",
    "GeneratorConfig",
]
