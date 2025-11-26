"""
Metaphor Machine: Algorithmic generation of rich, plot-like style descriptions for AI music systems.

This package provides tools for generating structured "metaphor" prompts that guide
AI music generators (Suno, Producer.ai, etc.) to produce tracks with specific textures,
dynamics, and emotional arcs.

Example:
    >>> from metaphor_machine import MetaphorGenerator, StyleComponents
    >>> components = StyleComponents.from_yaml("style_components.yaml")
    >>> generator = MetaphorGenerator(components, seed=42)
    >>> metaphor = generator.generate_single()
    >>> print(metaphor)
    cinematic pop-ballad, whispered bedside confessions, slow-bloom piano harmonies, ...

The package supports:
- Single 5-slot metaphor generation
- 3-act chain generation (Intro → Mid → Outro)
- Seed-based reproducibility
- Hamming distance diversity constraints
- YAML schema validation
"""

__version__ = "0.1.0"
__author__ = "Metaphor Machine Contributors"

from metaphor_machine.core.generator import MetaphorGenerator
from metaphor_machine.core.metaphor import Metaphor, MetaphorChain, MetaphorSlot
from metaphor_machine.schemas.components import StyleComponents
from metaphor_machine.schemas.config import GeneratorConfig

__all__ = [
    # Core classes
    "MetaphorGenerator",
    "Metaphor",
    "MetaphorChain",
    "MetaphorSlot",
    # Schema classes
    "StyleComponents",
    "GeneratorConfig",
    # Version
    "__version__",
]
