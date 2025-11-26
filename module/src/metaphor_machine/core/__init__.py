"""Core metaphor data structures and generation engine."""

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

__all__ = [
    "SlotType",
    "ChainPosition",
    "MetaphorSlot",
    "Metaphor",
    "MetaphorChain",
    "batch_min_distance",
    "GenerationError",
    "MetaphorGenerator",
]
