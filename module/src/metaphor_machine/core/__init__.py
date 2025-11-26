"""Core generation and data structure modules."""

from metaphor_machine.core.generator import GenerationError, MetaphorGenerator
from metaphor_machine.core.metaphor import (
    ChainPosition,
    Metaphor,
    MetaphorChain,
    MetaphorSlot,
    SlotType,
    batch_min_distance,
)

__all__ = [
    "MetaphorGenerator",
    "GenerationError",
    "Metaphor",
    "MetaphorChain",
    "MetaphorSlot",
    "SlotType",
    "ChainPosition",
    "batch_min_distance",
]
