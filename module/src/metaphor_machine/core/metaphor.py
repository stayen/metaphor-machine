"""
Core data structures for metaphor representation.

This module defines the fundamental building blocks:
- MetaphorSlot: Individual slot within a metaphor (genre, gesture, etc.)
- Metaphor: A complete 5-slot metaphor description
- MetaphorChain: A sequence of metaphors forming a 3-act structure
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, Sequence


class SlotType(str, Enum):
    """Enumeration of the five metaphor slot types."""

    GENRE_ANCHOR = "genre_anchor"
    INTIMATE_GESTURE = "intimate_gesture"
    DYNAMIC_TENSION = "dynamic_tension"
    SENSORY_BRIDGE = "sensory_bridge"
    EMOTIONAL_ANCHOR = "emotional_anchor"

    def __str__(self) -> str:
        return self.value


class ChainPosition(str, Enum):
    """Position within a 3-act metaphor chain."""

    INTRO = "intro"
    MID = "mid"
    OUTRO = "outro"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class MetaphorSlot:
    """
    A single slot within a metaphor.

    Each slot has a type (genre, gesture, etc.) and a value drawn from
    the corresponding pool in style_components.yaml.

    Attributes:
        slot_type: The category of this slot (genre_anchor, intimate_gesture, etc.)
        value: The actual text value for this slot
        source_pool: Optional path to the pool this value was drawn from

    Example:
        >>> slot = MetaphorSlot(SlotType.GENRE_ANCHOR, "cinematic pop-ballad")
        >>> str(slot)
        'cinematic pop-ballad'
    """

    slot_type: SlotType
    value: str
    source_pool: str | None = None

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"MetaphorSlot({self.slot_type.value}={self.value!r})"


@dataclass(frozen=True, slots=True)
class Metaphor:
    """
    A complete 5-slot metaphor description.

    A metaphor consists of exactly five slots:
    1. Genre anchor - the base genre/era/style
    2. Intimate gesture - vocal/lead behavior
    3. Dynamic tension - motion and energy evolution
    4. Sensory bridge - environment/space/lens
    5. Emotional anchor - the feeling expressed metaphorically

    Attributes:
        genre_anchor: The base genre/style slot
        intimate_gesture: The vocal/lead behavior slot
        dynamic_tension: The motion/energy slot
        sensory_bridge: The environment/space slot
        emotional_anchor: The emotional resolution slot
        position: Optional position if part of a chain (intro/mid/outro)

    Example:
        >>> m = Metaphor(
        ...     genre_anchor=MetaphorSlot(SlotType.GENRE_ANCHOR, "cinematic pop-ballad"),
        ...     intimate_gesture=MetaphorSlot(SlotType.INTIMATE_GESTURE, "whispered confessions"),
        ...     dynamic_tension=MetaphorSlot(SlotType.DYNAMIC_TENSION, "slow-bloom harmonies"),
        ...     sensory_bridge=MetaphorSlot(SlotType.SENSORY_BRIDGE, "bedroom-lamp reverb haze"),
        ...     emotional_anchor=MetaphorSlot(SlotType.EMOTIONAL_ANCHOR, "fragile-hope heartbeat"),
        ... )
        >>> str(m)
        'cinematic pop-ballad, whispered confessions, slow-bloom harmonies, ...'
    """

    genre_anchor: MetaphorSlot
    intimate_gesture: MetaphorSlot
    dynamic_tension: MetaphorSlot
    sensory_bridge: MetaphorSlot
    emotional_anchor: MetaphorSlot
    position: ChainPosition | None = None

    def __post_init__(self) -> None:
        """Validate slot types match expected positions."""
        expected = [
            (self.genre_anchor, SlotType.GENRE_ANCHOR),
            (self.intimate_gesture, SlotType.INTIMATE_GESTURE),
            (self.dynamic_tension, SlotType.DYNAMIC_TENSION),
            (self.sensory_bridge, SlotType.SENSORY_BRIDGE),
            (self.emotional_anchor, SlotType.EMOTIONAL_ANCHOR),
        ]
        for slot, expected_type in expected:
            if slot.slot_type != expected_type:
                raise ValueError(
                    f"Slot type mismatch: expected {expected_type}, got {slot.slot_type}"
                )

    @property
    def slots(self) -> tuple[MetaphorSlot, ...]:
        """Return all slots as a tuple in canonical order."""
        return (
            self.genre_anchor,
            self.intimate_gesture,
            self.dynamic_tension,
            self.sensory_bridge,
            self.emotional_anchor,
        )

    @property
    def slot_values(self) -> tuple[str, ...]:
        """Return just the string values of all slots."""
        return tuple(slot.value for slot in self.slots)

    def __iter__(self) -> Iterator[MetaphorSlot]:
        """Iterate over slots in canonical order."""
        return iter(self.slots)

    def __str__(self) -> str:
        """Format as comma-separated style description."""
        return ", ".join(slot.value for slot in self.slots)

    def __repr__(self) -> str:
        pos = f", position={self.position}" if self.position else ""
        return f"Metaphor({self.genre_anchor.value!r}, ...{pos})"

    def to_dict(self) -> dict[str, str | None]:
        """Convert to dictionary representation."""
        return {
            "genre_anchor": self.genre_anchor.value,
            "intimate_gesture": self.intimate_gesture.value,
            "dynamic_tension": self.dynamic_tension.value,
            "sensory_bridge": self.sensory_bridge.value,
            "emotional_anchor": self.emotional_anchor.value,
            "position": str(self.position) if self.position else None,
        }

    def hamming_distance(self, other: Metaphor) -> int:
        """
        Compute Hamming distance to another metaphor.

        Distance is the count of slots with different values (0-5).

        Args:
            other: Another Metaphor to compare against

        Returns:
            Integer distance from 0 (identical) to 5 (completely different)

        Example:
            >>> m1.hamming_distance(m2)
            3  # 3 of 5 slots differ
        """
        return sum(
            1 for s1, s2 in zip(self.slot_values, other.slot_values, strict=True) if s1 != s2
        )


@dataclass(frozen=True, slots=True)
class MetaphorChain:
    """
    A 3-act sequence of metaphors (Intro → Mid → Outro).

    The chain represents a complete track arc, with each metaphor
    describing a different phase of the musical journey.

    Attributes:
        intro: Opening metaphor - how the track begins
        mid: Middle metaphor - the lift/chorus/peak energy
        outro: Closing metaphor - decay/resolution/afterimage

    Example:
        >>> chain = MetaphorChain(intro=m1, mid=m2, outro=m3)
        >>> print(chain.to_suno_style())
        Intro: cinematic pop-ballad, ... ; Mid: voice opens into...; Outro: melody dissolves...
    """

    intro: Metaphor
    mid: Metaphor
    outro: Metaphor

    def __post_init__(self) -> None:
        """Validate chain positions if set."""
        if self.intro.position and self.intro.position != ChainPosition.INTRO:
            raise ValueError(f"Intro metaphor has wrong position: {self.intro.position}")
        if self.mid.position and self.mid.position != ChainPosition.MID:
            raise ValueError(f"Mid metaphor has wrong position: {self.mid.position}")
        if self.outro.position and self.outro.position != ChainPosition.OUTRO:
            raise ValueError(f"Outro metaphor has wrong position: {self.outro.position}")

    @property
    def metaphors(self) -> tuple[Metaphor, Metaphor, Metaphor]:
        """Return all metaphors as a tuple in order."""
        return (self.intro, self.mid, self.outro)

    def __iter__(self) -> Iterator[Metaphor]:
        """Iterate over metaphors in order."""
        return iter(self.metaphors)

    def __len__(self) -> int:
        """Chain always contains exactly 3 metaphors."""
        return 3

    def to_suno_style(self, separator: str = " → ") -> str:
        """
        Format as a single Style of Music field for Suno/Producer.ai.

        Args:
            separator: String to use between sections (default: " → ")

        Returns:
            Formatted string ready for paste into Style of Music field

        Example:
            >>> chain.to_suno_style()
            'Intro: cinematic... → Mid: voice opens... → Outro: melody dissolves...'
        """
        parts = [
            f"Intro: {self.intro}",
            f"Mid: {self.mid}",
            f"Outro: {self.outro}",
        ]
        return separator.join(parts)

    def to_dict(self) -> dict[str, dict[str, str | None]]:
        """Convert to dictionary representation."""
        return {
            "intro": self.intro.to_dict(),
            "mid": self.mid.to_dict(),
            "outro": self.outro.to_dict(),
        }

    def min_pairwise_distance(self) -> int:
        """
        Compute minimum Hamming distance between any two metaphors in chain.

        Returns:
            Minimum distance (0-5) across all pairs
        """
        distances = [
            self.intro.hamming_distance(self.mid),
            self.mid.hamming_distance(self.outro),
            self.intro.hamming_distance(self.outro),
        ]
        return min(distances)


def batch_min_distance(metaphors: Sequence[Metaphor]) -> int:
    """
    Compute minimum pairwise Hamming distance across a batch.

    Args:
        metaphors: Sequence of metaphors to compare

    Returns:
        Minimum distance found between any pair (0 if fewer than 2 metaphors)

    Example:
        >>> batch = [m1, m2, m3, m4]
        >>> batch_min_distance(batch)
        2  # At least 2 slots differ between any pair
    """
    if len(metaphors) < 2:
        return 0

    min_dist = 5  # Maximum possible distance
    for i, m1 in enumerate(metaphors):
        for m2 in metaphors[i + 1 :]:
            dist = m1.hamming_distance(m2)
            min_dist = min(min_dist, dist)
            if min_dist == 0:
                return 0  # Early exit - can't get lower
    return min_dist
