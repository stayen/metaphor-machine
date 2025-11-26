"""
Tests for metaphor_machine.core.metaphor module.

Tests cover:
- MetaphorSlot creation and validation
- Metaphor creation, formatting, and distance calculation
- MetaphorChain creation and formatting
- Batch distance calculations
"""

import pytest

from metaphor_machine.core.metaphor import (
    ChainPosition,
    Metaphor,
    MetaphorChain,
    MetaphorSlot,
    SlotType,
    batch_min_distance,
)


# =============================================================================
# MetaphorSlot Tests
# =============================================================================


class TestMetaphorSlot:
    """Tests for MetaphorSlot dataclass."""

    def test_creation(self) -> None:
        """Test basic slot creation."""
        slot = MetaphorSlot(SlotType.GENRE_ANCHOR, "darkwave electro")
        assert slot.slot_type == SlotType.GENRE_ANCHOR
        assert slot.value == "darkwave electro"
        assert slot.source_pool is None

    def test_creation_with_source(self) -> None:
        """Test slot creation with source pool tracking."""
        slot = MetaphorSlot(
            SlotType.GENRE_ANCHOR,
            "lo-fi boom-bap",
            source_pool="genre.eras/lo-fi",
        )
        assert slot.source_pool == "genre.eras/lo-fi"

    def test_str_returns_value(self) -> None:
        """Test string conversion returns just the value."""
        slot = MetaphorSlot(SlotType.INTIMATE_GESTURE, "whispered confessions")
        assert str(slot) == "whispered confessions"

    def test_repr(self) -> None:
        """Test repr format."""
        slot = MetaphorSlot(SlotType.DYNAMIC_TENSION, "blooming harmonies")
        assert "MetaphorSlot" in repr(slot)
        assert "dynamic_tension" in repr(slot)
        assert "blooming harmonies" in repr(slot)

    def test_immutability(self) -> None:
        """Test that slots are immutable (frozen dataclass)."""
        slot = MetaphorSlot(SlotType.GENRE_ANCHOR, "test")
        with pytest.raises(AttributeError):
            slot.value = "modified"  # type: ignore


# =============================================================================
# Metaphor Tests
# =============================================================================


class TestMetaphor:
    """Tests for Metaphor dataclass."""

    @pytest.fixture
    def sample_metaphor(self) -> Metaphor:
        """Create a sample metaphor for testing."""
        return Metaphor(
            genre_anchor=MetaphorSlot(SlotType.GENRE_ANCHOR, "cinematic pop-ballad"),
            intimate_gesture=MetaphorSlot(SlotType.INTIMATE_GESTURE, "whispered confessions"),
            dynamic_tension=MetaphorSlot(SlotType.DYNAMIC_TENSION, "slow-bloom harmonies"),
            sensory_bridge=MetaphorSlot(SlotType.SENSORY_BRIDGE, "bedroom-lamp reverb"),
            emotional_anchor=MetaphorSlot(SlotType.EMOTIONAL_ANCHOR, "fragile-hope heartbeat"),
        )

    def test_creation(self, sample_metaphor: Metaphor) -> None:
        """Test basic metaphor creation."""
        assert sample_metaphor.genre_anchor.value == "cinematic pop-ballad"
        assert sample_metaphor.position is None

    def test_creation_with_position(self) -> None:
        """Test metaphor creation with chain position."""
        m = Metaphor(
            genre_anchor=MetaphorSlot(SlotType.GENRE_ANCHOR, "test"),
            intimate_gesture=MetaphorSlot(SlotType.INTIMATE_GESTURE, "test"),
            dynamic_tension=MetaphorSlot(SlotType.DYNAMIC_TENSION, "test"),
            sensory_bridge=MetaphorSlot(SlotType.SENSORY_BRIDGE, "test"),
            emotional_anchor=MetaphorSlot(SlotType.EMOTIONAL_ANCHOR, "test"),
            position=ChainPosition.INTRO,
        )
        assert m.position == ChainPosition.INTRO

    def test_slot_type_validation(self) -> None:
        """Test that slot types must match expected positions."""
        with pytest.raises(ValueError, match="Slot type mismatch"):
            Metaphor(
                genre_anchor=MetaphorSlot(SlotType.INTIMATE_GESTURE, "wrong"),  # Wrong type
                intimate_gesture=MetaphorSlot(SlotType.INTIMATE_GESTURE, "test"),
                dynamic_tension=MetaphorSlot(SlotType.DYNAMIC_TENSION, "test"),
                sensory_bridge=MetaphorSlot(SlotType.SENSORY_BRIDGE, "test"),
                emotional_anchor=MetaphorSlot(SlotType.EMOTIONAL_ANCHOR, "test"),
            )

    def test_slots_property(self, sample_metaphor: Metaphor) -> None:
        """Test slots property returns tuple in order."""
        slots = sample_metaphor.slots
        assert len(slots) == 5
        assert slots[0].slot_type == SlotType.GENRE_ANCHOR
        assert slots[4].slot_type == SlotType.EMOTIONAL_ANCHOR

    def test_slot_values_property(self, sample_metaphor: Metaphor) -> None:
        """Test slot_values property."""
        values = sample_metaphor.slot_values
        assert len(values) == 5
        assert values[0] == "cinematic pop-ballad"

    def test_str_formatting(self, sample_metaphor: Metaphor) -> None:
        """Test string formatting produces comma-separated output."""
        result = str(sample_metaphor)
        assert "cinematic pop-ballad" in result
        assert ", " in result
        parts = result.split(", ")
        assert len(parts) == 5

    def test_iteration(self, sample_metaphor: Metaphor) -> None:
        """Test metaphor is iterable over slots."""
        slots = list(sample_metaphor)
        assert len(slots) == 5
        assert all(isinstance(s, MetaphorSlot) for s in slots)

    def test_to_dict(self, sample_metaphor: Metaphor) -> None:
        """Test dictionary conversion."""
        d = sample_metaphor.to_dict()
        assert d["genre_anchor"] == "cinematic pop-ballad"
        assert d["position"] is None
        assert len(d) == 6  # 5 slots + position

    def test_hamming_distance_identical(self, sample_metaphor: Metaphor) -> None:
        """Test Hamming distance between identical metaphors is 0."""
        assert sample_metaphor.hamming_distance(sample_metaphor) == 0

    def test_hamming_distance_completely_different(self) -> None:
        """Test Hamming distance between completely different metaphors is 5."""
        m1 = Metaphor(
            genre_anchor=MetaphorSlot(SlotType.GENRE_ANCHOR, "a"),
            intimate_gesture=MetaphorSlot(SlotType.INTIMATE_GESTURE, "a"),
            dynamic_tension=MetaphorSlot(SlotType.DYNAMIC_TENSION, "a"),
            sensory_bridge=MetaphorSlot(SlotType.SENSORY_BRIDGE, "a"),
            emotional_anchor=MetaphorSlot(SlotType.EMOTIONAL_ANCHOR, "a"),
        )
        m2 = Metaphor(
            genre_anchor=MetaphorSlot(SlotType.GENRE_ANCHOR, "b"),
            intimate_gesture=MetaphorSlot(SlotType.INTIMATE_GESTURE, "b"),
            dynamic_tension=MetaphorSlot(SlotType.DYNAMIC_TENSION, "b"),
            sensory_bridge=MetaphorSlot(SlotType.SENSORY_BRIDGE, "b"),
            emotional_anchor=MetaphorSlot(SlotType.EMOTIONAL_ANCHOR, "b"),
        )
        assert m1.hamming_distance(m2) == 5

    def test_hamming_distance_partial(self) -> None:
        """Test Hamming distance with some slots different."""
        m1 = Metaphor(
            genre_anchor=MetaphorSlot(SlotType.GENRE_ANCHOR, "same"),
            intimate_gesture=MetaphorSlot(SlotType.INTIMATE_GESTURE, "same"),
            dynamic_tension=MetaphorSlot(SlotType.DYNAMIC_TENSION, "different1"),
            sensory_bridge=MetaphorSlot(SlotType.SENSORY_BRIDGE, "different1"),
            emotional_anchor=MetaphorSlot(SlotType.EMOTIONAL_ANCHOR, "different1"),
        )
        m2 = Metaphor(
            genre_anchor=MetaphorSlot(SlotType.GENRE_ANCHOR, "same"),
            intimate_gesture=MetaphorSlot(SlotType.INTIMATE_GESTURE, "same"),
            dynamic_tension=MetaphorSlot(SlotType.DYNAMIC_TENSION, "different2"),
            sensory_bridge=MetaphorSlot(SlotType.SENSORY_BRIDGE, "different2"),
            emotional_anchor=MetaphorSlot(SlotType.EMOTIONAL_ANCHOR, "different2"),
        )
        assert m1.hamming_distance(m2) == 3


# =============================================================================
# MetaphorChain Tests
# =============================================================================


class TestMetaphorChain:
    """Tests for MetaphorChain dataclass."""

    @pytest.fixture
    def make_metaphor(self):
        """Factory for creating test metaphors."""
        def _make(prefix: str, position: ChainPosition | None = None) -> Metaphor:
            return Metaphor(
                genre_anchor=MetaphorSlot(SlotType.GENRE_ANCHOR, f"{prefix}-genre"),
                intimate_gesture=MetaphorSlot(SlotType.INTIMATE_GESTURE, f"{prefix}-gesture"),
                dynamic_tension=MetaphorSlot(SlotType.DYNAMIC_TENSION, f"{prefix}-tension"),
                sensory_bridge=MetaphorSlot(SlotType.SENSORY_BRIDGE, f"{prefix}-bridge"),
                emotional_anchor=MetaphorSlot(SlotType.EMOTIONAL_ANCHOR, f"{prefix}-anchor"),
                position=position,
            )
        return _make

    def test_creation(self, make_metaphor) -> None:
        """Test basic chain creation."""
        chain = MetaphorChain(
            intro=make_metaphor("intro"),
            mid=make_metaphor("mid"),
            outro=make_metaphor("outro"),
        )
        assert chain.intro.genre_anchor.value == "intro-genre"
        assert chain.mid.genre_anchor.value == "mid-genre"
        assert chain.outro.genre_anchor.value == "outro-genre"

    def test_metaphors_property(self, make_metaphor) -> None:
        """Test metaphors property returns tuple in order."""
        chain = MetaphorChain(
            intro=make_metaphor("a"),
            mid=make_metaphor("b"),
            outro=make_metaphor("c"),
        )
        metaphors = chain.metaphors
        assert len(metaphors) == 3
        assert metaphors[0].genre_anchor.value == "a-genre"

    def test_len(self, make_metaphor) -> None:
        """Test chain length is always 3."""
        chain = MetaphorChain(
            intro=make_metaphor("a"),
            mid=make_metaphor("b"),
            outro=make_metaphor("c"),
        )
        assert len(chain) == 3

    def test_iteration(self, make_metaphor) -> None:
        """Test chain is iterable."""
        chain = MetaphorChain(
            intro=make_metaphor("a"),
            mid=make_metaphor("b"),
            outro=make_metaphor("c"),
        )
        metaphors = list(chain)
        assert len(metaphors) == 3

    def test_to_suno_style_default(self, make_metaphor) -> None:
        """Test Suno style formatting with default separator."""
        chain = MetaphorChain(
            intro=make_metaphor("intro"),
            mid=make_metaphor("mid"),
            outro=make_metaphor("outro"),
        )
        result = chain.to_suno_style()
        assert "Intro:" in result
        assert "Mid:" in result
        assert "Outro:" in result
        assert " → " in result

    def test_to_suno_style_custom_separator(self, make_metaphor) -> None:
        """Test Suno style formatting with custom separator."""
        chain = MetaphorChain(
            intro=make_metaphor("intro"),
            mid=make_metaphor("mid"),
            outro=make_metaphor("outro"),
        )
        result = chain.to_suno_style(separator="; ")
        assert "; " in result
        assert " → " not in result

    def test_to_dict(self, make_metaphor) -> None:
        """Test dictionary conversion."""
        chain = MetaphorChain(
            intro=make_metaphor("intro"),
            mid=make_metaphor("mid"),
            outro=make_metaphor("outro"),
        )
        d = chain.to_dict()
        assert "intro" in d
        assert "mid" in d
        assert "outro" in d
        assert d["intro"]["genre_anchor"] == "intro-genre"

    def test_min_pairwise_distance(self, make_metaphor) -> None:
        """Test minimum pairwise distance calculation."""
        # All different
        chain = MetaphorChain(
            intro=make_metaphor("a"),
            mid=make_metaphor("b"),
            outro=make_metaphor("c"),
        )
        assert chain.min_pairwise_distance() == 5  # All slots differ

    def test_position_validation(self, make_metaphor) -> None:
        """Test that wrong positions are rejected."""
        with pytest.raises(ValueError, match="wrong position"):
            MetaphorChain(
                intro=make_metaphor("intro", position=ChainPosition.MID),  # Wrong!
                mid=make_metaphor("mid"),
                outro=make_metaphor("outro"),
            )


# =============================================================================
# Batch Distance Tests
# =============================================================================


class TestBatchMinDistance:
    """Tests for batch_min_distance function."""

    def make_metaphor(self, value: str) -> Metaphor:
        """Create metaphor with same value in all slots."""
        return Metaphor(
            genre_anchor=MetaphorSlot(SlotType.GENRE_ANCHOR, value),
            intimate_gesture=MetaphorSlot(SlotType.INTIMATE_GESTURE, value),
            dynamic_tension=MetaphorSlot(SlotType.DYNAMIC_TENSION, value),
            sensory_bridge=MetaphorSlot(SlotType.SENSORY_BRIDGE, value),
            emotional_anchor=MetaphorSlot(SlotType.EMOTIONAL_ANCHOR, value),
        )

    def test_empty_batch(self) -> None:
        """Test empty batch returns 0."""
        assert batch_min_distance([]) == 0

    def test_single_item(self) -> None:
        """Test single-item batch returns 0."""
        assert batch_min_distance([self.make_metaphor("a")]) == 0

    def test_identical_items(self) -> None:
        """Test batch of identical items returns 0."""
        m = self.make_metaphor("same")
        assert batch_min_distance([m, m, m]) == 0

    def test_all_different(self) -> None:
        """Test batch of completely different items returns 5."""
        batch = [self.make_metaphor(str(i)) for i in range(5)]
        assert batch_min_distance(batch) == 5

    def test_mixed_distances(self) -> None:
        """Test batch with varying distances."""
        m1 = self.make_metaphor("a")
        m2 = self.make_metaphor("b")
        m3 = self.make_metaphor("a")  # Same as m1
        assert batch_min_distance([m1, m2, m3]) == 0  # m1 and m3 are identical
