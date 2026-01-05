"""
Tests for metaphor_machine.core.generator module.

Tests cover:
- Generator initialization and seeding
- Single metaphor generation
- Chain generation
- Batch generation with diversity constraints
- Genre hint biasing
"""

import pytest

from metaphor_machine.core.generator import GenerationError, MetaphorGenerator
from metaphor_machine.core.metaphor import Metaphor, MetaphorChain, batch_min_distance
from metaphor_machine.schemas.components import StyleComponents
from metaphor_machine.schemas.config import DiversityConfig, GeneratorConfig


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def minimal_components() -> StyleComponents:
    """Create minimal valid StyleComponents for testing."""
    return StyleComponents.from_dict({
        "genre": {
            "electronic": ["lo-fi", "darkwave", "hyperpop", "synthwave"],
            "hip_hop_urban": ["boom bap", "trap"],
            "world_ethnic": ["afrobeat", "reggae"],
            "rock_guitar": ["rock", "punk"],
            "traditional_acoustic": ["cinematic", "ritual", "jazz"],
            "modifiers": {
                "mood": ["dark", "dreamy"],
                "intensity": ["hyper-", "liquid"],
                "style": ["prog", "electro-"],
            },
            "regional": ["arabic", "japanese"],
            "location": ["tokyo", "new orleans"],
            "instruments": ["piano", "sitar"],
            "experimental": ["vaporwave"],
        },
        "intimate_gesture": {
            "intensity_adjectives": {
                "energy": ["whispered", "hushed", "breathy"],
                "texture": ["creaking", "crackled", "glassy"],
            },
            "delivery_nouns": {
                "spoken": ["confessions", "murmurs", "mantras"],
                "sung": ["falsetto-runs", "lullaby-vocals"],
            },
        },
        "dynamic_tension": {
            "motion_verbs": ["blooming", "decaying", "spiraling", "pulsing"],
            "musical_objects": {
                "harmonic": ["harmonies", "arpeggios"],
                "percussive": ["808s", "hi-hats"],
            },
        },
        "sensory_bridge": {
            "environments": ["forest", "subway-tunnel", "cathedral", "neon-alley"],
            "sensory_mediums": {
                "visual-lens": ["VHS", "neon-lens"],
                "audio-effect": ["reverb", "echo"],
            },
            "descriptors": ["haze", "glow", "shimmer"],
        },
        "emotional_anchor": {
            "emotions": {
                "negative": ["heartbreak", "dread", "longing"],
                "positive": ["euphoria", "relief"],
                "complex": ["nostalgia", "bittersweet"],
            },
            "arcs": {
                "musical": ["crescendo", "comedown", "afterglow"],
                "temporal": ["surrender", "quiet-resolve"],
            },
        },
    })


@pytest.fixture
def generator(minimal_components: StyleComponents) -> MetaphorGenerator:
    """Create a seeded generator for reproducible tests."""
    return MetaphorGenerator(minimal_components, seed=42)


# =============================================================================
# Initialization Tests
# =============================================================================


class TestGeneratorInit:
    """Tests for generator initialization."""

    def test_basic_init(self, minimal_components: StyleComponents) -> None:
        """Test basic initialization."""
        gen = MetaphorGenerator(minimal_components)
        assert gen.components is minimal_components
        assert gen.config is not None

    def test_init_with_seed(self, minimal_components: StyleComponents) -> None:
        """Test initialization with seed."""
        gen = MetaphorGenerator(minimal_components, seed=42)
        assert gen.seed == 42

    def test_init_with_config(self, minimal_components: StyleComponents) -> None:
        """Test initialization with config."""
        config = GeneratorConfig(seed=123, genre_hint="darkwave")
        gen = MetaphorGenerator(minimal_components, config=config)
        assert gen.seed == 123
        assert gen.config.genre_hint == "darkwave"

    def test_config_seed_takes_precedence(self, minimal_components: StyleComponents) -> None:
        """Test that config seed overrides seed parameter."""
        config = GeneratorConfig(seed=100)
        gen = MetaphorGenerator(minimal_components, config=config, seed=200)
        assert gen.seed == 100  # Config takes precedence

    def test_reseed(self, generator: MetaphorGenerator) -> None:
        """Test reseeding the generator."""
        generator.reseed(999)
        assert generator.seed == 999


# =============================================================================
# Reproducibility Tests
# =============================================================================


class TestReproducibility:
    """Tests for seed-based reproducibility."""

    def test_same_seed_same_output(self, minimal_components: StyleComponents) -> None:
        """Test that same seed produces same output."""
        gen1 = MetaphorGenerator(minimal_components, seed=42)
        gen2 = MetaphorGenerator(minimal_components, seed=42)

        m1 = gen1.generate_single()
        m2 = gen2.generate_single()

        assert str(m1) == str(m2)

    def test_different_seed_different_output(self, minimal_components: StyleComponents) -> None:
        """Test that different seeds produce different output."""
        gen1 = MetaphorGenerator(minimal_components, seed=42)
        gen2 = MetaphorGenerator(minimal_components, seed=43)

        m1 = gen1.generate_single()
        m2 = gen2.generate_single()

        # Very unlikely to be identical with different seeds
        assert str(m1) != str(m2)

    def test_state_save_restore(self, generator: MetaphorGenerator) -> None:
        """Test saving and restoring RNG state."""
        # Generate one metaphor
        m1 = generator.generate_single()

        # Save state
        state = generator.get_state()

        # Generate another
        m2 = generator.generate_single()

        # Restore state
        generator.set_state(state)

        # Should reproduce m2
        m3 = generator.generate_single()
        assert str(m2) == str(m3)


# =============================================================================
# Single Generation Tests
# =============================================================================


class TestSingleGeneration:
    """Tests for single metaphor generation."""

    def test_generate_single_returns_metaphor(self, generator: MetaphorGenerator) -> None:
        """Test generate_single returns a Metaphor instance."""
        m = generator.generate_single()
        assert isinstance(m, Metaphor)

    def test_generate_single_has_all_slots(self, generator: MetaphorGenerator) -> None:
        """Test generated metaphor has all 5 slots filled."""
        m = generator.generate_single()
        assert len(m.slots) == 5
        assert all(slot.value for slot in m.slots)

    def test_generate_single_with_position(self, generator: MetaphorGenerator) -> None:
        """Test generate_single can set chain position."""
        from metaphor_machine.core.metaphor import ChainPosition

        m = generator.generate_single(position=ChainPosition.INTRO)
        assert m.position == ChainPosition.INTRO

    def test_generate_single_values_from_pools(
        self, generator: MetaphorGenerator, minimal_components: StyleComponents
    ) -> None:
        """Test generated values come from component pools."""
        m = generator.generate_single()

        # Genre should be from core genres (possibly with prefix)
        genre_value = m.genre_anchor.value
        # At minimum, should contain a core genre or experimental genre
        all_genres = minimal_components.genre.all_genres
        assert any(genre in genre_value for genre in all_genres)


# =============================================================================
# Chain Generation Tests
# =============================================================================


class TestChainGeneration:
    """Tests for 3-act chain generation."""

    def test_generate_chain_returns_chain(self, generator: MetaphorGenerator) -> None:
        """Test generate_chain returns a MetaphorChain instance."""
        chain = generator.generate_chain()
        assert isinstance(chain, MetaphorChain)

    def test_generate_chain_has_three_parts(self, generator: MetaphorGenerator) -> None:
        """Test chain has intro, mid, and outro."""
        chain = generator.generate_chain()
        assert chain.intro is not None
        assert chain.mid is not None
        assert chain.outro is not None

    def test_generate_chain_parts_are_different(self, generator: MetaphorGenerator) -> None:
        """Test chain parts are typically different."""
        chain = generator.generate_chain()
        # With reasonable pool sizes, parts should differ
        assert chain.intro.hamming_distance(chain.mid) > 0 or \
               chain.mid.hamming_distance(chain.outro) > 0


# =============================================================================
# Batch Generation Tests
# =============================================================================


class TestBatchGeneration:
    """Tests for batch generation with diversity constraints."""

    def test_generate_batch_count(self, generator: MetaphorGenerator) -> None:
        """Test batch generates requested count."""
        batch = generator.generate_batch(5, enforce_diversity=False)
        assert len(batch) == 5

    def test_generate_batch_empty(self, generator: MetaphorGenerator) -> None:
        """Test batch with count=0 returns empty list."""
        batch = generator.generate_batch(0)
        assert batch == []

    def test_generate_batch_diversity_enforced(
        self, minimal_components: StyleComponents
    ) -> None:
        """Test diversity constraints are enforced."""
        config = GeneratorConfig(
            seed=42,
            diversity=DiversityConfig(min_hamming_distance=2),
        )
        gen = MetaphorGenerator(minimal_components, config=config)

        batch = gen.generate_batch(5, enforce_diversity=True)

        # Check all pairs meet minimum distance
        min_dist = batch_min_distance(batch)
        assert min_dist >= 2

    def test_generate_batch_partial_on_failure(
        self, minimal_components: StyleComponents
    ) -> None:
        """Test partial batch returned when constraints cannot be met."""
        # Create very strict constraints
        config = GeneratorConfig(
            seed=42,
            diversity=DiversityConfig(
                min_hamming_distance=5,  # Maximum possible distance
                max_retries=10,
                allow_partial=True,
            ),
        )
        gen = MetaphorGenerator(minimal_components, config=config)

        # Request more than possible with max distance constraint
        batch = gen.generate_batch(100, enforce_diversity=True)

        # Should return partial batch
        assert 0 < len(batch) < 100

    def test_generate_batch_raises_on_strict_failure(
        self, minimal_components: StyleComponents
    ) -> None:
        """Test error raised when constraints cannot be met and partial not allowed."""
        config = GeneratorConfig(
            seed=42,
            diversity=DiversityConfig(
                min_hamming_distance=5,
                max_retries=5,
                allow_partial=False,
            ),
        )
        gen = MetaphorGenerator(minimal_components, config=config)

        with pytest.raises(GenerationError):
            gen.generate_batch(100, enforce_diversity=True)

    def test_generate_chain_batch(self, generator: MetaphorGenerator) -> None:
        """Test chain batch generation."""
        chains = generator.generate_chain_batch(3, enforce_diversity=False)
        assert len(chains) == 3
        assert all(isinstance(c, MetaphorChain) for c in chains)


# =============================================================================
# Genre Hint Tests
# =============================================================================


class TestGenreHint:
    """Tests for genre hint biasing."""

    def test_genre_hint_influences_output(
        self, minimal_components: StyleComponents
    ) -> None:
        """Test that genre hint biases generation toward matching genres."""
        config = GeneratorConfig(seed=42, genre_hint="darkwave")
        gen = MetaphorGenerator(minimal_components, config=config)

        # Generate several and check for darkwave presence
        metaphors = [gen.generate_single() for _ in range(10)]
        darkwave_count = sum(
            1 for m in metaphors if "darkwave" in m.genre_anchor.value.lower()
        )

        # Should have more darkwave than without hint
        assert darkwave_count > 0

    def test_genre_hint_fallback(self, minimal_components: StyleComponents) -> None:
        """Test that non-matching hint still produces output."""
        config = GeneratorConfig(seed=42, genre_hint="nonexistent-genre")
        gen = MetaphorGenerator(minimal_components, config=config)

        # Should still generate (falls back to all genres)
        m = gen.generate_single()
        assert m is not None


# =============================================================================
# Iterator Tests
# =============================================================================


class TestIterators:
    """Tests for iterator interfaces."""

    def test_iter_singles_finite(self, generator: MetaphorGenerator) -> None:
        """Test finite iteration over singles."""
        metaphors = list(generator.iter_singles(count=5))
        assert len(metaphors) == 5

    def test_iter_chains_finite(self, generator: MetaphorGenerator) -> None:
        """Test finite iteration over chains."""
        chains = list(generator.iter_chains(count=3))
        assert len(chains) == 3

    def test_iter_singles_infinite(self, generator: MetaphorGenerator) -> None:
        """Test infinite iteration (take first N)."""
        iterator = generator.iter_singles(count=None)
        first_10 = [next(iterator) for _ in range(10)]
        assert len(first_10) == 10


# =============================================================================
# Clone Tests
# =============================================================================


class TestClone:
    """Tests for generator cloning."""

    def test_clone_same_seed(self, generator: MetaphorGenerator) -> None:
        """Test cloning with same seed produces same output."""
        cloned = generator.clone()

        m1 = generator.generate_single()
        generator.reseed(generator.seed)
        m2 = cloned.generate_single()

        # Both should produce same first output with same seed
        assert str(m1) == str(m2)

    def test_clone_new_seed(self, generator: MetaphorGenerator) -> None:
        """Test cloning with new seed produces different output."""
        cloned = generator.clone(new_seed=999)

        m1 = generator.generate_single()
        m2 = cloned.generate_single()

        assert str(m1) != str(m2)
