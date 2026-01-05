"""
Core MetaphorGenerator class for producing structured metaphor descriptions.

This module contains the main generation engine that:
- Draws from component pools defined in style_components.yaml
- Supports seed-based reproducibility
- Enforces diversity constraints via Hamming distance
- Generates both single metaphors and 3-act chains
"""

from __future__ import annotations

import random
from typing import Callable, Iterator, Sequence

from metaphor_machine.core.metaphor import (
    ChainPosition,
    Metaphor,
    MetaphorChain,
    MetaphorSlot,
    SlotType,
    batch_min_distance,
)
from metaphor_machine.schemas.components import StyleComponents
from metaphor_machine.schemas.config import GeneratorConfig, SlotBias


class GenerationError(Exception):
    """Raised when generation fails (e.g., diversity constraints cannot be met)."""

    pass


class MetaphorGenerator:
    """
    Generator for structured metaphor descriptions.

    The generator draws from component pools to create 5-slot metaphors
    that can guide AI music systems toward specific textures and emotions.

    Attributes:
        components: StyleComponents instance containing all pools
        config: GeneratorConfig controlling generation behavior
        rng: Random number generator (seeded if config.seed is set)

    Example:
        >>> components = StyleComponents.from_yaml("style_components.yaml")
        >>> generator = MetaphorGenerator(components, seed=42)
        >>> metaphor = generator.generate_single()
        >>> print(metaphor)
        darkwave electro, whispered mantras, spiraling synths, neon-alley reverb, dread crescendo

        >>> chain = generator.generate_chain()
        >>> print(chain.to_suno_style())
        Intro: ... → Mid: ... → Outro: ...
    """

    def __init__(
        self,
        components: StyleComponents,
        config: GeneratorConfig | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Initialize the generator.

        Args:
            components: StyleComponents containing all pools
            config: Optional GeneratorConfig (uses defaults if not provided)
            seed: Shorthand for config.seed (config takes precedence)
        """
        self.components = components
        self.config = config or GeneratorConfig(seed=seed)
        
        # Initialize RNG
        effective_seed = self.config.seed if self.config.seed is not None else seed
        self.rng = random.Random(effective_seed)
        self._initial_seed = effective_seed

    def reseed(self, seed: int | None = None) -> None:
        """
        Reset the random number generator with a new seed.

        Args:
            seed: New seed value (None = random)
        """
        self._initial_seed = seed
        self.rng = random.Random(seed)

    @property
    def seed(self) -> int | None:
        """Return the initial seed used for this generator."""
        return self._initial_seed

    # -------------------------------------------------------------------------
    # Pool Selection Methods
    # -------------------------------------------------------------------------

    def _apply_bias(self, pool: list[str], slot_type: SlotType) -> list[str]:
        """Apply any configured bias to a pool."""
        bias = self.config.slot_biases.get(slot_type.value)
        if bias:
            pool = bias.filter_pool(pool)
        return pool

    def _weighted_choice(self, pool: list[str], slot_type: SlotType) -> str:
        """Select from pool with optional weighting."""
        pool = self._apply_bias(pool, slot_type)
        if not pool:
            raise GenerationError(f"No valid choices remain for {slot_type} after bias filtering")

        bias = self.config.slot_biases.get(slot_type.value)
        if bias and bias.weights:
            # Weighted selection
            weights = [bias.weights.get(v, 1.0) for v in pool]
            return self.rng.choices(pool, weights=weights, k=1)[0]
        else:
            return self.rng.choice(pool)

    def _select_genre_anchor(self) -> MetaphorSlot:
        """Select a genre anchor (genre with optional prefix)."""
        genre = self.components.genre

        # Get all core genres as the base pool
        genre_pool = genre.all_core_genres.copy()

        # Apply genre hint if configured
        if self.config.genre_hint:
            hint_lower = self.config.genre_hint.lower()
            matching_genres = [g for g in genre_pool if hint_lower in g.lower()]
            if matching_genres:
                genre_pool = matching_genres

        genre_pool = self._apply_bias(genre_pool, SlotType.GENRE_ANCHOR)
        if not genre_pool:
            raise GenerationError("No valid genres after filtering")

        selected_genre = self.rng.choice(genre_pool)

        # Optionally add a prefix (30% chance)
        use_prefix = self.rng.random() < 0.3
        prefix_pool = genre.all_prefixes

        if use_prefix and prefix_pool:
            prefix = self.rng.choice(prefix_pool)
            value = genre.build_genre_with_prefix(prefix, selected_genre)
            source_pool = f"genre/{genre.get_family_for_genre(selected_genre) or 'unknown'}+prefix"
        else:
            value = selected_genre
            source_pool = f"genre/{genre.get_family_for_genre(selected_genre) or 'unknown'}"

        return MetaphorSlot(
            slot_type=SlotType.GENRE_ANCHOR,
            value=value,
            source_pool=source_pool,
        )

    def _select_intimate_gesture(self) -> MetaphorSlot:
        """Select an intimate gesture (adjective + noun combination)."""
        gesture = self.components.intimate_gesture

        adj_pool = self._apply_bias(gesture.all_adjectives, SlotType.INTIMATE_GESTURE)
        noun_pool = gesture.all_nouns  # Nouns share same slot type

        if not adj_pool or not noun_pool:
            raise GenerationError("No valid intimate gesture components after filtering")

        adjective = self.rng.choice(adj_pool)
        noun = self.rng.choice(noun_pool)

        value = gesture.build_gesture(adjective, noun)
        return MetaphorSlot(
            slot_type=SlotType.INTIMATE_GESTURE,
            value=value,
            source_pool="intimate_gesture",
        )

    def _select_dynamic_tension(self) -> MetaphorSlot:
        """Select a dynamic tension phrase (verb + object combination)."""
        tension = self.components.dynamic_tension

        verb_pool = self._apply_bias(tension.motion_verbs, SlotType.DYNAMIC_TENSION)
        obj_pool = tension.all_objects

        if not verb_pool or not obj_pool:
            raise GenerationError("No valid dynamic tension components after filtering")

        verb = self.rng.choice(verb_pool)
        obj = self.rng.choice(obj_pool)

        value = tension.build_tension(verb, obj)
        return MetaphorSlot(
            slot_type=SlotType.DYNAMIC_TENSION,
            value=value,
            source_pool="dynamic_tension",
        )

    def _select_sensory_bridge(self) -> MetaphorSlot:
        """Select a sensory bridge (environment + descriptor combination)."""
        bridge = self.components.sensory_bridge

        env_pool = self._apply_bias(bridge.environments, SlotType.SENSORY_BRIDGE)

        # Choose from either mediums or descriptors
        descriptor_pool = bridge.all_mediums + bridge.descriptors
        if not descriptor_pool:
            descriptor_pool = ["reverb", "haze", "echo"]  # Fallback

        if not env_pool:
            raise GenerationError("No valid sensory bridge environments after filtering")

        environment = self.rng.choice(env_pool)
        descriptor = self.rng.choice(descriptor_pool)

        value = bridge.build_bridge(environment, descriptor)
        return MetaphorSlot(
            slot_type=SlotType.SENSORY_BRIDGE,
            value=value,
            source_pool="sensory_bridge",
        )

    def _select_emotional_anchor(self) -> MetaphorSlot:
        """Select an emotional anchor (emotion + arc combination)."""
        anchor = self.components.emotional_anchor

        emotion_pool = self._apply_bias(anchor.all_emotions, SlotType.EMOTIONAL_ANCHOR)
        arc_pool = anchor.all_arcs

        if not emotion_pool or not arc_pool:
            raise GenerationError("No valid emotional anchor components after filtering")

        emotion = self.rng.choice(emotion_pool)
        arc = self.rng.choice(arc_pool)

        value = anchor.build_anchor(emotion, arc)
        return MetaphorSlot(
            slot_type=SlotType.EMOTIONAL_ANCHOR,
            value=value,
            source_pool="emotional_anchor",
        )

    # -------------------------------------------------------------------------
    # Metaphor Generation Methods
    # -------------------------------------------------------------------------

    def generate_single(self, position: ChainPosition | None = None) -> Metaphor:
        """
        Generate a single 5-slot metaphor.

        Args:
            position: Optional chain position (intro/mid/outro) for context

        Returns:
            Complete Metaphor instance

        Example:
            >>> m = generator.generate_single()
            >>> str(m)
            'lo-fi boom-bap, breathy confessions, pulsing 808s, subway-tunnel echo, nostalgia comedown'
        """
        return Metaphor(
            genre_anchor=self._select_genre_anchor(),
            intimate_gesture=self._select_intimate_gesture(),
            dynamic_tension=self._select_dynamic_tension(),
            sensory_bridge=self._select_sensory_bridge(),
            emotional_anchor=self._select_emotional_anchor(),
            position=position,
        )

    def generate_chain(self) -> MetaphorChain:
        """
        Generate a 3-act metaphor chain (Intro → Mid → Outro).

        The chain represents a complete track arc with distinct
        metaphors for opening, peak, and resolution phases.

        Returns:
            MetaphorChain with intro, mid, and outro metaphors

        Example:
            >>> chain = generator.generate_chain()
            >>> print(chain.to_suno_style())
            Intro: cinematic orchestral, hushed lullaby-vocals, ...
        """
        return MetaphorChain(
            intro=self.generate_single(position=ChainPosition.INTRO),
            mid=self.generate_single(position=ChainPosition.MID),
            outro=self.generate_single(position=ChainPosition.OUTRO),
        )

    # -------------------------------------------------------------------------
    # Batch Generation with Diversity Constraints
    # -------------------------------------------------------------------------

    def generate_batch(
        self,
        count: int,
        enforce_diversity: bool = True,
    ) -> list[Metaphor]:
        """
        Generate a batch of metaphors with optional diversity constraints.

        When diversity is enforced, each new metaphor must have at least
        `min_hamming_distance` difference from all existing metaphors in
        the batch.

        Args:
            count: Number of metaphors to generate
            enforce_diversity: Whether to enforce Hamming distance constraints

        Returns:
            List of Metaphor instances

        Raises:
            GenerationError: If diversity constraints cannot be satisfied
                and allow_partial is False

        Example:
            >>> batch = generator.generate_batch(5, enforce_diversity=True)
            >>> batch_min_distance(batch)
            3  # At least 3 slots differ between any pair
        """
        if count < 1:
            return []

        batch: list[Metaphor] = []
        min_dist = self.config.diversity.min_hamming_distance
        max_retries = self.config.diversity.max_retries

        for i in range(count):
            if not enforce_diversity or i == 0:
                # First metaphor or no diversity constraint
                batch.append(self.generate_single())
            else:
                # Need to find a sufficiently different metaphor
                found = False
                for _ in range(max_retries):
                    candidate = self.generate_single()
                    # Check distance to all existing
                    if all(candidate.hamming_distance(m) >= min_dist for m in batch):
                        batch.append(candidate)
                        found = True
                        break

                if not found:
                    if self.config.diversity.allow_partial:
                        # Return what we have
                        break
                    else:
                        raise GenerationError(
                            f"Could not generate metaphor {i + 1}/{count} with "
                            f"min_hamming_distance={min_dist} after {max_retries} attempts"
                        )

        return batch

    def generate_chain_batch(
        self,
        count: int,
        enforce_diversity: bool = True,
    ) -> list[MetaphorChain]:
        """
        Generate a batch of 3-act chains with optional diversity constraints.

        Diversity is measured between intro metaphors of different chains.

        Args:
            count: Number of chains to generate
            enforce_diversity: Whether to enforce diversity between chains

        Returns:
            List of MetaphorChain instances
        """
        if count < 1:
            return []

        chains: list[MetaphorChain] = []
        min_dist = self.config.diversity.min_hamming_distance
        max_retries = self.config.diversity.max_retries

        for i in range(count):
            if not enforce_diversity or i == 0:
                chains.append(self.generate_chain())
            else:
                found = False
                for _ in range(max_retries):
                    candidate = self.generate_chain()
                    # Compare intro metaphors for diversity
                    if all(
                        candidate.intro.hamming_distance(c.intro) >= min_dist for c in chains
                    ):
                        chains.append(candidate)
                        found = True
                        break

                if not found:
                    if self.config.diversity.allow_partial:
                        break
                    else:
                        raise GenerationError(
                            f"Could not generate chain {i + 1}/{count} with sufficient diversity"
                        )

        return chains

    # -------------------------------------------------------------------------
    # Iterator Interface
    # -------------------------------------------------------------------------

    def iter_singles(self, count: int | None = None) -> Iterator[Metaphor]:
        """
        Iterate over generated single metaphors.

        Args:
            count: Maximum number to generate (None = infinite)

        Yields:
            Metaphor instances
        """
        generated = 0
        while count is None or generated < count:
            yield self.generate_single()
            generated += 1

    def iter_chains(self, count: int | None = None) -> Iterator[MetaphorChain]:
        """
        Iterate over generated chains.

        Args:
            count: Maximum number to generate (None = infinite)

        Yields:
            MetaphorChain instances
        """
        generated = 0
        while count is None or generated < count:
            yield self.generate_chain()
            generated += 1

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def get_state(self) -> tuple[int, ...]:
        """
        Get the current RNG state for later restoration.

        Returns:
            Tuple representing RNG state
        """
        return self.rng.getstate()  # type: ignore

    def set_state(self, state: tuple[int, ...]) -> None:
        """
        Restore RNG state from a previous get_state() call.

        Args:
            state: State tuple from get_state()
        """
        self.rng.setstate(state)  # type: ignore

    def clone(self, new_seed: int | None = None) -> MetaphorGenerator:
        """
        Create a copy of this generator with optional new seed.

        Args:
            new_seed: Seed for the new generator (None = same as current)

        Returns:
            New MetaphorGenerator instance
        """
        new_config = self.config.with_seed(new_seed if new_seed is not None else self.seed)
        return MetaphorGenerator(self.components, config=new_config)

    def __repr__(self) -> str:
        return f"MetaphorGenerator(seed={self.seed}, genre_hint={self.config.genre_hint!r})"
