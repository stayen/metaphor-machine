"""
Pydantic schema models for style_components.yaml validation.

This module defines the complete schema for validating and loading
style component pools from YAML configuration files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class ModifiersComponents(BaseModel):
    """Schema for genre modifier pools (prefixes)."""

    mood: list[str] = Field(
        default_factory=list,
        description="Mood modifiers (dark, dreamy, etc.)",
    )
    intensity: list[str] = Field(
        default_factory=list,
        description="Intensity modifiers (hyper-, liquid, etc.)",
    )
    style: list[str] = Field(
        default_factory=list,
        description="Style modifiers (prog, symphonic, electro-, etc.)",
    )

    @property
    def all_modifiers(self) -> list[str]:
        """Flatten all modifier pools into single list."""
        return self.mood + self.intensity + self.style


class GenreComponents(BaseModel):
    """Schema for genre-related component pools organized by family."""

    # Genre families - each contains a list of genres
    electronic: list[str] = Field(
        default_factory=list,
        description="Electronic genres (house, techno, synthwave, etc.)",
    )
    hip_hop_urban: list[str] = Field(
        default_factory=list,
        description="Hip-hop and urban genres (boom bap, trap, drill, etc.)",
    )
    world_ethnic: list[str] = Field(
        default_factory=list,
        description="World and ethnic genres (afrobeat, reggae, salsa, etc.)",
    )
    rock_guitar: list[str] = Field(
        default_factory=list,
        description="Rock and guitar-based genres (rock, punk, shoegaze, etc.)",
    )
    traditional_acoustic: list[str] = Field(
        default_factory=list,
        description="Traditional and acoustic genres (jazz, blues, folk, etc.)",
    )

    # Modifier prefixes
    modifiers: ModifiersComponents = Field(
        default_factory=ModifiersComponents,
        description="Verified modifiers that work as prefixes",
    )

    # Additional prefixes
    regional: list[str] = Field(
        default_factory=list,
        description="Regional/language prefixes (arabic, japanese, etc.)",
    )
    location: list[str] = Field(
        default_factory=list,
        description="Location prefixes (dakar, havana, tokyo, etc.)",
    )
    instruments: list[str] = Field(
        default_factory=list,
        description="Instrument prefixes (piano, sitar, etc.)",
    )

    # Experimental/unverified genres
    experimental: list[str] = Field(
        default_factory=list,
        description="Experimental/unverified genres (use with caution)",
    )

    @property
    def genre_families(self) -> dict[str, list[str]]:
        """Return all genre families as a dictionary."""
        return {
            "electronic": self.electronic,
            "hip_hop_urban": self.hip_hop_urban,
            "world_ethnic": self.world_ethnic,
            "rock_guitar": self.rock_guitar,
            "traditional_acoustic": self.traditional_acoustic,
        }

    @property
    def all_core_genres(self) -> list[str]:
        """Flatten all core genre family pools into single list."""
        result = []
        for genres in self.genre_families.values():
            result.extend(genres)
        return result

    @property
    def all_genres(self) -> list[str]:
        """Flatten all genres including experimental."""
        return self.all_core_genres + self.experimental

    @property
    def all_prefixes(self) -> list[str]:
        """Flatten all prefix pools (modifiers, regional, location, instruments)."""
        return (
            self.modifiers.all_modifiers
            + self.regional
            + self.location
            + self.instruments
        )

    def get_family_for_genre(self, genre: str) -> str | None:
        """Find which family a genre belongs to."""
        genre_lower = genre.lower()
        for family, genres in self.genre_families.items():
            if genre_lower in [g.lower() for g in genres]:
                return family
        return None

    def build_genre_with_prefix(self, prefix: str, genre: str) -> str:
        """Combine prefix and genre into full genre anchor."""
        # Handle prefixes that already end with hyphen
        if prefix.endswith("-"):
            return f"{prefix}{genre}"
        return f"{prefix} {genre}"


class IntimateGestureComponents(BaseModel):
    """Schema for intimate gesture (vocal/lead behavior) component pools."""

    intensity_adjectives: dict[str, list[str]] = Field(
        ...,
        description="Adjective pools by category (energy, texture, emotional)",
    )
    delivery_nouns: dict[str, list[str]] = Field(
        ...,
        description="Delivery noun pools by category (spoken, sung, hybrid)",
    )

    @property
    def all_adjectives(self) -> list[str]:
        """Flatten all adjective pools into single list."""
        return [adj for pool in self.intensity_adjectives.values() for adj in pool]

    @property
    def all_nouns(self) -> list[str]:
        """Flatten all delivery noun pools into single list."""
        return [noun for pool in self.delivery_nouns.values() for noun in pool]

    def build_gesture(self, adjective: str, noun: str) -> str:
        """Combine adjective and noun into intimate gesture phrase."""
        return f"{adjective} {noun}"


class DynamicTensionComponents(BaseModel):
    """Schema for dynamic tension (motion/energy) component pools."""

    motion_verbs: list[str] = Field(
        ...,
        min_length=1,
        description="Verbs describing temporal motion (blooming, decaying, etc.)",
    )
    musical_objects: dict[str, list[str]] = Field(
        ...,
        description="Musical element pools by category (harmonic, percussive, etc.)",
    )

    @property
    def all_objects(self) -> list[str]:
        """Flatten all musical object pools into single list."""
        return [obj for pool in self.musical_objects.values() for obj in pool]

    def build_tension(self, verb: str, obj: str) -> str:
        """Combine motion verb and musical object into tension phrase."""
        return f"{verb} {obj}"


class SensoryBridgeComponents(BaseModel):
    """Schema for sensory bridge (environment/space) component pools."""

    environments: list[str] = Field(
        ...,
        min_length=1,
        description="Physical/metaphorical environments (forest, subway-tunnel, etc.)",
    )
    sensory_mediums: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Sensory medium pools by category (visual-lens, audio-effect)",
    )
    descriptors: list[str] = Field(
        default_factory=list,
        description="General sensory descriptors (reverb, haze, glow, etc.)",
    )

    @property
    def all_mediums(self) -> list[str]:
        """Flatten all sensory medium pools into single list."""
        return [med for pool in self.sensory_mediums.values() for med in pool]

    def build_bridge(self, environment: str, descriptor: str) -> str:
        """Combine environment and descriptor into sensory bridge phrase."""
        return f"{environment}-{descriptor}"


class EmotionalAnchorComponents(BaseModel):
    """Schema for emotional anchor component pools."""

    emotions: dict[str, list[str]] = Field(
        ...,
        description="Emotion pools by valence (negative, positive, complex)",
    )
    arcs: dict[str, list[str]] = Field(
        ...,
        description="Arc pools by type (musical, temporal)",
    )

    @property
    def all_emotions(self) -> list[str]:
        """Flatten all emotion pools into single list."""
        return [emo for pool in self.emotions.values() for emo in pool]

    @property
    def all_arcs(self) -> list[str]:
        """Flatten all arc pools into single list."""
        return [arc for pool in self.arcs.values() for arc in pool]

    def build_anchor(self, emotion: str, arc: str) -> str:
        """Combine emotion and arc into emotional anchor phrase."""
        return f"{emotion} {arc}"


class StyleComponents(BaseModel):
    """
    Complete schema for style_components.yaml.

    This is the main configuration class that holds all component pools
    used by the MetaphorGenerator.

    Attributes:
        genre: Genre component pools organized by family with modifiers
        intimate_gesture: Vocal/lead behavior component pools
        dynamic_tension: Motion/energy component pools
        sensory_bridge: Environment/space component pools
        emotional_anchor: Emotional resolution component pools

    Example:
        >>> components = StyleComponents.from_yaml("style_components.yaml")
        >>> len(components.genre.all_core_genres)
        153
        >>> components.genre.electronic[:3]
        ['acid house', 'acid jazz', 'acid rock']
        >>> components.intimate_gesture.intensity_adjectives["energy"]
        ['whispered', 'hushed', 'breathy', ...]
    """

    genre: GenreComponents
    intimate_gesture: IntimateGestureComponents
    dynamic_tension: DynamicTensionComponents
    sensory_bridge: SensoryBridgeComponents
    emotional_anchor: EmotionalAnchorComponents

    model_config = {"extra": "allow"}

    @classmethod
    def from_yaml(cls, path: str | Path) -> StyleComponents:
        """
        Load and validate style components from a YAML file.

        Args:
            path: Path to the YAML configuration file

        Returns:
            Validated StyleComponents instance

        Raises:
            FileNotFoundError: If the YAML file doesn't exist
            ValidationError: If the YAML structure is invalid
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Style components file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls.model_validate(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StyleComponents:
        """
        Create StyleComponents from a dictionary.

        Args:
            data: Dictionary matching the expected schema

        Returns:
            Validated StyleComponents instance
        """
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        """
        Save style components to a YAML file.

        Args:
            path: Destination path for the YAML file
        """
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    def get_pool_sizes(self) -> dict[str, int]:
        """
        Get approximate sizes of all component pools.

        Returns:
            Dictionary mapping pool names to item counts
        """
        return {
            "genre_core": len(self.genre.all_core_genres),
            "genre_experimental": len(self.genre.experimental),
            "genre_prefixes": len(self.genre.all_prefixes),
            "intimate_adjectives": len(self.intimate_gesture.all_adjectives),
            "intimate_nouns": len(self.intimate_gesture.all_nouns),
            "motion_verbs": len(self.dynamic_tension.motion_verbs),
            "musical_objects": len(self.dynamic_tension.all_objects),
            "environments": len(self.sensory_bridge.environments),
            "sensory_mediums": len(self.sensory_bridge.all_mediums),
            "emotions": len(self.emotional_anchor.all_emotions),
            "arcs": len(self.emotional_anchor.all_arcs),
        }

    def estimate_combinatorial_space(self) -> int:
        """
        Estimate the total number of possible unique metaphors.

        This is a rough lower bound based on minimum pool sizes.

        Returns:
            Approximate count of possible combinations
        """
        sizes = self.get_pool_sizes()

        # Genre combinations: core genres + optional prefixes
        # Each genre can be used alone or with any prefix
        genre_base = sizes["genre_core"] + sizes["genre_experimental"]
        prefix_count = sizes["genre_prefixes"]
        # Genres alone + genres with prefixes
        genre_count = genre_base + (genre_base * prefix_count)

        # Intimate gesture combinations
        gesture_count = sizes["intimate_adjectives"] * sizes["intimate_nouns"]

        # Dynamic tension combinations
        tension_count = sizes["motion_verbs"] * sizes["musical_objects"]

        # Sensory bridge combinations
        bridge_count = sizes["environments"] * max(
            sizes["sensory_mediums"], len(self.sensory_bridge.descriptors)
        )

        # Emotional anchor combinations
        anchor_count = sizes["emotions"] * sizes["arcs"]

        return genre_count * gesture_count * tension_count * bridge_count * anchor_count


# Convenience type alias
Components = StyleComponents
