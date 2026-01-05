"""Shared test fixtures for Phase 2 modules."""

import pytest
from pathlib import Path


@pytest.fixture
def minimal_components_yaml(tmp_path):
    """Create a minimal components YAML file for testing."""
    yaml_content = """
genre:
  electronic:
    - darkwave
    - synthpop
    - ambient
    - house
  hip_hop_urban:
    - trip-hop
    - boom bap
  world_ethnic:
    - afrobeat
    - reggae
  rock_guitar:
    - rock
    - punk
  traditional_acoustic:
    - jazz
    - folk
  modifiers:
    mood:
      - dark
      - dreamy
    intensity:
      - hyper-
      - liquid
    style:
      - prog
      - electro-
  regional:
    - arabic
    - japanese
  location:
    - tokyo
    - new orleans
  instruments:
    - piano
    - sitar
  experimental:
    - vaporwave
    - hyperpop

intimate_gesture:
  intensity_adjectives:
    energy:
      - breathy
      - whispered
      - urgent
    texture:
      - reedy
      - silken
      - gravelly
    emotional:
      - vulnerable
      - ecstatic
  delivery_nouns:
    spoken:
      - confessions
      - murmurs
      - incantations
    sung:
      - hooks
      - melodies
      - harmonies
    hybrid:
      - chant-hooks
      - call-and-response

dynamic_tension:
  motion_verbs:
    - swelling
    - crackling
    - surging
    - decaying
    - rupturing
  musical_objects:
    harmonic:
      - synth-pads
      - chords
      - harmonies
    percussive:
      - beats
      - kicks
      - snares
    textural:
      - noise-beds
      - drones

sensory_bridge:
  environments:
    - basement
    - cathedral
    - neon-alley
    - shipwreck
    - attic
  sensory_mediums:
    visual_lens:
      - Polaroid
      - neon-glow
      - film-grain
    audio_effect:
      - reverb
      - static
      - echo
  descriptors:
    - reverb
    - haze
    - glow
    - shimmer

emotional_anchor:
  emotions:
    negative:
      - melancholy
      - dread
      - numbness
    positive:
      - hope
      - joy
      - awe
    complex:
      - bittersweet
      - longing
  arcs:
    musical:
      - drift
      - surge
      - flare
    temporal:
      - midnight-reckoning
      - dawn-break
"""
    yaml_file = tmp_path / "components.yaml"
    yaml_file.write_text(yaml_content)
    return yaml_file


@pytest.fixture
def temp_yaml_file(tmp_path, minimal_components_yaml):
    """Alias for minimal_components_yaml for compatibility."""
    return minimal_components_yaml
