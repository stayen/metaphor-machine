"""
Shared pytest fixtures for metaphor_machine tests.
"""

from pathlib import Path

import pytest

from metaphor_machine.schemas.components import StyleComponents


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_yaml_content() -> str:
    """Return sample YAML content for testing."""
    return """
genre:
  eras:
    - lo-fi
    - darkwave
    - cinematic
  subgenres:
    lo-fi:
      - boom-bap
      - chill-hop
  fallback_subgenres:
    - fusion

intimate_gesture:
  intensity_adjectives:
    energy:
      - whispered
      - breathy
    texture:
      - creaking
  delivery_nouns:
    spoken:
      - confessions
    sung:
      - lullaby-vocals

dynamic_tension:
  motion_verbs:
    - blooming
    - decaying
  musical_objects:
    harmonic:
      - harmonies
    percussive:
      - 808s

sensory_bridge:
  environments:
    - forest
    - cathedral
  sensory_mediums:
    audio-effect:
      - reverb
  descriptors:
    - haze

emotional_anchor:
  emotions:
    negative:
      - heartbreak
    positive:
      - euphoria
  arcs:
    musical:
      - crescendo
"""


@pytest.fixture
def temp_yaml_file(tmp_path: Path, sample_yaml_content: str) -> Path:
    """Create a temporary YAML file for testing."""
    yaml_file = tmp_path / "test_components.yaml"
    yaml_file.write_text(sample_yaml_content)
    return yaml_file


@pytest.fixture
def loaded_components(temp_yaml_file: Path) -> StyleComponents:
    """Load StyleComponents from temporary YAML file."""
    return StyleComponents.from_yaml(temp_yaml_file)
