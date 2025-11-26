# Metaphor Machine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Algorithmic generation of rich, plot-like style descriptions for AI music systems.**

The Metaphor Machine produces structured 5-slot "metaphors" that guide AI music generators (Suno, Producer.ai, Udio) toward specific textures, dynamics, and emotional arcs—transforming simple genre labels into cinematic sound narratives.

## What is a "Metaphor"?

A **metaphor** is a short, evocative description that behaves like a *mini-plot* for a musical track:

```
cinematic pop-ballad, whispered bedside confessions, slow-bloom piano harmonies,
bedroom-lamp reverb haze, fragile-hope heartbeat
```

Each metaphor consists of **5 slots**:

| Slot | Purpose | Example |
|------|---------|---------|
| **Genre Anchor** | Base genre/era/style | `darkwave electro` |
| **Intimate Gesture** | Vocal/lead behavior | `whispered confessions` |
| **Dynamic Tension** | Motion through time | `slow-bloom harmonies` |
| **Sensory Bridge** | Environment/space/lens | `neon-alley reverb haze` |
| **Emotional Anchor** | Inner feeling | `heartbreak crescendo` |

## Why Use It?

When tested with Suno v5.0 and Producer.ai FUZZ-2.0-Pro:
- **100% success rate** producing rich vocal textures and "songs in unknown language" effects
- Dramatically more varied outputs than bare genre labels
- Reproducible results via seed-based generation
- Systematic exploration of the vast combinatorial space

## Installation

```bash
# From PyPI (when published)
pip install metaphor-machine

# From source
git clone https://github.com/stayen/metaphor-machine.git
cd module
pip install -e ".[dev]"
```

## Quick Start

### Python API

```python
from metaphor_machine import MetaphorGenerator, StyleComponents

# Load component pools
components = StyleComponents.from_yaml("style_components.yaml")

# Create generator with seed for reproducibility
generator = MetaphorGenerator(components, seed=42)

# Generate a single metaphor
metaphor = generator.generate_single()
print(metaphor)
# → darkwave electro, whispered mantras, spiraling synths, neon-alley reverb, dread crescendo

# Generate a 3-act chain (Intro → Mid → Outro)
chain = generator.generate_chain()
print(chain.to_suno_style())
# → Intro: cinematic orchestral, ... → Mid: voice soars... → Outro: melody dissolves...

# Generate a diverse batch
batch = generator.generate_batch(10, enforce_diversity=True)
```

### Command Line

```bash
# Single metaphor
metaphor generate --seed 42

# 3-act chain
metaphor chain --seed 42 --format suno

# Diverse batch
metaphor batch --count 10 --min-distance 3 --format json

# Explore seeds around a known good value
metaphor explore --seed 42 --range 10

# Show pool statistics
metaphor info
```

## Configuration

### style_components.yaml

The `style_components.yaml` file defines all component pools:

```yaml
genre:
  eras:
    - lo-fi
    - darkwave
    - cinematic
    # ...
  subgenres:
    lo-fi:
      - boom-bap
      - chill-hop
    # ...

intimate_gesture:
  intensity_adjectives:
    energy:
      - whispered
      - hushed
      - breathy
    # ...
  delivery_nouns:
    spoken:
      - confessions
      - murmurs
    # ...

# ... (dynamic_tension, sensory_bridge, emotional_anchor)
```

### Generator Configuration

```python
from metaphor_machine import GeneratorConfig, DiversityConfig

config = GeneratorConfig(
    seed=42,                           # Random seed for reproducibility
    genre_hint="darkwave",             # Bias toward specific genre
    diversity=DiversityConfig(
        min_hamming_distance=3,        # Min slots different between batch items
        max_retries=100,               # Attempts before giving up
        allow_partial=True,            # Return partial batch on failure
    ),
)

generator = MetaphorGenerator(components, config=config)
```

## Key Features

### Seed-Based Reproducibility

Same seed + same components = same outputs:

```python
gen1 = MetaphorGenerator(components, seed=42)
gen2 = MetaphorGenerator(components, seed=42)

assert str(gen1.generate_single()) == str(gen2.generate_single())
```

### Diversity Constraints

Ensure batch variety via Hamming distance (count of differing slots):

```python
batch = generator.generate_batch(10, enforce_diversity=True)
# All pairs differ in at least 3 of 5 slots
```

### 3-Act Chains

Generate complete track arcs:

```python
chain = generator.generate_chain()
# Intro: how it begins
# Mid: the peak/chorus
# Outro: the resolution

print(chain.to_suno_style(separator=" → "))
```

### Platform Formatting

```python
from metaphor_machine.utils.formatting import format_for_suno, format_for_producer_ai

# Suno (120 char limit)
suno_style = format_for_suno(metaphor)

# Producer.ai (longer prompts allowed)
producer_prompt = format_for_producer_ai(chain)
```

## Development

### Setup

```bash
git clone https://github.com/stayen/metaphor-machine.git
cd module
pip install -e ".[dev]"
pre-commit install
```

### Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=metaphor_machine --cov-report=html

# Type checking
mypy src/metaphor_machine
```

### Code Quality

```bash
# Lint and format
ruff check src tests
ruff format src tests
```

## Architecture

```
src/metaphor_machine/
├── __init__.py           # Package exports
├── core/
│   ├── metaphor.py       # Data structures (Metaphor, MetaphorChain, etc.)
│   └── generator.py      # MetaphorGenerator class
├── schemas/
│   ├── components.py     # Pydantic models for style_components.yaml
│   └── config.py         # GeneratorConfig and related
├── cli/
│   └── main.py           # Click CLI implementation
└── utils/
    ├── formatting.py     # Platform-specific formatters
    └── validation.py     # YAML validation utilities
```

## Roadmap

- **Phase 1** ✅ Core package structure, CLI, tests
- **Phase 2** 🔜 Optimization layer (genetic algorithms, Bayesian optimization)
- **Phase 3** 🔜 Learning layer (Markov chains on successful prompts, embeddings)
- **Phase 4** 🔜 Platform integrations, analytics dashboard

## Contributing

Contributions welcome! Please read our contributing guidelines and submit PRs.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Inspired by community discoveries on r/SunoAI and AI music Discord servers
- Built on research into transformer-based music generation
- Component pools refined through 400+ generation tests
