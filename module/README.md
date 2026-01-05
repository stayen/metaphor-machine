# Metaphor Machine

[![PyPI version](https://badge.fury.io/py/metaphor-machine.svg)](https://badge.fury.io/py/metaphor-machine)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Algorithmic generation and optimization of AI music style prompts.**

Metaphor Machine creates structured 5-slot style descriptions for AI music platforms like Suno and Producer.ai, then optimizes them through genetic and Bayesian algorithms to discover high-performing prompt patterns.

## Installation

From source, with optimization extras:
```bash
git clone git@github.com:stayen/metaphor-machine.git
cd module
pip -e ".[optimization]"
```
(see also below, for other installation modes)

From Pypi:
```bash
pip install metaphor-machine
```

With optimization extras:
```bash
pip install metaphor-machine[optimization]  # numpy, scipy
pip install metaphor-machine[llm]           # anthropic, openai
pip install metaphor-machine[all]           # everything
```

## Quick Start

### Basic Generation

```python
from metaphor_machine import StyleComponents, MetaphorGenerator

# Load component pools
components = StyleComponents.from_yaml("style_components.yaml")

# Create generator with seed for reproducibility
generator = MetaphorGenerator(components, seed=42)

# Generate a single metaphor
metaphor = generator.generate_single()
print(metaphor)
# → "dark synthwave, whispered mantras, spiraling synths, neon-alley reverb, dread crescendo"

# Generate a 3-act chain
chain = generator.generate_chain()
print(chain.to_suno_style())
# → "Intro: cinematic orchestral... → Mid: voice opens into... → Outro: melody dissolves..."
```

### Optimization (v1.0)

```python
from metaphor_machine.optimization import (
    GeneticOptimizer,
    GeneticConfig,
    RuleBasedEvaluator,
)

# Create fitness evaluator
evaluator = RuleBasedEvaluator(
    preferred_terms=["darkwave", "ethereal", "haunting"],
    avoided_terms=["pop", "bright"],
)

# Configure genetic algorithm
config = GeneticConfig(
    population_size=30,
    generations=50,
    elite_size=2,
    mutation_rate=0.2,
)

# Run optimization
optimizer = GeneticOptimizer(generator, evaluator, config)
best = optimizer.run(verbose=True)

print(f"Best: {best.metaphor}")
print(f"Score: {best.score:.3f}")
```

### Corpus Storage

```python
from metaphor_machine.corpus import SQLiteCorpus, CorpusEntry

# Store optimization results
corpus = SQLiteCorpus("prompts.db")

for individual in optimizer.get_top_n(20):
    entry = CorpusEntry.from_metaphor(
        individual.metaphor,
        fitness_score=individual.score,
        tags={"genetic", "darkwave"},
        source="genetic_optimizer",
    )
    corpus.add(entry)

# Query high-performing prompts
results = corpus.query(min_score=0.8, tags={"darkwave"})
for entry in results:
    print(f"[{entry.fitness_score:.2f}] {entry.metaphor_text}")

corpus.close()
```

## Metaphor Structure

Each metaphor consists of 5 slots:

| Slot | Purpose | Example |
|------|---------|---------|
| **Genre Anchor** | Base genre (with optional prefix) | "dark synthwave", "japanese house" |
| **Intimate Gesture** | Vocal/lead behavior | "whispered mantras" |
| **Dynamic Tension** | Motion and energy | "spiraling synths" |
| **Sensory Bridge** | Environment/space | "neon-alley reverb" |
| **Emotional Anchor** | Feeling/resolution | "dread crescendo" |

Genres are organized by family (electronic, hip-hop/urban, world/ethnic, rock/guitar, traditional/acoustic) with optional modifier prefixes (mood, intensity, style, regional, location, instruments).

The combinatorial space exceeds **35 trillion** unique combinations.

## Optimization Methods

### Genetic Algorithm

Evolutionary search through selection, crossover, and mutation:

- **Selection**: Tournament, roulette, rank, truncation
- **Crossover**: Single-point, two-point, uniform (slot-level)
- **Mutation**: Per-slot probability replacement
- **Elitism**: Preserve top performers across generations

### Bayesian Optimization

Sample-efficient search for expensive evaluations (human rating, actual audio generation):

- **Surrogate model**: RBF kernel interpolation
- **Acquisition functions**: Expected Improvement, UCB, Thompson sampling
- **External observation**: Support for async evaluation workflows

## Fitness Evaluators

| Evaluator | Speed | Use Case |
|-----------|-------|----------|
| `RuleBasedEvaluator` | Fast | Keyword matching, length, diversity |
| `SemanticCoherenceEvaluator` | Fast | Slot compatibility scoring |
| `LLMEvaluator` | Slow | Claude/GPT quality assessment |
| `HumanFeedbackEvaluator` | Slow | Interactive ground truth |
| `CompositeEvaluator` | Varies | Weighted combination |

## CLI Usage

```bash
# Generate single metaphor
mm generate

# Generate with seed
mm generate --seed 42

# Generate diverse batch
mm batch --count 10 --min-distance 3

# Generate 3-act chain
mm chain --seed 42
```

## Architecture

```
metaphor_machine/
├── core/              # Metaphor, Generator, Chain
├── schemas/           # Pydantic models, config
├── optimization/      # Genetic, Bayesian, Fitness
├── corpus/            # SQLite/JSON storage
└── cli/               # Click commands
```

## Roadmap

- [x] **v0.9.9**: Core generation engine, CLI, seed reproducibility
- [x] **v1.0.0**: Optimization layer (genetic, Bayesian, corpus)
- [ ] **v1.1**: Variation engine (Markov chains, latent navigation)
- [ ] **v1.2**: Audio feedback loop (librosa features, embedding similarity)

## License

MIT

## Links

- [GitHub](https://github.com/stayen/metaphor-machine)
- [PyPI](https://pypi.org/project/metaphor-machine/)
