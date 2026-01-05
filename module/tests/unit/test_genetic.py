"""Unit tests for genetic algorithm optimization module."""

import pytest
import random

from metaphor_machine.core.metaphor import Metaphor, MetaphorSlot, SlotType
from metaphor_machine.core.generator import MetaphorGenerator
from metaphor_machine.schemas.components import StyleComponents
from metaphor_machine.optimization.fitness import (
    FitnessScore,
    EvaluationResult,
    FitnessEvaluator,
    RandomEvaluator,
    RuleBasedEvaluator,
)
from metaphor_machine.optimization.genetic import (
    Individual,
    Population,
    GeneticConfig,
    GeneticOptimizer,
    SelectionStrategy,
    CrossoverType,
    tournament_select,
    roulette_select,
    rank_select,
    single_point_crossover,
    two_point_crossover,
    uniform_crossover,
    mutate_metaphor,
    mutate_single_slot,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def minimal_components_yaml(tmp_path):
    """Create a minimal components YAML file."""
    yaml_content = """
genre:
  electronic:
    - darkwave
    - synthpop
  hip_hop_urban:
    - trip-hop
  world_ethnic: []
  rock_guitar: []
  traditional_acoustic: []
  modifiers:
    mood:
      - dark
    intensity: []
    style: []
  regional: []
  location: []
  instruments: []
  experimental: []

intimate_gesture:
  intensity_adjectives:
    energy:
      - breathy
      - whispered
    texture:
      - reedy
      - silken
  delivery_nouns:
    spoken:
      - confessions
      - murmurs
    sung:
      - hooks
      - melodies

dynamic_tension:
  motion_verbs:
    - swelling
    - crackling
    - surging
  musical_objects:
    harmonic:
      - synth-pads
      - chords
    percussive:
      - beats
      - kicks

sensory_bridge:
  environments:
    - basement
    - cathedral
    - neon-alley
  sensory_mediums:
    visual_lens:
      - Polaroid
      - neon-glow

emotional_anchor:
  emotions:
    negative:
      - melancholy
      - dread
    positive:
      - hope
      - joy
  arcs:
    musical:
      - drift
      - surge
"""
    yaml_file = tmp_path / "components.yaml"
    yaml_file.write_text(yaml_content)
    return yaml_file


@pytest.fixture
def components(minimal_components_yaml):
    """Load style components."""
    return StyleComponents.from_yaml(str(minimal_components_yaml))


@pytest.fixture
def generator(components):
    """Create a metaphor generator."""
    return MetaphorGenerator(components, seed=42)


@pytest.fixture
def sample_metaphor():
    """Create a sample metaphor."""
    return Metaphor(
        genre_anchor=MetaphorSlot(SlotType.GENRE_ANCHOR, "darkwave fusion"),
        intimate_gesture=MetaphorSlot(SlotType.INTIMATE_GESTURE, "breathy confessions"),
        dynamic_tension=MetaphorSlot(SlotType.DYNAMIC_TENSION, "swelling synth-pads"),
        sensory_bridge=MetaphorSlot(SlotType.SENSORY_BRIDGE, "basement-reverb"),
        emotional_anchor=MetaphorSlot(SlotType.EMOTIONAL_ANCHOR, "melancholy drift"),
    )


@pytest.fixture
def second_metaphor():
    """Create another sample metaphor."""
    return Metaphor(
        genre_anchor=MetaphorSlot(SlotType.GENRE_ANCHOR, "synthpop blend"),
        intimate_gesture=MetaphorSlot(SlotType.INTIMATE_GESTURE, "whispered hooks"),
        dynamic_tension=MetaphorSlot(SlotType.DYNAMIC_TENSION, "crackling beats"),
        sensory_bridge=MetaphorSlot(SlotType.SENSORY_BRIDGE, "neon-alley-glow"),
        emotional_anchor=MetaphorSlot(SlotType.EMOTIONAL_ANCHOR, "hope surge"),
    )


@pytest.fixture
def evaluated_individual(sample_metaphor):
    """Create an evaluated individual."""
    ind = Individual(metaphor=sample_metaphor, generation=0)
    ind.fitness = EvaluationResult(score=FitnessScore(0.75))
    return ind


# =============================================================================
# Individual Tests
# =============================================================================


class TestIndividual:
    """Tests for the Individual class."""
    
    def test_creation(self, sample_metaphor):
        """Test creating an individual."""
        ind = Individual(metaphor=sample_metaphor, generation=0)
        
        assert ind.metaphor == sample_metaphor
        assert ind.generation == 0
        assert ind.fitness is None
        assert not ind.is_evaluated
    
    def test_score_unevaluated(self, sample_metaphor):
        """Test score for unevaluated individual."""
        ind = Individual(metaphor=sample_metaphor)
        assert ind.score == 0.0
    
    def test_score_evaluated(self, evaluated_individual):
        """Test score for evaluated individual."""
        assert evaluated_individual.score == 0.75
        assert evaluated_individual.is_evaluated
    
    def test_comparison(self, sample_metaphor, second_metaphor):
        """Test comparison by fitness."""
        ind1 = Individual(metaphor=sample_metaphor)
        ind1.fitness = EvaluationResult(score=FitnessScore(0.6))
        
        ind2 = Individual(metaphor=second_metaphor)
        ind2.fitness = EvaluationResult(score=FitnessScore(0.8))
        
        assert ind1 < ind2  # Lower fitness
    
    def test_parent_tracking(self, sample_metaphor):
        """Test parent ID tracking."""
        parent1_id = 100
        parent2_id = 200
        
        child = Individual(
            metaphor=sample_metaphor,
            generation=1,
            parent_ids=(parent1_id, parent2_id),
        )
        
        assert child.parent_ids == (100, 200)


# =============================================================================
# Population Tests
# =============================================================================


class TestPopulation:
    """Tests for the Population class."""
    
    def test_empty_population(self):
        """Test empty population."""
        pop = Population()
        
        assert len(pop) == 0
        assert pop.best is None
        assert pop.worst is None
        assert pop.mean_fitness == 0.0
    
    def test_population_with_individuals(self, sample_metaphor, second_metaphor):
        """Test population with individuals."""
        ind1 = Individual(metaphor=sample_metaphor)
        ind1.fitness = EvaluationResult(score=FitnessScore(0.6))
        
        ind2 = Individual(metaphor=second_metaphor)
        ind2.fitness = EvaluationResult(score=FitnessScore(0.8))
        
        pop = Population(individuals=[ind1, ind2])
        
        assert len(pop) == 2
        assert pop.best.score == 0.8
        assert pop.worst.score == 0.6
        assert pop.mean_fitness == 0.7
    
    def test_sort_by_fitness(self, sample_metaphor, second_metaphor):
        """Test sorting by fitness."""
        ind1 = Individual(metaphor=sample_metaphor)
        ind1.fitness = EvaluationResult(score=FitnessScore(0.3))
        
        ind2 = Individual(metaphor=second_metaphor)
        ind2.fitness = EvaluationResult(score=FitnessScore(0.9))
        
        pop = Population(individuals=[ind1, ind2])
        pop.sort_by_fitness(descending=True)
        
        assert pop[0].score == 0.9
        assert pop[1].score == 0.3
    
    def test_best_ever_tracking(self, sample_metaphor, second_metaphor):
        """Test best_ever tracking across generations."""
        ind1 = Individual(metaphor=sample_metaphor)
        ind1.fitness = EvaluationResult(score=FitnessScore(0.9))
        
        pop = Population(individuals=[ind1])
        pop.update_best_ever()
        
        assert pop.best_ever.score == 0.9
        
        # Add worse individual
        ind2 = Individual(metaphor=second_metaphor)
        ind2.fitness = EvaluationResult(score=FitnessScore(0.5))
        pop.individuals.append(ind2)
        pop.update_best_ever()
        
        # Best ever should not change
        assert pop.best_ever.score == 0.9
    
    def test_get_stats(self, sample_metaphor, second_metaphor):
        """Test getting population statistics."""
        ind1 = Individual(metaphor=sample_metaphor)
        ind1.fitness = EvaluationResult(score=FitnessScore(0.4))
        
        ind2 = Individual(metaphor=second_metaphor)
        ind2.fitness = EvaluationResult(score=FitnessScore(0.8))
        
        pop = Population(individuals=[ind1, ind2], generation=5)
        
        stats = pop.get_stats()
        
        assert stats["generation"] == 5
        assert stats["size"] == 2
        assert stats["evaluated"] == 2
        assert abs(stats["mean_fitness"] - 0.6) < 0.001
        assert stats["best_score"] == 0.8
        assert stats["worst_score"] == 0.4


# =============================================================================
# Selection Strategy Tests
# =============================================================================


class TestSelectionStrategies:
    """Tests for selection strategies."""
    
    def test_tournament_select(self, sample_metaphor, second_metaphor):
        """Test tournament selection."""
        ind1 = Individual(metaphor=sample_metaphor)
        ind1.fitness = EvaluationResult(score=FitnessScore(0.3))
        
        ind2 = Individual(metaphor=second_metaphor)
        ind2.fitness = EvaluationResult(score=FitnessScore(0.9))
        
        pop = Population(individuals=[ind1, ind2])
        rng = random.Random(42)
        
        # With tournament size = 2, should always get the better one
        selected = tournament_select(pop, tournament_size=2, rng=rng)
        assert selected.score == 0.9
    
    def test_roulette_select(self, sample_metaphor, second_metaphor):
        """Test roulette selection."""
        ind1 = Individual(metaphor=sample_metaphor)
        ind1.fitness = EvaluationResult(score=FitnessScore(0.1))
        
        ind2 = Individual(metaphor=second_metaphor)
        ind2.fitness = EvaluationResult(score=FitnessScore(0.9))
        
        pop = Population(individuals=[ind1, ind2])
        rng = random.Random(42)
        
        # Run multiple selections, higher fitness should be selected more often
        selections = [roulette_select(pop, rng).score for _ in range(100)]
        high_count = sum(1 for s in selections if s == 0.9)
        
        assert high_count > 50  # Should be selected more than half the time
    
    def test_rank_select(self, sample_metaphor, second_metaphor):
        """Test rank-based selection."""
        ind1 = Individual(metaphor=sample_metaphor)
        ind1.fitness = EvaluationResult(score=FitnessScore(0.1))
        
        ind2 = Individual(metaphor=second_metaphor)
        ind2.fitness = EvaluationResult(score=FitnessScore(0.9))
        
        pop = Population(individuals=[ind1, ind2])
        rng = random.Random(42)
        
        # Should select based on rank, not raw fitness
        selected = rank_select(pop, rng)
        assert selected is not None


# =============================================================================
# Crossover Tests
# =============================================================================


class TestCrossover:
    """Tests for crossover operators."""
    
    def test_single_point_crossover(self, sample_metaphor, second_metaphor):
        """Test single-point crossover."""
        rng = random.Random(42)
        
        child1, child2 = single_point_crossover(sample_metaphor, second_metaphor, rng)
        
        # Children should be valid metaphors
        assert isinstance(child1, Metaphor)
        assert isinstance(child2, Metaphor)
        
        # Children should have mix of parent slots
        assert child1 != sample_metaphor or child2 != sample_metaphor
        assert child1 != second_metaphor or child2 != second_metaphor
    
    def test_two_point_crossover(self, sample_metaphor, second_metaphor):
        """Test two-point crossover."""
        rng = random.Random(42)
        
        child1, child2 = two_point_crossover(sample_metaphor, second_metaphor, rng)
        
        assert isinstance(child1, Metaphor)
        assert isinstance(child2, Metaphor)
    
    def test_uniform_crossover(self, sample_metaphor, second_metaphor):
        """Test uniform crossover."""
        rng = random.Random(42)
        
        child1, child2 = uniform_crossover(sample_metaphor, second_metaphor, rng, swap_prob=0.5)
        
        assert isinstance(child1, Metaphor)
        assert isinstance(child2, Metaphor)
    
    def test_uniform_crossover_swap_prob_1(self, sample_metaphor, second_metaphor):
        """Test uniform crossover with swap_prob=1.0 (always swap)."""
        rng = random.Random(42)
        
        child1, child2 = uniform_crossover(sample_metaphor, second_metaphor, rng, swap_prob=1.0)
        
        # With swap_prob=1, children should be swapped versions
        assert child1.slot_values == second_metaphor.slot_values
        assert child2.slot_values == sample_metaphor.slot_values
    
    def test_uniform_crossover_swap_prob_0(self, sample_metaphor, second_metaphor):
        """Test uniform crossover with swap_prob=0.0 (never swap)."""
        rng = random.Random(42)
        
        child1, child2 = uniform_crossover(sample_metaphor, second_metaphor, rng, swap_prob=0.0)
        
        # With swap_prob=0, children should match parents
        assert child1.slot_values == sample_metaphor.slot_values
        assert child2.slot_values == second_metaphor.slot_values


# =============================================================================
# Mutation Tests
# =============================================================================


class TestMutation:
    """Tests for mutation operators."""
    
    def test_mutate_metaphor(self, sample_metaphor, generator):
        """Test metaphor mutation."""
        rng = random.Random(42)
        
        mutated = mutate_metaphor(sample_metaphor, generator, mutation_rate=1.0, rng=rng)
        
        # With mutation_rate=1.0, should always mutate
        assert mutated != sample_metaphor
        assert isinstance(mutated, Metaphor)
    
    def test_mutate_metaphor_zero_rate(self, sample_metaphor, generator):
        """Test mutation with rate=0 (no mutation)."""
        rng = random.Random(42)
        
        mutated = mutate_metaphor(sample_metaphor, generator, mutation_rate=0.0, rng=rng)
        
        # With mutation_rate=0, should never mutate
        assert mutated == sample_metaphor
    
    def test_mutate_single_slot(self, sample_metaphor, generator):
        """Test single slot mutation."""
        rng = random.Random(42)
        
        mutated = mutate_single_slot(sample_metaphor, generator, slot_index=0, rng=rng)
        
        # Only first slot should change
        assert mutated.genre_anchor != sample_metaphor.genre_anchor
        # Other slots should be the same
        assert mutated.intimate_gesture == sample_metaphor.intimate_gesture
        assert mutated.dynamic_tension == sample_metaphor.dynamic_tension
        assert mutated.sensory_bridge == sample_metaphor.sensory_bridge
        assert mutated.emotional_anchor == sample_metaphor.emotional_anchor
    
    def test_mutate_single_slot_random(self, sample_metaphor, generator):
        """Test single slot mutation with random slot selection."""
        rng = random.Random(42)
        
        # Run multiple times to verify randomness
        mutations = []
        for i in range(10):
            rng = random.Random(i)
            mutated = mutate_single_slot(sample_metaphor, generator, rng=rng)
            mutations.append(mutated)
        
        # Should have at least some different mutations
        unique_mutations = len(set(str(m) for m in mutations))
        assert unique_mutations > 1


# =============================================================================
# GeneticConfig Tests
# =============================================================================


class TestGeneticConfig:
    """Tests for GeneticConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = GeneticConfig()
        
        assert config.population_size == 30
        assert config.generations == 50
        assert config.elite_size == 2
        assert config.tournament_size == 3
        assert config.crossover_rate == 0.8
        assert config.mutation_rate == 0.2
        assert config.selection_strategy == SelectionStrategy.TOURNAMENT
        assert config.crossover_type == CrossoverType.UNIFORM
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = GeneticConfig(
            population_size=50,
            generations=100,
            selection_strategy=SelectionStrategy.ROULETTE,
            crossover_type=CrossoverType.TWO_POINT,
        )
        
        assert config.population_size == 50
        assert config.generations == 100
        assert config.selection_strategy == SelectionStrategy.ROULETTE
        assert config.crossover_type == CrossoverType.TWO_POINT


# =============================================================================
# GeneticOptimizer Tests
# =============================================================================


class TestGeneticOptimizer:
    """Tests for the GeneticOptimizer class."""
    
    def test_initialization(self, generator):
        """Test optimizer initialization."""
        evaluator = RandomEvaluator(seed=42)
        config = GeneticConfig(population_size=10, generations=5)
        
        optimizer = GeneticOptimizer(
            generator=generator,
            evaluator=evaluator,
            config=config,
        )
        
        assert optimizer.generator == generator
        assert optimizer.evaluator == evaluator
        assert optimizer.config.population_size == 10
    
    def test_initialize_population(self, generator):
        """Test population initialization."""
        evaluator = RandomEvaluator(seed=42)
        config = GeneticConfig(population_size=10, generations=5)
        
        optimizer = GeneticOptimizer(
            generator=generator,
            evaluator=evaluator,
            config=config,
        )
        
        pop = optimizer.initialize_population()
        
        assert len(pop) == 10
        assert all(isinstance(ind.metaphor, Metaphor) for ind in pop)
    
    def test_evaluate_population(self, generator):
        """Test population evaluation."""
        evaluator = RandomEvaluator(seed=42)
        config = GeneticConfig(population_size=5, generations=2)
        
        optimizer = GeneticOptimizer(
            generator=generator,
            evaluator=evaluator,
            config=config,
        )
        
        pop = optimizer.initialize_population()
        assert all(not ind.is_evaluated for ind in pop)
        
        optimizer.evaluate_population(pop)
        assert all(ind.is_evaluated for ind in pop)
    
    def test_run_optimization(self, generator):
        """Test running full optimization."""
        evaluator = RuleBasedEvaluator(preferred_terms=["darkwave"])
        config = GeneticConfig(
            population_size=10,
            generations=5,
            seed=42,
        )
        
        optimizer = GeneticOptimizer(
            generator=generator,
            evaluator=evaluator,
            config=config,
        )
        
        best = optimizer.run(verbose=False)
        
        assert isinstance(best, Individual)
        assert best.is_evaluated
        assert 0 <= best.score <= 1
    
    def test_history_tracking(self, generator):
        """Test that optimization history is tracked."""
        evaluator = RandomEvaluator(seed=42)
        config = GeneticConfig(
            population_size=10,
            generations=5,
            seed=42,
        )
        
        optimizer = GeneticOptimizer(
            generator=generator,
            evaluator=evaluator,
            config=config,
        )
        
        optimizer.run(verbose=False)
        
        assert len(optimizer.history) == 5  # One entry per generation
        assert "mean_fitness" in optimizer.history[0]
        assert "best_score" in optimizer.history[0]
    
    def test_elitism(self, generator):
        """Test that elite individuals are preserved."""
        evaluator = RandomEvaluator(seed=42)
        config = GeneticConfig(
            population_size=10,
            generations=3,
            elite_size=2,
            seed=42,
        )
        
        optimizer = GeneticOptimizer(
            generator=generator,
            evaluator=evaluator,
            config=config,
        )
        
        pop1 = optimizer.initialize_population()
        optimizer.evaluate_population(pop1)
        pop1.sort_by_fitness()
        
        top_metaphors = [pop1[0].metaphor, pop1[1].metaphor]
        
        pop2 = optimizer.create_next_generation(pop1)
        optimizer.evaluate_population(pop2)
        
        # Elite metaphors should be in next generation
        pop2_metaphor_strs = [str(ind.metaphor) for ind in pop2]
        assert str(top_metaphors[0]) in pop2_metaphor_strs or str(top_metaphors[1]) in pop2_metaphor_strs
    
    def test_get_top_n(self, generator):
        """Test getting top N individuals."""
        evaluator = RandomEvaluator(seed=42)
        config = GeneticConfig(
            population_size=20,
            generations=3,
            seed=42,
        )
        
        optimizer = GeneticOptimizer(
            generator=generator,
            evaluator=evaluator,
            config=config,
        )
        
        optimizer.run(verbose=False)
        
        top_5 = optimizer.get_top_n(5)
        
        assert len(top_5) == 5
        assert all(top_5[i].score >= top_5[i + 1].score for i in range(len(top_5) - 1))
    
    def test_callback(self, generator):
        """Test callback functionality."""
        evaluator = RandomEvaluator(seed=42)
        config = GeneticConfig(
            population_size=10,
            generations=3,
            seed=42,
        )
        
        callback_calls = []
        def my_callback(pop, gen):
            callback_calls.append((len(pop), gen))
        
        optimizer = GeneticOptimizer(
            generator=generator,
            evaluator=evaluator,
            config=config,
            callback=my_callback,
        )
        
        optimizer.run(verbose=False)
        
        assert len(callback_calls) == 3
        assert callback_calls[0][1] == 0  # First generation
        assert callback_calls[2][1] == 2  # Last generation
    
    def test_early_stopping(self, generator):
        """Test early stopping when no improvement."""
        # Use evaluator that always returns same score (no improvement)
        class ConstantEvaluator(FitnessEvaluator):
            def evaluate(self, target):
                return EvaluationResult(score=FitnessScore(0.5))
        
        evaluator = ConstantEvaluator()
        config = GeneticConfig(
            population_size=10,
            generations=100,  # Many generations
            patience=3,  # Stop after 3 generations without improvement
            seed=42,
        )
        
        optimizer = GeneticOptimizer(
            generator=generator,
            evaluator=evaluator,
            config=config,
        )
        
        optimizer.run(verbose=False)
        
        # Should stop early due to no improvement
        assert len(optimizer.history) < 100
    
    def test_selection_strategies(self, generator):
        """Test different selection strategies."""
        evaluator = RandomEvaluator(seed=42)
        
        for strategy in SelectionStrategy:
            config = GeneticConfig(
                population_size=10,
                generations=2,
                selection_strategy=strategy,
                seed=42,
            )
            
            optimizer = GeneticOptimizer(
                generator=generator,
                evaluator=evaluator,
                config=config,
            )
            
            best = optimizer.run(verbose=False)
            assert isinstance(best, Individual)
    
    def test_crossover_types(self, generator):
        """Test different crossover types."""
        evaluator = RandomEvaluator(seed=42)
        
        for crossover_type in CrossoverType:
            config = GeneticConfig(
                population_size=10,
                generations=2,
                crossover_type=crossover_type,
                seed=42,
            )
            
            optimizer = GeneticOptimizer(
                generator=generator,
                evaluator=evaluator,
                config=config,
            )
            
            best = optimizer.run(verbose=False)
            assert isinstance(best, Individual)
