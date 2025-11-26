"""
Genetic algorithm module for metaphor prompt optimization.

This module implements evolutionary optimization for discovering high-fitness
metaphors through selection, crossover, and mutation operations.

Key components:
- Individual: A metaphor with associated fitness score
- Population: Collection of individuals evolving over generations
- GeneticOptimizer: Main class orchestrating the evolutionary process

The genetic operators work at the slot level:
- Crossover: Combines slots from two parent metaphors
- Mutation: Replaces slots with new values from component pools
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterator, Sequence

from metaphor_machine.core.generator import MetaphorGenerator
from metaphor_machine.core.metaphor import Metaphor, MetaphorSlot, SlotType
from metaphor_machine.optimization.fitness import EvaluationResult, FitnessEvaluator, FitnessScore
from metaphor_machine.schemas.components import StyleComponents


# =============================================================================
# Individual and Population Types
# =============================================================================


@dataclass
class Individual:
    """
    An individual in the genetic population.
    
    Represents a metaphor with its fitness evaluation results and
    genealogy information for tracking evolutionary history.
    
    Attributes:
        metaphor: The metaphor this individual represents
        fitness: Evaluation result (None if not yet evaluated)
        generation: Which generation this individual was created in
        parent_ids: IDs of parent individuals (empty for initial population)
        id: Unique identifier for this individual
    """
    
    metaphor: Metaphor
    fitness: EvaluationResult | None = None
    generation: int = 0
    parent_ids: tuple[int, ...] = ()
    id: int = field(default_factory=lambda: id(object()))
    
    @property
    def score(self) -> float:
        """Return fitness score, or 0 if not evaluated."""
        return float(self.fitness.score) if self.fitness else 0.0
    
    @property
    def is_evaluated(self) -> bool:
        """Check if this individual has been evaluated."""
        return self.fitness is not None
    
    def __lt__(self, other: Individual) -> bool:
        """Compare by fitness for sorting (higher is better)."""
        return self.score < other.score
    
    def __repr__(self) -> str:
        score_str = f"{self.score:.3f}" if self.is_evaluated else "?"
        return f"Individual(gen={self.generation}, score={score_str})"


@dataclass
class Population:
    """
    A population of individuals for genetic optimization.
    
    Tracks generation statistics and maintains sorted order by fitness.
    
    Attributes:
        individuals: List of Individual instances
        generation: Current generation number
        best_ever: Best individual seen across all generations
    """
    
    individuals: list[Individual] = field(default_factory=list)
    generation: int = 0
    best_ever: Individual | None = None
    
    def __len__(self) -> int:
        return len(self.individuals)
    
    def __iter__(self) -> Iterator[Individual]:
        return iter(self.individuals)
    
    def __getitem__(self, idx: int) -> Individual:
        return self.individuals[idx]
    
    @property
    def evaluated(self) -> list[Individual]:
        """Return only evaluated individuals."""
        return [ind for ind in self.individuals if ind.is_evaluated]
    
    @property
    def best(self) -> Individual | None:
        """Return the best individual in current population."""
        evaluated = self.evaluated
        return max(evaluated, key=lambda i: i.score) if evaluated else None
    
    @property
    def worst(self) -> Individual | None:
        """Return the worst individual in current population."""
        evaluated = self.evaluated
        return min(evaluated, key=lambda i: i.score) if evaluated else None
    
    @property
    def mean_fitness(self) -> float:
        """Return mean fitness of evaluated individuals."""
        evaluated = self.evaluated
        if not evaluated:
            return 0.0
        return sum(i.score for i in evaluated) / len(evaluated)
    
    @property
    def fitness_std(self) -> float:
        """Return standard deviation of fitness."""
        evaluated = self.evaluated
        if len(evaluated) < 2:
            return 0.0
        mean = self.mean_fitness
        variance = sum((i.score - mean) ** 2 for i in evaluated) / len(evaluated)
        return variance ** 0.5
    
    def sort_by_fitness(self, descending: bool = True) -> None:
        """Sort individuals by fitness score."""
        self.individuals.sort(key=lambda i: i.score, reverse=descending)
    
    def update_best_ever(self) -> None:
        """Update best_ever if current best exceeds it."""
        current_best = self.best
        if current_best:
            if self.best_ever is None or current_best.score > self.best_ever.score:
                self.best_ever = current_best
    
    def get_stats(self) -> dict[str, Any]:
        """Return population statistics."""
        return {
            "generation": self.generation,
            "size": len(self),
            "evaluated": len(self.evaluated),
            "mean_fitness": self.mean_fitness,
            "std_fitness": self.fitness_std,
            "best_score": self.best.score if self.best else None,
            "worst_score": self.worst.score if self.worst else None,
            "best_ever_score": self.best_ever.score if self.best_ever else None,
        }


# =============================================================================
# Selection Strategies
# =============================================================================


class SelectionStrategy(str, Enum):
    """Available selection strategies for parent selection."""
    
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"
    TRUNCATION = "truncation"


def tournament_select(
    population: Population,
    tournament_size: int,
    rng: random.Random,
) -> Individual:
    """
    Select an individual via tournament selection.
    
    Randomly samples tournament_size individuals and returns the best.
    Higher tournament_size = more selection pressure.
    """
    contestants = rng.sample(population.evaluated, min(tournament_size, len(population.evaluated)))
    return max(contestants, key=lambda i: i.score)


def roulette_select(
    population: Population,
    rng: random.Random,
) -> Individual:
    """
    Select an individual via fitness-proportionate (roulette) selection.
    
    Probability of selection is proportional to fitness score.
    """
    evaluated = population.evaluated
    if not evaluated:
        raise ValueError("No evaluated individuals in population")
    
    total_fitness = sum(i.score for i in evaluated)
    if total_fitness == 0:
        return rng.choice(evaluated)
    
    pick = rng.uniform(0, total_fitness)
    cumulative = 0.0
    for ind in evaluated:
        cumulative += ind.score
        if cumulative >= pick:
            return ind
    
    return evaluated[-1]  # Fallback


def rank_select(
    population: Population,
    rng: random.Random,
) -> Individual:
    """
    Select an individual via rank-based selection.
    
    Selection probability based on rank rather than raw fitness,
    reducing selection pressure from fitness outliers.
    """
    evaluated = sorted(population.evaluated, key=lambda i: i.score)
    n = len(evaluated)
    if n == 0:
        raise ValueError("No evaluated individuals in population")
    
    # Linear ranking: worst has rank 1, best has rank n
    ranks = list(range(1, n + 1))
    total_rank = sum(ranks)
    
    pick = rng.uniform(0, total_rank)
    cumulative = 0.0
    for ind, rank in zip(evaluated, ranks):
        cumulative += rank
        if cumulative >= pick:
            return ind
    
    return evaluated[-1]


# =============================================================================
# Crossover Operators
# =============================================================================


class CrossoverType(str, Enum):
    """Available crossover strategies."""
    
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"


def single_point_crossover(
    parent1: Metaphor,
    parent2: Metaphor,
    rng: random.Random,
) -> tuple[Metaphor, Metaphor]:
    """
    Single-point crossover at slot level.
    
    Picks a random crossover point and swaps slots after that point.
    """
    slots1 = list(parent1.slots)
    slots2 = list(parent2.slots)
    
    # Pick crossover point (1 to 4, so at least one slot from each parent)
    point = rng.randint(1, 4)
    
    # Create children
    child1_slots = slots1[:point] + slots2[point:]
    child2_slots = slots2[:point] + slots1[point:]
    
    return (
        _slots_to_metaphor(child1_slots),
        _slots_to_metaphor(child2_slots),
    )


def two_point_crossover(
    parent1: Metaphor,
    parent2: Metaphor,
    rng: random.Random,
) -> tuple[Metaphor, Metaphor]:
    """
    Two-point crossover at slot level.
    
    Picks two random points and swaps the middle segment.
    """
    slots1 = list(parent1.slots)
    slots2 = list(parent2.slots)
    
    # Pick two distinct crossover points
    points = sorted(rng.sample(range(1, 5), 2))
    p1, p2 = points
    
    # Create children by swapping middle segment
    child1_slots = slots1[:p1] + slots2[p1:p2] + slots1[p2:]
    child2_slots = slots2[:p1] + slots1[p1:p2] + slots2[p2:]
    
    return (
        _slots_to_metaphor(child1_slots),
        _slots_to_metaphor(child2_slots),
    )


def uniform_crossover(
    parent1: Metaphor,
    parent2: Metaphor,
    rng: random.Random,
    swap_prob: float = 0.5,
) -> tuple[Metaphor, Metaphor]:
    """
    Uniform crossover at slot level.
    
    Each slot independently chosen from either parent with given probability.
    """
    slots1 = list(parent1.slots)
    slots2 = list(parent2.slots)
    
    child1_slots = []
    child2_slots = []
    
    for s1, s2 in zip(slots1, slots2):
        if rng.random() < swap_prob:
            child1_slots.append(s2)
            child2_slots.append(s1)
        else:
            child1_slots.append(s1)
            child2_slots.append(s2)
    
    return (
        _slots_to_metaphor(child1_slots),
        _slots_to_metaphor(child2_slots),
    )


def _slots_to_metaphor(slots: Sequence[MetaphorSlot]) -> Metaphor:
    """Convert a sequence of slots back to a Metaphor."""
    return Metaphor(
        genre_anchor=slots[0],
        intimate_gesture=slots[1],
        dynamic_tension=slots[2],
        sensory_bridge=slots[3],
        emotional_anchor=slots[4],
    )


# =============================================================================
# Mutation Operators
# =============================================================================


def mutate_metaphor(
    metaphor: Metaphor,
    generator: MetaphorGenerator,
    mutation_rate: float = 0.2,
    rng: random.Random | None = None,
) -> Metaphor:
    """
    Mutate a metaphor by potentially replacing each slot.
    
    Each slot has mutation_rate probability of being replaced with
    a newly generated value from the component pools.
    
    Args:
        metaphor: The metaphor to mutate
        generator: MetaphorGenerator for generating new slot values
        mutation_rate: Probability of mutating each slot
        rng: Random number generator (uses generator's rng if None)
    
    Returns:
        Mutated metaphor (or original if no mutations occurred)
    """
    rng = rng or generator.rng
    
    slots = list(metaphor.slots)
    mutated = False
    
    # Potentially mutate each slot by generating a new metaphor and borrowing slots
    for i in range(5):
        if rng.random() < mutation_rate:
            # Generate a fresh metaphor and take the corresponding slot
            fresh = generator.generate_single()
            slots[i] = fresh.slots[i]
            mutated = True
    
    return _slots_to_metaphor(slots) if mutated else metaphor


def mutate_single_slot(
    metaphor: Metaphor,
    generator: MetaphorGenerator,
    slot_index: int | None = None,
    rng: random.Random | None = None,
) -> Metaphor:
    """
    Mutate exactly one slot in the metaphor.
    
    Args:
        metaphor: The metaphor to mutate
        generator: MetaphorGenerator for generating new slot values
        slot_index: Which slot to mutate (random if None)
        rng: Random number generator
    
    Returns:
        Metaphor with one slot mutated
    """
    rng = rng or generator.rng
    
    slots = list(metaphor.slots)
    idx = slot_index if slot_index is not None else rng.randint(0, 4)
    
    # Generate a fresh metaphor and borrow the slot at idx
    fresh = generator.generate_single()
    slots[idx] = fresh.slots[idx]
    
    return _slots_to_metaphor(slots)


# =============================================================================
# Main Genetic Optimizer Class
# =============================================================================


@dataclass
class GeneticConfig:
    """Configuration for the genetic optimizer."""
    
    population_size: int = 30
    generations: int = 50
    elite_size: int = 2
    tournament_size: int = 3
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2
    selection_strategy: SelectionStrategy = SelectionStrategy.TOURNAMENT
    crossover_type: CrossoverType = CrossoverType.UNIFORM
    seed: int | None = None
    
    # Early stopping
    patience: int = 10  # Stop if no improvement for this many generations
    min_fitness_delta: float = 0.001  # Minimum improvement to count as progress


class GeneticOptimizer:
    """
    Genetic algorithm optimizer for metaphor generation.
    
    Evolves a population of metaphors toward high fitness through
    selection, crossover, and mutation operations.
    
    Example:
        >>> from metaphor_machine import StyleComponents, MetaphorGenerator
        >>> from metaphor_machine.optimization import GeneticOptimizer, RuleBasedEvaluator
        >>> 
        >>> components = StyleComponents.from_yaml("style_components.yaml")
        >>> generator = MetaphorGenerator(components)
        >>> evaluator = RuleBasedEvaluator(preferred_terms=["darkwave", "ethereal"])
        >>> 
        >>> optimizer = GeneticOptimizer(
        ...     generator=generator,
        ...     evaluator=evaluator,
        ...     config=GeneticConfig(population_size=30, generations=50),
        ... )
        >>> 
        >>> best = optimizer.run()
        >>> print(f"Best metaphor: {best.metaphor}")
        >>> print(f"Fitness: {best.score:.3f}")
    """
    
    def __init__(
        self,
        generator: MetaphorGenerator,
        evaluator: FitnessEvaluator,
        config: GeneticConfig | None = None,
        callback: Callable[[Population, int], None] | None = None,
    ) -> None:
        """
        Initialize the optimizer.
        
        Args:
            generator: MetaphorGenerator for creating and mutating metaphors
            evaluator: FitnessEvaluator for scoring metaphors
            config: GeneticConfig with algorithm parameters
            callback: Optional callback called after each generation
        """
        self.generator = generator
        self.evaluator = evaluator
        self.config = config or GeneticConfig()
        self.callback = callback
        
        self.rng = random.Random(self.config.seed)
        self.population: Population | None = None
        self.history: list[dict[str, Any]] = []
    
    def initialize_population(self) -> Population:
        """Create initial random population."""
        individuals = []
        for _ in range(self.config.population_size):
            metaphor = self.generator.generate_single()
            individuals.append(Individual(metaphor=metaphor, generation=0))
        
        return Population(individuals=individuals, generation=0)
    
    def evaluate_population(self, population: Population) -> None:
        """Evaluate all unevaluated individuals in the population."""
        for ind in population:
            if not ind.is_evaluated:
                ind.fitness = self.evaluator(ind.metaphor)
        
        population.update_best_ever()
    
    def select_parent(self, population: Population) -> Individual:
        """Select a parent individual based on configured strategy."""
        strategy = self.config.selection_strategy
        
        if strategy == SelectionStrategy.TOURNAMENT:
            return tournament_select(population, self.config.tournament_size, self.rng)
        elif strategy == SelectionStrategy.ROULETTE:
            return roulette_select(population, self.rng)
        elif strategy == SelectionStrategy.RANK:
            return rank_select(population, self.rng)
        elif strategy == SelectionStrategy.TRUNCATION:
            # Select from top half
            population.sort_by_fitness()
            top_half = population.individuals[:len(population) // 2]
            return self.rng.choice(top_half)
        else:
            return tournament_select(population, self.config.tournament_size, self.rng)
    
    def crossover(
        self, 
        parent1: Metaphor, 
        parent2: Metaphor
    ) -> tuple[Metaphor, Metaphor]:
        """Perform crossover based on configured type."""
        crossover_type = self.config.crossover_type
        
        if crossover_type == CrossoverType.SINGLE_POINT:
            return single_point_crossover(parent1, parent2, self.rng)
        elif crossover_type == CrossoverType.TWO_POINT:
            return two_point_crossover(parent1, parent2, self.rng)
        else:  # UNIFORM
            return uniform_crossover(parent1, parent2, self.rng)
    
    def create_next_generation(self, population: Population) -> Population:
        """Create the next generation through selection, crossover, and mutation."""
        population.sort_by_fitness()
        new_individuals: list[Individual] = []
        next_gen = population.generation + 1
        
        # Elitism: preserve best individuals
        for i in range(min(self.config.elite_size, len(population))):
            elite = Individual(
                metaphor=population[i].metaphor,
                fitness=population[i].fitness,
                generation=next_gen,
                parent_ids=(population[i].id,),
            )
            new_individuals.append(elite)
        
        # Fill rest of population
        while len(new_individuals) < self.config.population_size:
            # Select parents
            parent1 = self.select_parent(population)
            parent2 = self.select_parent(population)
            
            # Crossover
            if self.rng.random() < self.config.crossover_rate:
                child1_m, child2_m = self.crossover(parent1.metaphor, parent2.metaphor)
            else:
                child1_m, child2_m = parent1.metaphor, parent2.metaphor
            
            # Mutation
            child1_m = mutate_metaphor(
                child1_m, self.generator, self.config.mutation_rate, self.rng
            )
            child2_m = mutate_metaphor(
                child2_m, self.generator, self.config.mutation_rate, self.rng
            )
            
            # Create individuals
            child1 = Individual(
                metaphor=child1_m,
                generation=next_gen,
                parent_ids=(parent1.id, parent2.id),
            )
            child2 = Individual(
                metaphor=child2_m,
                generation=next_gen,
                parent_ids=(parent1.id, parent2.id),
            )
            
            new_individuals.append(child1)
            if len(new_individuals) < self.config.population_size:
                new_individuals.append(child2)
        
        return Population(
            individuals=new_individuals,
            generation=next_gen,
            best_ever=population.best_ever,
        )
    
    def run(self, verbose: bool = False) -> Individual:
        """
        Run the genetic algorithm optimization.
        
        Args:
            verbose: If True, print progress information
            
        Returns:
            Best individual found across all generations
        """
        # Initialize
        self.population = self.initialize_population()
        self.history = []
        generations_without_improvement = 0
        last_best_score = 0.0
        
        for gen in range(self.config.generations):
            # Evaluate
            self.evaluate_population(self.population)
            
            # Record stats
            stats = self.population.get_stats()
            self.history.append(stats)
            
            if verbose:
                print(
                    f"Generation {gen}: "
                    f"best={stats['best_score']:.3f}, "
                    f"mean={stats['mean_fitness']:.3f}, "
                    f"std={stats['std_fitness']:.3f}"
                )
            
            # Callback
            if self.callback:
                self.callback(self.population, gen)
            
            # Early stopping check
            current_best = stats["best_score"] or 0
            if current_best > last_best_score + self.config.min_fitness_delta:
                generations_without_improvement = 0
                last_best_score = current_best
            else:
                generations_without_improvement += 1
            
            if generations_without_improvement >= self.config.patience:
                if verbose:
                    print(f"Early stopping at generation {gen}")
                break
            
            # Create next generation (unless last iteration)
            if gen < self.config.generations - 1:
                self.population = self.create_next_generation(self.population)
        
        return self.population.best_ever or self.population.best or self.population[0]
    
    def get_top_n(self, n: int = 10) -> list[Individual]:
        """Return top N individuals from current population."""
        if self.population is None:
            return []
        
        self.population.sort_by_fitness()
        return self.population.individuals[:n]
    
    def get_history_df(self) -> Any:
        """
        Return optimization history as a pandas DataFrame.
        
        Requires pandas to be installed.
        """
        try:
            import pandas as pd
            return pd.DataFrame(self.history)
        except ImportError:
            return self.history
