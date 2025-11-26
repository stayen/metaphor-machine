"""
Optimization module for metaphor prompt discovery.

This module provides tools for systematically exploring the metaphor space
to find high-performing prompts for AI music generation.

Components:
- Fitness evaluation: Pluggable scoring systems
- Genetic optimization: Evolutionary search
- Bayesian optimization: Sample-efficient exploration
"""

from metaphor_machine.optimization.fitness import (
    FitnessScore,
    EvaluationResult,
    FitnessEvaluator,
    RandomEvaluator,
    RuleBasedEvaluator,
    SemanticCoherenceEvaluator,
    HumanFeedbackEvaluator,
    LLMEvaluator,
    CompositeEvaluator,
)

from metaphor_machine.optimization.genetic import (
    Individual,
    Population,
    GeneticConfig,
    GeneticOptimizer,
    SelectionStrategy,
    CrossoverType,
    mutate_metaphor,
    mutate_single_slot,
)

from metaphor_machine.optimization.bayesian import (
    MetaphorEmbedder,
    BayesianConfig,
    BayesianResult,
    BayesianOptimizer,
    AcquisitionType,
)


__all__ = [
    # Fitness evaluation
    "FitnessScore",
    "EvaluationResult",
    "FitnessEvaluator",
    "RandomEvaluator",
    "RuleBasedEvaluator",
    "SemanticCoherenceEvaluator",
    "HumanFeedbackEvaluator",
    "LLMEvaluator",
    "CompositeEvaluator",
    # Genetic algorithm
    "Individual",
    "Population",
    "GeneticConfig",
    "GeneticOptimizer",
    "SelectionStrategy",
    "CrossoverType",
    "mutate_metaphor",
    "mutate_single_slot",
    # Bayesian optimization
    "MetaphorEmbedder",
    "BayesianConfig",
    "BayesianResult",
    "BayesianOptimizer",
    "AcquisitionType",
]
