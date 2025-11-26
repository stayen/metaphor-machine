"""
Fitness function interfaces for evaluating generated metaphors.

This module defines the abstract interface for fitness evaluation and provides
concrete implementations for different evaluation strategies:
- Audio feature extraction (requires generated audio)
- LLM-based scoring (uses language models to evaluate semantic quality)
- Human feedback integration (interactive rating)
- Composite scoring (weighted combination of multiple evaluators)

The fitness functions are designed to be pluggable into optimization loops
(genetic algorithms, Bayesian optimization) for systematic prompt exploration.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from metaphor_machine.core.metaphor import Metaphor, MetaphorChain, MetaphorSlot, SlotType


# =============================================================================
# Core Types and Protocols
# =============================================================================


class FitnessScore(float):
    """
    A fitness score in the range [0.0, 1.0].
    
    Higher scores indicate better fitness. Scores are normalized to enable
    comparison across different evaluation methods.
    """
    
    def __new__(cls, value: float) -> FitnessScore:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Fitness score must be in [0, 1], got {value}")
        return super().__new__(cls, value)
    
    @classmethod
    def from_unbounded(cls, value: float, min_val: float, max_val: float) -> FitnessScore:
        """Normalize an unbounded value to [0, 1] range."""
        if max_val <= min_val:
            return cls(0.5)
        normalized = (value - min_val) / (max_val - min_val)
        return cls(max(0.0, min(1.0, normalized)))


@dataclass(frozen=True)
class EvaluationResult:
    """
    Complete result of a fitness evaluation.
    
    Attributes:
        score: The normalized fitness score [0, 1]
        raw_scores: Dictionary of raw scores from individual criteria
        metadata: Additional information about the evaluation
        evaluation_time: Time taken for evaluation in seconds
        evaluator_name: Name of the evaluator that produced this result
    """
    score: FitnessScore
    raw_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    evaluation_time: float = 0.0
    evaluator_name: str = "unknown"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "score": float(self.score),
            "raw_scores": self.raw_scores,
            "metadata": self.metadata,
            "evaluation_time": self.evaluation_time,
            "evaluator_name": self.evaluator_name,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationResult:
        """Create from dictionary."""
        return cls(
            score=FitnessScore(data["score"]),
            raw_scores=data.get("raw_scores", {}),
            metadata=data.get("metadata", {}),
            evaluation_time=data.get("evaluation_time", 0.0),
            evaluator_name=data.get("evaluator_name", "unknown"),
        )


class EvaluationTarget(Protocol):
    """Protocol for objects that can be evaluated."""
    
    def __str__(self) -> str: ...
    def to_dict(self) -> dict[str, Any]: ...


# =============================================================================
# Abstract Base Class
# =============================================================================


class FitnessEvaluator(ABC):
    """
    Abstract base class for fitness evaluators.
    
    Subclasses must implement the `evaluate` method to score metaphors
    or chains according to specific criteria.
    
    Example:
        >>> class MyEvaluator(FitnessEvaluator):
        ...     def evaluate(self, target):
        ...         # Custom scoring logic
        ...         return EvaluationResult(score=FitnessScore(0.8))
        >>> 
        >>> evaluator = MyEvaluator(name="my_evaluator")
        >>> result = evaluator(metaphor)
    """
    
    def __init__(self, name: str | None = None, weight: float = 1.0) -> None:
        """
        Initialize the evaluator.
        
        Args:
            name: Human-readable name for this evaluator
            weight: Weight for use in composite evaluators
        """
        self.name = name or self.__class__.__name__
        self.weight = weight
        self._evaluation_count = 0
        self._total_time = 0.0
    
    @abstractmethod
    def evaluate(self, target: Metaphor | MetaphorChain) -> EvaluationResult:
        """
        Evaluate a metaphor or chain and return a fitness score.
        
        Args:
            target: The metaphor or chain to evaluate
            
        Returns:
            EvaluationResult with score and metadata
        """
        pass
    
    def __call__(self, target: Metaphor | MetaphorChain) -> EvaluationResult:
        """Evaluate with timing and counting."""
        start = time.perf_counter()
        result = self.evaluate(target)
        elapsed = time.perf_counter() - start
        
        self._evaluation_count += 1
        self._total_time += elapsed
        
        # Add timing to result
        return EvaluationResult(
            score=result.score,
            raw_scores=result.raw_scores,
            metadata=result.metadata,
            evaluation_time=elapsed,
            evaluator_name=self.name,
        )
    
    def evaluate_batch(
        self, 
        targets: Sequence[Metaphor | MetaphorChain]
    ) -> list[EvaluationResult]:
        """
        Evaluate multiple targets.
        
        Default implementation calls evaluate() for each target.
        Subclasses may override for batch optimization.
        
        Args:
            targets: Sequence of metaphors or chains
            
        Returns:
            List of evaluation results in same order
        """
        return [self(target) for target in targets]
    
    @property
    def stats(self) -> dict[str, float]:
        """Return evaluation statistics."""
        return {
            "evaluation_count": self._evaluation_count,
            "total_time": self._total_time,
            "avg_time": self._total_time / max(1, self._evaluation_count),
        }
    
    def reset_stats(self) -> None:
        """Reset evaluation statistics."""
        self._evaluation_count = 0
        self._total_time = 0.0


# =============================================================================
# Concrete Evaluators
# =============================================================================


class RandomEvaluator(FitnessEvaluator):
    """
    Random fitness evaluator for testing and baseline comparisons.
    
    Returns uniformly random scores. Useful for:
    - Testing optimization pipelines
    - Establishing random baselines
    - Debugging without expensive evaluations
    """
    
    def __init__(self, seed: int | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        import random
        self.rng = random.Random(seed)
    
    def evaluate(self, target: Metaphor | MetaphorChain) -> EvaluationResult:
        score = self.rng.random()
        return EvaluationResult(
            score=FitnessScore(score),
            metadata={"method": "random"},
        )


class RuleBasedEvaluator(FitnessEvaluator):
    """
    Rule-based evaluator using configurable heuristics.
    
    Scores metaphors based on:
    - Presence of preferred terms (positive keywords)
    - Absence of avoided terms (negative keywords)
    - Slot diversity (penalizes repetitive patterns)
    - Length constraints
    
    This evaluator is fast and deterministic, suitable for
    filtering during generation or as a component in composite scoring.
    """
    
    def __init__(
        self,
        preferred_terms: list[str] | None = None,
        avoided_terms: list[str] | None = None,
        min_length: int = 50,
        max_length: int = 200,
        diversity_weight: float = 0.2,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.preferred_terms = [t.lower() for t in (preferred_terms or [])]
        self.avoided_terms = [t.lower() for t in (avoided_terms or [])]
        self.min_length = min_length
        self.max_length = max_length
        self.diversity_weight = diversity_weight
    
    def evaluate(self, target: Metaphor | MetaphorChain) -> EvaluationResult:
        text = str(target).lower()
        raw_scores: dict[str, float] = {}
        
        # Preferred terms score
        if self.preferred_terms:
            matches = sum(1 for t in self.preferred_terms if t in text)
            raw_scores["preferred_terms"] = matches / len(self.preferred_terms)
        else:
            raw_scores["preferred_terms"] = 0.5
        
        # Avoided terms penalty
        if self.avoided_terms:
            matches = sum(1 for t in self.avoided_terms if t in text)
            raw_scores["avoided_terms"] = 1.0 - (matches / len(self.avoided_terms))
        else:
            raw_scores["avoided_terms"] = 1.0
        
        # Length score
        length = len(text)
        if length < self.min_length:
            raw_scores["length"] = length / self.min_length
        elif length > self.max_length:
            raw_scores["length"] = max(0, 1.0 - (length - self.max_length) / self.max_length)
        else:
            raw_scores["length"] = 1.0
        
        # Diversity score (unique words ratio)
        words = text.split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            raw_scores["diversity"] = unique_ratio
        else:
            raw_scores["diversity"] = 0.0
        
        # Weighted combination
        weights = {
            "preferred_terms": 0.3,
            "avoided_terms": 0.3,
            "length": 0.2,
            "diversity": self.diversity_weight,
        }
        
        total_weight = sum(weights.values())
        final_score = sum(
            raw_scores[k] * weights[k] for k in raw_scores
        ) / total_weight
        
        return EvaluationResult(
            score=FitnessScore(final_score),
            raw_scores=raw_scores,
            metadata={
                "text_length": length,
                "word_count": len(words),
            },
        )


class SemanticCoherenceEvaluator(FitnessEvaluator):
    """
    Evaluates semantic coherence between metaphor slots.
    
    Uses word co-occurrence statistics or embedding similarity
    to assess whether slot combinations make semantic sense.
    
    This is a lightweight alternative to full LLM evaluation.
    """
    
    def __init__(
        self,
        coherence_pairs: dict[tuple[str, str], float] | None = None,
        default_coherence: float = 0.5,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            coherence_pairs: Dict mapping (term1, term2) to coherence score
            default_coherence: Score for unknown pairs
        """
        super().__init__(**kwargs)
        self.coherence_pairs = coherence_pairs or {}
        self.default_coherence = default_coherence
    
    def evaluate(self, target: Metaphor | MetaphorChain) -> EvaluationResult:
        if isinstance(target, MetaphorChain):
            # Average coherence across chain parts
            scores = [self._evaluate_metaphor(m) for m in target.metaphors]
            avg_score = sum(scores) / len(scores)
            return EvaluationResult(
                score=FitnessScore(avg_score),
                raw_scores={"chain_coherence": avg_score},
                metadata={"part_scores": scores},
            )
        else:
            score = self._evaluate_metaphor(target)
            return EvaluationResult(
                score=FitnessScore(score),
                raw_scores={"metaphor_coherence": score},
            )
    
    def _evaluate_metaphor(self, metaphor: Metaphor) -> float:
        """Evaluate coherence of a single metaphor."""
        slots = metaphor.slot_values
        
        if not self.coherence_pairs:
            # Without coherence data, use simple heuristics
            # Check for word overlap between slots (indicates thematic consistency)
            all_words: set[str] = set()
            slot_words: list[set[str]] = []
            
            for slot in slots:
                words = set(slot.lower().replace("-", " ").split())
                slot_words.append(words)
                all_words.update(words)
            
            # Penalize too much overlap (repetitive) or too little (incoherent)
            if not all_words:
                return 0.5
            
            overlap_count = 0
            for i, w1 in enumerate(slot_words):
                for w2 in slot_words[i + 1:]:
                    overlap_count += len(w1 & w2)
            
            # Normalize: some overlap is good, too much is bad
            max_overlaps = len(slots) * (len(slots) - 1) / 2
            if max_overlaps > 0:
                overlap_ratio = overlap_count / (max_overlaps * 2)  # Expect ~2 overlaps
                score = 1.0 - abs(overlap_ratio - 0.3) * 2  # Peak at 0.3 overlap
                return max(0.0, min(1.0, score))
            
            return 0.5
        
        # With coherence pairs, compute average pairwise coherence
        total = 0.0
        count = 0
        for i, s1 in enumerate(slots):
            for s2 in slots[i + 1:]:
                key = (s1.lower(), s2.lower())
                rev_key = (s2.lower(), s1.lower())
                coherence = self.coherence_pairs.get(
                    key,
                    self.coherence_pairs.get(rev_key, self.default_coherence)
                )
                total += coherence
                count += 1
        
        return total / max(1, count)


class HumanFeedbackEvaluator(FitnessEvaluator):
    """
    Interactive evaluator that solicits human feedback.
    
    Presents metaphors to a human rater and collects scores.
    Supports caching to avoid re-rating previously seen metaphors.
    
    This evaluator is expensive (requires human time) but provides
    ground truth for training or validating other evaluators.
    """
    
    def __init__(
        self,
        cache_path: str | Path | None = None,
        prompt_fn: Callable[[str], float] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            cache_path: Path to JSON file for caching ratings
            prompt_fn: Custom function to prompt for rating (default: console input)
        """
        super().__init__(**kwargs)
        self.cache_path = Path(cache_path) if cache_path else None
        self.prompt_fn = prompt_fn or self._default_prompt
        self._cache: dict[str, float] = {}
        
        if self.cache_path and self.cache_path.exists():
            self._load_cache()
    
    def _load_cache(self) -> None:
        """Load cached ratings from disk."""
        if self.cache_path:
            with open(self.cache_path, "r") as f:
                self._cache = json.load(f)
    
    def _save_cache(self) -> None:
        """Save cached ratings to disk."""
        if self.cache_path:
            with open(self.cache_path, "w") as f:
                json.dump(self._cache, f, indent=2)
    
    def _default_prompt(self, text: str) -> float:
        """Default console-based rating prompt."""
        print("\n" + "=" * 60)
        print("Please rate this metaphor (0.0 - 1.0):")
        print("-" * 60)
        print(text)
        print("-" * 60)
        
        while True:
            try:
                rating = float(input("Rating [0.0-1.0]: ").strip())
                if 0.0 <= rating <= 1.0:
                    return rating
                print("Rating must be between 0.0 and 1.0")
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\nSkipping (default 0.5)")
                return 0.5
    
    def evaluate(self, target: Metaphor | MetaphorChain) -> EvaluationResult:
        text = str(target)
        cache_key = text
        
        # Check cache
        if cache_key in self._cache:
            return EvaluationResult(
                score=FitnessScore(self._cache[cache_key]),
                metadata={"source": "cache"},
            )
        
        # Get human rating
        rating = self.prompt_fn(text)
        
        # Cache and save
        self._cache[cache_key] = rating
        self._save_cache()
        
        return EvaluationResult(
            score=FitnessScore(rating),
            metadata={"source": "human"},
        )


class LLMEvaluator(FitnessEvaluator):
    """
    Evaluator that uses a language model to score metaphors.
    
    Sends the metaphor to an LLM with a scoring prompt and parses
    the numeric response. Supports various criteria:
    - Creativity/novelty
    - Coherence/consistency
    - Evocativeness
    - Genre appropriateness
    
    This evaluator requires an LLM API (OpenAI, Anthropic, etc.).
    """
    
    DEFAULT_PROMPT = """Rate the following music style description on a scale of 0 to 10,
where 0 is completely incoherent/uninteresting and 10 is highly creative and evocative.

Consider these criteria:
- Creativity: Does it combine elements in novel, unexpected ways?
- Coherence: Do the elements work together thematically?
- Evocativeness: Does it paint a vivid sonic picture?
- Specificity: Does it give clear direction for music generation?

Style description:
{metaphor}

Respond with ONLY a single number from 0 to 10."""
    
    def __init__(
        self,
        api_fn: Callable[[str], str] | None = None,
        prompt_template: str | None = None,
        score_parser: Callable[[str], float] | None = None,
        cache_path: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            api_fn: Function that takes a prompt string and returns LLM response
            prompt_template: Template with {metaphor} placeholder
            score_parser: Function to extract score from LLM response
            cache_path: Path to cache LLM responses
        """
        super().__init__(**kwargs)
        self.api_fn = api_fn
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT
        self.score_parser = score_parser or self._default_parser
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, dict[str, Any]] = {}
        
        if self.cache_path and self.cache_path.exists():
            with open(self.cache_path, "r") as f:
                self._cache = json.load(f)
    
    def _default_parser(self, response: str) -> float:
        """Extract numeric score from LLM response."""
        import re
        # Find first number in response
        match = re.search(r"(\d+(?:\.\d+)?)", response.strip())
        if match:
            value = float(match.group(1))
            # Normalize to [0, 1] assuming 0-10 scale
            return min(1.0, max(0.0, value / 10.0))
        return 0.5  # Default if no number found
    
    def _save_cache(self) -> None:
        """Save cache to disk."""
        if self.cache_path:
            with open(self.cache_path, "w") as f:
                json.dump(self._cache, f, indent=2)
    
    def evaluate(self, target: Metaphor | MetaphorChain) -> EvaluationResult:
        if self.api_fn is None:
            raise RuntimeError("LLMEvaluator requires api_fn to be set")
        
        text = str(target)
        cache_key = text
        
        # Check cache
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            return EvaluationResult(
                score=FitnessScore(cached["score"]),
                metadata={"source": "cache", "response": cached["response"]},
            )
        
        # Build prompt and call LLM
        prompt = self.prompt_template.format(metaphor=text)
        response = self.api_fn(prompt)
        
        # Parse score
        score = self.score_parser(response)
        
        # Cache
        self._cache[cache_key] = {"score": score, "response": response}
        self._save_cache()
        
        return EvaluationResult(
            score=FitnessScore(score),
            raw_scores={"llm_score": score * 10},  # Original 0-10 scale
            metadata={"source": "llm", "response": response},
        )


# =============================================================================
# Composite Evaluator
# =============================================================================


class CompositeEvaluator(FitnessEvaluator):
    """
    Combines multiple evaluators with weighted averaging.
    
    Useful for balancing different evaluation criteria:
    - Quick heuristics (rule-based)
    - Semantic quality (coherence)
    - Ground truth (human feedback or LLM)
    
    Example:
        >>> composite = CompositeEvaluator([
        ...     (RuleBasedEvaluator(), 0.3),
        ...     (SemanticCoherenceEvaluator(), 0.3),
        ...     (LLMEvaluator(api_fn=call_claude), 0.4),
        ... ])
        >>> result = composite(metaphor)
    """
    
    def __init__(
        self,
        evaluators: list[tuple[FitnessEvaluator, float]],
        aggregation: str = "weighted_mean",
        **kwargs: Any,
    ) -> None:
        """
        Args:
            evaluators: List of (evaluator, weight) tuples
            aggregation: How to combine scores ("weighted_mean", "min", "max", "product")
        """
        super().__init__(**kwargs)
        self.evaluators = evaluators
        self.aggregation = aggregation
        
        # Normalize weights
        total_weight = sum(w for _, w in evaluators)
        self.normalized_weights = [w / total_weight for _, w in evaluators]
    
    def evaluate(self, target: Metaphor | MetaphorChain) -> EvaluationResult:
        results = [(e(target), w) for (e, _), w in zip(self.evaluators, self.normalized_weights)]
        
        scores = [r.score for r, _ in results]
        weights = [w for _, w in results]
        
        # Aggregate scores
        if self.aggregation == "weighted_mean":
            final = sum(s * w for s, w in zip(scores, weights))
        elif self.aggregation == "min":
            final = min(scores)
        elif self.aggregation == "max":
            final = max(scores)
        elif self.aggregation == "product":
            final = 1.0
            for s in scores:
                final *= s
        else:
            final = sum(s * w for s, w in zip(scores, weights))
        
        # Collect raw scores from all evaluators
        raw_scores = {}
        for (result, _), (evaluator, _) in zip(results, self.evaluators):
            for key, value in result.raw_scores.items():
                raw_scores[f"{evaluator.name}.{key}"] = value
        
        return EvaluationResult(
            score=FitnessScore(max(0.0, min(1.0, final))),
            raw_scores=raw_scores,
            metadata={
                "component_scores": {
                    e.name: float(r.score) for (e, _), (r, _) in zip(self.evaluators, results)
                },
                "aggregation": self.aggregation,
            },
        )
