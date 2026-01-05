"""
Bayesian optimization module for expensive metaphor evaluation.

When fitness evaluation is expensive (e.g., generating actual audio with Suno
and having humans rate it), Bayesian optimization provides sample-efficient
exploration by building a surrogate model of the fitness landscape.

This module implements:
- Surrogate models (Gaussian Process approximation via RBF kernel)
- Acquisition functions (Expected Improvement, Upper Confidence Bound)
- BayesianOptimizer class for managing the optimization loop

The optimizer works in a latent space derived from metaphor embeddings,
making it compatible with the discrete, structured nature of metaphors.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Sequence

import random

from metaphor_machine.core.generator import MetaphorGenerator
from metaphor_machine.core.metaphor import Metaphor
from metaphor_machine.optimization.fitness import EvaluationResult, FitnessEvaluator, FitnessScore


# =============================================================================
# Metaphor Embedding
# =============================================================================


class MetaphorEmbedder:
    """
    Converts metaphors to numerical vectors for Bayesian optimization.
    
    Two embedding strategies:
    1. One-hot encoding of slot indices (deterministic, no dependencies)
    2. Sentence embedding via transformers (requires sentence-transformers)
    
    The embedding enables distance computation and interpolation in a
    continuous space, which is required for Gaussian Process modeling.
    """
    
    def __init__(
        self,
        components: Any,  # StyleComponents
        use_sentence_embeddings: bool = False,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        """
        Args:
            components: StyleComponents for one-hot encoding dimensions
            use_sentence_embeddings: If True, use transformer embeddings
            embedding_model: Name of sentence-transformers model
        """
        self.components = components
        self.use_sentence_embeddings = use_sentence_embeddings
        self._sentence_model = None
        self._embedding_model_name = embedding_model
        
        # Build index maps for one-hot encoding
        self._build_index_maps()
    
    def _build_index_maps(self) -> None:
        """Build mappings from slot values to indices."""
        self.slot_indices: dict[str, dict[str, int]] = {}
        self.slot_sizes: dict[str, int] = {}

        # Core genres (from all families)
        genres = self.components.genre.all_core_genres
        self.slot_indices["genre"] = {genre: i for i, genre in enumerate(genres)}
        self.slot_sizes["genre"] = len(genres)
        
        # Intimate gesture adjectives
        adjs = self.components.intimate_gesture.all_adjectives
        self.slot_indices["gesture_adj"] = {adj: i for i, adj in enumerate(adjs)}
        self.slot_sizes["gesture_adj"] = len(adjs)
        
        # Intimate gesture nouns
        nouns = self.components.intimate_gesture.all_nouns
        self.slot_indices["gesture_noun"] = {noun: i for i, noun in enumerate(nouns)}
        self.slot_sizes["gesture_noun"] = len(nouns)
        
        # Motion verbs
        verbs = self.components.dynamic_tension.motion_verbs
        self.slot_indices["motion_verb"] = {verb: i for i, verb in enumerate(verbs)}
        self.slot_sizes["motion_verb"] = len(verbs)
        
        # Musical objects
        objs = self.components.dynamic_tension.all_objects
        self.slot_indices["music_obj"] = {obj: i for i, obj in enumerate(objs)}
        self.slot_sizes["music_obj"] = len(objs)
        
        # Environments
        envs = self.components.sensory_bridge.environments
        self.slot_indices["environment"] = {env: i for i, env in enumerate(envs)}
        self.slot_sizes["environment"] = len(envs)
        
        # Emotions
        emotions = self.components.emotional_anchor.all_emotions
        self.slot_indices["emotion"] = {emo: i for i, emo in enumerate(emotions)}
        self.slot_sizes["emotion"] = len(emotions)
        
        # Arcs
        arcs = self.components.emotional_anchor.all_arcs
        self.slot_indices["arc"] = {arc: i for i, arc in enumerate(arcs)}
        self.slot_sizes["arc"] = len(arcs)
        
        # Total dimension
        self.embedding_dim = sum(self.slot_sizes.values())
    
    def _get_sentence_model(self) -> Any:
        """Lazy load sentence transformer model."""
        if self._sentence_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._sentence_model = SentenceTransformer(self._embedding_model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for sentence embeddings. "
                    "Install with: pip install sentence-transformers"
                )
        return self._sentence_model
    
    def embed(self, metaphor: Metaphor) -> list[float]:
        """
        Convert a metaphor to a numerical vector.
        
        Args:
            metaphor: The metaphor to embed
            
        Returns:
            List of floats representing the metaphor
        """
        if self.use_sentence_embeddings:
            return self._sentence_embed(metaphor)
        else:
            return self._onehot_embed(metaphor)
    
    def _sentence_embed(self, metaphor: Metaphor) -> list[float]:
        """Embed using sentence transformer."""
        model = self._get_sentence_model()
        text = str(metaphor)
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def _onehot_embed(self, metaphor: Metaphor) -> list[float]:
        """
        Create a relaxed one-hot encoding.
        
        Instead of strict one-hot, uses partial matches for robustness
        to unseen slot values.
        """
        vector: list[float] = []
        
        # Extract components from each slot
        slots = metaphor.slot_values
        
        # Genre (slot 0) - match era
        genre_vec = [0.0] * self.slot_sizes["genre"]
        for era, idx in self.slot_indices["genre"].items():
            if era.lower() in slots[0].lower():
                genre_vec[idx] = 1.0
                break
        vector.extend(genre_vec)
        
        # Gesture adjective (slot 1)
        adj_vec = [0.0] * self.slot_sizes["gesture_adj"]
        for adj, idx in self.slot_indices["gesture_adj"].items():
            if adj.lower() in slots[1].lower():
                adj_vec[idx] = 1.0
                break
        vector.extend(adj_vec)
        
        # Gesture noun (slot 1)
        noun_vec = [0.0] * self.slot_sizes["gesture_noun"]
        for noun, idx in self.slot_indices["gesture_noun"].items():
            if noun.lower() in slots[1].lower():
                noun_vec[idx] = 1.0
                break
        vector.extend(noun_vec)
        
        # Motion verb (slot 2)
        verb_vec = [0.0] * self.slot_sizes["motion_verb"]
        for verb, idx in self.slot_indices["motion_verb"].items():
            if verb.lower() in slots[2].lower():
                verb_vec[idx] = 1.0
                break
        vector.extend(verb_vec)
        
        # Musical object (slot 2)
        obj_vec = [0.0] * self.slot_sizes["music_obj"]
        for obj, idx in self.slot_indices["music_obj"].items():
            if obj.lower() in slots[2].lower():
                obj_vec[idx] = 1.0
                break
        vector.extend(obj_vec)
        
        # Environment (slot 3)
        env_vec = [0.0] * self.slot_sizes["environment"]
        for env, idx in self.slot_indices["environment"].items():
            if env.lower() in slots[3].lower():
                env_vec[idx] = 1.0
                break
        vector.extend(env_vec)
        
        # Emotion (slot 4)
        emo_vec = [0.0] * self.slot_sizes["emotion"]
        for emo, idx in self.slot_indices["emotion"].items():
            if emo.lower() in slots[4].lower():
                emo_vec[idx] = 1.0
                break
        vector.extend(emo_vec)
        
        # Arc (slot 4)
        arc_vec = [0.0] * self.slot_sizes["arc"]
        for arc, idx in self.slot_indices["arc"].items():
            if arc.lower() in slots[4].lower():
                arc_vec[idx] = 1.0
                break
        vector.extend(arc_vec)
        
        return vector
    
    def embed_batch(self, metaphors: Sequence[Metaphor]) -> list[list[float]]:
        """Embed multiple metaphors."""
        if self.use_sentence_embeddings:
            model = self._get_sentence_model()
            texts = [str(m) for m in metaphors]
            embeddings = model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        else:
            return [self._onehot_embed(m) for m in metaphors]


# =============================================================================
# Surrogate Model (Simple GP approximation)
# =============================================================================


def rbf_kernel(x1: list[float], x2: list[float], length_scale: float = 1.0) -> float:
    """Radial Basis Function (squared exponential) kernel."""
    sq_dist = sum((a - b) ** 2 for a, b in zip(x1, x2))
    return math.exp(-sq_dist / (2 * length_scale ** 2))


class SimpleSurrogate:
    """
    Simple surrogate model using RBF kernel interpolation.
    
    This is a lightweight approximation to Gaussian Process regression,
    suitable when scipy is not available. For production use, consider
    using sklearn.gaussian_process.GaussianProcessRegressor.
    
    The model predicts fitness at new points by weighted averaging of
    observed points, with weights determined by RBF kernel similarity.
    """
    
    def __init__(
        self,
        length_scale: float = 1.0,
        noise: float = 0.1,
    ) -> None:
        self.length_scale = length_scale
        self.noise = noise
        self.X: list[list[float]] = []
        self.y: list[float] = []
    
    def fit(self, X: list[list[float]], y: list[float]) -> None:
        """Store training data."""
        self.X = X
        self.y = y
    
    def add_observation(self, x: list[float], y_val: float) -> None:
        """Add a single observation."""
        self.X.append(x)
        self.y.append(y_val)
    
    def predict(self, x: list[float]) -> tuple[float, float]:
        """
        Predict mean and uncertainty at a point.
        
        Returns:
            Tuple of (mean prediction, uncertainty estimate)
        """
        if not self.X:
            return 0.5, 1.0  # Uninformed prior
        
        # Compute kernel weights
        weights = [rbf_kernel(x, xi, self.length_scale) for xi in self.X]
        total_weight = sum(weights) + 1e-10
        
        # Weighted mean
        mean = sum(w * yi for w, yi in zip(weights, self.y)) / total_weight
        
        # Uncertainty: lower when close to observed points
        max_weight = max(weights) if weights else 0
        uncertainty = 1.0 - max_weight + self.noise
        
        return mean, uncertainty
    
    def predict_batch(
        self, 
        X: list[list[float]]
    ) -> tuple[list[float], list[float]]:
        """Predict for multiple points."""
        means = []
        uncertainties = []
        for x in X:
            m, u = self.predict(x)
            means.append(m)
            uncertainties.append(u)
        return means, uncertainties


# =============================================================================
# Acquisition Functions
# =============================================================================


class AcquisitionType(str, Enum):
    """Available acquisition functions."""
    
    EI = "expected_improvement"
    UCB = "upper_confidence_bound"
    PI = "probability_improvement"
    THOMPSON = "thompson_sampling"


def expected_improvement(
    mean: float,
    std: float,
    best_so_far: float,
    xi: float = 0.01,
) -> float:
    """
    Expected Improvement acquisition function.
    
    Balances exploitation (high mean) with exploration (high uncertainty).
    
    Args:
        mean: Predicted mean at point
        std: Predicted standard deviation
        best_so_far: Best observed value
        xi: Exploration-exploitation tradeoff (higher = more exploration)
    """
    if std <= 0:
        return 0.0
    
    z = (mean - best_so_far - xi) / std
    
    # Approximate CDF and PDF of standard normal
    # Using simple approximation to avoid scipy dependency
    cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    pdf = math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
    
    ei = (mean - best_so_far - xi) * cdf + std * pdf
    return max(0.0, ei)


def upper_confidence_bound(
    mean: float,
    std: float,
    beta: float = 2.0,
) -> float:
    """
    Upper Confidence Bound acquisition function.
    
    Simply adds beta * std to the mean prediction.
    Higher beta = more exploration.
    """
    return mean + beta * std


def probability_of_improvement(
    mean: float,
    std: float,
    best_so_far: float,
    xi: float = 0.01,
) -> float:
    """
    Probability of Improvement acquisition function.
    
    Returns probability that point improves over best_so_far.
    """
    if std <= 0:
        return 0.0 if mean < best_so_far + xi else 1.0
    
    z = (mean - best_so_far - xi) / std
    cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return cdf


# =============================================================================
# Bayesian Optimizer
# =============================================================================


@dataclass
class BayesianConfig:
    """Configuration for Bayesian optimization."""
    
    n_initial: int = 10  # Initial random samples
    n_iterations: int = 50  # Optimization iterations
    n_candidates: int = 100  # Candidates to evaluate acquisition function
    acquisition: AcquisitionType = AcquisitionType.EI
    exploration_weight: float = 0.1  # xi for EI, beta for UCB
    length_scale: float = 1.0  # RBF kernel length scale
    use_sentence_embeddings: bool = False
    seed: int | None = None


@dataclass
class BayesianResult:
    """Result of Bayesian optimization."""
    
    best_metaphor: Metaphor
    best_score: float
    n_evaluations: int
    history: list[dict[str, Any]]


class BayesianOptimizer:
    """
    Bayesian optimizer for sample-efficient metaphor optimization.
    
    Builds a surrogate model of the fitness landscape and uses
    acquisition functions to decide which metaphors to evaluate next.
    Particularly useful when evaluation is expensive (actual audio
    generation + human rating).
    
    Example:
        >>> optimizer = BayesianOptimizer(
        ...     generator=generator,
        ...     evaluator=human_evaluator,
        ...     config=BayesianConfig(n_initial=5, n_iterations=20),
        ... )
        >>> result = optimizer.run(verbose=True)
        >>> print(f"Best: {result.best_metaphor}")
    """
    
    def __init__(
        self,
        generator: MetaphorGenerator,
        evaluator: FitnessEvaluator,
        config: BayesianConfig | None = None,
    ) -> None:
        self.generator = generator
        self.evaluator = evaluator
        self.config = config or BayesianConfig()
        
        self.rng = random.Random(self.config.seed)
        self.embedder = MetaphorEmbedder(
            generator.components,
            use_sentence_embeddings=self.config.use_sentence_embeddings,
        )
        self.surrogate = SimpleSurrogate(length_scale=self.config.length_scale)
        
        # Observation history
        self.observed_metaphors: list[Metaphor] = []
        self.observed_scores: list[float] = []
        self.observed_embeddings: list[list[float]] = []
        
        self.history: list[dict[str, Any]] = []
    
    def _evaluate_and_record(self, metaphor: Metaphor) -> float:
        """Evaluate a metaphor and record the observation."""
        result = self.evaluator(metaphor)
        score = float(result.score)
        embedding = self.embedder.embed(metaphor)
        
        self.observed_metaphors.append(metaphor)
        self.observed_scores.append(score)
        self.observed_embeddings.append(embedding)
        self.surrogate.add_observation(embedding, score)
        
        return score
    
    def _generate_candidates(self, n: int) -> list[Metaphor]:
        """Generate random candidate metaphors."""
        return [self.generator.generate_single() for _ in range(n)]
    
    def _select_next(self, candidates: list[Metaphor]) -> Metaphor:
        """Select the best candidate according to acquisition function."""
        if not self.observed_scores:
            return self.rng.choice(candidates)
        
        best_so_far = max(self.observed_scores)
        best_acquisition = float("-inf")
        best_candidate = candidates[0]
        
        for candidate in candidates:
            embedding = self.embedder.embed(candidate)
            mean, std = self.surrogate.predict(embedding)
            
            # Compute acquisition value
            if self.config.acquisition == AcquisitionType.EI:
                acq = expected_improvement(
                    mean, std, best_so_far, self.config.exploration_weight
                )
            elif self.config.acquisition == AcquisitionType.UCB:
                acq = upper_confidence_bound(
                    mean, std, self.config.exploration_weight
                )
            elif self.config.acquisition == AcquisitionType.PI:
                acq = probability_of_improvement(
                    mean, std, best_so_far, self.config.exploration_weight
                )
            else:  # Thompson sampling
                acq = self.rng.gauss(mean, std)
            
            if acq > best_acquisition:
                best_acquisition = acq
                best_candidate = candidate
        
        return best_candidate
    
    def run(self, verbose: bool = False) -> BayesianResult:
        """
        Run Bayesian optimization.
        
        Args:
            verbose: Print progress information
            
        Returns:
            BayesianResult with best metaphor and history
        """
        # Phase 1: Initial random sampling
        if verbose:
            print(f"Phase 1: Initial sampling ({self.config.n_initial} points)")
        
        for i in range(self.config.n_initial):
            metaphor = self.generator.generate_single()
            score = self._evaluate_and_record(metaphor)
            
            self.history.append({
                "iteration": i,
                "phase": "initial",
                "score": score,
                "best_so_far": max(self.observed_scores),
            })
            
            if verbose:
                print(f"  [{i+1}/{self.config.n_initial}] score={score:.3f}")
        
        # Phase 2: Bayesian optimization loop
        if verbose:
            print(f"\nPhase 2: Optimization ({self.config.n_iterations} iterations)")
        
        for i in range(self.config.n_iterations):
            # Generate candidates
            candidates = self._generate_candidates(self.config.n_candidates)
            
            # Select best according to acquisition function
            next_metaphor = self._select_next(candidates)
            
            # Evaluate
            score = self._evaluate_and_record(next_metaphor)
            
            self.history.append({
                "iteration": self.config.n_initial + i,
                "phase": "optimization",
                "score": score,
                "best_so_far": max(self.observed_scores),
            })
            
            if verbose:
                best = max(self.observed_scores)
                print(f"  [{i+1}/{self.config.n_iterations}] score={score:.3f}, best={best:.3f}")
        
        # Find best
        best_idx = max(range(len(self.observed_scores)), key=lambda i: self.observed_scores[i])
        
        return BayesianResult(
            best_metaphor=self.observed_metaphors[best_idx],
            best_score=self.observed_scores[best_idx],
            n_evaluations=len(self.observed_scores),
            history=self.history,
        )
    
    def suggest_next(self, n: int = 1) -> list[Metaphor]:
        """
        Suggest next metaphors to evaluate without actually evaluating them.
        
        Useful for batch evaluation scenarios where you want to select
        multiple points before getting any results back.
        
        Args:
            n: Number of suggestions to return
            
        Returns:
            List of suggested metaphors
        """
        suggestions = []
        
        for _ in range(n):
            candidates = self._generate_candidates(self.config.n_candidates)
            best = self._select_next(candidates)
            suggestions.append(best)
            
            # Temporarily add to observations with predicted score
            # to encourage diversity in suggestions
            embedding = self.embedder.embed(best)
            mean, _ = self.surrogate.predict(embedding)
            self.surrogate.add_observation(embedding, mean)
        
        return suggestions
    
    def add_external_observation(self, metaphor: Metaphor, score: float) -> None:
        """
        Add an observation from external evaluation.
        
        Useful when evaluations happen outside the optimizer
        (e.g., async human rating).
        """
        embedding = self.embedder.embed(metaphor)
        self.observed_metaphors.append(metaphor)
        self.observed_scores.append(score)
        self.observed_embeddings.append(embedding)
        self.surrogate.add_observation(embedding, score)
