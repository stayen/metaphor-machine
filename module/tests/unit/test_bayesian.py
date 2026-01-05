"""Unit tests for Bayesian optimization module."""

import pytest
import math

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
from metaphor_machine.optimization.bayesian import (
    MetaphorEmbedder,
    SimpleSurrogate,
    BayesianConfig,
    BayesianResult,
    BayesianOptimizer,
    AcquisitionType,
    rbf_kernel,
    expected_improvement,
    upper_confidence_bound,
    probability_of_improvement,
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


# =============================================================================
# RBF Kernel Tests
# =============================================================================


class TestRBFKernel:
    """Tests for the RBF kernel function."""
    
    def test_identical_points(self):
        """Test that identical points have kernel value 1."""
        x = [1.0, 2.0, 3.0]
        assert rbf_kernel(x, x) == 1.0
    
    def test_different_points(self):
        """Test that different points have kernel value < 1."""
        x1 = [0.0, 0.0]
        x2 = [1.0, 0.0]
        
        k = rbf_kernel(x1, x2, length_scale=1.0)
        assert 0 < k < 1
    
    def test_length_scale_effect(self):
        """Test that larger length scale increases similarity."""
        x1 = [0.0, 0.0]
        x2 = [1.0, 0.0]
        
        k_small = rbf_kernel(x1, x2, length_scale=0.5)
        k_large = rbf_kernel(x1, x2, length_scale=2.0)
        
        assert k_large > k_small
    
    def test_symmetry(self):
        """Test that kernel is symmetric."""
        x1 = [1.0, 2.0, 3.0]
        x2 = [4.0, 5.0, 6.0]
        
        assert rbf_kernel(x1, x2) == rbf_kernel(x2, x1)


# =============================================================================
# Acquisition Function Tests
# =============================================================================


class TestExpectedImprovement:
    """Tests for Expected Improvement acquisition function."""
    
    def test_high_mean_high_ei(self):
        """Test that high mean gives higher EI."""
        ei_high = expected_improvement(0.9, 0.1, 0.5)
        ei_low = expected_improvement(0.3, 0.1, 0.5)
        
        assert ei_high > ei_low
    
    def test_high_std_exploration(self):
        """Test that high uncertainty encourages exploration."""
        ei_certain = expected_improvement(0.6, 0.01, 0.5)
        ei_uncertain = expected_improvement(0.6, 0.3, 0.5)
        
        assert ei_uncertain > ei_certain
    
    def test_below_best_with_uncertainty(self):
        """Test EI for points below best but with uncertainty."""
        ei = expected_improvement(0.4, 0.2, 0.5)
        
        # Should still be positive due to uncertainty
        assert ei > 0
    
    def test_zero_std_zero_ei(self):
        """Test that zero uncertainty gives zero EI."""
        ei = expected_improvement(0.4, 0.0, 0.5)
        assert ei == 0.0


class TestUpperConfidenceBound:
    """Tests for Upper Confidence Bound acquisition function."""
    
    def test_ucb_formula(self):
        """Test UCB formula: mean + beta * std."""
        mean, std, beta = 0.5, 0.2, 2.0
        ucb = upper_confidence_bound(mean, std, beta)
        
        assert ucb == mean + beta * std
    
    def test_higher_beta_more_exploration(self):
        """Test that higher beta leads to more exploration."""
        mean, std = 0.5, 0.3
        
        ucb_low = upper_confidence_bound(mean, std, beta=1.0)
        ucb_high = upper_confidence_bound(mean, std, beta=3.0)
        
        assert ucb_high > ucb_low


class TestProbabilityOfImprovement:
    """Tests for Probability of Improvement acquisition function."""
    
    def test_above_best_high_pi(self):
        """Test that points above best have high PI."""
        pi = probability_of_improvement(0.9, 0.1, 0.5)
        assert pi > 0.9
    
    def test_below_best_low_pi(self):
        """Test that points below best have lower PI."""
        pi = probability_of_improvement(0.3, 0.1, 0.5)
        assert pi < 0.1
    
    def test_zero_std_deterministic(self):
        """Test PI with zero uncertainty is deterministic."""
        pi_above = probability_of_improvement(0.6, 0.0, 0.5)
        pi_below = probability_of_improvement(0.4, 0.0, 0.5)
        
        assert pi_above == 1.0
        assert pi_below == 0.0


# =============================================================================
# SimpleSurrogate Tests
# =============================================================================


class TestSimpleSurrogate:
    """Tests for SimpleSurrogate model."""
    
    def test_no_data_prior(self):
        """Test prediction with no data returns prior."""
        surrogate = SimpleSurrogate()
        mean, std = surrogate.predict([0.5, 0.5])
        
        assert mean == 0.5
        assert std == 1.0
    
    def test_prediction_near_data(self):
        """Test prediction near observed point."""
        surrogate = SimpleSurrogate(length_scale=1.0)
        
        # Add observation
        surrogate.add_observation([0.0, 0.0], 0.8)
        
        # Predict at same point
        mean, std = surrogate.predict([0.0, 0.0])
        
        assert abs(mean - 0.8) < 0.1
        assert std < 0.5  # Low uncertainty near data
    
    def test_prediction_far_from_data(self):
        """Test prediction far from observed point."""
        surrogate = SimpleSurrogate(length_scale=0.5)
        
        # Add observation
        surrogate.add_observation([0.0, 0.0], 0.8)
        
        # Predict far away
        mean, std = surrogate.predict([10.0, 10.0])
        
        # Should have high uncertainty
        assert std > 0.9
    
    def test_multiple_observations(self):
        """Test with multiple observations."""
        surrogate = SimpleSurrogate(length_scale=1.0)
        
        surrogate.add_observation([0.0, 0.0], 0.3)
        surrogate.add_observation([1.0, 0.0], 0.7)
        
        # Midpoint should interpolate
        mean, std = surrogate.predict([0.5, 0.0])
        
        assert 0.3 < mean < 0.7
    
    def test_fit_method(self):
        """Test fit method."""
        surrogate = SimpleSurrogate()
        
        X = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        y = [0.2, 0.5, 0.8]
        
        surrogate.fit(X, y)
        
        assert len(surrogate.X) == 3
        assert len(surrogate.y) == 3
    
    def test_predict_batch(self):
        """Test batch prediction."""
        surrogate = SimpleSurrogate()
        surrogate.add_observation([0.0, 0.0], 0.5)
        
        X = [[0.0, 0.0], [1.0, 1.0]]
        means, stds = surrogate.predict_batch(X)
        
        assert len(means) == 2
        assert len(stds) == 2


# =============================================================================
# MetaphorEmbedder Tests
# =============================================================================


class TestMetaphorEmbedder:
    """Tests for MetaphorEmbedder class."""
    
    def test_initialization(self, components):
        """Test embedder initialization."""
        embedder = MetaphorEmbedder(components)
        
        assert embedder.embedding_dim > 0
        assert len(embedder.slot_indices) > 0
    
    def test_onehot_embed(self, components, sample_metaphor):
        """Test one-hot embedding."""
        embedder = MetaphorEmbedder(components, use_sentence_embeddings=False)
        
        embedding = embedder.embed(sample_metaphor)
        
        assert isinstance(embedding, list)
        assert len(embedding) == embedder.embedding_dim
        assert all(isinstance(v, float) for v in embedding)
    
    def test_embed_batch(self, components, sample_metaphor, second_metaphor):
        """Test batch embedding."""
        embedder = MetaphorEmbedder(components, use_sentence_embeddings=False)
        
        embeddings = embedder.embed_batch([sample_metaphor, second_metaphor])
        
        assert len(embeddings) == 2
        assert len(embeddings[0]) == embedder.embedding_dim
    
    def test_different_metaphors_different_embeddings(self, components, sample_metaphor, second_metaphor):
        """Test that different metaphors have different embeddings."""
        embedder = MetaphorEmbedder(components, use_sentence_embeddings=False)
        
        emb1 = embedder.embed(sample_metaphor)
        emb2 = embedder.embed(second_metaphor)
        
        # Should not be identical
        assert emb1 != emb2


# =============================================================================
# BayesianConfig Tests
# =============================================================================


class TestBayesianConfig:
    """Tests for BayesianConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = BayesianConfig()
        
        assert config.n_initial == 10
        assert config.n_iterations == 50
        assert config.n_candidates == 100
        assert config.acquisition == AcquisitionType.EI
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = BayesianConfig(
            n_initial=5,
            n_iterations=20,
            acquisition=AcquisitionType.UCB,
        )
        
        assert config.n_initial == 5
        assert config.n_iterations == 20
        assert config.acquisition == AcquisitionType.UCB


# =============================================================================
# BayesianOptimizer Tests
# =============================================================================


class TestBayesianOptimizer:
    """Tests for BayesianOptimizer class."""
    
    def test_initialization(self, generator):
        """Test optimizer initialization."""
        evaluator = RandomEvaluator(seed=42)
        config = BayesianConfig(n_initial=3, n_iterations=5)
        
        optimizer = BayesianOptimizer(
            generator=generator,
            evaluator=evaluator,
            config=config,
        )
        
        assert optimizer.generator == generator
        assert optimizer.evaluator == evaluator
        assert len(optimizer.observed_metaphors) == 0
    
    def test_run_optimization(self, generator):
        """Test running Bayesian optimization."""
        evaluator = RuleBasedEvaluator(preferred_terms=["darkwave"])
        config = BayesianConfig(
            n_initial=3,
            n_iterations=5,
            n_candidates=20,
            seed=42,
        )
        
        optimizer = BayesianOptimizer(
            generator=generator,
            evaluator=evaluator,
            config=config,
        )
        
        result = optimizer.run(verbose=False)
        
        assert isinstance(result, BayesianResult)
        assert isinstance(result.best_metaphor, Metaphor)
        assert 0 <= result.best_score <= 1
        assert result.n_evaluations == 8  # 3 initial + 5 iterations
    
    def test_history_tracking(self, generator):
        """Test that optimization history is tracked."""
        evaluator = RandomEvaluator(seed=42)
        config = BayesianConfig(
            n_initial=3,
            n_iterations=5,
            seed=42,
        )
        
        optimizer = BayesianOptimizer(
            generator=generator,
            evaluator=evaluator,
            config=config,
        )
        
        result = optimizer.run(verbose=False)
        
        assert len(result.history) == 8
        assert all("score" in h for h in result.history)
        assert all("best_so_far" in h for h in result.history)
    
    def test_suggest_next(self, generator):
        """Test suggestion without evaluation."""
        evaluator = RandomEvaluator(seed=42)
        config = BayesianConfig(
            n_initial=3,
            n_iterations=5,
            seed=42,
        )
        
        optimizer = BayesianOptimizer(
            generator=generator,
            evaluator=evaluator,
            config=config,
        )
        
        # Add some initial observations
        for _ in range(3):
            m = generator.generate_single()
            optimizer.add_external_observation(m, 0.5)
        
        suggestions = optimizer.suggest_next(n=3)
        
        assert len(suggestions) == 3
        assert all(isinstance(s, Metaphor) for s in suggestions)
    
    def test_add_external_observation(self, generator):
        """Test adding external observations."""
        evaluator = RandomEvaluator(seed=42)
        config = BayesianConfig()
        
        optimizer = BayesianOptimizer(
            generator=generator,
            evaluator=evaluator,
            config=config,
        )
        
        metaphor = generator.generate_single()
        optimizer.add_external_observation(metaphor, 0.75)
        
        assert len(optimizer.observed_metaphors) == 1
        assert len(optimizer.observed_scores) == 1
        assert optimizer.observed_scores[0] == 0.75
    
    def test_acquisition_types(self, generator):
        """Test different acquisition function types."""
        evaluator = RandomEvaluator(seed=42)
        
        for acq_type in AcquisitionType:
            config = BayesianConfig(
                n_initial=2,
                n_iterations=2,
                acquisition=acq_type,
                seed=42,
            )
            
            optimizer = BayesianOptimizer(
                generator=generator,
                evaluator=evaluator,
                config=config,
            )
            
            result = optimizer.run(verbose=False)
            assert isinstance(result.best_metaphor, Metaphor)
    
    def test_best_improves_over_time(self, generator):
        """Test that best_so_far improves or stays same."""
        evaluator = RandomEvaluator(seed=42)
        config = BayesianConfig(
            n_initial=3,
            n_iterations=10,
            seed=42,
        )
        
        optimizer = BayesianOptimizer(
            generator=generator,
            evaluator=evaluator,
            config=config,
        )
        
        result = optimizer.run(verbose=False)
        
        # best_so_far should be monotonically non-decreasing
        best_values = [h["best_so_far"] for h in result.history]
        for i in range(1, len(best_values)):
            assert best_values[i] >= best_values[i - 1]


# =============================================================================
# Integration Tests
# =============================================================================


class TestBayesianIntegration:
    """Integration tests for Bayesian optimization."""
    
    def test_optimize_toward_target(self, generator):
        """Test that optimization can find high-scoring metaphors."""
        # Create evaluator that rewards specific terms
        evaluator = RuleBasedEvaluator(
            preferred_terms=["darkwave", "swelling", "melancholy"],
            avoided_terms=["pop", "bright"],
        )
        
        config = BayesianConfig(
            n_initial=5,
            n_iterations=10,
            n_candidates=50,
            seed=42,
        )
        
        optimizer = BayesianOptimizer(
            generator=generator,
            evaluator=evaluator,
            config=config,
        )
        
        result = optimizer.run(verbose=False)
        
        # Should find relatively good solutions
        assert result.best_score > 0.5
    
    def test_exploration_vs_exploitation(self, generator):
        """Test exploration-exploitation tradeoff."""
        evaluator = RandomEvaluator(seed=42)
        
        # High exploration weight (UCB with high beta)
        config_explore = BayesianConfig(
            n_initial=3,
            n_iterations=5,
            acquisition=AcquisitionType.UCB,
            exploration_weight=5.0,
            seed=42,
        )
        
        # Low exploration weight
        config_exploit = BayesianConfig(
            n_initial=3,
            n_iterations=5,
            acquisition=AcquisitionType.UCB,
            exploration_weight=0.1,
            seed=42,
        )
        
        optimizer_explore = BayesianOptimizer(generator, evaluator, config_explore)
        optimizer_exploit = BayesianOptimizer(generator, evaluator, config_exploit)
        
        result_explore = optimizer_explore.run(verbose=False)
        result_exploit = optimizer_exploit.run(verbose=False)
        
        # Both should complete successfully
        assert isinstance(result_explore.best_metaphor, Metaphor)
        assert isinstance(result_exploit.best_metaphor, Metaphor)
