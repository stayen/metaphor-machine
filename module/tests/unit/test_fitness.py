"""Unit tests for fitness evaluation module."""

import pytest
import tempfile
from pathlib import Path

from metaphor_machine.core.metaphor import Metaphor, MetaphorSlot, SlotType
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


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_metaphor():
    """Create a sample metaphor for testing."""
    return Metaphor(
        genre_anchor=MetaphorSlot(SlotType.GENRE_ANCHOR, "darkwave fusion"),
        intimate_gesture=MetaphorSlot(SlotType.INTIMATE_GESTURE, "ethereal whisper-hooks"),
        dynamic_tension=MetaphorSlot(SlotType.DYNAMIC_TENSION, "swelling synth-pads"),
        sensory_bridge=MetaphorSlot(SlotType.SENSORY_BRIDGE, "neon-glow"),
        emotional_anchor=MetaphorSlot(SlotType.EMOTIONAL_ANCHOR, "melancholy drift"),
    )


@pytest.fixture
def second_metaphor():
    """Create another sample metaphor for testing."""
    return Metaphor(
        genre_anchor=MetaphorSlot(SlotType.GENRE_ANCHOR, "trip-hop blend"),
        intimate_gesture=MetaphorSlot(SlotType.INTIMATE_GESTURE, "breathy confessions"),
        dynamic_tension=MetaphorSlot(SlotType.DYNAMIC_TENSION, "crackling beats"),
        sensory_bridge=MetaphorSlot(SlotType.SENSORY_BRIDGE, "basement-reverb"),
        emotional_anchor=MetaphorSlot(SlotType.EMOTIONAL_ANCHOR, "longing surge"),
    )


# =============================================================================
# FitnessScore Tests
# =============================================================================


class TestFitnessScore:
    """Tests for the FitnessScore class."""
    
    def test_valid_score(self):
        """Test creating valid fitness scores."""
        score = FitnessScore(0.5)
        assert score == 0.5
        
        score_zero = FitnessScore(0.0)
        assert score_zero == 0.0
        
        score_one = FitnessScore(1.0)
        assert score_one == 1.0
    
    def test_invalid_score_too_high(self):
        """Test that scores above 1.0 raise ValueError."""
        with pytest.raises(ValueError):
            FitnessScore(1.1)
    
    def test_invalid_score_too_low(self):
        """Test that scores below 0.0 raise ValueError."""
        with pytest.raises(ValueError):
            FitnessScore(-0.1)
    
    def test_from_unbounded_normalization(self):
        """Test normalizing unbounded values."""
        score = FitnessScore.from_unbounded(50, 0, 100)
        assert score == 0.5
        
        score_min = FitnessScore.from_unbounded(0, 0, 100)
        assert score_min == 0.0
        
        score_max = FitnessScore.from_unbounded(100, 0, 100)
        assert score_max == 1.0
    
    def test_from_unbounded_clamping(self):
        """Test that values outside range are clamped."""
        score_over = FitnessScore.from_unbounded(150, 0, 100)
        assert score_over == 1.0
        
        score_under = FitnessScore.from_unbounded(-50, 0, 100)
        assert score_under == 0.0


# =============================================================================
# EvaluationResult Tests
# =============================================================================


class TestEvaluationResult:
    """Tests for the EvaluationResult class."""
    
    def test_creation(self):
        """Test creating an evaluation result."""
        result = EvaluationResult(
            score=FitnessScore(0.75),
            raw_scores={"criterion1": 0.8, "criterion2": 0.7},
            metadata={"method": "test"},
            evaluation_time=0.1,
            evaluator_name="test_evaluator",
        )
        
        assert result.score == 0.75
        assert result.raw_scores["criterion1"] == 0.8
        assert result.metadata["method"] == "test"
        assert result.evaluation_time == 0.1
        assert result.evaluator_name == "test_evaluator"
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        result = EvaluationResult(
            score=FitnessScore(0.5),
            raw_scores={"a": 1.0},
        )
        
        d = result.to_dict()
        assert d["score"] == 0.5
        assert d["raw_scores"]["a"] == 1.0
    
    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "score": 0.6,
            "raw_scores": {"b": 0.7},
            "metadata": {"key": "value"},
            "evaluation_time": 0.2,
            "evaluator_name": "restored",
        }
        
        result = EvaluationResult.from_dict(data)
        assert result.score == 0.6
        assert result.raw_scores["b"] == 0.7
        assert result.metadata["key"] == "value"


# =============================================================================
# RandomEvaluator Tests
# =============================================================================


class TestRandomEvaluator:
    """Tests for the RandomEvaluator class."""
    
    def test_returns_valid_score(self, sample_metaphor):
        """Test that random evaluator returns valid scores."""
        evaluator = RandomEvaluator(seed=42)
        result = evaluator(sample_metaphor)
        
        assert 0.0 <= result.score <= 1.0
        assert result.evaluator_name == "RandomEvaluator"
    
    def test_reproducibility(self, sample_metaphor):
        """Test that same seed produces same results."""
        evaluator1 = RandomEvaluator(seed=123)
        evaluator2 = RandomEvaluator(seed=123)
        
        result1 = evaluator1.evaluate(sample_metaphor)
        result2 = evaluator2.evaluate(sample_metaphor)
        
        assert result1.score == result2.score
    
    def test_stats_tracking(self, sample_metaphor):
        """Test that evaluation statistics are tracked."""
        evaluator = RandomEvaluator()
        
        evaluator(sample_metaphor)
        evaluator(sample_metaphor)
        
        stats = evaluator.stats
        assert stats["evaluation_count"] == 2
        assert stats["total_time"] > 0


# =============================================================================
# RuleBasedEvaluator Tests
# =============================================================================


class TestRuleBasedEvaluator:
    """Tests for the RuleBasedEvaluator class."""
    
    def test_preferred_terms_boost(self, sample_metaphor):
        """Test that preferred terms increase score."""
        evaluator_with = RuleBasedEvaluator(
            preferred_terms=["darkwave", "ethereal"],
            name="with_prefs",
        )
        evaluator_without = RuleBasedEvaluator(
            preferred_terms=["jazz", "blues"],
            name="without_prefs",
        )
        
        result_with = evaluator_with(sample_metaphor)
        result_without = evaluator_without(sample_metaphor)
        
        # Sample metaphor contains "darkwave" and "ethereal"
        assert result_with.raw_scores["preferred_terms"] > result_without.raw_scores["preferred_terms"]
    
    def test_avoided_terms_penalty(self, sample_metaphor):
        """Test that avoided terms decrease score."""
        evaluator = RuleBasedEvaluator(
            avoided_terms=["darkwave"],  # Present in sample
        )
        
        result = evaluator(sample_metaphor)
        
        # Should be penalized
        assert result.raw_scores["avoided_terms"] < 1.0
    
    def test_length_scoring(self):
        """Test length-based scoring."""
        evaluator = RuleBasedEvaluator(min_length=50, max_length=100)
        
        short_metaphor = Metaphor(
            genre_anchor=MetaphorSlot(SlotType.GENRE_ANCHOR, "a"),
            intimate_gesture=MetaphorSlot(SlotType.INTIMATE_GESTURE, "b"),
            dynamic_tension=MetaphorSlot(SlotType.DYNAMIC_TENSION, "c"),
            sensory_bridge=MetaphorSlot(SlotType.SENSORY_BRIDGE, "d"),
            emotional_anchor=MetaphorSlot(SlotType.EMOTIONAL_ANCHOR, "e"),
        )
        
        result = evaluator(short_metaphor)
        
        # Very short, should have low length score
        assert result.raw_scores["length"] < 1.0
    
    def test_diversity_scoring(self, sample_metaphor):
        """Test word diversity scoring."""
        evaluator = RuleBasedEvaluator()
        result = evaluator(sample_metaphor)
        
        # Sample metaphor has diverse words
        assert "diversity" in result.raw_scores
        assert result.raw_scores["diversity"] > 0


# =============================================================================
# SemanticCoherenceEvaluator Tests
# =============================================================================


class TestSemanticCoherenceEvaluator:
    """Tests for the SemanticCoherenceEvaluator class."""
    
    def test_default_coherence(self, sample_metaphor):
        """Test coherence evaluation without custom pairs."""
        evaluator = SemanticCoherenceEvaluator()
        result = evaluator(sample_metaphor)
        
        assert 0.0 <= result.score <= 1.0
    
    def test_custom_coherence_pairs(self, sample_metaphor):
        """Test with custom coherence pairs."""
        evaluator = SemanticCoherenceEvaluator(
            coherence_pairs={
                ("darkwave", "ethereal"): 0.9,
                ("darkwave", "melancholy"): 0.8,
            },
            default_coherence=0.3,
        )
        
        result = evaluator(sample_metaphor)
        assert result.score > 0


# =============================================================================
# HumanFeedbackEvaluator Tests
# =============================================================================


class TestHumanFeedbackEvaluator:
    """Tests for the HumanFeedbackEvaluator class."""
    
    def test_custom_prompt_function(self, sample_metaphor):
        """Test with custom prompt function."""
        def mock_prompt(text: str) -> float:
            return 0.75  # Always return 0.75
        
        evaluator = HumanFeedbackEvaluator(prompt_fn=mock_prompt)
        result = evaluator(sample_metaphor)
        
        assert result.score == 0.75
    
    def test_caching(self, sample_metaphor, tmp_path):
        """Test that ratings are cached."""
        cache_path = tmp_path / "ratings.json"
        
        call_count = 0
        def counting_prompt(text: str) -> float:
            nonlocal call_count
            call_count += 1
            return 0.8
        
        evaluator = HumanFeedbackEvaluator(
            cache_path=cache_path,
            prompt_fn=counting_prompt,
        )
        
        # First call
        result1 = evaluator(sample_metaphor)
        assert call_count == 1
        
        # Second call should use cache
        result2 = evaluator(sample_metaphor)
        assert call_count == 1  # Not incremented
        assert result2.score == result1.score
        assert result2.metadata.get("source") == "cache"


# =============================================================================
# LLMEvaluator Tests
# =============================================================================


class TestLLMEvaluator:
    """Tests for the LLMEvaluator class."""
    
    def test_without_api_fn_raises(self, sample_metaphor):
        """Test that missing api_fn raises error."""
        evaluator = LLMEvaluator()
        
        with pytest.raises(RuntimeError):
            evaluator(sample_metaphor)
    
    def test_with_mock_api(self, sample_metaphor):
        """Test with mock API function."""
        def mock_api(prompt: str) -> str:
            return "I rate this a 7 out of 10."
        
        evaluator = LLMEvaluator(api_fn=mock_api)
        result = evaluator(sample_metaphor)
        
        assert result.score == 0.7  # 7/10 normalized
    
    def test_custom_parser(self, sample_metaphor):
        """Test with custom score parser."""
        def mock_api(prompt: str) -> str:
            return "Score: HIGH"
        
        def custom_parser(response: str) -> float:
            if "HIGH" in response:
                return 0.9
            return 0.5
        
        evaluator = LLMEvaluator(api_fn=mock_api, score_parser=custom_parser)
        result = evaluator(sample_metaphor)
        
        assert result.score == 0.9
    
    def test_caching(self, sample_metaphor, tmp_path):
        """Test that LLM responses are cached."""
        cache_path = tmp_path / "llm_cache.json"
        
        call_count = 0
        def counting_api(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return "8"
        
        evaluator = LLMEvaluator(api_fn=counting_api, cache_path=cache_path)
        
        evaluator(sample_metaphor)
        assert call_count == 1
        
        evaluator(sample_metaphor)
        assert call_count == 1  # Cached


# =============================================================================
# CompositeEvaluator Tests
# =============================================================================


class TestCompositeEvaluator:
    """Tests for the CompositeEvaluator class."""
    
    def test_weighted_mean(self, sample_metaphor):
        """Test weighted mean aggregation."""
        eval1 = RandomEvaluator(seed=42, name="eval1")
        eval2 = RandomEvaluator(seed=43, name="eval2")
        
        composite = CompositeEvaluator([
            (eval1, 0.6),
            (eval2, 0.4),
        ])
        
        result = composite(sample_metaphor)
        
        # Check that component scores are recorded
        assert "eval1" in str(result.metadata.get("component_scores", {}))
        assert "eval2" in str(result.metadata.get("component_scores", {}))
    
    def test_min_aggregation(self, sample_metaphor):
        """Test min aggregation."""
        class FixedEvaluator(FitnessEvaluator):
            def __init__(self, score: float, **kwargs):
                super().__init__(**kwargs)
                self._score = score
            
            def evaluate(self, target):
                return EvaluationResult(score=FitnessScore(self._score))
        
        composite = CompositeEvaluator(
            [
                (FixedEvaluator(0.9, name="high"), 1.0),
                (FixedEvaluator(0.3, name="low"), 1.0),
            ],
            aggregation="min",
        )
        
        result = composite(sample_metaphor)
        assert result.score == 0.3
    
    def test_max_aggregation(self, sample_metaphor):
        """Test max aggregation."""
        class FixedEvaluator(FitnessEvaluator):
            def __init__(self, score: float, **kwargs):
                super().__init__(**kwargs)
                self._score = score
            
            def evaluate(self, target):
                return EvaluationResult(score=FitnessScore(self._score))
        
        composite = CompositeEvaluator(
            [
                (FixedEvaluator(0.9, name="high"), 1.0),
                (FixedEvaluator(0.3, name="low"), 1.0),
            ],
            aggregation="max",
        )
        
        result = composite(sample_metaphor)
        assert result.score == 0.9
    
    def test_batch_evaluation(self, sample_metaphor, second_metaphor):
        """Test batch evaluation."""
        evaluator = RuleBasedEvaluator()
        
        results = evaluator.evaluate_batch([sample_metaphor, second_metaphor])
        
        assert len(results) == 2
        assert all(0 <= r.score <= 1 for r in results)
