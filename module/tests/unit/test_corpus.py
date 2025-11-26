"""Unit tests for corpus storage module."""

import pytest
import json
from datetime import datetime
from pathlib import Path

from metaphor_machine.core.metaphor import Metaphor, MetaphorSlot, SlotType
from metaphor_machine.corpus.storage import (
    CorpusEntry,
    QueryResult,
    SQLiteCorpus,
    JSONCorpus,
    open_corpus,
)


# =============================================================================
# Fixtures
# =============================================================================


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
def sample_entry(sample_metaphor):
    """Create a sample corpus entry."""
    return CorpusEntry.from_metaphor(
        sample_metaphor,
        fitness_score=0.85,
        raw_scores={"criterion1": 0.9, "criterion2": 0.8},
        tags={"darkwave", "experimental"},
        source="genetic",
        generator_seed=42,
        metadata={"generation": 10},
    )


@pytest.fixture
def sqlite_corpus(tmp_path):
    """Create a temporary SQLite corpus."""
    db_path = tmp_path / "test_corpus.db"
    corpus = SQLiteCorpus(db_path)
    yield corpus
    corpus.close()


@pytest.fixture
def json_corpus(tmp_path):
    """Create a temporary JSON corpus."""
    json_path = tmp_path / "test_corpus.json"
    corpus = JSONCorpus(json_path)
    yield corpus
    corpus.close()


# =============================================================================
# CorpusEntry Tests
# =============================================================================


class TestCorpusEntry:
    """Tests for CorpusEntry class."""
    
    def test_from_metaphor(self, sample_metaphor):
        """Test creating entry from metaphor."""
        entry = CorpusEntry.from_metaphor(
            sample_metaphor,
            fitness_score=0.8,
            tags={"test"},
        )
        
        assert entry.metaphor_text == str(sample_metaphor)
        assert entry.fitness_score == 0.8
        assert "test" in entry.tags
    
    def test_to_dict(self, sample_entry):
        """Test serialization to dictionary."""
        d = sample_entry.to_dict()
        
        assert d["fitness_score"] == 0.85
        assert "darkwave" in d["tags"]
        assert d["source"] == "genetic"
        assert d["generator_seed"] == 42
    
    def test_from_dict(self, sample_entry):
        """Test deserialization from dictionary."""
        d = sample_entry.to_dict()
        restored = CorpusEntry.from_dict(d)
        
        assert restored.fitness_score == sample_entry.fitness_score
        assert restored.tags == sample_entry.tags
        assert restored.source == sample_entry.source
        assert restored.generator_seed == sample_entry.generator_seed
    
    def test_round_trip(self, sample_entry):
        """Test serialization round-trip."""
        d = sample_entry.to_dict()
        restored = CorpusEntry.from_dict(d)
        d2 = restored.to_dict()
        
        # Compare key fields
        assert d["fitness_score"] == d2["fitness_score"]
        assert set(d["tags"]) == set(d2["tags"])
        assert d["source"] == d2["source"]


# =============================================================================
# SQLiteCorpus Tests
# =============================================================================


class TestSQLiteCorpus:
    """Tests for SQLiteCorpus class."""
    
    def test_add_entry(self, sqlite_corpus, sample_entry):
        """Test adding an entry."""
        entry_id = sqlite_corpus.add(sample_entry)
        
        assert entry_id is not None
        assert entry_id > 0
        assert sqlite_corpus.count() == 1
    
    def test_get_entry(self, sqlite_corpus, sample_entry):
        """Test retrieving an entry."""
        entry_id = sqlite_corpus.add(sample_entry)
        retrieved = sqlite_corpus.get(entry_id)
        
        assert retrieved is not None
        assert retrieved.fitness_score == sample_entry.fitness_score
        assert retrieved.tags == sample_entry.tags
    
    def test_get_nonexistent(self, sqlite_corpus):
        """Test retrieving nonexistent entry."""
        result = sqlite_corpus.get(9999)
        assert result is None
    
    def test_update_entry(self, sqlite_corpus, sample_entry):
        """Test updating an entry."""
        entry_id = sqlite_corpus.add(sample_entry)
        
        retrieved = sqlite_corpus.get(entry_id)
        retrieved.fitness_score = 0.95
        retrieved.tags.add("updated")
        
        success = sqlite_corpus.update(retrieved)
        assert success
        
        updated = sqlite_corpus.get(entry_id)
        assert updated.fitness_score == 0.95
        assert "updated" in updated.tags
    
    def test_update_nonexistent(self, sqlite_corpus, sample_entry):
        """Test updating nonexistent entry."""
        sample_entry.id = 9999
        success = sqlite_corpus.update(sample_entry)
        assert not success
    
    def test_delete_entry(self, sqlite_corpus, sample_entry):
        """Test deleting an entry."""
        entry_id = sqlite_corpus.add(sample_entry)
        assert sqlite_corpus.count() == 1
        
        success = sqlite_corpus.delete(entry_id)
        assert success
        assert sqlite_corpus.count() == 0
    
    def test_delete_nonexistent(self, sqlite_corpus):
        """Test deleting nonexistent entry."""
        success = sqlite_corpus.delete(9999)
        assert not success
    
    def test_query_by_score(self, sqlite_corpus, sample_metaphor, second_metaphor):
        """Test querying by score range."""
        entry1 = CorpusEntry.from_metaphor(sample_metaphor, fitness_score=0.3)
        entry2 = CorpusEntry.from_metaphor(second_metaphor, fitness_score=0.8)
        
        sqlite_corpus.add(entry1)
        sqlite_corpus.add(entry2)
        
        result = sqlite_corpus.query(min_score=0.5)
        assert len(result) == 1
        assert result.entries[0].fitness_score == 0.8
    
    def test_query_by_tags(self, sqlite_corpus, sample_metaphor, second_metaphor):
        """Test querying by tags."""
        entry1 = CorpusEntry.from_metaphor(sample_metaphor, tags={"dark", "experimental"})
        entry2 = CorpusEntry.from_metaphor(second_metaphor, tags={"bright", "pop"})
        
        sqlite_corpus.add(entry1)
        sqlite_corpus.add(entry2)
        
        result = sqlite_corpus.query(tags={"dark"})
        assert len(result) == 1
        assert "dark" in result.entries[0].tags
    
    def test_query_by_source(self, sqlite_corpus, sample_metaphor, second_metaphor):
        """Test querying by source."""
        entry1 = CorpusEntry.from_metaphor(sample_metaphor, source="genetic")
        entry2 = CorpusEntry.from_metaphor(second_metaphor, source="bayesian")
        
        sqlite_corpus.add(entry1)
        sqlite_corpus.add(entry2)
        
        result = sqlite_corpus.query(source="genetic")
        assert len(result) == 1
        assert result.entries[0].source == "genetic"
    
    def test_query_by_text(self, sqlite_corpus, sample_metaphor, second_metaphor):
        """Test querying by text content."""
        entry1 = CorpusEntry.from_metaphor(sample_metaphor)  # Contains "darkwave"
        entry2 = CorpusEntry.from_metaphor(second_metaphor)  # Contains "synthpop"
        
        sqlite_corpus.add(entry1)
        sqlite_corpus.add(entry2)
        
        result = sqlite_corpus.query(text_contains="darkwave")
        assert len(result) == 1
        assert "darkwave" in result.entries[0].metaphor_text
    
    def test_query_ordering(self, sqlite_corpus, sample_metaphor, second_metaphor):
        """Test query result ordering."""
        entry1 = CorpusEntry.from_metaphor(sample_metaphor, fitness_score=0.3)
        entry2 = CorpusEntry.from_metaphor(second_metaphor, fitness_score=0.9)
        
        sqlite_corpus.add(entry1)
        sqlite_corpus.add(entry2)
        
        # Descending order (default)
        result_desc = sqlite_corpus.query(order_by="fitness_score", descending=True)
        assert result_desc.entries[0].fitness_score == 0.9
        
        # Ascending order
        result_asc = sqlite_corpus.query(order_by="fitness_score", descending=False)
        assert result_asc.entries[0].fitness_score == 0.3
    
    def test_query_pagination(self, sqlite_corpus, sample_metaphor):
        """Test query pagination."""
        # Add multiple entries
        for i in range(10):
            entry = CorpusEntry.from_metaphor(sample_metaphor, fitness_score=i / 10)
            sqlite_corpus.add(entry)
        
        # Get first page
        result1 = sqlite_corpus.query(limit=3, offset=0)
        assert len(result1) == 3
        assert result1.total_count == 10
        
        # Get second page
        result2 = sqlite_corpus.query(limit=3, offset=3)
        assert len(result2) == 3
    
    def test_get_top_n(self, sqlite_corpus, sample_metaphor, second_metaphor):
        """Test getting top N entries."""
        entry1 = CorpusEntry.from_metaphor(sample_metaphor, fitness_score=0.3)
        entry2 = CorpusEntry.from_metaphor(second_metaphor, fitness_score=0.9)
        
        sqlite_corpus.add(entry1)
        sqlite_corpus.add(entry2)
        
        top = sqlite_corpus.get_top_n(1)
        assert len(top) == 1
        assert top[0].fitness_score == 0.9
    
    def test_get_all_tags(self, sqlite_corpus, sample_metaphor, second_metaphor):
        """Test getting all unique tags."""
        entry1 = CorpusEntry.from_metaphor(sample_metaphor, tags={"tag1", "tag2"})
        entry2 = CorpusEntry.from_metaphor(second_metaphor, tags={"tag2", "tag3"})
        
        sqlite_corpus.add(entry1)
        sqlite_corpus.add(entry2)
        
        tags = sqlite_corpus.get_all_tags()
        assert tags == {"tag1", "tag2", "tag3"}
    
    def test_export_texts(self, sqlite_corpus, sample_metaphor, second_metaphor):
        """Test exporting metaphor texts."""
        entry1 = CorpusEntry.from_metaphor(sample_metaphor, fitness_score=0.3)
        entry2 = CorpusEntry.from_metaphor(second_metaphor, fitness_score=0.9)
        
        sqlite_corpus.add(entry1)
        sqlite_corpus.add(entry2)
        
        texts = sqlite_corpus.export_texts()
        assert len(texts) == 2
        assert texts[0] == str(second_metaphor)  # Higher score first
        
        texts_filtered = sqlite_corpus.export_texts(min_score=0.5)
        assert len(texts_filtered) == 1
    
    def test_add_batch(self, sqlite_corpus, sample_metaphor, second_metaphor):
        """Test adding multiple entries at once."""
        entry1 = CorpusEntry.from_metaphor(sample_metaphor)
        entry2 = CorpusEntry.from_metaphor(second_metaphor)
        
        ids = sqlite_corpus.add_batch([entry1, entry2])
        
        assert len(ids) == 2
        assert sqlite_corpus.count() == 2
    
    def test_context_manager(self, tmp_path):
        """Test using corpus as context manager."""
        db_path = tmp_path / "context_test.db"
        
        with SQLiteCorpus(db_path) as corpus:
            corpus.add(CorpusEntry(
                metaphor_text="test",
                metaphor_data={},
                fitness_score=0.5,
            ))
        
        # Should be able to reopen
        with SQLiteCorpus(db_path) as corpus:
            assert corpus.count() == 1


# =============================================================================
# JSONCorpus Tests
# =============================================================================


class TestJSONCorpus:
    """Tests for JSONCorpus class."""
    
    def test_add_entry(self, json_corpus, sample_entry):
        """Test adding an entry."""
        entry_id = json_corpus.add(sample_entry)
        
        assert entry_id is not None
        assert json_corpus.count() == 1
    
    def test_get_entry(self, json_corpus, sample_entry):
        """Test retrieving an entry."""
        entry_id = json_corpus.add(sample_entry)
        retrieved = json_corpus.get(entry_id)
        
        assert retrieved is not None
        assert retrieved.fitness_score == sample_entry.fitness_score
    
    def test_persistence(self, tmp_path, sample_entry):
        """Test that data persists after closing."""
        json_path = tmp_path / "persist_test.json"
        
        # Add entry and close
        corpus1 = JSONCorpus(json_path)
        entry_id = corpus1.add(sample_entry)
        corpus1.close()
        
        # Reopen and verify
        corpus2 = JSONCorpus(json_path)
        retrieved = corpus2.get(entry_id)
        corpus2.close()
        
        assert retrieved is not None
        assert retrieved.fitness_score == sample_entry.fitness_score
    
    def test_query(self, json_corpus, sample_metaphor, second_metaphor):
        """Test querying entries."""
        entry1 = CorpusEntry.from_metaphor(sample_metaphor, fitness_score=0.3)
        entry2 = CorpusEntry.from_metaphor(second_metaphor, fitness_score=0.9)
        
        json_corpus.add(entry1)
        json_corpus.add(entry2)
        
        result = json_corpus.query(min_score=0.5)
        assert len(result) == 1
        assert result.entries[0].fitness_score == 0.9
    
    def test_update(self, json_corpus, sample_entry):
        """Test updating an entry."""
        entry_id = json_corpus.add(sample_entry)
        
        retrieved = json_corpus.get(entry_id)
        retrieved.fitness_score = 0.99
        
        success = json_corpus.update(retrieved)
        assert success
        
        updated = json_corpus.get(entry_id)
        assert updated.fitness_score == 0.99
    
    def test_delete(self, json_corpus, sample_entry):
        """Test deleting an entry."""
        entry_id = json_corpus.add(sample_entry)
        
        success = json_corpus.delete(entry_id)
        assert success
        assert json_corpus.count() == 0


# =============================================================================
# open_corpus Tests
# =============================================================================


class TestOpenCorpus:
    """Tests for the open_corpus convenience function."""
    
    def test_open_sqlite(self, tmp_path):
        """Test opening SQLite corpus by extension."""
        for ext in [".db", ".sqlite", ".sqlite3"]:
            path = tmp_path / f"test{ext}"
            corpus = open_corpus(path)
            assert isinstance(corpus, SQLiteCorpus)
            corpus.close()
    
    def test_open_json(self, tmp_path):
        """Test opening JSON corpus by extension."""
        path = tmp_path / "test.json"
        corpus = open_corpus(path)
        assert isinstance(corpus, JSONCorpus)
        corpus.close()
    
    def test_open_default(self, tmp_path):
        """Test opening with unknown extension defaults to SQLite."""
        path = tmp_path / "test.unknown"
        corpus = open_corpus(path)
        assert isinstance(corpus, SQLiteCorpus)
        corpus.close()


# =============================================================================
# Integration Tests
# =============================================================================


class TestCorpusIntegration:
    """Integration tests for corpus functionality."""
    
    def test_full_workflow(self, sqlite_corpus, sample_metaphor, second_metaphor):
        """Test complete workflow: add, query, update, export."""
        # Add entries with different scores and tags
        entries = [
            CorpusEntry.from_metaphor(
                sample_metaphor,
                fitness_score=0.85,
                tags={"darkwave", "high-quality"},
                source="genetic",
            ),
            CorpusEntry.from_metaphor(
                second_metaphor,
                fitness_score=0.45,
                tags={"pop", "low-quality"},
                source="bayesian",
            ),
        ]
        
        ids = sqlite_corpus.add_batch(entries)
        assert len(ids) == 2
        
        # Query high-quality entries
        high_quality = sqlite_corpus.query(min_score=0.7)
        assert len(high_quality) == 1
        assert "darkwave" in high_quality.entries[0].tags
        
        # Update an entry
        entry = sqlite_corpus.get(ids[1])
        entry.fitness_score = 0.75
        entry.tags.add("improved")
        sqlite_corpus.update(entry)
        
        # Verify update
        updated = sqlite_corpus.get(ids[1])
        assert updated.fitness_score == 0.75
        assert "improved" in updated.tags
        
        # Export texts for training
        texts = sqlite_corpus.export_texts(min_score=0.7)
        assert len(texts) == 2  # Both entries now >= 0.7
    
    def test_corpus_stats(self, sqlite_corpus, sample_metaphor):
        """Test gathering corpus statistics."""
        # Add multiple entries
        for i in range(20):
            entry = CorpusEntry.from_metaphor(
                sample_metaphor,
                fitness_score=i / 20,
                source="genetic" if i % 2 == 0 else "bayesian",
            )
            sqlite_corpus.add(entry)
        
        # Count
        assert sqlite_corpus.count() == 20
        
        # Query by source
        genetic_results = sqlite_corpus.query(source="genetic")
        bayesian_results = sqlite_corpus.query(source="bayesian")
        
        assert genetic_results.total_count == 10
        assert bayesian_results.total_count == 10
        
        # Get top entries
        top_5 = sqlite_corpus.get_top_n(5)
        assert len(top_5) == 5
        assert top_5[0].fitness_score > top_5[4].fitness_score
