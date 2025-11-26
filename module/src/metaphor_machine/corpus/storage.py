"""
Corpus storage module for tracking metaphor prompt-outcome pairs.

This module provides persistent storage for:
- Generated metaphors and their fitness scores
- Successful "exploits" (high-performing prompt patterns)
- Tagging and categorization for analysis
- Export formats for training Markov models

Storage backends:
- SQLite (default): Single-file database, good for local use
- JSON: Human-readable, portable, good for small corpora
"""

from __future__ import annotations

import json
import sqlite3
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Sequence

from metaphor_machine.core.metaphor import Metaphor, MetaphorChain, SlotType


# =============================================================================
# Data Types
# =============================================================================


@dataclass
class CorpusEntry:
    """
    A single entry in the corpus.
    
    Stores a metaphor/chain with its evaluation results, metadata,
    and tags for categorization.
    
    Attributes:
        id: Unique identifier (auto-generated if None)
        metaphor_text: String representation of the metaphor
        metaphor_data: Serialized metaphor structure
        fitness_score: Evaluation score [0, 1]
        raw_scores: Component scores from evaluation
        tags: User-defined tags for categorization
        source: How this entry was generated (e.g., "genetic", "bayesian", "manual")
        generator_seed: Seed used for generation (for reproducibility)
        created_at: Timestamp of creation
        metadata: Additional arbitrary metadata
    """
    
    metaphor_text: str
    metaphor_data: dict[str, Any]
    fitness_score: float | None = None
    raw_scores: dict[str, float] = field(default_factory=dict)
    tags: set[str] = field(default_factory=set)
    source: str = "unknown"
    generator_seed: int | None = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    id: int | None = None
    
    @classmethod
    def from_metaphor(
        cls,
        metaphor: Metaphor | MetaphorChain,
        fitness_score: float | None = None,
        raw_scores: dict[str, float] | None = None,
        tags: set[str] | None = None,
        source: str = "unknown",
        generator_seed: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CorpusEntry:
        """Create a corpus entry from a metaphor or chain."""
        return cls(
            metaphor_text=str(metaphor),
            metaphor_data=metaphor.to_dict(),
            fitness_score=fitness_score,
            raw_scores=raw_scores or {},
            tags=tags or set(),
            source=source,
            generator_seed=generator_seed,
            metadata=metadata or {},
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "metaphor_text": self.metaphor_text,
            "metaphor_data": self.metaphor_data,
            "fitness_score": self.fitness_score,
            "raw_scores": self.raw_scores,
            "tags": list(self.tags),
            "source": self.source,
            "generator_seed": self.generator_seed,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CorpusEntry:
        """Deserialize from dictionary."""
        return cls(
            id=data.get("id"),
            metaphor_text=data["metaphor_text"],
            metaphor_data=data["metaphor_data"],
            fitness_score=data.get("fitness_score"),
            raw_scores=data.get("raw_scores", {}),
            tags=set(data.get("tags", [])),
            source=data.get("source", "unknown"),
            generator_seed=data.get("generator_seed"),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            metadata=data.get("metadata", {}),
        )


@dataclass
class QueryResult:
    """Result of a corpus query."""
    
    entries: list[CorpusEntry]
    total_count: int
    query_time: float
    
    def __len__(self) -> int:
        return len(self.entries)
    
    def __iter__(self) -> Iterator[CorpusEntry]:
        return iter(self.entries)


# =============================================================================
# Abstract Storage Backend
# =============================================================================


class CorpusStorage(ABC):
    """
    Abstract base class for corpus storage backends.
    
    Defines the interface for storing and retrieving corpus entries.
    """
    
    @abstractmethod
    def add(self, entry: CorpusEntry) -> int:
        """Add an entry and return its ID."""
        pass
    
    @abstractmethod
    def add_batch(self, entries: Sequence[CorpusEntry]) -> list[int]:
        """Add multiple entries and return their IDs."""
        pass
    
    @abstractmethod
    def get(self, entry_id: int) -> CorpusEntry | None:
        """Get an entry by ID."""
        pass
    
    @abstractmethod
    def update(self, entry: CorpusEntry) -> bool:
        """Update an existing entry. Returns True if found and updated."""
        pass
    
    @abstractmethod
    def delete(self, entry_id: int) -> bool:
        """Delete an entry by ID. Returns True if found and deleted."""
        pass
    
    @abstractmethod
    def query(
        self,
        min_score: float | None = None,
        max_score: float | None = None,
        tags: set[str] | None = None,
        source: str | None = None,
        text_contains: str | None = None,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "fitness_score",
        descending: bool = True,
    ) -> QueryResult:
        """Query entries with filters."""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Return total number of entries."""
        pass
    
    @abstractmethod
    def get_all_tags(self) -> set[str]:
        """Return all unique tags in the corpus."""
        pass
    
    @abstractmethod
    def get_top_n(self, n: int = 10) -> list[CorpusEntry]:
        """Return top N entries by fitness score."""
        pass
    
    @abstractmethod
    def export_texts(self, min_score: float | None = None) -> list[str]:
        """Export metaphor texts for training."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close any open resources."""
        pass


# =============================================================================
# SQLite Backend
# =============================================================================


class SQLiteCorpus(CorpusStorage):
    """
    SQLite-based corpus storage.
    
    Provides efficient querying and persistent storage in a single file.
    
    Example:
        >>> corpus = SQLiteCorpus("my_corpus.db")
        >>> entry = CorpusEntry.from_metaphor(metaphor, fitness_score=0.85)
        >>> entry_id = corpus.add(entry)
        >>> top_entries = corpus.get_top_n(10)
    """
    
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()
    
    def _init_schema(self) -> None:
        """Initialize database schema."""
        cursor = self.conn.cursor()
        
        # Main entries table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metaphor_text TEXT NOT NULL,
                metaphor_data TEXT NOT NULL,
                fitness_score REAL,
                raw_scores TEXT,
                source TEXT,
                generator_seed INTEGER,
                created_at TEXT NOT NULL,
                metadata TEXT
            )
        """)
        
        # Tags table (many-to-many)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entry_tags (
                entry_id INTEGER,
                tag_id INTEGER,
                PRIMARY KEY (entry_id, tag_id),
                FOREIGN KEY (entry_id) REFERENCES entries(id) ON DELETE CASCADE,
                FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
            )
        """)
        
        # Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_fitness ON entries(fitness_score)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_source ON entries(source)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_created ON entries(created_at)")
        
        self.conn.commit()
    
    @contextmanager
    def _transaction(self):
        """Context manager for transactions."""
        try:
            yield
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
    
    def _get_or_create_tag(self, tag_name: str, cursor: sqlite3.Cursor) -> int:
        """Get tag ID, creating if necessary."""
        cursor.execute("SELECT id FROM tags WHERE name = ?", (tag_name,))
        row = cursor.fetchone()
        if row:
            return row["id"]
        
        cursor.execute("INSERT INTO tags (name) VALUES (?)", (tag_name,))
        return cursor.lastrowid
    
    def add(self, entry: CorpusEntry) -> int:
        cursor = self.conn.cursor()
        
        with self._transaction():
            cursor.execute("""
                INSERT INTO entries 
                (metaphor_text, metaphor_data, fitness_score, raw_scores, 
                 source, generator_seed, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.metaphor_text,
                json.dumps(entry.metaphor_data),
                entry.fitness_score,
                json.dumps(entry.raw_scores),
                entry.source,
                entry.generator_seed,
                entry.created_at.isoformat(),
                json.dumps(entry.metadata),
            ))
            
            entry_id = cursor.lastrowid
            
            # Add tags
            for tag in entry.tags:
                tag_id = self._get_or_create_tag(tag, cursor)
                cursor.execute(
                    "INSERT OR IGNORE INTO entry_tags (entry_id, tag_id) VALUES (?, ?)",
                    (entry_id, tag_id)
                )
        
        return entry_id
    
    def add_batch(self, entries: Sequence[CorpusEntry]) -> list[int]:
        ids = []
        for entry in entries:
            ids.append(self.add(entry))
        return ids
    
    def _row_to_entry(self, row: sqlite3.Row, cursor: sqlite3.Cursor) -> CorpusEntry:
        """Convert database row to CorpusEntry."""
        entry_id = row["id"]
        
        # Get tags
        cursor.execute("""
            SELECT t.name FROM tags t
            JOIN entry_tags et ON t.id = et.tag_id
            WHERE et.entry_id = ?
        """, (entry_id,))
        tags = {r["name"] for r in cursor.fetchall()}
        
        return CorpusEntry(
            id=entry_id,
            metaphor_text=row["metaphor_text"],
            metaphor_data=json.loads(row["metaphor_data"]),
            fitness_score=row["fitness_score"],
            raw_scores=json.loads(row["raw_scores"]) if row["raw_scores"] else {},
            tags=tags,
            source=row["source"] or "unknown",
            generator_seed=row["generator_seed"],
            created_at=datetime.fromisoformat(row["created_at"]),
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )
    
    def get(self, entry_id: int) -> CorpusEntry | None:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM entries WHERE id = ?", (entry_id,))
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        return self._row_to_entry(row, cursor)
    
    def update(self, entry: CorpusEntry) -> bool:
        if entry.id is None:
            return False
        
        cursor = self.conn.cursor()
        
        with self._transaction():
            cursor.execute("""
                UPDATE entries SET
                    metaphor_text = ?,
                    metaphor_data = ?,
                    fitness_score = ?,
                    raw_scores = ?,
                    source = ?,
                    generator_seed = ?,
                    metadata = ?
                WHERE id = ?
            """, (
                entry.metaphor_text,
                json.dumps(entry.metaphor_data),
                entry.fitness_score,
                json.dumps(entry.raw_scores),
                entry.source,
                entry.generator_seed,
                json.dumps(entry.metadata),
                entry.id,
            ))
            
            if cursor.rowcount == 0:
                return False
            
            # Update tags
            cursor.execute("DELETE FROM entry_tags WHERE entry_id = ?", (entry.id,))
            for tag in entry.tags:
                tag_id = self._get_or_create_tag(tag, cursor)
                cursor.execute(
                    "INSERT INTO entry_tags (entry_id, tag_id) VALUES (?, ?)",
                    (entry.id, tag_id)
                )
        
        return True
    
    def delete(self, entry_id: int) -> bool:
        cursor = self.conn.cursor()
        
        with self._transaction():
            cursor.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
            return cursor.rowcount > 0
    
    def query(
        self,
        min_score: float | None = None,
        max_score: float | None = None,
        tags: set[str] | None = None,
        source: str | None = None,
        text_contains: str | None = None,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "fitness_score",
        descending: bool = True,
    ) -> QueryResult:
        start_time = time.perf_counter()
        cursor = self.conn.cursor()
        
        # Build query
        conditions = []
        params: list[Any] = []
        
        if min_score is not None:
            conditions.append("fitness_score >= ?")
            params.append(min_score)
        
        if max_score is not None:
            conditions.append("fitness_score <= ?")
            params.append(max_score)
        
        if source is not None:
            conditions.append("source = ?")
            params.append(source)
        
        if text_contains is not None:
            conditions.append("metaphor_text LIKE ?")
            params.append(f"%{text_contains}%")
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        # Handle tag filtering with subquery
        if tags:
            tag_placeholders = ",".join("?" * len(tags))
            where_clause += f"""
                AND id IN (
                    SELECT et.entry_id FROM entry_tags et
                    JOIN tags t ON et.tag_id = t.id
                    WHERE t.name IN ({tag_placeholders})
                    GROUP BY et.entry_id
                    HAVING COUNT(DISTINCT t.id) = ?
                )
            """
            params.extend(tags)
            params.append(len(tags))
        
        # Validate order_by to prevent SQL injection
        valid_columns = {"fitness_score", "created_at", "id", "source"}
        if order_by not in valid_columns:
            order_by = "fitness_score"
        
        direction = "DESC" if descending else "ASC"
        
        # Get total count
        cursor.execute(f"SELECT COUNT(*) FROM entries WHERE {where_clause}", params)
        total_count = cursor.fetchone()[0]
        
        # Get entries
        query = f"""
            SELECT * FROM entries 
            WHERE {where_clause}
            ORDER BY {order_by} {direction}
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        entries = [self._row_to_entry(row, cursor) for row in rows]
        query_time = time.perf_counter() - start_time
        
        return QueryResult(entries=entries, total_count=total_count, query_time=query_time)
    
    def count(self) -> int:
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM entries")
        return cursor.fetchone()[0]
    
    def get_all_tags(self) -> set[str]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM tags")
        return {row["name"] for row in cursor.fetchall()}
    
    def get_top_n(self, n: int = 10) -> list[CorpusEntry]:
        result = self.query(limit=n, order_by="fitness_score", descending=True)
        return result.entries
    
    def export_texts(self, min_score: float | None = None) -> list[str]:
        cursor = self.conn.cursor()
        
        if min_score is not None:
            cursor.execute(
                "SELECT metaphor_text FROM entries WHERE fitness_score >= ? ORDER BY fitness_score DESC",
                (min_score,)
            )
        else:
            cursor.execute("SELECT metaphor_text FROM entries ORDER BY fitness_score DESC")
        
        return [row["metaphor_text"] for row in cursor.fetchall()]
    
    def get_slot_frequency(self, slot_type: str) -> dict[str, int]:
        """
        Get frequency distribution of values for a specific slot type.
        
        Useful for analyzing which slot values correlate with high fitness.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT metaphor_data, fitness_score FROM entries")
        
        frequencies: dict[str, int] = {}
        
        for row in cursor.fetchall():
            data = json.loads(row["metaphor_data"])
            
            # Handle both Metaphor and MetaphorChain structures
            if "intro" in data:  # Chain
                for part in ["intro", "mid", "outro"]:
                    if part in data and slot_type in data[part]:
                        value = data[part][slot_type]
                        if isinstance(value, dict):
                            value = value.get("value", str(value))
                        frequencies[value] = frequencies.get(value, 0) + 1
            else:  # Single metaphor
                if slot_type in data:
                    value = data[slot_type]
                    if isinstance(value, dict):
                        value = value.get("value", str(value))
                    frequencies[value] = frequencies.get(value, 0) + 1
        
        return frequencies
    
    def close(self) -> None:
        self.conn.close()
    
    def __enter__(self) -> SQLiteCorpus:
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# =============================================================================
# JSON Backend (for portability)
# =============================================================================


class JSONCorpus(CorpusStorage):
    """
    JSON file-based corpus storage.
    
    Stores all entries in a single JSON file. Good for small corpora
    and when human-readable format is needed.
    
    Note: Loads entire corpus into memory. For large corpora, use SQLiteCorpus.
    """
    
    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)
        self._entries: dict[int, CorpusEntry] = {}
        self._next_id = 1
        self._load()
    
    def _load(self) -> None:
        """Load entries from file."""
        if self.file_path.exists():
            with open(self.file_path, "r") as f:
                data = json.load(f)
            
            self._entries = {
                int(k): CorpusEntry.from_dict(v) 
                for k, v in data.get("entries", {}).items()
            }
            self._next_id = data.get("next_id", 1)
    
    def _save(self) -> None:
        """Save entries to file."""
        data = {
            "entries": {str(k): v.to_dict() for k, v in self._entries.items()},
            "next_id": self._next_id,
        }
        
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.file_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def add(self, entry: CorpusEntry) -> int:
        entry_id = self._next_id
        self._next_id += 1
        
        entry.id = entry_id
        self._entries[entry_id] = entry
        self._save()
        
        return entry_id
    
    def add_batch(self, entries: Sequence[CorpusEntry]) -> list[int]:
        ids = []
        for entry in entries:
            ids.append(self.add(entry))
        return ids
    
    def get(self, entry_id: int) -> CorpusEntry | None:
        return self._entries.get(entry_id)
    
    def update(self, entry: CorpusEntry) -> bool:
        if entry.id is None or entry.id not in self._entries:
            return False
        
        self._entries[entry.id] = entry
        self._save()
        return True
    
    def delete(self, entry_id: int) -> bool:
        if entry_id not in self._entries:
            return False
        
        del self._entries[entry_id]
        self._save()
        return True
    
    def query(
        self,
        min_score: float | None = None,
        max_score: float | None = None,
        tags: set[str] | None = None,
        source: str | None = None,
        text_contains: str | None = None,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "fitness_score",
        descending: bool = True,
    ) -> QueryResult:
        start_time = time.perf_counter()
        
        # Filter entries
        filtered = list(self._entries.values())
        
        if min_score is not None:
            filtered = [e for e in filtered if e.fitness_score is not None and e.fitness_score >= min_score]
        
        if max_score is not None:
            filtered = [e for e in filtered if e.fitness_score is not None and e.fitness_score <= max_score]
        
        if tags is not None:
            filtered = [e for e in filtered if tags.issubset(e.tags)]
        
        if source is not None:
            filtered = [e for e in filtered if e.source == source]
        
        if text_contains is not None:
            text_lower = text_contains.lower()
            filtered = [e for e in filtered if text_lower in e.metaphor_text.lower()]
        
        # Sort
        def sort_key(e: CorpusEntry) -> Any:
            if order_by == "fitness_score":
                return e.fitness_score or 0
            elif order_by == "created_at":
                return e.created_at
            elif order_by == "id":
                return e.id or 0
            else:
                return e.fitness_score or 0
        
        filtered.sort(key=sort_key, reverse=descending)
        
        total_count = len(filtered)
        entries = filtered[offset:offset + limit]
        query_time = time.perf_counter() - start_time
        
        return QueryResult(entries=entries, total_count=total_count, query_time=query_time)
    
    def count(self) -> int:
        return len(self._entries)
    
    def get_all_tags(self) -> set[str]:
        tags: set[str] = set()
        for entry in self._entries.values():
            tags.update(entry.tags)
        return tags
    
    def get_top_n(self, n: int = 10) -> list[CorpusEntry]:
        result = self.query(limit=n, order_by="fitness_score", descending=True)
        return result.entries
    
    def export_texts(self, min_score: float | None = None) -> list[str]:
        entries = self._entries.values()
        
        if min_score is not None:
            entries = [e for e in entries if e.fitness_score is not None and e.fitness_score >= min_score]
        
        sorted_entries = sorted(entries, key=lambda e: e.fitness_score or 0, reverse=True)
        return [e.metaphor_text for e in sorted_entries]
    
    def close(self) -> None:
        self._save()
    
    def __enter__(self) -> JSONCorpus:
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# =============================================================================
# Convenience Functions
# =============================================================================


def open_corpus(path: str | Path) -> CorpusStorage:
    """
    Open a corpus file, automatically selecting the appropriate backend.
    
    Uses file extension to determine format:
    - .db, .sqlite, .sqlite3 -> SQLiteCorpus
    - .json -> JSONCorpus
    
    Example:
        >>> with open_corpus("my_corpus.db") as corpus:
        ...     corpus.add(entry)
    """
    path = Path(path)
    
    if path.suffix in {".db", ".sqlite", ".sqlite3"}:
        return SQLiteCorpus(path)
    elif path.suffix == ".json":
        return JSONCorpus(path)
    else:
        # Default to SQLite
        return SQLiteCorpus(path)
