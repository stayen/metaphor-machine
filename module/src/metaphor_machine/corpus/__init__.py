"""
Corpus module for storing and querying metaphor prompt-outcome pairs.

Provides persistent storage for tracking:
- Generated metaphors and their fitness scores
- Successful prompt patterns ("exploits")
- Tags and metadata for analysis
"""

from metaphor_machine.corpus.storage import (
    CorpusEntry,
    QueryResult,
    CorpusStorage,
    SQLiteCorpus,
    JSONCorpus,
    open_corpus,
)


__all__ = [
    "CorpusEntry",
    "QueryResult",
    "CorpusStorage",
    "SQLiteCorpus",
    "JSONCorpus",
    "open_corpus",
]
