"""Phase 5: Multi-Level Memory System for SIVAC.

Provides dual-layer episodic trajectory storage and semantic vector retrieval:
    - storage       : Relational audit store (SQLite + SQLAlchemy 2.0)
    - vector_store  : Local persistent vector collections (ChromaDB)
    - episodic      : High-level episodic task lifecycle manager
    - semantic      : Learned heuristic knowledge & lesson store
    - retrieval     : Context-aware experience retriever for planner prompting
"""

from .schema import (
    TaskStatus,
    TaskRecord,
    TrajectoryRecord,
    StepRecord,
    VerificationRecord,
    ReflectionRecord,
    LearnedSkill,
    SimilarTaskResult,
    HeuristicQueryResult,
    RetrievedExperience,
)
from .storage import DatabaseManager
from .vector_store import (
    ChromaVectorStore,
    FastDeterministicEmbeddingFunction,
)
from .episodic import EpisodicMemoryManager
from .semantic import SemanticMemoryManager
from .retrieval import ExperienceRetriever

__all__ = [
    "TaskStatus",
    "TaskRecord",
    "TrajectoryRecord",
    "StepRecord",
    "VerificationRecord",
    "ReflectionRecord",
    "LearnedSkill",
    "SimilarTaskResult",
    "HeuristicQueryResult",
    "RetrievedExperience",
    "DatabaseManager",
    "ChromaVectorStore",
    "FastDeterministicEmbeddingFunction",
    "EpisodicMemoryManager",
    "SemanticMemoryManager",
    "ExperienceRetriever",
]
