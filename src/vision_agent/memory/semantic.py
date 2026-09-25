"""Semantic Memory Subsystem for SIVAC Phase 5.

Manages application-specific heuristic rules, behavioral strategies, and
cross-task knowledge extraction, storing structured records in SQLite and
semantic embeddings in ChromaDB.
"""

import logging
from typing import Optional, List

from .schema import LearnedSkill, HeuristicQueryResult
from .storage import DatabaseManager
from .vector_store import ChromaVectorStore

logger = logging.getLogger("sivac.memory.semantic")


class SemanticMemoryManager:
    """Manages persistent operational heuristics and learned interaction rules."""

    def __init__(
        self,
        db: Optional[DatabaseManager] = None,
        vector_store: Optional[ChromaVectorStore] = None,
    ) -> None:
        """Initialise SemanticMemoryManager with relational and vector stores.

        Args:
            db: Relational database manager instance (creates default if None).
            vector_store: Chroma vector store instance (creates default if None).
        """
        self.db = db or DatabaseManager()
        self.vector_store = vector_store or ChromaVectorStore()

    def store_heuristic(
        self,
        application_name: str,
        trigger_condition: str,
        heuristic_rule: str,
        reflection_id: Optional[str] = None,
        confidence_score: float = 0.8,
    ) -> LearnedSkill:
        """Store a new learned heuristic rule in SQLite and index in ChromaDB.

        Args:
            application_name: Target application (e.g. 'Chrome', 'Notepad', 'Excel').
            trigger_condition: Context or error state triggering this rule.
            heuristic_rule: Concrete actionable advice (e.g. 'Wait 2s before clicking submit').
            reflection_id: Optional ID of the reflection that derived this rule.
            confidence_score: Initial confidence rating (0.0 to 1.0).

        Returns:
            The created LearnedSkill model.
        """
        # 1. Persist in SQLite
        skill = self.db.save_learned_skill(
            application_name=application_name,
            trigger_condition=trigger_condition,
            heuristic_rule=heuristic_rule,
            reflection_id=reflection_id,
            confidence_score=confidence_score,
        )

        # 2. Embed and index in ChromaDB
        self.vector_store.index_heuristic(
            skill_id=skill.id,
            application_name=skill.application_name,
            trigger_condition=skill.trigger_condition,
            heuristic_rule=skill.heuristic_rule,
            confidence_score=skill.confidence_score,
            success_count=skill.success_count,
            failure_count=skill.failure_count,
        )

        logger.info(
            f"Stored heuristic [{skill.id}] for '{application_name}': "
            f"'{heuristic_rule[:60]}...'"
        )
        return skill

    def query_heuristics(
        self,
        query_context: str,
        application_name: Optional[str] = None,
        top_k: int = 5,
        min_confidence: float = 0.5,
    ) -> List[HeuristicQueryResult]:
        """Search for relevant heuristic rules given current screen/task context.

        Args:
            query_context: Natural language description of current screen state or error.
            application_name: Optional application name filter.
            top_k: Maximum number of heuristic rules to retrieve.
            min_confidence: Threshold below which heuristics are filtered out.

        Returns:
            List of matching HeuristicQueryResult objects.
        """
        return self.vector_store.query_heuristics(
            query_text=query_context,
            application_name=application_name,
            top_k=top_k,
            min_confidence=min_confidence,
        )

    def record_feedback(self, skill_id: str, success: bool) -> Optional[LearnedSkill]:
        """Update confidence and success/failure counters for a heuristic.

        Args:
            skill_id: Unique heuristic ID.
            success: True if following this rule led to success, False if it failed.

        Returns:
            Updated LearnedSkill or None if not found.
        """
        updated_skill = self.db.update_skill_stats(skill_id=skill_id, success=success)
        if updated_skill:
            # Sync metadata in ChromaDB
            self.vector_store.update_heuristic_stats(
                skill_id=skill_id,
                confidence_score=updated_skill.confidence_score,
                success_count=updated_skill.success_count,
                failure_count=updated_skill.failure_count,
            )
            logger.info(
                f"Updated heuristic [{skill_id}] feedback (success={success}) -> "
                f"confidence={updated_skill.confidence_score:.2f}"
            )
        return updated_skill

    def get_heuristics_for_app(
        self, application_name: str, limit: int = 50
    ) -> List[LearnedSkill]:
        """Fetch all stored heuristics for a given application from relational store."""
        return self.db.get_learned_skills(application_name=application_name, limit=limit)

    def list_top_heuristics(self, limit: int = 50) -> List[LearnedSkill]:
        """Fetch highest-confidence stored heuristics across all applications."""
        return self.db.get_learned_skills(application_name=None, limit=limit)
