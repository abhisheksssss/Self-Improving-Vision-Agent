"""Lesson Extractor for SIVAC Phase 7.

Validates, deduplicates, and persists synthesized operational heuristic rules
into the Phase 5 Dual-Layer Memory (SQLite + ChromaDB).
"""

import logging
from typing import List, Optional

from ..memory.semantic import SemanticMemoryManager
from ..memory.schema import LearnedSkill
from .schema import ReflectionOutput, ExtractedLesson

logger = logging.getLogger("sivac.learning.lesson_extractor")


class LessonExtractor:
    """Extracts, deduplicates, and commits heuristic rules to memory."""

    def __init__(self, semantic_memory: Optional[SemanticMemoryManager] = None) -> None:
        """Initialise LessonExtractor with semantic memory manager."""
        self.semantic = semantic_memory or SemanticMemoryManager()

    def extract_and_persist(
        self,
        reflection: ReflectionOutput,
        dedup_similarity_threshold: float = 0.85,
    ) -> List[LearnedSkill]:
        """Process extracted lessons from reflection and persist unique rules.

        Args:
            reflection: ReflectionOutput from ReflectionEngine.
            dedup_similarity_threshold: Similarity score above which a rule is
                                       treated as an existing duplicate.

        Returns:
            List of created or updated LearnedSkill instances.
        """
        committed_skills: List[LearnedSkill] = []

        for lesson in reflection.lessons:
            # Basic validation
            if not lesson.heuristic_rule or not lesson.trigger_condition:
                continue

            app = lesson.application_name or "Desktop"
            trigger = lesson.trigger_condition.strip()
            rule = lesson.heuristic_rule.strip()
            confidence = max(0.5, min(1.0, lesson.confidence_score))

            # Deduplication check against existing ChromaDB heuristics
            query_text = f"Trigger: {trigger} | Rule: {rule}"
            existing_matches = self.semantic.query_heuristics(
                query_context=query_text,
                application_name=app,
                top_k=1,
            )

            # If an existing rule matches closely, reinforce it rather than duplicate
            if existing_matches and existing_matches[0].similarity_score >= dedup_similarity_threshold:
                existing = existing_matches[0]
                updated = self.semantic.record_feedback(
                    skill_id=existing.skill_id,
                    success=reflection.overall_success,
                )
                if updated:
                    logger.info(
                        f"Deduplicated heuristic: Reinforced existing [{existing.skill_id}] "
                        f"(similarity={existing.similarity_score:.2f}) -> confidence={updated.confidence_score:.2f}"
                    )
                    committed_skills.append(updated)
                continue

            # Persist new heuristic in SQLite and ChromaDB
            skill = self.semantic.store_heuristic(
                application_name=app,
                trigger_condition=trigger,
                heuristic_rule=rule,
                confidence_score=confidence,
            )
            committed_skills.append(skill)
            logger.info(
                f"Persisted new operational heuristic [{skill.id}] for '{app}': "
                f"'{rule[:50]}...'"
            )

        return committed_skills
