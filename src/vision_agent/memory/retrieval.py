"""Context-Aware Experience Retriever for SIVAC Phase 5.

Combines episodic past-task demonstrations with application-specific heuristic
rules into a structured context bundle formatted for the LangGraph Hierarchical
Planner prompt.
"""

import logging
from typing import Optional, List

from .schema import RetrievedExperience, SimilarTaskResult, HeuristicQueryResult
from .episodic import EpisodicMemoryManager
from .semantic import SemanticMemoryManager

logger = logging.getLogger("sivac.memory.retrieval")


class ExperienceRetriever:
    """Retrieves relevant past experiences and heuristic rules to inform planning."""

    def __init__(
        self,
        episodic_memory: Optional[EpisodicMemoryManager] = None,
        semantic_memory: Optional[SemanticMemoryManager] = None,
    ) -> None:
        """Initialise ExperienceRetriever with episodic and semantic managers.

        Args:
            episodic_memory: Episodic memory manager instance.
            semantic_memory: Semantic memory manager instance.
        """
        self.episodic = episodic_memory or EpisodicMemoryManager()
        self.semantic = semantic_memory or SemanticMemoryManager()

    def retrieve_context(
        self,
        goal: str,
        application: str = "Desktop",
        screen_context: str = "",
        top_tasks: int = 2,
        top_heuristics: int = 3,
        min_confidence: float = 0.5,
    ) -> RetrievedExperience:
        """Retrieve past similar tasks and applicable application heuristics.

        Args:
            goal: Current high-level user goal.
            application: Current active application name.
            screen_context: Optional summary of current screen elements/error.
            top_tasks: Number of past similar tasks to retrieve.
            top_heuristics: Number of application heuristic rules to retrieve.
            min_confidence: Minimum confidence required for heuristics.

        Returns:
            RetrievedExperience container.
        """
        # 1. Query past similar tasks (few-shot demonstrations)
        similar_tasks = self.episodic.query_similar_tasks(
            goal=goal,
            application=application,
            top_k=top_tasks,
            only_successful=True,
        )

        # 2. Query application heuristics using screen context + goal
        heuristic_query = f"{goal} {screen_context}".strip()
        heuristics = self.semantic.query_heuristics(
            query_context=heuristic_query,
            application_name=application,
            top_k=top_heuristics,
            min_confidence=min_confidence,
        )

        # If app-specific heuristics yielded few results, broaden search
        if len(heuristics) < top_heuristics:
            broad_heuristics = self.semantic.query_heuristics(
                query_context=heuristic_query,
                application_name=None,
                top_k=top_heuristics - len(heuristics),
                min_confidence=min_confidence,
            )
            existing_ids = {h.skill_id for h in heuristics}
            for h in broad_heuristics:
                if h.skill_id not in existing_ids:
                    heuristics.append(h)

        logger.debug(
            f"Retrieved {len(similar_tasks)} past tasks and {len(heuristics)} "
            f"heuristics for goal: '{goal}'"
        )

        return RetrievedExperience(
            query_goal=goal,
            target_application=application,
            similar_tasks=similar_tasks,
            heuristics=heuristics,
        )

    def format_for_planner(self, experience: RetrievedExperience) -> str:
        """Format retrieved experience into an LLM prompt context block.

        Args:
            experience: RetrievedExperience returned by retrieve_context.

        Returns:
            Markdown-formatted text ready for injection into the planner prompt.
            Returns empty string if no relevant memories exist.
        """
        sections: List[str] = []

        # 1. Past Demonstrations
        if experience.similar_tasks:
            lines = ["### Relevant Prior Task Experiences:"]
            for task in experience.similar_tasks:
                lines.append(
                    f"- Prior Goal: \"{task.goal}\" | Application: {task.application} "
                    f"| Status: {task.status} | Steps: {task.total_steps} "
                    f"| Similarity: {task.similarity_score:.2f}"
                )
            sections.append("\n".join(lines))

        # 2. Learned Operational Heuristics
        if experience.heuristics:
            lines = ["### Learned Interaction Rules & Heuristics:"]
            for h in experience.heuristics:
                lines.append(
                    f"- [{h.application_name}] Condition: \"{h.trigger_condition}\" "
                    f"-> Strategy: \"{h.heuristic_rule}\" (Confidence: {h.confidence_score:.2f})"
                )
            sections.append("\n".join(lines))

        if not sections:
            return ""

        header = "## Experience-Informed Prior Guidance"
        return f"{header}\n" + "\n\n".join(sections)
