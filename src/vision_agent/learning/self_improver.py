"""Self-Improvement Pipeline Coordinator for SIVAC Phase 7.

Orchestrates post-task trajectory evaluation, cognitive reflection, heuristic
lesson extraction, and confidence reinforcement into a unified learning step.
"""

import logging
from typing import Optional, List, Dict, Any

from ..memory.storage import DatabaseManager
from ..memory.semantic import SemanticMemoryManager
from .schema import TrajectoryEvaluation, ReflectionOutput
from .evaluator import TrajectoryEvaluator
from .reflection import ReflectionEngine
from .lesson_extractor import LessonExtractor
from .strategy import StrategyManager

logger = logging.getLogger("sivac.learning.self_improver")


class SelfImprovementPipeline:
    """Unified coordinator executing post-task learning and self-improvement."""

    def __init__(
        self,
        db: Optional[DatabaseManager] = None,
        semantic_memory: Optional[SemanticMemoryManager] = None,
        evaluator: Optional[TrajectoryEvaluator] = None,
        reflector: Optional[ReflectionEngine] = None,
        extractor: Optional[LessonExtractor] = None,
        strategy: Optional[StrategyManager] = None,
    ) -> None:
        """Initialise pipeline with all learning subsystem components."""
        self.db = db or DatabaseManager()
        self.semantic = semantic_memory or SemanticMemoryManager(db=self.db)
        self.evaluator = evaluator or TrajectoryEvaluator()
        self.reflector = reflector or ReflectionEngine(db=self.db)
        self.extractor = extractor or LessonExtractor(semantic_memory=self.semantic)
        self.strategy = strategy or StrategyManager(semantic_memory=self.semantic)

    def improve_from_task(
        self,
        task_id: str,
        goal: str,
        application: str = "Desktop",
        success: bool = True,
        steps: Optional[List[Any]] = None,
        action_history: Optional[List[Dict[str, Any]]] = None,
        verification_history: Optional[List[Dict[str, Any]]] = None,
        retrieved_skill_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Execute post-task reflection, lesson extraction, and reinforcement.

        Args:
            task_id: Unique task identifier.
            goal: Natural language user instruction.
            application: Interacted application.
            success: Whether the overall task succeeded.
            steps: List of StepRecord objects.
            action_history: List of ActionCommand dictionaries.
            verification_history: List of VerificationResult dictionaries.
            retrieved_skill_ids: Skill IDs that were retrieved and used during planning.

        Returns:
            Dictionary containing evaluation metrics, reflection analysis,
            and newly persisted learned skills.
        """
        logger.info(f"Starting self-improvement pipeline for task [{task_id}]...")

        # 1. Trajectory Evaluation (Efficiency & Friction)
        evaluation = self.evaluator.evaluate(
            task_id=task_id,
            steps=steps,
            verification_history=verification_history,
            action_history=action_history,
        )

        # 2. Cognitive Post-Mortem Reflection
        reflection = self.reflector.reflect(
            task_id=task_id,
            goal=goal,
            application=application,
            overall_success=success,
            evaluation=evaluation,
            action_history=action_history,
            verification_history=verification_history,
        )

        # 3. Lesson Extraction & Heuristic Rule Persistence
        persisted_skills = self.extractor.extract_and_persist(reflection)

        # 4. Feedback Reinforcement of previously used skills
        reinforcement_updates = {}
        if retrieved_skill_ids:
            reinforcement_updates = self.strategy.reinforce_skills_used(
                skill_ids=retrieved_skill_ids,
                success=success,
            )

        logger.info(
            f"Self-improvement complete for [{task_id}]: "
            f"efficiency={evaluation.overall_efficiency:.2f}, "
            f"skills_learned={len(persisted_skills)}, "
            f"skills_reinforced={len(reinforcement_updates)}"
        )

        return {
            "task_id": task_id,
            "evaluation": evaluation,
            "reflection": reflection,
            "learned_skills": persisted_skills,
            "reinforcement_updates": reinforcement_updates,
        }
