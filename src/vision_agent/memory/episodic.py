"""Episodic Memory Manager for SIVAC Phase 5.

Orchestrates relational audit logging (SQLite) and semantic vector indexing
(ChromaDB) across the complete task execution lifecycle.
"""

import time
import logging
from typing import Optional, List, Dict, Any

from ..perception.state import UIState
from ..actions.schema import ActionCommand, ActionResult
from ..verifier.schema import VerificationResult
from .schema import (
    TaskStatus,
    TaskRecord,
    TrajectoryRecord,
    StepRecord,
    VerificationRecord,
    SimilarTaskResult,
)
from .storage import DatabaseManager
from .vector_store import ChromaVectorStore

logger = logging.getLogger("sivac.memory.episodic")


class EpisodicMemoryManager:
    """Manages the full lifecycle of task trajectories and episodic recall."""

    def __init__(
        self,
        db: Optional[DatabaseManager] = None,
        vector_store: Optional[ChromaVectorStore] = None,
    ) -> None:
        """Initialise EpisodicMemoryManager with relational and vector stores.

        Args:
            db: Relational database manager instance (creates default if None).
            vector_store: Chroma vector store instance (creates default if None).
        """
        self.db = db or DatabaseManager()
        self.vector_store = vector_store or ChromaVectorStore()

    # ------------------------------------------------------------------
    # Task Lifecycle Operations
    # ------------------------------------------------------------------

    def start_task(
        self,
        goal: str,
        initial_app: str = "Desktop",
        task_id: Optional[str] = None,
        model_used: str = "default",
    ) -> str:
        """Start and record a new agent task session.

        Args:
            goal: Natural language user objective.
            initial_app: Active desktop window or browser application.
            task_id: Optional explicit task ID (UUID generated if None).
            model_used: Primary model identifier used for this task.

        Returns:
            The created task_id string.
        """
        task = self.db.create_task(goal=goal, task_id=task_id)
        traj = self.db.create_trajectory(task_id=task.id, model_used=model_used)
        logger.info(f"Started task [{task.id}]: '{goal}' (trajectory: {traj.id})")
        return task.id

    def log_step(
        self,
        task_id: str,
        step_number: int,
        planned_subgoal: str,
        action: ActionCommand,
        result: ActionResult,
        verification: Optional[VerificationResult] = None,
        before_state: Optional[UIState] = None,
        after_state: Optional[UIState] = None,
    ) -> StepRecord:
        """Log an execution step along with observation and verification audits.

        Args:
            task_id: Active task session ID.
            step_number: 1-indexed step sequence number.
            planned_subgoal: Subgoal description decided by the planner.
            action: ActionCommand that was dispatched.
            result: ActionResult returned by ActionExecutor.
            verification: Optional VerificationResult from OutcomeVerifier.
            before_state: UIState snapshot before action dispatch.
            after_state: UIState snapshot after action dispatch.

        Returns:
            The created StepRecord.
        """
        traj = self.db.get_trajectory(task_id)
        if not traj:
            traj = self.db.create_trajectory(task_id=task_id)

        # Extract screenshot paths
        screenshot_before = before_state.screenshot_path if before_state and before_state.screenshot_path else ""
        screenshot_after = after_state.screenshot_path if after_state and after_state.screenshot_path else None

        # Compact observation element summaries (ids, types, texts)
        obs_elements: Optional[List[Dict[str, Any]]] = None
        if before_state and before_state.elements:
            obs_elements = [
                {"id": e.id, "type": e.type, "text": e.text, "bbox": e.bbox.to_list()}
                for e in before_state.elements[:50]  # store up to 50 primary elements
            ]

        # Serialise action and result
        action_dict = action.model_dump()
        result_dict = result.model_dump()

        step = self.db.add_step(
            trajectory_id=traj.id,
            step_number=step_number,
            screenshot_before=screenshot_before,
            screenshot_after=screenshot_after,
            observation_elements=obs_elements,
            planned_subgoal=planned_subgoal,
            action_command=action_dict,
            execution_result=result_dict,
        )

        # Record verification if provided
        if verification:
            self.db.add_verification(
                step_id=step.id,
                passed=verification.passed,
                visual_diff_score=verification.visual_diff_score,
                dom_delta=verification.metadata if verification.metadata else None,
                failure_reason=verification.details if not verification.passed else None,
                retry_count=verification.retry_count,
            )

        logger.debug(
            f"Logged step {step_number} for task {task_id}: action={action.action.value}, "
            f"success={result.success}, verified={verification.passed if verification else 'N/A'}"
        )
        return step

    def complete_task(
        self,
        task_id: str,
        success: bool,
        duration_seconds: float = 0.0,
        total_steps: Optional[int] = None,
        application: str = "Desktop",
        model_used: str = "default",
    ) -> Optional[TaskRecord]:
        """Finalize task status in SQLite and index trajectory into ChromaDB.

        Args:
            task_id: Task session ID.
            success: Whether the high-level user goal was achieved.
            duration_seconds: Total elapsed time in seconds.
            total_steps: Total steps executed (computed from DB if None).
            application: Primary application interacted with.
            model_used: Model identifier used during execution.

        Returns:
            Updated TaskRecord.
        """
        status = TaskStatus.COMPLETED if success else TaskStatus.FAILED

        if total_steps is None:
            traj = self.db.get_trajectory(task_id)
            if traj:
                steps = self.db.get_steps_for_trajectory(traj.id)
                total_steps = len(steps)
            else:
                total_steps = 0

        updated_task = self.db.update_task_status(
            task_id=task_id,
            status=status,
            total_steps=total_steps,
            total_duration_seconds=duration_seconds,
        )

        if updated_task:
            # Index into ChromaDB vector store
            self.vector_store.index_task(
                task_id=task_id,
                goal=updated_task.goal,
                application=application,
                status=status.value,
                total_steps=total_steps,
                duration_seconds=duration_seconds,
                model_used=model_used,
                timestamp=time.time(),
            )
            logger.info(
                f"Task [{task_id}] marked as {status.value} with {total_steps} steps "
                f"({duration_seconds:.1f}s) and indexed in ChromaDB"
            )

        return updated_task

    # ------------------------------------------------------------------
    # Query & Retrieval Operations
    # ------------------------------------------------------------------

    def get_task(self, task_id: str) -> Optional[TaskRecord]:
        """Fetch task record by ID."""
        return self.db.get_task(task_id)

    def get_steps(self, task_id: str) -> List[StepRecord]:
        """Fetch ordered step records for a task."""
        traj = self.db.get_trajectory(task_id)
        if not traj:
            return []
        return self.db.get_steps_for_trajectory(traj.id)

    def list_tasks(self, limit: int = 20, status: Optional[TaskStatus] = None) -> List[TaskRecord]:
        """List recent tasks."""
        return self.db.list_tasks(limit=limit, status=status)

    def query_similar_tasks(
        self,
        goal: str,
        application: Optional[str] = None,
        top_k: int = 3,
        only_successful: bool = True,
    ) -> List[SimilarTaskResult]:
        """Query semantic vector memory for similar past tasks."""
        return self.vector_store.query_similar_tasks(
            goal=goal,
            application=application,
            top_k=top_k,
            only_successful=only_successful,
        )
