"""Cognitive Reflection Engine for SIVAC Phase 7.

Performs post-mortem analysis on completed agent trajectories, determines
root causes of friction, synthesizes operational lessons, and persists
reflection records into SQLite.
"""

import re
import json
import logging
from typing import Optional, List, Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage

from ..model.factory import ModelFactory
from ..memory.storage import DatabaseManager
from ..memory.schema import ReflectionRecord
from .schema import TrajectoryEvaluation, ExtractedLesson, ReflectionOutput

logger = logging.getLogger("sivac.learning.reflection")

REFLECTION_SYSTEM_PROMPT = """You are SIVAC's Cognitive Reflection Engine.
Your task is to conduct an objective post-mortem evaluation of an autonomous computer-use trajectory.
Analyze the user's goal, the executed steps, verification outcomes, and identified friction points.

You must output ONLY a valid JSON object in the following format:
{
  "root_cause_analysis": "Concise paragraph explaining what worked, what caused delays or failures, and how obstacles were overcome.",
  "friction_points": ["List of specific friction events or delays encountered"],
  "lessons": [
    {
      "application_name": "Chrome|Notepad|Excel|Desktop",
      "trigger_condition": "Concrete UI condition or state where this rule applies (e.g. 'Modal dialog blocks page')",
      "heuristic_rule": "Concrete actionable strategy to handle this condition (e.g. 'Press ESC to dismiss modal before clicking')",
      "confidence_score": 0.85,
      "rationale": "Why this rule improves efficiency or reliability"
    }
  ]
}
"""


class ReflectionEngine:
    """Performs post-task cognitive reflection and root-cause analysis."""

    def __init__(
        self,
        db: Optional[DatabaseManager] = None,
        model_override: Optional[Any] = None,
    ) -> None:
        """Initialise ReflectionEngine with database manager and reasoning model."""
        self.db = db or DatabaseManager()
        self._model = model_override

    def _get_model(self):
        if self._model == "fallback":
            return None
        if self._model:
            return self._model
        try:
            return ModelFactory.get_reasoning_model()
        except Exception as e:
            logger.warning(f"Could not initialize reasoning model for reflection: {e}")
            return None

    def reflect(
        self,
        task_id: str,
        goal: str,
        application: str,
        overall_success: bool,
        evaluation: TrajectoryEvaluation,
        action_history: Optional[List[Dict[str, Any]]] = None,
        verification_history: Optional[List[Dict[str, Any]]] = None,
    ) -> ReflectionOutput:
        """Conduct post-mortem reflection and extract reusable heuristic lessons.

        Args:
            task_id: Unique task identifier.
            goal: Natural language user objective.
            application: Application interacted with.
            overall_success: Whether the final goal was achieved.
            evaluation: TrajectoryEvaluation metrics from TrajectoryEvaluator.
            action_history: List of executed action dictionaries.
            verification_history: List of step verification dictionaries.

        Returns:
            ReflectionOutput containing root cause analysis and extracted lessons.
        """
        model = self._get_model()
        reflection_out: Optional[ReflectionOutput] = None

        if model is not None:
            try:
                prompt_content = self._build_reflection_prompt(
                    goal=goal,
                    application=application,
                    overall_success=overall_success,
                    evaluation=evaluation,
                    action_history=action_history or [],
                    verification_history=verification_history or [],
                )
                response = model.invoke([
                    SystemMessage(content=REFLECTION_SYSTEM_PROMPT),
                    HumanMessage(content=prompt_content),
                ])
                reflection_out = self._parse_llm_response(task_id, overall_success, response.content)
            except Exception as e:
                logger.warning(f"Reasoning model reflection failed ({e}), falling back to deterministic reflection")

        if reflection_out is None:
            # Deterministic heuristic reflection fallback
            reflection_out = self._heuristic_fallback_reflection(
                task_id=task_id,
                goal=goal,
                application=application,
                overall_success=overall_success,
                evaluation=evaluation,
            )

        # Persist ReflectionRecord in SQLite
        try:
            self._persist_reflection(reflection_out)
        except Exception as e:
            logger.warning(f"Could not persist reflection record to SQLite: {e}")

        logger.info(
            f"Reflection completed for task [{task_id}]: success={overall_success}, "
            f"lessons_extracted={len(reflection_out.lessons)}"
        )
        return reflection_out

    def _build_reflection_prompt(
        self,
        goal: str,
        application: str,
        overall_success: bool,
        evaluation: TrajectoryEvaluation,
        action_history: List[Dict[str, Any]],
        verification_history: List[Dict[str, Any]],
    ) -> str:
        """Build user prompt for post-mortem reflection."""
        lines = [
            f"Task Goal: {goal}",
            f"Target Application: {application}",
            f"Task Outcome: {'SUCCESS' if overall_success else 'FAILED'}",
            f"Total Steps: {evaluation.total_steps}",
            f"Trajectory Efficiency: {evaluation.overall_efficiency:.2f}",
            f"Recovery Interventions: {evaluation.recovery_count}",
        ]

        if evaluation.friction_points:
            lines.append("\nIdentified Friction Points:")
            for fp in evaluation.friction_points:
                lines.append(f"  - {fp}")

        if action_history:
            lines.append("\nExecuted Actions Sequence:")
            for idx, a in enumerate(action_history):
                lines.append(f"  {idx + 1}. Action: {a.get('action')} (Reason: {a.get('reason')})")

        if verification_history:
            lines.append("\nStep Verification Verdicts:")
            for idx, v in enumerate(verification_history):
                lines.append(f"  Step {idx + 1}: Passed={v.get('passed')}, Mode={v.get('failure_mode')}")

        lines.append("\nProvide your structured JSON reflection post-mortem now:")
        return "\n".join(lines)

    def _parse_llm_response(
        self, task_id: str, overall_success: bool, text: str
    ) -> Optional[ReflectionOutput]:
        """Parse structured reflection output from LLM JSON response."""
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None

        try:
            data = json.loads(match.group(0))
            lessons = []
            for item in data.get("lessons", []):
                lessons.append(
                    ExtractedLesson(
                        application_name=item.get("application_name", "Desktop"),
                        trigger_condition=item.get("trigger_condition", ""),
                        heuristic_rule=item.get("heuristic_rule", ""),
                        confidence_score=float(item.get("confidence_score", 0.80)),
                        rationale=item.get("rationale", ""),
                    )
                )

            return ReflectionOutput(
                task_id=task_id,
                overall_success=overall_success,
                root_cause_analysis=data.get("root_cause_analysis", ""),
                friction_points=data.get("friction_points", []),
                lessons=lessons,
            )
        except Exception as e:
            logger.warning(f"Failed to parse LLM reflection JSON: {e}")
            return None

    def _heuristic_fallback_reflection(
        self,
        task_id: str,
        goal: str,
        application: str,
        overall_success: bool,
        evaluation: TrajectoryEvaluation,
    ) -> ReflectionOutput:
        """Deterministic rule-based reflection when LLM is unavailable."""
        friction_points = evaluation.friction_points
        lessons: List[ExtractedLesson] = []

        if overall_success and not friction_points:
            root_cause = (
                f"Task succeeded cleanly with 100% efficiency. Actions were validated "
                f"and verified without requiring recovery interventions."
            )
            lessons.append(
                ExtractedLesson(
                    application_name=application,
                    trigger_condition=f"Goal involves {goal[:40]} in {application}",
                    heuristic_rule=f"Direct sequential execution effective; maintain standard click and wait intervals",
                    confidence_score=0.85,
                    rationale="Clean execution baseline",
                )
            )
        else:
            root_cause = (
                f"Task executed with {evaluation.recovery_count} recovery interventions. "
                f"Friction points encountered were analyzed to extract operational heuristics."
            )

            # Analyze friction points to synthesize rules
            fp_text = " ".join(friction_points).lower()

            if "modal" in fp_text or "dialog" in fp_text:
                lessons.append(
                    ExtractedLesson(
                        application_name=application,
                        trigger_condition="Unexpected modal dialog or alert blocks interactive elements",
                        heuristic_rule="Press 'escape' or click dismiss button to clear blocking dialogs before main actions",
                        confidence_score=0.88,
                        rationale="Prevents repeated blocked clicks on obscured target buttons",
                    )
                )

            if "focus" in fp_text or "type" in fp_text:
                lessons.append(
                    ExtractedLesson(
                        application_name=application,
                        trigger_condition="Typing into an input field or text area",
                        heuristic_rule="Explicitly click target input field to restore focus before dispatching type action",
                        confidence_score=0.90,
                        rationale="Ensures keyboard focus is active so typed characters are not dropped",
                    )
                )

            if "missed" in fp_text or "dead zone" in fp_text:
                lessons.append(
                    ExtractedLesson(
                        application_name=application,
                        trigger_condition="Click produces no visual delta or target element state change",
                        heuristic_rule="Apply a +5px to +10px coordinate offset or re-ground on element center coordinates",
                        confidence_score=0.82,
                        rationale="Avoids inactive borders or unclickable whitespace padding",
                    )
                )

            if "lag" in fp_text or "latency" in fp_text:
                lessons.append(
                    ExtractedLesson(
                        application_name=application,
                        trigger_condition="Asynchronous page navigation or heavy UI rendering",
                        heuristic_rule="Wait 1.5s to 2.0s for loading spinners and DOM elements to settle",
                        confidence_score=0.85,
                        rationale="Prevents premature action dispatch while UI is still rendering",
                    )
                )

            if "repeat" in fp_text or "identical" in fp_text or "click" in fp_text:
                lessons.append(
                    ExtractedLesson(
                        application_name=application,
                        trigger_condition="Consecutive repeated clicks on the same coordinate or button",
                        heuristic_rule="Allow a 1.0s settle interval or verify target state change before retrying click",
                        confidence_score=0.85,
                        rationale="Prevents racing UI state transitions caused by rapid duplicate clicks",
                    )
                )

            if not lessons:
                lessons.append(
                    ExtractedLesson(
                        application_name=application,
                        trigger_condition=f"Interacting with {application} interface for {goal[:40]}",
                        heuristic_rule="Ensure UI elements are settled and visible before dispatching interactions",
                        confidence_score=0.80,
                        rationale="General operational reliability heuristic",
                    )
                )

        return ReflectionOutput(
            task_id=task_id,
            overall_success=overall_success,
            root_cause_analysis=root_cause,
            friction_points=friction_points,
            lessons=lessons,
        )

    def _persist_reflection(self, reflection: ReflectionOutput) -> None:
        """Persist reflection record in SQLite reflections table."""
        with self.db.session_scope() as session:
            # Check if task exists in database
            from ..memory.storage import TaskModel, ReflectionModel
            task = session.get(TaskModel, reflection.task_id)
            if not task:
                return

            # Check if reflection already recorded
            existing = session.query(ReflectionModel).filter_by(task_id=reflection.task_id).first()
            if existing:
                existing.overall_success = 1 if reflection.overall_success else 0
                existing.root_cause_analysis = reflection.root_cause_analysis
                existing.friction_points = json.dumps(reflection.friction_points)
            else:
                record = ReflectionModel(
                    task_id=reflection.task_id,
                    overall_success=1 if reflection.overall_success else 0,
                    root_cause_analysis=reflection.root_cause_analysis,
                    friction_points=json.dumps(reflection.friction_points),
                )
                session.add(record)
