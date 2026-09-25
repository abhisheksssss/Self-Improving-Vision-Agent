"""Trajectory Evaluator for SIVAC Phase 7.

Inspects executed step sequences, verification outcomes, and recovery events
to measure trajectory efficiency and identify friction points.
"""

import logging
from typing import List, Dict, Any, Optional

from ..memory.schema import StepRecord, VerificationRecord
from .schema import TrajectoryEvaluation

logger = logging.getLogger("sivac.learning.evaluator")


class TrajectoryEvaluator:
    """Analyzes task trajectories for operational friction and efficiency."""

    def evaluate(
        self,
        task_id: str,
        steps: Optional[List[Any]] = None,
        verification_history: Optional[List[Dict[str, Any]]] = None,
        action_history: Optional[List[Dict[str, Any]]] = None,
    ) -> TrajectoryEvaluation:
        """Evaluate task step sequence and identify friction points.

        Args:
            task_id: Unique task identifier.
            steps: List of StepRecord objects or raw step dictionaries.
            verification_history: List of VerificationResult dictionaries.
            action_history: List of ActionCommand dictionaries.

        Returns:
            TrajectoryEvaluation containing efficiency score and friction points.
        """
        step_items = steps or []
        verif_items = verification_history or []
        action_items = action_history or []

        total_steps = len(step_items) or len(action_items)
        if total_steps == 0:
            return TrajectoryEvaluation(
                task_id=task_id,
                overall_efficiency=1.0,
                total_steps=0,
                successful_steps=0,
                recovery_count=0,
                friction_points=[],
                recommendations=["Task had no executed steps to evaluate."],
            )

        recovery_count = 0
        friction_points: List[str] = []
        recommendations: List[str] = []

        # 1. Analyze Verification History
        for idx, verif in enumerate(verif_items):
            passed = verif.get("passed", True)
            mode = verif.get("failure_mode", "none")
            step_num = idx + 1

            if not passed:
                recovery_count += 1
                if mode == "modal_blocked":
                    friction_points.append(f"Step {step_num}: Modal / unexpected dialog blocked execution")
                    recommendations.append("Proactively check for and dismiss modal dialogs prior to element interactions.")
                elif mode == "missed_click":
                    friction_points.append(f"Step {step_num}: Click missed target or landed on dead zone")
                    recommendations.append("Use semantic element target centers or apply coordinate offset nudges.")
                elif mode == "type_failed":
                    friction_points.append(f"Step {step_num}: Keyboard focus was missing during text input")
                    recommendations.append("Always click the target input field to restore keyboard focus before typing.")
                elif mode == "ui_lag":
                    friction_points.append(f"Step {step_num}: Screen update lagged during execution")
                    recommendations.append("Inject a 1.5s - 2.0s settle pause after heavy UI transitions.")
                elif mode == "navigation_error":
                    friction_points.append(f"Step {step_num}: Browser navigation failed to load target page")
                    recommendations.append("Verify network connectivity and trigger page reloads upon timeout.")
                elif mode == "no_visual_change":
                    friction_points.append(f"Step {step_num}: Action produced no visual screen change")
                else:
                    friction_points.append(f"Step {step_num}: Verification failed with mode '{mode}'")

        # 2. Check for Injected Recovery Actions in Action History
        for idx, action in enumerate(action_items):
            reason = action.get("reason", "")
            if "[Recovery Step]" in reason or "Recovery:" in reason:
                if f"Step {idx + 1}" not in "".join(friction_points):
                    recovery_count += 1
                    friction_points.append(f"Step {idx + 1}: Executed corrective recovery action: {reason}")

        # 3. Detect Redundant Consecutive Actions (e.g. identical clicks without progress)
        for i in range(len(action_items) - 1):
            a1 = action_items[i]
            a2 = action_items[i + 1]
            if (
                a1.get("action") == a2.get("action") == "click"
                and a1.get("x") == a2.get("x")
                and a1.get("y") == a2.get("y")
                and a1.get("x") is not None
            ):
                friction_points.append(f"Steps {i+1}-{i+2}: Repeated identical click at ({a1.get('x')}, {a1.get('y')})")
                recommendations.append("Avoid rapid identical repeat clicks; allow UI time to process input events.")

        # 4. Compute Efficiency Score
        successful_steps = max(0, total_steps - recovery_count)
        # Efficiency penalty: 0.5 per recovery intervention
        efficiency = (total_steps - (0.5 * recovery_count)) / total_steps if total_steps > 0 else 1.0
        efficiency = round(max(0.1, min(1.0, efficiency)), 2)

        # Deduplicate recommendations
        unique_recs = list(dict.fromkeys(recommendations))

        logger.debug(
            f"Evaluated task {task_id}: total_steps={total_steps}, "
            f"recoveries={recovery_count}, efficiency={efficiency:.2f}, "
            f"friction_count={len(friction_points)}"
        )

        return TrajectoryEvaluation(
            task_id=task_id,
            overall_efficiency=efficiency,
            total_steps=total_steps,
            successful_steps=successful_steps,
            recovery_count=recovery_count,
            friction_points=friction_points,
            recommendations=unique_recs,
        )
