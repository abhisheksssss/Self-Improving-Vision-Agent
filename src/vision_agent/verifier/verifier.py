"""Unified Outcome Verifier for SIVAC Phase 4.

Orchestrates the two-tier verification pipeline:
    Tier 1: Perceptual visual diff (VisualDiffEngine)
    Tier 2: Structural DOM / accessibility tree delta (StateDiffEngine)

Followed by:
    - Failure classification (FailureClassifier)
    - Autonomous corrective plan generation (RecoveryEngine)
"""

import os
import logging
from typing import Optional, Tuple, Dict, Any, List

from ..perception.state import UIState, UIElement
from ..actions.schema import ActionCommand, ActionType
from .schema import VerificationResult, FailureMode, RecoveryPlan, RecoveryStep
from .visual_diff import (
    VisualDiffEngine,
    DEFAULT_NO_CHANGE_THRESHOLD,
    DEFAULT_LAG_THRESHOLD,
)
from .state_diff import StateDiffEngine, ElementDelta
from .classifier import FailureClassifier
from .recovery import RecoveryEngine, MAX_RETRIES

logger = logging.getLogger("sivac.verifier")


class OutcomeVerifier:
    """Two-tier outcome verifier and closed-loop recovery orchestrator.

    Verifies whether an executed ActionCommand had the intended effect on the
    target environment by comparing UI state snapshots before and after action
    execution.

    If verification fails, it invokes the RecoveryEngine to generate a corrective
    action plan that can be injected back into the execution loop.
    """

    def __init__(
        self,
        visual_engine: Optional[VisualDiffEngine] = None,
        state_engine: Optional[StateDiffEngine] = None,
        classifier: Optional[FailureClassifier] = None,
        recovery_engine: Optional[RecoveryEngine] = None,
        no_change_threshold: float = DEFAULT_NO_CHANGE_THRESHOLD,
        lag_threshold: float = DEFAULT_LAG_THRESHOLD,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        """Initialise OutcomeVerifier with engine instances and thresholds.

        Args:
            visual_engine:       Visual diff engine (creates default if None).
            state_engine:        Structural state diff engine (creates default if None).
            classifier:          Failure mode classifier (creates default if None).
            recovery_engine:     Autonomous recovery engine (creates default if None).
            no_change_threshold: Minimum visual change score to avoid NO_VISUAL_CHANGE.
            lag_threshold:       Visual change threshold separating lag from full update.
            max_retries:         Maximum retries before aborting via MAX_RETRIES_EXCEEDED.
        """
        self.visual_engine = visual_engine or VisualDiffEngine(
            no_change_threshold=no_change_threshold,
            lag_threshold=lag_threshold,
        )
        self.state_engine = state_engine or StateDiffEngine()
        self.classifier = classifier or FailureClassifier(visual_engine=self.visual_engine)
        self.recovery_engine = recovery_engine or RecoveryEngine(max_retries=max_retries)

    # ------------------------------------------------------------------
    # Tier 1 & Tier 2 Inspection Helpers
    # ------------------------------------------------------------------

    def calculate_visual_diff(
        self,
        img1_path: Optional[str],
        img2_path: Optional[str],
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate perceptual visual difference score between two screenshot paths.

        Args:
            img1_path: Path to screenshot before action execution.
            img2_path: Path to screenshot after action execution.

        Returns:
            Tuple of (diff_score, metadata_dict). If screenshots are not
            available or invalid, returns (0.0, {"error": "..."}).
        """
        if not img1_path or not img2_path:
            return 0.0, {"warning": "One or both screenshot paths not provided"}

        if not os.path.isfile(img1_path) or not os.path.isfile(img2_path):
            return 0.0, {"warning": "One or both screenshot files do not exist on disk"}

        return self.visual_engine.compute_score(img1_path, img2_path)

    def verify_dom_change(
        self,
        before: UIState,
        after: UIState,
    ) -> ElementDelta:
        """Inspect structural accessibility/DOM changes between two UIStates.

        Args:
            before: UIState captured immediately before action execution.
            after:  UIState captured immediately after action execution.

        Returns:
            ElementDelta containing appeared, disappeared, and changed elements.
        """
        return self.state_engine.compute_delta(before, after)

    # ------------------------------------------------------------------
    # Step Verification
    # ------------------------------------------------------------------

    def verify_step(
        self,
        before: UIState,
        after: UIState,
        cmd: ActionCommand,
        retry_count: int = 0,
    ) -> VerificationResult:
        """Verify the execution outcome of an action step.

        Runs perceptual visual diff and structural DOM diff, passes the signals
        to the failure classifier, and produces a strongly-typed VerificationResult.

        Args:
            before:      UIState snapshot captured immediately before execution.
            after:       UIState snapshot captured immediately after execution.
            cmd:         The ActionCommand that was dispatched.
            retry_count: Number of previous recovery retries for this step.

        Returns:
            VerificationResult indicating pass/fail, classified failure mode,
            and diagnostic details.
        """
        action_type = cmd.action

        # Handle FINISH action — always succeeds
        if action_type == ActionType.FINISH:
            return VerificationResult(
                passed=True,
                failure_mode=FailureMode.NONE,
                visual_diff_score=0.0,
                element_delta_count=0,
                details="Task completed: FINISH command verified.",
                retry_count=retry_count,
            )

        # 1. Structural DOM delta (Tier 2)
        delta = self.verify_dom_change(before, after)

        # 2. Perceptual visual diff (Tier 1)
        visual_score, visual_meta = self.calculate_visual_diff(
            before.screenshot_path,
            after.screenshot_path,
        )

        # Handle WAIT action — waiting succeeds unless an unexpected blocking modal appeared
        if action_type == ActionType.WAIT:
            if delta.has_modal:
                failure_mode = FailureMode.MODAL_BLOCKED
                passed = False
                details = "Modal appeared during wait period."
            else:
                failure_mode = FailureMode.NONE
                passed = True
                details = f"Wait action verified ({cmd.seconds or 1.0}s). Screen settled."

            return VerificationResult(
                passed=passed,
                failure_mode=failure_mode,
                visual_diff_score=visual_score,
                element_delta_count=delta.total_changes,
                details=details,
                retry_count=retry_count,
                metadata={
                    "delta_summary": delta.summary(),
                    "visual_meta": visual_meta,
                },
            )

        # 3. Classify failure mode
        failure_mode = self.classifier.classify(
            visual_score=visual_score,
            delta=delta,
            action_type=action_type.value,
        )

        # Check if max retries exceeded
        if failure_mode != FailureMode.NONE and retry_count >= self.recovery_engine.max_retries:
            logger.warning(
                f"Action {action_type.value} failed and retry limit "
                f"({self.recovery_engine.max_retries}) reached."
            )
            failure_mode = FailureMode.MAX_RETRIES_EXCEEDED

        passed = (failure_mode == FailureMode.NONE)

        # Build diagnostic explanation
        if passed:
            details = (
                f"Action '{action_type.value}' verified successfully. "
                f"Visual change: {visual_score:.3f} | {delta.summary()}"
            )
        else:
            details = (
                f"Action '{action_type.value}' verification failed: {failure_mode.value}. "
                f"Visual score: {visual_score:.3f} | {delta.summary()}"
            )

        return VerificationResult(
            passed=passed,
            failure_mode=failure_mode,
            visual_diff_score=visual_score,
            element_delta_count=delta.total_changes,
            details=details,
            retry_count=retry_count,
            metadata={
                "action": action_type.value,
                "delta_summary": delta.summary(),
                "appeared_ids": [e.id for e in delta.appeared],
                "disappeared_ids": [e.id for e in delta.disappeared],
                "url_changed": delta.url_changed,
                "title_changed": delta.title_changed,
                "has_modal": delta.has_modal,
                "visual_meta": visual_meta,
            },
        )

    # ------------------------------------------------------------------
    # Recovery Planning
    # ------------------------------------------------------------------

    def generate_recovery(
        self,
        result: VerificationResult,
        cmd: ActionCommand,
    ) -> RecoveryPlan:
        """Generate a corrective RecoveryPlan for a failed verification result.

        Args:
            result: VerificationResult produced by verify_step (must have passed=False).
            cmd:    The ActionCommand that failed.

        Returns:
            RecoveryPlan containing ordered corrective actions and routing flags.
        """
        if result.passed:
            logger.info("Verification passed — no recovery needed.")
            return RecoveryPlan(
                failure_mode=FailureMode.NONE,
                steps=[],
                should_replan=False,
                should_abort=False,
                explanation="No recovery required; action verification passed.",
            )

        return self.recovery_engine.generate_plan(
            failure_mode=result.failure_mode,
            retry_count=result.retry_count,
            original_action_type=cmd.action.value,
            original_x=cmd.x,
            original_y=cmd.y,
            original_text=cmd.text,
            original_target_id=cmd.target_id,
            original_url=cmd.url,
            original_app_name=cmd.app_name,
        )

    def verify_and_recover(
        self,
        before: UIState,
        after: UIState,
        cmd: ActionCommand,
        retry_count: int = 0,
    ) -> Tuple[VerificationResult, Optional[RecoveryPlan]]:
        """Verify the step outcome and immediately generate a recovery plan if failed.

        Convenience orchestration method for the agent execution graph.

        Args:
            before:      UIState snapshot before execution.
            after:       UIState snapshot after execution.
            cmd:         The ActionCommand executed.
            retry_count: Number of previous recovery retries for this step.

        Returns:
            Tuple of (VerificationResult, Optional[RecoveryPlan]).
            RecoveryPlan is None if verification passed.
        """
        result = self.verify_step(before, after, cmd, retry_count=retry_count)

        if result.passed:
            return result, None

        recovery_plan = self.generate_recovery(result, cmd)
        return result, recovery_plan
