"""Autonomous Recovery Engine for SIVAC Phase 4.

Generates a tailored RecoveryPlan (a short ordered sequence of corrective
ActionCommands) in response to a classified FailureMode.  These steps are
injected into the execution pipeline BEFORE the Hierarchical Planner is
invoked again, so the agent operates against a more stable UIState.

Recovery Strategies by FailureMode:
┌───────────────────────┬──────────────────────────────────────────────────────┐
│ FailureMode           │ Recovery Strategy                                    │
├───────────────────────┼──────────────────────────────────────────────────────┤
│ NO_VISUAL_CHANGE      │ Wait 1s, then retry the original action with a small │
│                       │ coordinate offset (+10px, +10px) to avoid dead zones │
│ UI_LAG                │ Wait 2s, then re-verify (no action injection)        │
│ MISSED_CLICK          │ Slight coordinate offset retry (+5px, +5px)          │
│ TYPE_FAILED           │ Click target to refocus, then retype                 │
│ MODAL_BLOCKED         │ Press ESC → wait 0.5s → try clicking 'Close'/'OK'   │
│ NAVIGATION_ERROR      │ Wait 2s → press F5 (browser reload) → wait 2s       │
│ APP_LAUNCH_FAILED     │ Wait 1.5s → retry APP_OPEN                          │
│ MAX_RETRIES_EXCEEDED  │ Abort (should_abort=True)                            │
│ UNKNOWN               │ Wait 1.5s, replan from scratch                       │
└───────────────────────┴──────────────────────────────────────────────────────┘
"""

import logging
from typing import Optional
from .schema import FailureMode, RecoveryPlan, RecoveryStep

logger = logging.getLogger("sivac.verifier.recovery")

# Maximum number of automatic retries before the task is aborted
MAX_RETRIES = 3


class RecoveryEngine:
    """Generates deterministic RecoveryPlans for each classified FailureMode.

    Each strategy is a pure function of (failure_mode, retry_count, original_cmd)
    so the recovery logic is fully testable and reproducible.
    """

    def __init__(self, max_retries: int = MAX_RETRIES) -> None:
        """Initialise the recovery engine.

        Args:
            max_retries: How many recovery attempts to allow before aborting.
        """
        self.max_retries = max_retries

    def generate_plan(
        self,
        failure_mode: FailureMode,
        retry_count: int,
        original_action_type: Optional[str] = None,
        original_x: Optional[int] = None,
        original_y: Optional[int] = None,
        original_text: Optional[str] = None,
        original_target_id: Optional[str] = None,
        original_url: Optional[str] = None,
        original_app_name: Optional[str] = None,
    ) -> RecoveryPlan:
        """Generate a RecoveryPlan for the given failure mode and retry context.

        Args:
            failure_mode:         Classified failure from FailureClassifier.
            retry_count:          How many retries have already been attempted.
            original_action_type: The ActionType string of the failed action.
            original_x / y:       Coordinates used in the failed action.
            original_text:        Text that was supposed to be typed.
            original_target_id:   Semantic element ID targeted.
            original_url:         URL for browser navigation failures.
            original_app_name:    Application name for launch failures.

        Returns:
            RecoveryPlan with ordered corrective steps and routing flags.
        """
        # Hard abort when max retries are exceeded
        if retry_count >= self.max_retries:
            logger.warning(
                f"Max retries ({self.max_retries}) exceeded — aborting task"
            )
            return RecoveryPlan(
                failure_mode=FailureMode.MAX_RETRIES_EXCEEDED,
                steps=[],
                should_replan=False,
                should_abort=True,
                explanation=(
                    f"Recovery aborted: {self.max_retries} attempts made without success. "
                    f"Last failure: {failure_mode.value}"
                ),
            )

        # Dispatch to the correct strategy
        if failure_mode == FailureMode.NO_VISUAL_CHANGE:
            return self._strategy_no_visual_change(
                retry_count, original_action_type, original_x, original_y, original_target_id
            )
        elif failure_mode == FailureMode.UI_LAG:
            return self._strategy_ui_lag(retry_count)
        elif failure_mode == FailureMode.MISSED_CLICK:
            return self._strategy_missed_click(
                retry_count, original_x, original_y, original_target_id
            )
        elif failure_mode == FailureMode.TYPE_FAILED:
            return self._strategy_type_failed(
                retry_count, original_text, original_x, original_y, original_target_id
            )
        elif failure_mode == FailureMode.MODAL_BLOCKED:
            return self._strategy_modal_blocked(retry_count)
        elif failure_mode == FailureMode.NAVIGATION_ERROR:
            return self._strategy_navigation_error(retry_count, original_url)
        elif failure_mode == FailureMode.APP_LAUNCH_FAILED:
            return self._strategy_app_launch_failed(retry_count, original_app_name)
        else:
            # UNKNOWN or anything else — conservative replan after a wait
            return self._strategy_unknown(retry_count)

    # ------------------------------------------------------------------
    # Individual recovery strategies
    # ------------------------------------------------------------------

    def _strategy_no_visual_change(
        self,
        retry_count: int,
        action_type: Optional[str],
        x: Optional[int],
        y: Optional[int],
        target_id: Optional[str],
    ) -> RecoveryPlan:
        """Screen showed no change after action — wait then retry with coordinate nudge."""
        # Exponential backoff: 1s, 2s, 4s
        wait_secs = 1.0 * (2 ** retry_count)
        steps = [
            RecoveryStep(
                action_type="wait",
                params={"seconds": wait_secs},
                reason=f"Waiting {wait_secs}s before retrying (attempt {retry_count + 1})",
            )
        ]

        # If we know the original coordinates, nudge by +10px to avoid dead zones
        if x is not None and y is not None and action_type and "click" in action_type:
            nudge = 10 * (retry_count + 1)  # increase nudge each retry
            steps.append(
                RecoveryStep(
                    action_type="click",
                    params={
                        "x": x + nudge,
                        "y": y + nudge,
                        "button": "left",
                    },
                    reason=f"Retrying click with {nudge}px coordinate nudge",
                )
            )

        return RecoveryPlan(
            failure_mode=FailureMode.NO_VISUAL_CHANGE,
            steps=steps,
            should_replan=True,
            explanation=f"No visual change detected. Waiting {wait_secs}s then requesting replan.",
        )

    def _strategy_ui_lag(self, retry_count: int) -> RecoveryPlan:
        """Screen shows minor change — UI is still loading, just wait longer."""
        wait_secs = 1.5 + (1.0 * retry_count)  # 1.5s, 2.5s, 3.5s
        return RecoveryPlan(
            failure_mode=FailureMode.UI_LAG,
            steps=[
                RecoveryStep(
                    action_type="wait",
                    params={"seconds": wait_secs},
                    reason=f"UI appears to be loading — waiting {wait_secs}s",
                )
            ],
            should_replan=False,  # Re-verify the same step after waiting
            explanation=(
                f"Minor visual change detected (UI_LAG). Waiting {wait_secs}s "
                f"for the UI to settle before re-verifying."
            ),
        )

    def _strategy_missed_click(
        self,
        retry_count: int,
        x: Optional[int],
        y: Optional[int],
        target_id: Optional[str],
    ) -> RecoveryPlan:
        """Click landed on wrong element — re-attempt with a small offset."""
        wait_secs = 0.5
        steps: list = [
            RecoveryStep(
                action_type="wait",
                params={"seconds": wait_secs},
                reason="Brief pause before retry",
            )
        ]
        if x is not None and y is not None:
            nudge = 5 * (retry_count + 1)
            steps.append(
                RecoveryStep(
                    action_type="click",
                    params={"x": x + nudge, "y": y, "button": "left"},
                    reason=f"Retrying click with {nudge}px horizontal offset",
                )
            )
        return RecoveryPlan(
            failure_mode=FailureMode.MISSED_CLICK,
            steps=steps,
            should_replan=True,
            explanation=f"Click appears to have missed target. Retrying with coordinate offset.",
        )

    def _strategy_type_failed(
        self,
        retry_count: int,
        text: Optional[str],
        x: Optional[int],
        y: Optional[int],
        target_id: Optional[str],
    ) -> RecoveryPlan:
        """Typed text did not appear — click to refocus field then retype."""
        steps: list = []

        # Step 1: Click to re-focus the input field
        if x is not None and y is not None:
            steps.append(
                RecoveryStep(
                    action_type="click",
                    params={"x": x, "y": y},
                    reason="Re-clicking input field to ensure keyboard focus",
                )
            )
        elif target_id:
            steps.append(
                RecoveryStep(
                    action_type="click",
                    params={"target_id": target_id},
                    reason="Re-clicking target element by ID to restore focus",
                )
            )

        # Step 2: Clear any existing text with Ctrl+A then Delete
        steps.append(
            RecoveryStep(
                action_type="hotkey",
                params={"keys": ["ctrl", "a"]},
                reason="Select all existing text in field before retyping",
            )
        )
        steps.append(
            RecoveryStep(
                action_type="press",
                params={"key": "delete"},
                reason="Clear the selected text",
            )
        )

        # Step 3: Retype the text
        if text:
            steps.append(
                RecoveryStep(
                    action_type="type",
                    params={"text": text},
                    reason=f"Retyping the text: '{text[:30]}'",
                )
            )

        return RecoveryPlan(
            failure_mode=FailureMode.TYPE_FAILED,
            steps=steps,
            should_replan=True,
            explanation="Typed text did not appear in target element. Refocusing and retyping.",
        )

    def _strategy_modal_blocked(self, retry_count: int) -> RecoveryPlan:
        """An unexpected modal appeared — dismiss it then replan."""
        return RecoveryPlan(
            failure_mode=FailureMode.MODAL_BLOCKED,
            steps=[
                # Attempt 1: Press ESC to dismiss dialog
                RecoveryStep(
                    action_type="press",
                    params={"key": "escape"},
                    reason="Pressing ESC to dismiss unexpected modal / dialog",
                ),
                # Brief wait for dialog close animation
                RecoveryStep(
                    action_type="wait",
                    params={"seconds": 0.5},
                    reason="Waiting for dialog close animation",
                ),
            ],
            should_replan=True,
            explanation=(
                "Unexpected modal dialog detected. Pressing ESC to dismiss it "
                "before replanning the current step."
            ),
        )

    def _strategy_navigation_error(
        self, retry_count: int, url: Optional[str]
    ) -> RecoveryPlan:
        """Browser failed to navigate — wait and reload."""
        return RecoveryPlan(
            failure_mode=FailureMode.NAVIGATION_ERROR,
            steps=[
                RecoveryStep(
                    action_type="wait",
                    params={"seconds": 2.0},
                    reason="Waiting 2s for network to recover",
                ),
                RecoveryStep(
                    action_type="press",
                    params={"key": "f5"},
                    reason="Pressing F5 to reload the page",
                ),
                RecoveryStep(
                    action_type="wait",
                    params={"seconds": 2.0},
                    reason="Waiting 2s for page to fully reload",
                ),
            ],
            should_replan=True,
            explanation=(
                f"Browser navigation to '{url}' failed. "
                f"Waiting and reloading before replanning."
            ),
        )

    def _strategy_app_launch_failed(
        self, retry_count: int, app_name: Optional[str]
    ) -> RecoveryPlan:
        """Application did not appear after APP_OPEN — wait and retry."""
        wait_secs = 1.5 + (0.5 * retry_count)
        steps: list = [
            RecoveryStep(
                action_type="wait",
                params={"seconds": wait_secs},
                reason=f"Waiting {wait_secs}s for application to start",
            )
        ]
        if app_name:
            steps.append(
                RecoveryStep(
                    action_type="app_open",
                    params={"app_name": app_name},
                    reason=f"Retrying launch of '{app_name}'",
                )
            )
        return RecoveryPlan(
            failure_mode=FailureMode.APP_LAUNCH_FAILED,
            steps=steps,
            should_replan=True,
            explanation=(
                f"Application '{app_name}' did not appear after launch. "
                f"Waiting {wait_secs}s then retrying."
            ),
        )

    def _strategy_unknown(self, retry_count: int) -> RecoveryPlan:
        """Ambiguous failure — conservative wait + full replan."""
        wait_secs = 1.5 * (retry_count + 1)
        return RecoveryPlan(
            failure_mode=FailureMode.UNKNOWN,
            steps=[
                RecoveryStep(
                    action_type="wait",
                    params={"seconds": wait_secs},
                    reason=f"Unknown failure — waiting {wait_secs}s before replanning",
                )
            ],
            should_replan=True,
            explanation=(
                f"Failure mode could not be classified. Waiting {wait_secs}s "
                f"and triggering a full replan of the current step."
            ),
        )
