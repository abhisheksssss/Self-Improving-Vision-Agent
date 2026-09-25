"""Failure Mode Classifier for SIVAC Phase 4.

Takes the outputs of VisualDiffEngine and StateDiffEngine together with
the original ActionCommand and deterministically maps the combined signal
to a single FailureMode enum value.

Classification Rules (evaluated top-to-bottom, first match wins):
┌──────────────────────────────────────────────────────────────────────────────┐
│ Rule │ Visual Score │ State Delta │ Action Type     │ → FailureMode          │
├──────┼──────────────┼─────────────┼─────────────────┼────────────────────────┤
│  1   │ any          │ modal appeared │ any           │ MODAL_BLOCKED          │
│  2   │ ≥ threshold  │ ≥ 1 change  │ any             │ NONE  (success)        │
│  3   │ ≥ threshold  │ 0 changes   │ CLICK           │ MISSED_CLICK           │
│  4   │ ≥ threshold  │ 0 changes   │ TYPE_TEXT       │ TYPE_FAILED            │
│  5   │ ≥ threshold  │ 0 changes   │ BROWSER_NAVIGATE│ NAVIGATION_ERROR       │
│  6   │ ≥ threshold  │ 0 changes   │ APP_OPEN        │ APP_LAUNCH_FAILED      │
│  7   │ 0–lag_thresh │ any         │ any             │ UI_LAG                 │
│  8   │ ≈ 0          │ 0 changes   │ any             │ NO_VISUAL_CHANGE       │
│  9   │ (fallback)   │ (fallback)  │ any             │ UNKNOWN                │
└──────────────────────────────────────────────────────────────────────────────┘
"""

import logging
from .schema import FailureMode
from .visual_diff import VisualDiffEngine
from .state_diff import ElementDelta

logger = logging.getLogger("sivac.verifier.classifier")


class FailureClassifier:
    """Deterministic rule engine that maps diff signals to a FailureMode.

    Designed to be stateless and called fresh for every verification step.
    """

    def __init__(self, visual_engine: VisualDiffEngine) -> None:
        """Initialise with a configured VisualDiffEngine to reuse its thresholds.

        Args:
            visual_engine: VisualDiffEngine instance (threshold values are read from it).
        """
        self._engine = visual_engine

    def classify(
        self,
        visual_score: float,
        delta: ElementDelta,
        action_type: str,
    ) -> FailureMode:
        """Classify the failure mode given diff signals and the action type.

        Args:
            visual_score: Blended perceptual change score from VisualDiffEngine (0–1).
            delta:        Structural ElementDelta from StateDiffEngine.
            action_type:  String value of the ActionType that was executed.

        Returns:
            FailureMode enum value (NONE = success, anything else = failure).
        """
        action = action_type.lower()

        is_meaningful = self._engine.is_meaningfully_changed(visual_score)
        has_state_change = delta.total_changes > 0

        # Rule 1 — Modal / unexpected dialog blocked the action
        if delta.has_modal:
            logger.info("Classifier -> MODAL_BLOCKED: unexpected dialog appeared")
            return FailureMode.MODAL_BLOCKED

        # Rule 2 — Structural state delta and/or visual confirmation verifies success
        if has_state_change:
            logger.debug(
                f"Classifier -> NONE (success): score={visual_score:.3f}, "
                f"delta_elements={delta.total_changes}"
            )
            return FailureMode.NONE

        # Rules 3–6 — Screen changed visually but structure did not update as expected
        if is_meaningful and not has_state_change:
            if "click" in action or "double_click" in action or "right_click" in action:
                logger.info("Classifier -> MISSED_CLICK: visual changed but no element delta")
                return FailureMode.MISSED_CLICK
            if "type" in action:
                logger.info("Classifier -> TYPE_FAILED: text did not appear in target element")
                return FailureMode.TYPE_FAILED
            if "navigate" in action or "browser" in action:
                logger.info("Classifier -> NAVIGATION_ERROR: page did not update")
                return FailureMode.NAVIGATION_ERROR
            if "app_open" in action:
                logger.info("Classifier -> APP_LAUNCH_FAILED: application did not appear")
                return FailureMode.APP_LAUNCH_FAILED

        # Rule 7 — Screen changed slightly (loading spinner, cursor blink, micro-animation)
        if self._engine.is_lagging(visual_score):
            logger.info(
                f"Classifier -> UI_LAG: minor visual change (score={visual_score:.3f}), "
                f"likely still loading"
            )
            return FailureMode.UI_LAG

        # Rule 8 — Absolutely nothing changed
        if self._engine.is_unchanged(visual_score) and delta.total_changes == 0:
            logger.info(
                "Classifier -> NO_VISUAL_CHANGE: screen identical, no element delta"
            )
            return FailureMode.NO_VISUAL_CHANGE

        # Rule 9 — Fallback for ambiguous signals
        logger.warning(
            f"Classifier -> UNKNOWN: could not classify "
            f"(score={visual_score:.3f}, delta={delta.total_changes})"
        )
        return FailureMode.UNKNOWN
