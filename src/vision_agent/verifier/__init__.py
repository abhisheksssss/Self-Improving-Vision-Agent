"""Phase 4: Outcome Verifier & Closed-Loop Recovery Engine for SIVAC.

This package closes the agent loop by verifying every action outcome
before the planner moves to the next step.

Components:
    - visual_diff   : Perceptual image hashing & pixel SSIM comparison
    - state_diff    : Structural UIState element-set delta comparison
    - classifier    : Failure mode classifier (UI_LAG, MISSED_CLICK, etc.)
    - recovery      : Autonomous recovery action plan generator
    - verifier      : Unified OutcomeVerifier orchestrating all above
"""

from .schema import (
    VerificationResult,
    FailureMode,
    RecoveryPlan,
    RecoveryStep,
)
from .visual_diff import VisualDiffEngine
from .state_diff import StateDiffEngine
from .classifier import FailureClassifier
from .recovery import RecoveryEngine
from .verifier import OutcomeVerifier

__all__ = [
    "VerificationResult",
    "FailureMode",
    "RecoveryPlan",
    "RecoveryStep",
    "VisualDiffEngine",
    "StateDiffEngine",
    "FailureClassifier",
    "RecoveryEngine",
    "OutcomeVerifier",
]
