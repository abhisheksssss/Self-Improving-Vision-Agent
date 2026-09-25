"""Agent State Definition for SIVAC Phase 6 (LangGraph Orchestration).

Defines the centralized AgentState dictionary tracked and transformed across
all LangGraph nodes in the closed-loop execution lifecycle.
"""

import uuid
from typing import Optional, List, Dict, Any
from typing_extensions import TypedDict

from ..perception.state import UIState
from ..actions.schema import ActionCommand, ActionResult
from ..verifier.schema import VerificationResult, RecoveryPlan


class AgentState(TypedDict, total=False):
    """The complete state dictionary passed across LangGraph nodes."""

    # 1. Identity & Objective
    task_id: str
    goal: str
    application: str
    step_count: int
    max_steps: int

    # 2. Perception Snapshots
    current_ui_state: Optional[UIState]
    previous_ui_state: Optional[UIState]

    # 3. Planning & Safety
    planned_subgoal: str
    current_action: Optional[ActionCommand]
    is_safe: bool
    safety_reason: str

    # 4. Action Execution & Verification
    last_execution_result: Optional[ActionResult]
    last_verification_result: Optional[VerificationResult]

    # 5. Recovery & Closed-Loop Control
    recovery_plan: Optional[RecoveryPlan]
    retry_count: int
    max_retries: int

    # 6. Memory & Retrieval Guidance
    retrieved_guidance: str
    retrieved_skill_ids: List[str]
    action_history: List[Dict[str, Any]]
    execution_history: List[Dict[str, Any]]
    verification_history: List[Dict[str, Any]]

    # 7. Final Lifecycle Flags
    is_complete: bool
    is_failed: bool
    error_summary: str


def create_initial_state(
    goal: str,
    application: str = "Desktop",
    max_steps: int = 25,
    max_retries: int = 3,
    task_id: Optional[str] = None,
) -> AgentState:
    """Helper to allocate a pristine initial AgentState."""
    return {
        "task_id": task_id or str(uuid.uuid4()),
        "goal": goal,
        "application": application,
        "step_count": 0,
        "max_steps": max_steps,
        "current_ui_state": None,
        "previous_ui_state": None,
        "planned_subgoal": "",
        "current_action": None,
        "is_safe": True,
        "safety_reason": "",
        "last_execution_result": None,
        "last_verification_result": None,
        "recovery_plan": None,
        "retry_count": 0,
        "max_retries": max_retries,
        "retrieved_guidance": "",
        "retrieved_skill_ids": [],
        "action_history": [],
        "execution_history": [],
        "verification_history": [],
        "is_complete": False,
        "is_failed": False,
        "error_summary": "",
    }
