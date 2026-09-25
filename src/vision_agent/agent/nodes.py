"""Node Implementations for SIVAC LangGraph Agent State Machine.

Each function acts as a pure LangGraph node receiving AgentState and returning
a partial state dictionary to update the graph context.
"""

import time
import logging
from typing import Dict, Any, Optional

from ..actions.schema import ActionCommand, ActionType, ActionResult
from ..actions.executor import ActionExecutor
from ..actions.safety import ActionSafetyGuard
from ..verifier.verifier import OutcomeVerifier
from ..verifier.schema import VerificationResult, FailureMode, RecoveryPlan
from ..perception.hybrid_merger import HybridPerceptionEngine
from ..perception.state import UIState
from ..memory.episodic import EpisodicMemoryManager
from ..memory.retrieval import ExperienceRetriever
from ..learning.self_improver import SelfImprovementPipeline
from .state import AgentState
from .planner import HierarchicalPlanner

logger = logging.getLogger("sivac.agent.nodes")


class AgentNodes:
    """Encapsulates node functions bound to subsystem instances."""

    def __init__(
        self,
        perception: Optional[HybridPerceptionEngine] = None,
        executor: Optional[ActionExecutor] = None,
        safety_guard: Optional[ActionSafetyGuard] = None,
        verifier: Optional[OutcomeVerifier] = None,
        episodic_memory: Optional[EpisodicMemoryManager] = None,
        experience_retriever: Optional[ExperienceRetriever] = None,
        planner: Optional[HierarchicalPlanner] = None,
        self_improver: Optional[SelfImprovementPipeline] = None,
        dry_run: bool = False,
    ) -> None:
        self.perception = perception or HybridPerceptionEngine()
        self.safety_guard = safety_guard or ActionSafetyGuard()
        self.executor = executor or ActionExecutor(safety_guard=self.safety_guard)
        self.verifier = verifier or OutcomeVerifier()
        self.episodic_memory = episodic_memory or EpisodicMemoryManager()
        self.experience_retriever = experience_retriever or ExperienceRetriever(episodic_memory=self.episodic_memory)
        self.self_improver = self_improver or SelfImprovementPipeline(
            db=self.episodic_memory.db,
            semantic_memory=self.experience_retriever.semantic,
        )
        self.planner = planner or HierarchicalPlanner()
        self.dry_run = dry_run

    # ------------------------------------------------------------------
    # 1. Initialize Node
    # ------------------------------------------------------------------

    def initialize(self, state: AgentState) -> Dict[str, Any]:
        """Record task start in relational episodic memory."""
        task_id = state.get("task_id")
        goal = state.get("goal", "")
        application = state.get("application", "Desktop")

        # Persist task initiation in SQLite
        self.episodic_memory.start_task(
            goal=goal,
            initial_app=application,
            task_id=task_id,
        )
        logger.info(f"[Node: Initialize] Task [{task_id}] initialized: '{goal}'")
        return {
            "step_count": 0,
            "retry_count": 0,
            "is_complete": False,
            "is_failed": False,
        }

    # ------------------------------------------------------------------
    # 2. Retrieve Experience Node
    # ------------------------------------------------------------------

    def retrieve(self, state: AgentState) -> Dict[str, Any]:
        """Query dual-layer memory for past task demonstrations and heuristics."""
        goal = state.get("goal", "")
        application = state.get("application", "Desktop")

        context = self.experience_retriever.retrieve_context(
            goal=goal,
            application=application,
        )
        guidance_text = self.experience_retriever.format_for_planner(context)

        skill_ids = [h.skill_id for h in context.heuristics if hasattr(h, "skill_id")]

        logger.debug(
            f"[Node: Retrieve] Found {len(context.similar_tasks)} past tasks & "
            f"{len(context.heuristics)} heuristics"
        )
        return {
            "retrieved_guidance": guidance_text,
            "retrieved_skill_ids": skill_ids,
        }

    # ------------------------------------------------------------------
    # 3. Perceive Node
    # ------------------------------------------------------------------

    def perceive(self, state: AgentState) -> Dict[str, Any]:
        """Capture screen buffer and build unified multi-modal UIState."""
        previous_ui = state.get("current_ui_state")

        # In dry-run or mock mode where UIState was pre-injected
        if self.dry_run and state.get("current_ui_state") is not None:
            return {
                "previous_ui_state": previous_ui,
                "current_ui_state": state["current_ui_state"],
            }

        try:
            # Capture live perception — skip VLM/OCR to avoid API-dependent failures
            is_web = "http" in state.get("goal", "") or state.get("application", "").lower() in ("chrome", "browser", "firefox", "edge")
            current_ui = self.perception.perceive(
                use_dom=is_web,
                use_uia=True,
                use_vlm=False,   # VLM disabled: requires external API — use UIA/DOM instead
                use_ocr=False,   # OCR disabled: slow and unreliable without GPU
            )
        except Exception as e:
            logger.warning(f"[Node: Perceive] Perception capture failed ({e}), using fallback state")
            current_ui = UIState(application=state.get("application", "Desktop"))


        return {
            "previous_ui_state": previous_ui,
            "current_ui_state": current_ui,
        }

    # ------------------------------------------------------------------
    # 4. Plan Node
    # ------------------------------------------------------------------

    def plan(self, state: AgentState) -> Dict[str, Any]:
        """Decide the immediate next subgoal and ActionCommand."""
        # Check if an autonomous recovery plan is currently queued
        recovery_plan = state.get("recovery_plan")
        if recovery_plan and recovery_plan.steps:
            # Pop next recovery step
            next_step = recovery_plan.steps.pop(0)
            action_type = ActionType(next_step.action_type)
            params = next_step.params

            recovery_cmd = ActionCommand(
                action=action_type,
                target_id=params.get("target_id"),
                x=params.get("x"),
                y=params.get("y"),
                text=params.get("text"),
                key=params.get("key"),
                keys=params.get("keys"),
                seconds=params.get("seconds", 1.0),
                reason=f"[Recovery Step] {next_step.reason}",
            )
            logger.info(f"[Node: Plan] Injected recovery action: {action_type.value} ({next_step.reason})")
            return {
                "planned_subgoal": f"Recovery: {next_step.reason}",
                "current_action": recovery_cmd,
                "recovery_plan": recovery_plan,
            }

        # Standard hierarchical planning
        subgoal, cmd = self.planner.plan_next_action(state)
        logger.info(f"[Node: Plan] Subgoal: '{subgoal}' -> Action: {cmd.action.value}")
        return {
            "planned_subgoal": subgoal,
            "current_action": cmd,
        }

    # ------------------------------------------------------------------
    # 5. Safety Validation Node
    # ------------------------------------------------------------------

    def validate_safety(self, state: AgentState) -> Dict[str, Any]:
        """Check planned command against security and boundary guardrails."""
        cmd: Optional[ActionCommand] = state.get("current_action")
        if not cmd:
            return {"is_safe": False, "safety_reason": "No action command provided"}

        is_safe, reason = self.safety_guard.validate_action(cmd)
        if not is_safe:
            logger.warning(f"[Node: Safety] Action blocked by guardrails: {reason}")
        else:
            logger.debug(f"[Node: Safety] Action approved: {cmd.action.value}")

        return {
            "is_safe": is_safe,
            "safety_reason": reason,
        }

    # ------------------------------------------------------------------
    # 6. Action Execution Node
    # ------------------------------------------------------------------

    def act(self, state: AgentState) -> Dict[str, Any]:
        """Dispatch approved ActionCommand to hardware / browser primitives."""
        cmd: ActionCommand = state["current_action"]
        current_ui = state.get("current_ui_state")

        # Execute command
        result: ActionResult = self.executor.execute(cmd, state=current_ui, dry_run=self.dry_run)

        # Update action and execution histories
        action_history = list(state.get("action_history", []))
        execution_history = list(state.get("execution_history", []))

        action_history.append(cmd.model_dump())
        execution_history.append(result.model_dump())

        logger.info(
            f"[Node: Act] Executed {cmd.action.value}: success={result.success} "
            f"({result.execution_time_ms:.1f}ms)"
        )
        return {
            "last_execution_result": result,
            "action_history": action_history,
            "execution_history": execution_history,
        }

    # ------------------------------------------------------------------
    # 7. Outcome Verification Node
    # ------------------------------------------------------------------

    def verify(self, state: AgentState) -> Dict[str, Any]:
        """Verify action outcome by inspecting visual and structural state deltas."""
        cmd: ActionCommand = state["current_action"]
        before_state: Optional[UIState] = state.get("current_ui_state")

        # Capture post-action state
        if self.dry_run:
            after_state = state.get("current_ui_state") or UIState()
        else:
            try:
                is_web = "http" in state.get("goal", "") or state.get("application", "").lower() in ("chrome", "browser")
                after_state = self.perception.perceive(
                    use_dom=is_web,
                    use_uia=True,
                    use_vlm=False,  # Skip VLM in verify to avoid slow/failing API calls
                    use_ocr=False,
                )
            except Exception as e:
                logger.warning(f"[Node: Verify] Post-action perception failed ({e}), reusing before_state")
                after_state = before_state or UIState()

        # Run two-tier verification
        v_res = self.verifier.verify_step(
            before=before_state or UIState(),
            after=after_state,
            cmd=cmd,
            retry_count=state.get("retry_count", 0),
        )

        verification_history = list(state.get("verification_history", []))
        verification_history.append(v_res.model_dump())

        logger.info(
            f"[Node: Verify] Step outcome: passed={v_res.passed} "
            f"(mode={v_res.failure_mode.value}, diff={v_res.visual_diff_score:.3f})"
        )
        return {
            "previous_ui_state": before_state,
            "current_ui_state": after_state,
            "last_verification_result": v_res,
            "verification_history": verification_history,
        }

    # ------------------------------------------------------------------
    # 8. Record Step Node
    # ------------------------------------------------------------------

    def record_step(self, state: AgentState) -> Dict[str, Any]:
        """Persist verified step and diagnostics into SQLite episodic store."""
        task_id = state["task_id"]
        step_number = state.get("step_count", 0) + 1
        cmd: ActionCommand = state["current_action"]
        exec_res: ActionResult = state["last_execution_result"]
        v_res: Optional[VerificationResult] = state.get("last_verification_result")

        # Persist audit record in SQLite
        self.episodic_memory.log_step(
            task_id=task_id,
            step_number=step_number,
            planned_subgoal=state.get("planned_subgoal", ""),
            action=cmd,
            result=exec_res,
            verification=v_res,
            before_state=state.get("previous_ui_state"),
            after_state=state.get("current_ui_state"),
        )

        logger.info(f"[Node: Record] Successfully audited Step {step_number} in SQLite")
        return {
            "step_count": step_number,
            "retry_count": 0,  # Reset retry counter on successful verified step
            "recovery_plan": None,
        }

    # ------------------------------------------------------------------
    # 9. Recovery Node
    # ------------------------------------------------------------------

    def recover(self, state: AgentState) -> Dict[str, Any]:
        """Formulate autonomous recovery plan for safety violations or failed steps."""
        retry_count = state.get("retry_count", 0) + 1
        max_retries = state.get("max_retries", 3)
        cmd: Optional[ActionCommand] = state.get("current_action")
        v_res: Optional[VerificationResult] = state.get("last_verification_result")

        # If failed due to safety violation
        if not state.get("is_safe", True):
            reason = state.get("safety_reason", "Action blocked by safety policy")
            logger.warning(f"[Node: Recover] Recovering from safety block: {reason}")
            # Generate wait and replan
            return {
                "retry_count": retry_count,
                "is_failed": retry_count >= max_retries,
                "error_summary": reason if retry_count >= max_retries else "",
                "recovery_plan": RecoveryPlan(
                    failure_mode=FailureMode.UNKNOWN,
                    steps=[],
                    should_replan=True,
                    explanation=f"Safety violation: {reason}. Replanning step.",
                ),
            }

        # Outcome verification failure
        if v_res and cmd:
            plan = self.verifier.generate_recovery(v_res, cmd)
            should_abort = plan.should_abort or (retry_count >= max_retries)
            logger.info(
                f"[Node: Recover] Generated recovery plan: mode={plan.failure_mode.value}, "
                f"steps={len(plan.steps)}, should_abort={should_abort}"
            )
            return {
                "retry_count": retry_count,
                "recovery_plan": plan,
                "is_failed": should_abort,
                "error_summary": plan.explanation if should_abort else "",
            }

        return {
            "retry_count": retry_count,
            "is_failed": retry_count >= max_retries,
        }

    # ------------------------------------------------------------------
    # 10. Finalize Node
    # ------------------------------------------------------------------

    def finalize(self, state: AgentState) -> Dict[str, Any]:
        """Finalize task status, persist trajectory, and index into ChromaDB."""
        task_id = state["task_id"]
        is_failed = state.get("is_failed", False)
        cmd: Optional[ActionCommand] = state.get("current_action")

        # If current action was FINISH and no failures flagged, task succeeded
        success = (cmd is not None and cmd.action == ActionType.FINISH) or (not is_failed)
        if is_failed:
            success = False

        total_steps = state.get("step_count", 0)
        application = state.get("application", "Desktop")

        # Finalize in SQLite & index in ChromaDB
        self.episodic_memory.complete_task(
            task_id=task_id,
            success=success,
            total_steps=total_steps,
            application=application,
        )

        # Trigger Post-Task Reflection & Self-Improvement Pipeline
        if self.self_improver:
            try:
                steps = self.episodic_memory.get_steps(task_id)
                self.self_improver.improve_from_task(
                    task_id=task_id,
                    goal=state.get("goal", ""),
                    application=application,
                    success=success,
                    steps=steps,
                    action_history=state.get("action_history", []),
                    verification_history=state.get("verification_history", []),
                    retrieved_skill_ids=state.get("retrieved_skill_ids", []),
                )
            except Exception as e:
                logger.warning(f"[Node: Finalize] Post-task self-improvement failed: {e}")

        logger.info(
            f"[Node: Finalize] Task [{task_id}] finalized: "
            f"success={success}, total_steps={total_steps}"
        )
        return {
            "is_complete": success,
            "is_failed": not success,
        }
