"""LangGraph State Graph Definition & SIVACAgent Orchestrator for Phase 6.

Constructs the compiled cyclic state machine coordinating perception, planning,
safety validation, action execution, outcome verification, and closed-loop recovery.
"""

import logging
from typing import Optional, Dict, Any, Generator

from langgraph.graph import StateGraph, START, END

from ..actions.schema import ActionType
from ..perception.hybrid_merger import HybridPerceptionEngine
from ..actions.executor import ActionExecutor
from ..actions.safety import ActionSafetyGuard
from ..verifier.verifier import OutcomeVerifier
from ..memory.episodic import EpisodicMemoryManager
from ..memory.retrieval import ExperienceRetriever
from .state import AgentState, create_initial_state
from .planner import HierarchicalPlanner
from .nodes import AgentNodes
from ..learning.self_improver import SelfImprovementPipeline

logger = logging.getLogger("sivac.agent.graph")


def build_agent_graph(nodes: AgentNodes):
    """Compile the LangGraph StateGraph governing the closed-loop agent lifecycle."""
    workflow = StateGraph(AgentState)

    # 1. Register Nodes
    workflow.add_node("initialize", nodes.initialize)
    workflow.add_node("retrieve", nodes.retrieve)
    workflow.add_node("perceive", nodes.perceive)
    workflow.add_node("plan", nodes.plan)
    workflow.add_node("validate_safety", nodes.validate_safety)
    workflow.add_node("act", nodes.act)
    workflow.add_node("verify", nodes.verify)
    workflow.add_node("record_step", nodes.record_step)
    workflow.add_node("recover", nodes.recover)
    workflow.add_node("finalize", nodes.finalize)

    # 2. Main Linear Trajectory Edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "retrieve")
    workflow.add_edge("retrieve", "perceive")
    workflow.add_edge("perceive", "plan")

    # 3. Conditional Routing after 'plan'
    def route_after_plan(state: AgentState) -> str:
        cmd = state.get("current_action")
        if cmd and cmd.action == ActionType.FINISH:
            return "finalize"
        return "validate_safety"

    workflow.add_conditional_edges(
        "plan",
        route_after_plan,
        {
            "finalize": "finalize",
            "validate_safety": "validate_safety",
        },
    )

    # 4. Conditional Routing after 'validate_safety'
    def route_after_safety(state: AgentState) -> str:
        if state.get("is_safe", True):
            return "act"
        return "recover"

    workflow.add_conditional_edges(
        "validate_safety",
        route_after_safety,
        {
            "act": "act",
            "recover": "recover",
        },
    )

    # 5. Fixed Edge after 'act' -> 'verify'
    workflow.add_edge("act", "verify")

    # 6. Conditional Routing after 'verify'
    def route_after_verify(state: AgentState) -> str:
        v_res = state.get("last_verification_result")
        if v_res and v_res.passed:
            return "record_step"
        return "recover"

    workflow.add_conditional_edges(
        "verify",
        route_after_verify,
        {
            "record_step": "record_step",
            "recover": "recover",
        },
    )

    # 7. Conditional Routing after 'recover'
    def route_after_recover(state: AgentState) -> str:
        if state.get("is_failed", False):
            return "finalize"
        return "plan"

    workflow.add_conditional_edges(
        "recover",
        route_after_recover,
        {
            "finalize": "finalize",
            "plan": "plan",
        },
    )

    # 8. Conditional Routing after 'record_step'
    def route_after_record(state: AgentState) -> str:
        if state.get("step_count", 0) >= state.get("max_steps", 25):
            return "finalize"
        return "perceive"

    workflow.add_conditional_edges(
        "record_step",
        route_after_record,
        {
            "finalize": "finalize",
            "perceive": "perceive",
        },
    )

    # 9. Final Edge
    workflow.add_edge("finalize", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# SIVACAgent Orchestrator
# ---------------------------------------------------------------------------

class SIVACAgent:
    """High-level autonomous computer-use agent orchestrator."""

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
        """Initialise SIVACAgent with all modular subsystems."""
        self.nodes = AgentNodes(
            perception=perception,
            executor=executor,
            safety_guard=safety_guard,
            verifier=verifier,
            episodic_memory=episodic_memory,
            experience_retriever=experience_retriever,
            planner=planner,
            self_improver=self_improver,
            dry_run=dry_run,
        )
        self.graph = build_agent_graph(self.nodes)
        self.dry_run = dry_run
        logger.info(f"SIVACAgent initialized (dry_run={dry_run})")

    def run(
        self,
        goal: str,
        application: str = "Desktop",
        max_steps: int = 25,
        max_retries: int = 3,
        task_id: Optional[str] = None,
    ) -> AgentState:
        """Execute task goal to completion through the LangGraph state machine.

        Args:
            goal: Natural language user instruction.
            application: Initial application or window context.
            max_steps: Upper safety limit on execution step count.
            max_retries: Consecutive recovery attempts before declaring failure.
            task_id: Optional specific task UUID.

        Returns:
            Final AgentState after graph execution terminates.
        """
        initial_state = create_initial_state(
            goal=goal,
            application=application,
            max_steps=max_steps,
            max_retries=max_retries,
            task_id=task_id,
        )
        logger.info(f"Running SIVACAgent task [{initial_state['task_id']}]: '{goal}'")
        final_state = self.graph.invoke(initial_state)
        return final_state

    def stream(
        self,
        goal: str,
        application: str = "Desktop",
        max_steps: int = 25,
        max_retries: int = 3,
        task_id: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream intermediate state changes for live dashboard monitoring."""
        initial_state = create_initial_state(
            goal=goal,
            application=application,
            max_steps=max_steps,
            max_retries=max_retries,
            task_id=task_id,
        )
        for update in self.graph.stream(initial_state):
            yield update
