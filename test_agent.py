"""Phase 6 verification script: Tests LangGraph Agent State Machine & Closed-Loop Controller.

Validates:
    1. AgentState: State allocation and data contract integrity.
    2. HierarchicalPlanner: Structured subgoal and ActionCommand generation.
    3. Safety Validation: Intercepting dangerous/out-of-bounds actions before execution.
    4. Closed-Loop Execution: End-to-end multi-step task execution terminating at FINISH.
    5. Autonomous Recovery: Modal blocker detection -> ESC injection -> replan recovery.
"""

import os
import tempfile
import chromadb

from vision_agent.perception import BBox, UIElement, UIState
from vision_agent.actions import ActionCommand, ActionType, ActionSafetyGuard
from vision_agent.verifier import OutcomeVerifier, FailureMode
from vision_agent.memory import (
    DatabaseManager,
    ChromaVectorStore,
    EpisodicMemoryManager,
    ExperienceRetriever,
)
from vision_agent.agent import (
    AgentState,
    create_initial_state,
    HierarchicalPlanner,
    AgentNodes,
    build_agent_graph,
    SIVACAgent,
)


def test_agent_state_initialization():
    print("\n--- 1. Testing AgentState Initialization ---")
    state = create_initial_state(
        goal="Download monthly bank statement in Chrome",
        application="Chrome",
        max_steps=10,
        max_retries=3,
    )
    assert state["task_id"]
    assert state["goal"] == "Download monthly bank statement in Chrome"
    assert state["application"] == "Chrome"
    assert state["step_count"] == 0
    assert state["max_steps"] == 10
    assert state["is_complete"] is False
    assert state["is_failed"] is False
    print(f"  [PASS] Initialized AgentState: task_id={state['task_id']}, goal='{state['goal']}'")


def test_hierarchical_planner():
    print("\n--- 2. Testing HierarchicalPlanner ---")
    planner = HierarchicalPlanner()

    # Case A: URL Goal
    state_url = create_initial_state(goal="Navigate to https://python.org to view docs")
    subgoal, cmd = planner.plan_next_action(state_url)
    assert cmd.action == ActionType.BROWSER_NAVIGATE
    assert cmd.url == "https://python.org"
    print(f"  [PASS] URL Planning: '{subgoal}' -> {cmd.action.value} (url={cmd.url})")

    # Case B: Desktop App Open Goal
    state_app = create_initial_state(goal="Launch notepad to draft notes")
    subgoal, cmd = planner.plan_next_action(state_app)
    assert cmd.action == ActionType.APP_OPEN
    assert cmd.app_name == "notepad"
    print(f"  [PASS] App Open Planning: '{subgoal}' -> {cmd.action.value} (app={cmd.app_name})")

    # Case C: Element Click Goal
    elem_search = UIElement(id="btn_search", type="button", text="Search", bbox=BBox(x_min=100, y_min=200, x_max=180, y_max=240))
    ui_state = UIState(elements=[elem_search])
    state_click = create_initial_state(goal="Click Search button")
    state_click["current_ui_state"] = ui_state
    subgoal, cmd = planner.plan_next_action(state_click)
    assert cmd.action == ActionType.CLICK
    assert cmd.target_id == "btn_search"
    print(f"  [PASS] Element Click Planning: '{subgoal}' -> {cmd.action.value} (target={cmd.target_id})")


def test_safety_validation_routing():
    print("\n--- 3. Testing Safety Validation Node & Routing ---")
    safety_guard = ActionSafetyGuard(screen_dimensions=(1920, 1080))
    nodes = AgentNodes(safety_guard=safety_guard, dry_run=True)

    # 1. Safe command
    safe_cmd = ActionCommand(action=ActionType.CLICK, x=500, y=500)
    state_safe = create_initial_state(goal="Click center")
    state_safe["current_action"] = safe_cmd
    res_safe = nodes.validate_safety(state_safe)
    assert res_safe["is_safe"] is True
    print(f"  [PASS] Approved safe action: {safe_cmd.action.value} at ({safe_cmd.x}, {safe_cmd.y})")

    # 2. Dangerous command (out of bounds)
    unsafe_cmd = ActionCommand(action=ActionType.CLICK, x=3500, y=500)
    state_unsafe = create_initial_state(goal="Click out of bounds")
    state_unsafe["current_action"] = unsafe_cmd
    res_unsafe = nodes.validate_safety(state_unsafe)
    assert res_unsafe["is_safe"] is False
    assert "outside screen bounds" in res_unsafe["safety_reason"]
    print(f"  [PASS] Blocked dangerous action: {res_unsafe['safety_reason']}")


def test_closed_loop_agent_execution():
    print("\n--- 4. Testing End-to-End Closed-Loop Agent Execution ---")
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = DatabaseManager(db_url=f"sqlite:///{db_path}")
    chroma_client = chromadb.EphemeralClient()
    vector_store = ChromaVectorStore(client=chroma_client)
    episodic = EpisodicMemoryManager(db=db, vector_store=vector_store)
    retriever = ExperienceRetriever(episodic_memory=episodic)

    try:
        # Construct UI elements that will be interacted with
        elem_search = UIElement(id="btn_download", type="button", text="Download Report", bbox=BBox(x_min=100, y_min=200, x_max=240, y_max=240))
        test_ui = UIState(elements=[elem_search], url="https://reports.internal/q3")

        agent = SIVACAgent(
            episodic_memory=episodic,
            experience_retriever=retriever,
            dry_run=True,
        )

        # Pre-seed initial UIState by overriding perceive node or running task
        initial_state = create_initial_state(
            goal="Download Report from reports page",
            application="Chrome",
            max_steps=5,
        )
        initial_state["current_ui_state"] = test_ui

        # Execute full graph
        final_state = agent.graph.invoke(initial_state)

        assert final_state["is_complete"] is True
        assert final_state["step_count"] >= 1
        assert len(final_state["action_history"]) >= 1
        print(
            f"  [PASS] Agent task completed successfully: steps={final_state['step_count']}, "
            f"actions={[a['action'] for a in final_state['action_history']]}"
        )

        # Verify audit was persisted in SQLite
        steps = episodic.get_steps(final_state["task_id"])
        assert len(steps) >= 1
        print(f"  [PASS] Verified {len(steps)} steps persisted in relational audit store")

    finally:
        db.close()
        if os.path.exists(db_path):
            os.remove(db_path)


def test_autonomous_recovery_closed_loop():
    print("\n--- 5. Testing Autonomous Recovery Closed-Loop ---")
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = DatabaseManager(db_url=f"sqlite:///{db_path}")
    chroma_client = chromadb.EphemeralClient()
    vector_store = ChromaVectorStore(client=chroma_client)
    episodic = EpisodicMemoryManager(db=db, vector_store=vector_store)

    try:
        nodes = AgentNodes(
            episodic_memory=episodic,
            dry_run=True,
        )

        # Simulate state where an unexpected modal appears
        modal_elem = UIElement(id="alert_cookie", type="dialog", text="Accept Cookies or Dismiss", bbox=BBox(x_min=50, y_min=50, x_max=300, y_max=200))
        state_with_modal = UIState(elements=[modal_elem])

        cmd_click = ActionCommand(action=ActionType.CLICK, x=100, y=200)

        # Verify step triggers MODAL_BLOCKED
        v_res = nodes.verifier.verify_step(
            before=UIState(elements=[]),
            after=state_with_modal,
            cmd=cmd_click,
        )
        assert v_res.passed is False
        assert v_res.failure_mode == FailureMode.MODAL_BLOCKED
        print(f"  [PASS] Detected failure mode: {v_res.failure_mode.value}")

        # Recover node produces recovery plan with ESC key
        state_recovery = create_initial_state(goal="Dismiss popup and proceed")
        state_recovery["current_action"] = cmd_click
        state_recovery["last_verification_result"] = v_res

        recovery_dict = nodes.recover(state_recovery)
        plan = recovery_dict["recovery_plan"]
        assert plan is not None
        assert plan.failure_mode == FailureMode.MODAL_BLOCKED
        assert any(s.action_type == "press" for s in plan.steps)
        print(f"  [PASS] Recover node generated corrective steps: {[s.action_type for s in plan.steps]}")

        # Plan node injects first recovery step
        state_recovery["recovery_plan"] = plan
        plan_dict = nodes.plan(state_recovery)
        injected_cmd = plan_dict["current_action"]
        assert injected_cmd.action == ActionType.PRESS_KEY
        assert injected_cmd.key == "escape"
        print(f"  [PASS] Plan node injected corrective recovery command: {injected_cmd.action.value} key={injected_cmd.key}")

    finally:
        db.close()
        if os.path.exists(db_path):
            os.remove(db_path)


if __name__ == "__main__":
    print("==================================================================")
    print("     SIVAC Phase 6 Verification: LangGraph Agent State Machine    ")
    print("==================================================================")
    test_agent_state_initialization()
    test_hierarchical_planner()
    test_safety_validation_routing()
    test_closed_loop_agent_execution()
    test_autonomous_recovery_closed_loop()
    print("\n[ALL PHASE 6 TESTS PASSED SUCCESSFULLY!]")
