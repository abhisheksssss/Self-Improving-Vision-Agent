"""Phase 5 verification script: Tests Multi-Level Memory System.

Validates:
    1. DatabaseManager: SQLite relational models, transactions, and foreign-key cascades.
    2. ChromaVectorStore: Persistent / ephemeral vector collections and cosine similarity search.
    3. EpisodicMemoryManager: Task lifecycle (start, step audit logging, completion indexing).
    4. SemanticMemoryManager: Operational heuristics, rule indexing, and feedback reinforcement.
    5. ExperienceRetriever: Context-aware retrieval and LLM prompt formatting.
"""

import os
import tempfile
import chromadb

from vision_agent.perception import BBox, UIElement, UIState
from vision_agent.actions import ActionCommand, ActionType, ActionResult
from vision_agent.verifier import VerificationResult, FailureMode
from vision_agent.memory import (
    TaskStatus,
    DatabaseManager,
    ChromaVectorStore,
    EpisodicMemoryManager,
    SemanticMemoryManager,
    ExperienceRetriever,
)


def test_database_manager():
    print("\n--- 1. Testing DatabaseManager (SQLite Relational Store) ---")
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db_url = f"sqlite:///{db_path}"

    try:
        db = DatabaseManager(db_url=db_url)

        # 1. Create Task
        task = db.create_task(goal="Test downloading an invoice in Chrome")
        assert task.id and task.goal == "Test downloading an invoice in Chrome"
        assert task.status == TaskStatus.RUNNING
        print(f"  [PASS] Created Task: id={task.id}, status={task.status.value}")

        # 2. Create Trajectory
        traj = db.create_trajectory(task_id=task.id, model_used="test-model-v1")
        assert traj.task_id == task.id
        print(f"  [PASS] Created Trajectory: id={traj.id}")

        # 3. Add Step
        step = db.add_step(
            trajectory_id=traj.id,
            step_number=1,
            screenshot_before="data/screen_before.png",
            planned_subgoal="Click the download button",
            action_command={"action": "click", "x": 100, "y": 200},
            execution_result={"success": True, "execution_time_ms": 45.2},
        )
        assert step.step_number == 1
        assert step.action_command.get("action") == "click"
        print(f"  [PASS] Added Step {step.step_number}: id={step.id}")

        # 4. Add Verification linked to step
        verif = db.add_verification(
            step_id=step.id,
            passed=True,
            visual_diff_score=0.18,
            dom_delta={"appeared": ["dlg_success"]},
        )
        assert verif.passed is True
        assert verif.visual_diff_score == 0.18
        print(f"  [PASS] Added Verification: passed={verif.passed}, diff={verif.visual_diff_score}")

        # 5. Retrieve steps
        steps = db.get_steps_for_trajectory(traj.id)
        assert len(steps) == 1
        assert steps[0].planned_subgoal == "Click the download button"
        print(f"  [PASS] Retrieved {len(steps)} steps for trajectory")

        # 6. Update Task Status
        completed_task = db.update_task_status(
            task_id=task.id,
            status=TaskStatus.COMPLETED,
            total_steps=1,
            total_duration_seconds=5.2,
        )
        assert completed_task is not None
        assert completed_task.status == TaskStatus.COMPLETED
        assert completed_task.total_duration_seconds == 5.2
        print(f"  [PASS] Updated Task status: {completed_task.status.value}")

        # 7. Cascade Deletion
        deleted = db.delete_task(task.id)
        assert deleted is True
        assert db.get_task(task.id) is None
        assert db.get_trajectory(task.id) is None
        assert len(db.get_steps_for_trajectory(traj.id)) == 0
        print("  [PASS] Cascade deletion cleaned up trajectories and steps")

    finally:
        db.close()
        if os.path.exists(db_path):
            os.remove(db_path)


def test_chroma_vector_store():
    print("\n--- 2. Testing ChromaVectorStore (Semantic Vector Indexing) ---")
    chroma_client = chromadb.EphemeralClient()
    vector_store = ChromaVectorStore(client=chroma_client)

    # 1. Index Tasks
    vector_store.index_task(
        task_id="task_chrome_pdf",
        goal="Download invoice PDF document in Google Chrome",
        application="Chrome",
        status="COMPLETED",
        total_steps=4,
        duration_seconds=12.5,
    )
    vector_store.index_task(
        task_id="task_calc_math",
        goal="Open Windows Calculator and compute monthly budget sum",
        application="Calculator",
        status="COMPLETED",
        total_steps=6,
        duration_seconds=15.0,
    )
    assert vector_store.count_tasks() == 2
    print(f"  [PASS] Indexed {vector_store.count_tasks()} tasks into episodic_trajectories")

    # 2. Query Similar Tasks (Semantic cosine search)
    results = vector_store.query_similar_tasks(
        goal="Retrieve invoice in Chrome browser",
        top_k=1,
    )
    assert len(results) == 1
    assert results[0].task_id == "task_chrome_pdf"
    assert results[0].similarity_score > 0.4
    print(f"  [PASS] Vector query match: '{results[0].goal}' (score={results[0].similarity_score:.3f})")

    # 3. Index Heuristic Rule
    vector_store.index_heuristic(
        skill_id="skill_chrome_shelf",
        application_name="Chrome",
        trigger_condition="File download shelf appears at screen bottom",
        heuristic_rule="Wait 2.0s for file save animation to complete before clicking",
        confidence_score=0.85,
    )
    assert vector_store.count_heuristics() == 1
    print(f"  [PASS] Indexed {vector_store.count_heuristics()} heuristic into learned_heuristics")

    # 4. Query Heuristic Rule
    h_matches = vector_store.query_heuristics(
        query_text="file download shelf animation waiting",
        application_name="Chrome",
        top_k=1,
    )
    assert len(h_matches) == 1
    assert h_matches[0].skill_id == "skill_chrome_shelf"
    print(f"  [PASS] Heuristic match: '{h_matches[0].heuristic_rule}'")


def test_episodic_memory_lifecycle():
    print("\n--- 3. Testing EpisodicMemoryManager Lifecycle ---")
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = DatabaseManager(db_url=f"sqlite:///{db_path}")
    vector_store = ChromaVectorStore(client=chromadb.EphemeralClient())
    episodic = EpisodicMemoryManager(db=db, vector_store=vector_store)

    try:
        # Start Task
        task_id = episodic.start_task(
            goal="Book a table on reservation site",
            initial_app="Chrome",
        )
        assert task_id

        # Log Step
        elem = UIElement(id="btn_reserve", type="button", text="Find Table", bbox=BBox(x_min=100, y_min=200, x_max=200, y_max=250))
        before_state = UIState(elements=[elem], screenshot_path="data/step1_before.png")
        after_state = UIState(elements=[], screenshot_path="data/step1_after.png")
        cmd = ActionCommand(action=ActionType.CLICK, x=150, y=225, target_id="btn_reserve")
        action_res = ActionResult(success=True, action=ActionType.CLICK, details="Clicked successfully")
        verif_res = VerificationResult(passed=True, visual_diff_score=0.22, details="Verified state change")

        step_record = episodic.log_step(
            task_id=task_id,
            step_number=1,
            planned_subgoal="Click the reservation button",
            action=cmd,
            result=action_res,
            verification=verif_res,
            before_state=before_state,
            after_state=after_state,
        )
        assert step_record.step_number == 1
        print(f"  [PASS] Logged Step {step_record.step_number} in Episodic Memory")

        # Complete Task
        completed_task = episodic.complete_task(
            task_id=task_id,
            success=True,
            duration_seconds=8.4,
            application="Chrome",
        )
        assert completed_task is not None
        assert completed_task.status == TaskStatus.COMPLETED
        print(f"  [PASS] Completed Task: status={completed_task.status.value}, steps={completed_task.total_steps}")

        # Semantic Query via Episodic Manager
        similar = episodic.query_similar_tasks(goal="Reserve a table online", top_k=1)
        assert len(similar) == 1
        assert similar[0].task_id == task_id
        print(f"  [PASS] Found past task via EpisodicManager: '{similar[0].goal}'")

    finally:
        episodic.db.close()
        if os.path.exists(db_path):
            os.remove(db_path)


def test_semantic_memory_lifecycle():
    print("\n--- 4. Testing SemanticMemoryManager Lifecycle ---")
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = DatabaseManager(db_url=f"sqlite:///{db_path}")
    vector_store = ChromaVectorStore(client=chromadb.EphemeralClient())
    semantic = SemanticMemoryManager(db=db, vector_store=vector_store)

    try:
        # Store Heuristic
        skill = semantic.store_heuristic(
            application_name="Excel",
            trigger_condition="Dialog asking to enable macros appears",
            heuristic_rule="Click 'Enable Content' yellow warning banner before editing",
            confidence_score=0.80,
        )
        assert skill.application_name == "Excel"
        print(f"  [PASS] Stored heuristic: id={skill.id}, confidence={skill.confidence_score}")

        # Reinforce with positive feedback
        reinforced = semantic.record_feedback(skill.id, success=True)
        assert reinforced is not None
        assert reinforced.success_count == 2
        assert round(reinforced.confidence_score, 2) == 0.85
        print(f"  [PASS] Positive feedback reinforced confidence to: {reinforced.confidence_score:.2f}")

        # Penalize with negative feedback
        penalized = semantic.record_feedback(skill.id, success=False)
        assert penalized is not None
        assert penalized.failure_count == 1
        assert round(penalized.confidence_score, 2) == 0.75
        print(f"  [PASS] Negative feedback adjusted confidence to: {penalized.confidence_score:.2f}")

        # Query Heuristics
        matches = semantic.query_heuristics(
            query_context="enable macros warning dialog in spreadsheet",
            application_name="Excel",
            top_k=1,
        )
        assert len(matches) == 1
        assert matches[0].skill_id == skill.id
        print(f"  [PASS] Semantic query retrieved correct heuristic: '{matches[0].heuristic_rule}'")

    finally:
        semantic.db.close()
        if os.path.exists(db_path):
            os.remove(db_path)


def test_experience_retriever():
    print("\n--- 5. Testing ExperienceRetriever & Prompt Formatting ---")
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = DatabaseManager(db_url=f"sqlite:///{db_path}")
    vector_store = ChromaVectorStore(client=chromadb.EphemeralClient())
    episodic = EpisodicMemoryManager(db=db, vector_store=vector_store)
    semantic = SemanticMemoryManager(db=db, vector_store=vector_store)

    try:
        # Seed an episodic demonstration
        tid = episodic.start_task(goal="Export monthly report as PDF in Chrome", initial_app="Chrome")
        episodic.complete_task(task_id=tid, success=True, duration_seconds=10.0, total_steps=3, application="Chrome")

        # Seed a semantic heuristic
        semantic.store_heuristic(
            application_name="Chrome",
            trigger_condition="Print preview modal loading",
            heuristic_rule="Wait 1.5s for print preview to render before pressing Enter",
            confidence_score=0.90,
        )

        retriever = ExperienceRetriever(episodic_memory=episodic, semantic_memory=semantic)

        # Retrieve context for a new similar goal
        context = retriever.retrieve_context(
            goal="Export expense report to PDF in Chrome",
            application="Chrome",
            screen_context="Print preview dialog open",
        )
        assert len(context.similar_tasks) >= 1
        assert len(context.heuristics) >= 1
        print(f"  [PASS] Retrieved {len(context.similar_tasks)} past tasks & {len(context.heuristics)} heuristics")

        # Format for LangGraph planner prompt
        prompt_block = retriever.format_for_planner(context)
        assert "## Experience-Informed Prior Guidance" in prompt_block
        assert "Relevant Prior Task Experiences" in prompt_block
        assert "Export monthly report as PDF" in prompt_block
        assert "Learned Interaction Rules & Heuristics" in prompt_block
        assert "Print preview modal loading" in prompt_block
        print("  [PASS] Formatted LLM Prompt Block:")
        for line in prompt_block.splitlines():
            print(f"    | {line}")

    finally:
        episodic.db.close()
        if os.path.exists(db_path):
            os.remove(db_path)


if __name__ == "__main__":
    print("==================================================================")
    print("        SIVAC Phase 5 Verification: Multi-Level Memory System     ")
    print("==================================================================")
    test_database_manager()
    test_chroma_vector_store()
    test_episodic_memory_lifecycle()
    test_semantic_memory_lifecycle()
    test_experience_retriever()
    print("\n[ALL PHASE 5 TESTS PASSED SUCCESSFULLY!]")
