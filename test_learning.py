"""Phase 7 verification script: Tests Self-Improvement & Reflection Engine.

Validates:
    1. TrajectoryEvaluator: Friction point detection and efficiency scoring.
    2. ReflectionEngine: Cognitive post-mortem analysis and SQLite reflection persistence.
    3. LessonExtractor: Heuristic lesson extraction, semantic deduplication, and persistence.
    4. StrategyManager: Dynamic modality selection (DOM vs UIA vs Hybrid) and confidence reinforcement.
    5. Closed-Loop Self-Improvement: Automated learning pipeline within LangGraph agent execution.
"""

import os
import tempfile
import chromadb

from vision_agent.perception import BBox, UIElement, UIState
from vision_agent.actions import ActionCommand, ActionType
from vision_agent.verifier import VerificationResult, FailureMode
from vision_agent.memory import (
    DatabaseManager,
    ChromaVectorStore,
    EpisodicMemoryManager,
    SemanticMemoryManager,
    ExperienceRetriever,
    LearnedSkill,
)
from vision_agent.learning import (
    TrajectoryEvaluation,
    ExtractedLesson,
    ReflectionOutput,
    TrajectoryEvaluator,
    ReflectionEngine,
    LessonExtractor,
    StrategyManager,
    SelfImprovementPipeline,
)
from vision_agent.agent import (
    AgentState,
    create_initial_state,
    HierarchicalPlanner,
    AgentNodes,
    build_agent_graph,
    SIVACAgent,
)


def test_trajectory_evaluator():
    print("\n--- 1. Testing TrajectoryEvaluator (Friction & Efficiency) ---")
    evaluator = TrajectoryEvaluator()

    # Case A: Friction Trajectory
    verif_history = [
        {"passed": True, "failure_mode": "none"},
        {"passed": False, "failure_mode": "modal_blocked"},
        {"passed": False, "failure_mode": "type_failed"},
        {"passed": True, "failure_mode": "none"},
    ]
    action_history = [
        {"action": "click", "x": 150, "y": 250, "reason": "Initial click"},
        {"action": "press_key", "key": "escape", "reason": "[Recovery Step] Dismiss modal"},
        {"action": "click", "x": 300, "y": 400, "reason": "Retry click"},
        {"action": "click", "x": 300, "y": 400, "reason": "Identical repeat click"},
    ]

    evaluation = evaluator.evaluate(
        task_id="eval-task-001",
        verification_history=verif_history,
        action_history=action_history,
    )

    assert evaluation.task_id == "eval-task-001"
    assert evaluation.total_steps == 4
    assert evaluation.recovery_count >= 2
    assert evaluation.overall_efficiency < 1.0
    assert len(evaluation.friction_points) >= 2
    assert len(evaluation.recommendations) >= 1

    print(
        f"  [PASS] Friction evaluated: efficiency={evaluation.overall_efficiency:.2f}, "
        f"recoveries={evaluation.recovery_count}, friction_points={len(evaluation.friction_points)}"
    )
    for fp in evaluation.friction_points:
        print(f"         - {fp}")

    # Case B: Clean Trajectory
    clean_verif = [
        {"passed": True, "failure_mode": "none"},
        {"passed": True, "failure_mode": "none"},
    ]
    clean_eval = evaluator.evaluate(
        task_id="eval-task-clean",
        verification_history=clean_verif,
        action_history=[{"action": "click"}, {"action": "finish"}],
    )
    assert clean_eval.overall_efficiency == 1.0
    assert clean_eval.recovery_count == 0
    assert len(clean_eval.friction_points) == 0
    print(f"  [PASS] Clean trajectory: efficiency={clean_eval.overall_efficiency:.2f}, recoveries=0")


def test_reflection_engine():
    print("\n--- 2. Testing ReflectionEngine (Cognitive Post-Mortem) ---")
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = DatabaseManager(db_url=f"sqlite:///{db_path}")

    try:
        # Pre-create task record to satisfy foreign key requirement
        db.create_task(goal="Sign in to user account", task_id="refl-task-001")

        engine = ReflectionEngine(db=db, model_override="fallback")
        evaluator = TrajectoryEvaluator()

        verif_history = [{"passed": False, "failure_mode": "modal_blocked"}]
        action_history = [{"action": "click", "x": 200, "y": 300, "reason": "Click link"}]

        eval_res = evaluator.evaluate(
            task_id="refl-task-001",
            verification_history=verif_history,
            action_history=action_history,
        )

        reflection = engine.reflect(
            task_id="refl-task-001",
            goal="Sign in to user account",
            application="Chrome",
            overall_success=False,
            evaluation=eval_res,
            action_history=action_history,
            verification_history=verif_history,
        )

        assert reflection.task_id == "refl-task-001"
        assert len(reflection.root_cause_analysis) > 0
        assert len(reflection.lessons) >= 1

        lesson = reflection.lessons[0]
        assert lesson.application_name == "Chrome"
        assert len(lesson.trigger_condition) > 0
        assert len(lesson.heuristic_rule) > 0
        assert 0.0 <= lesson.confidence_score <= 1.0

        print(
            f"  [PASS] Reflection output generated:\n"
            f"         - Analysis: {reflection.root_cause_analysis[:80]}...\n"
            f"         - Lesson: IF [{lesson.trigger_condition}] THEN [{lesson.heuristic_rule}]"
        )

        # Verify persisted in SQLite
        from vision_agent.memory.storage import ReflectionModel
        with db.session_scope() as session:
            record = session.query(ReflectionModel).filter_by(task_id="refl-task-001").first()
            assert record is not None
            assert record.overall_success == 0
            assert len(record.root_cause_analysis) > 0
            assert "modal" in record.friction_points.lower()
            print(f"  [PASS] Confirmed ReflectionRecord persisted in SQLite (id={record.id})")

        # Test LLM JSON Parsing with Mock Model
        class MockLLMResponse:
            content = """{
                "root_cause_analysis": "Target button was obscured by modal overlay.",
                "friction_points": ["Modal blocker popup"],
                "lessons": [
                    {
                        "application_name": "Chrome",
                        "trigger_condition": "Modal dialog blocking elements",
                        "heuristic_rule": "Send ESC key to close modal",
                        "confidence_score": 0.92,
                        "rationale": "Clears overlay cleanly"
                    }
                ]
            }"""

        class MockReasoningModel:
            def invoke(self, messages):
                return MockLLMResponse()

        db.create_task(goal="Sign in with mock LLM", task_id="refl-task-llm")
        engine_llm = ReflectionEngine(db=db, model_override=MockReasoningModel())
        reflection_llm = engine_llm.reflect(
            task_id="refl-task-llm",
            goal="Sign in with mock LLM",
            application="Chrome",
            overall_success=True,
            evaluation=eval_res,
        )
        assert len(reflection_llm.lessons) == 1
        assert reflection_llm.lessons[0].heuristic_rule == "Send ESC key to close modal"
        assert reflection_llm.lessons[0].confidence_score == 0.92
        print("  [PASS] Verified LLM structured JSON parsing in ReflectionEngine")

    finally:
        db.close()
        if os.path.exists(db_path):
            os.remove(db_path)


def test_lesson_extractor_and_deduplication():
    print("\n--- 3. Testing LessonExtractor & Heuristic Deduplication ---")
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = DatabaseManager(db_url=f"sqlite:///{db_path}")
    chroma_client = chromadb.EphemeralClient()
    vector_store = ChromaVectorStore(client=chroma_client)
    semantic = SemanticMemoryManager(db=db, vector_store=vector_store)
    extractor = LessonExtractor(semantic_memory=semantic)

    try:
        # Lesson A: New Heuristic Rule
        lesson_a = ExtractedLesson(
            application_name="Chrome",
            trigger_condition="Modal dialog blocks page interaction",
            heuristic_rule="Press Escape to dismiss modal before clicking buttons",
            confidence_score=0.85,
            rationale="Dismisses overlays reliably without navigating away",
        )
        reflection_1 = ReflectionOutput(
            task_id="task-learn-1",
            overall_success=True,
            root_cause_analysis="Overcame cookie consent overlay.",
            friction_points=["Cookie banner blocked click"],
            lessons=[lesson_a],
        )

        persisted_1 = extractor.extract_and_persist(reflection_1)
        assert len(persisted_1) == 1
        skill_1 = persisted_1[0]
        assert skill_1.application_name == "Chrome"
        assert skill_1.confidence_score == 0.85
        print(f"  [PASS] Stored new heuristic in SQLite & ChromaDB: [{skill_1.id}]")

        # Query semantic memory to verify indexing
        query_results = semantic.query_heuristics(
            query_context="Modal dialog overlay is blocking click",
            application_name="Chrome",
        )
        assert len(query_results) >= 1
        assert query_results[0].skill_id == skill_1.id
        print(f"  [PASS] Verified semantic retrieval match (score={query_results[0].similarity_score:.3f})")

        # Lesson B: Near-Duplicate Rule (Trigger deduplication & reinforcement)
        lesson_duplicate = ExtractedLesson(
            application_name="Chrome",
            trigger_condition="Modal dialog blocks page interaction",
            heuristic_rule="Press Escape to dismiss modal before clicking buttons",
            confidence_score=0.80,
            rationale="Repeated observation from another task",
        )
        reflection_2 = ReflectionOutput(
            task_id="task-learn-2",
            overall_success=True,
            root_cause_analysis="Encountered modal again and succeeded.",
            friction_points=["Modal blocker"],
            lessons=[lesson_duplicate],
        )

        persisted_2 = extractor.extract_and_persist(reflection_2, dedup_similarity_threshold=0.80)
        assert len(persisted_2) == 1
        reinforced_skill = persisted_2[0]
        # Should reinforce existing skill (confidence: 0.85 + 0.05 = 0.90)
        assert reinforced_skill.id == skill_1.id
        assert reinforced_skill.confidence_score == 0.90
        assert reinforced_skill.success_count == 2
        print(
            f"  [PASS] Deduplicated & reinforced existing heuristic: "
            f"new confidence={reinforced_skill.confidence_score:.2f}, success_count={reinforced_skill.success_count}"
        )

    finally:
        db.close()
        if os.path.exists(db_path):
            os.remove(db_path)


def test_strategy_prioritization_and_reinforcement():
    print("\n--- 4. Testing StrategyManager (Modality Prioritization & Reinforcement) ---")
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = DatabaseManager(db_url=f"sqlite:///{db_path}")
    chroma_client = chromadb.EphemeralClient()
    vector_store = ChromaVectorStore(client=chroma_client)
    semantic = SemanticMemoryManager(db=db, vector_store=vector_store)
    strategy = StrategyManager(semantic_memory=semantic)

    try:
        # Modality Prioritization
        assert strategy.get_preferred_modality("Chrome", "Open website") == "dom"
        assert strategy.get_preferred_modality("Edge", "https://bing.com") == "dom"
        assert strategy.get_preferred_modality("Notepad", "Write notes") == "uia"
        assert strategy.get_preferred_modality("Excel", "Sum row 5") == "uia"
        assert strategy.get_preferred_modality("Photoshop", "Paint canvas") == "hybrid"
        print("  [PASS] Verified context-aware modality prioritization: DOM / UIA / Hybrid")

        # Feedback Reinforcement
        skill = semantic.store_heuristic(
            application_name="Notepad",
            trigger_condition="Text area empty",
            heuristic_rule="Click editor surface before typing text",
            confidence_score=0.80,
        )

        # Successful task: confidence + 0.05 -> 0.85
        updates = strategy.reinforce_skills_used([skill.id], success=True)
        assert skill.id in updates
        assert round(updates[skill.id], 2) == 0.85
        print(f"  [PASS] Positive reinforcement: 0.80 -> {updates[skill.id]:.2f}")

        # Failed task: confidence - 0.10 -> 0.75
        updates_fail = strategy.reinforce_skills_used([skill.id], success=False)
        assert round(updates_fail[skill.id], 2) == 0.75
        print(f"  [PASS] Negative reinforcement penalty: 0.85 -> {updates_fail[skill.id]:.2f}")

    finally:
        db.close()
        if os.path.exists(db_path):
            os.remove(db_path)


def test_closed_loop_self_improvement():
    print("\n--- 5. Testing Closed-Loop Self-Improvement in Agent State Machine ---")
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = DatabaseManager(db_url=f"sqlite:///{db_path}")
    chroma_client = chromadb.EphemeralClient()
    vector_store = ChromaVectorStore(client=chroma_client)

    episodic = EpisodicMemoryManager(db=db, vector_store=vector_store)
    semantic = SemanticMemoryManager(db=db, vector_store=vector_store)
    retriever = ExperienceRetriever(episodic_memory=episodic, semantic_memory=semantic)
    reflector = ReflectionEngine(db=db, model_override="fallback")
    self_improver = SelfImprovementPipeline(db=db, semantic_memory=semantic, reflector=reflector)

    try:
        # Pre-seed UIState for Task 1
        elem_submit = UIElement(
            id="btn_submit",
            type="button",
            text="Submit Form",
            bbox=BBox(x_min=100, y_min=200, x_max=220, y_max=240),
        )
        test_ui = UIState(elements=[elem_submit], application="Chrome")

        planner = HierarchicalPlanner(model_override="fallback")
        agent = SIVACAgent(
            episodic_memory=episodic,
            experience_retriever=retriever,
            planner=planner,
            self_improver=self_improver,
            dry_run=True,
        )

        # Run Task 1: Autonomous execution with self-improver hook in finalize
        task1_state = create_initial_state(
            goal="Submit Form in Chrome",
            application="Chrome",
            max_steps=5,
        )
        task1_state["current_ui_state"] = test_ui

        final_state = agent.graph.invoke(task1_state)
        assert final_state["is_complete"] is True
        print(f"  [PASS] Task 1 completed (steps={final_state['step_count']})")

        # Verify that Self-Improvement Pipeline executed during finalize
        from vision_agent.memory.storage import ReflectionModel, LearnedSkillModel
        with db.session_scope() as s:
            reflections = s.query(ReflectionModel).all()
            assert len(reflections) >= 1
            print(f"  [PASS] Verified post-task reflection recorded in SQLite: {len(reflections)} reflection(s)")

            skills = s.query(LearnedSkillModel).all()
            assert len(skills) >= 1
            print(f"  [PASS] Verified new operational heuristic stored: '{skills[0].heuristic_rule[:60]}...'")

        # Run Task 2: Retriever should query memory and inject newly learned heuristic!
        context = retriever.retrieve_context(
            goal="Submit Form in Chrome",
            application="Chrome",
        )
        formatted_guidance = retriever.format_for_planner(context)
        assert len(context.heuristics) >= 1
        assert "Learned Interaction Rules & Heuristics" in formatted_guidance
        assert "Strategy:" in formatted_guidance
        print(f"  [PASS] Task 2 retrieved learned heuristic guidance into planner prompt context!")
        print(f"         Prompt Guidance:\n{formatted_guidance.strip()}")

    finally:
        db.close()
        if os.path.exists(db_path):
            os.remove(db_path)


if __name__ == "__main__":
    print("==================================================================")
    print("  SIVAC Phase 7 Verification: Self-Improvement & Reflection Engine")
    print("==================================================================")
    test_trajectory_evaluator()
    test_reflection_engine()
    test_lesson_extractor_and_deduplication()
    test_strategy_prioritization_and_reinforcement()
    test_closed_loop_self_improvement()
    print("\n[ALL PHASE 7 TESTS PASSED SUCCESSFULLY!]")
