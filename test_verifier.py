"""Phase 4 verification script: Tests Outcome Verifier and Closed-Loop Recovery Engine.

Validates:
    1. VisualDiffEngine: Perceptual hashing and Mean Absolute Difference (MAD).
    2. StateDiffEngine: Structural delta tracking (appear, disappear, text, URL, modal).
    3. FailureClassifier: Deterministic mapping from diff signals to FailureMode.
    4. RecoveryEngine: Tailored recovery plan generation for each failure mode.
    5. OutcomeVerifier: End-to-end two-tier verification and recovery orchestration.
"""

import os
import tempfile
from PIL import Image

from vision_agent.perception import BBox, UIElement, UIState
from vision_agent.actions import ActionCommand, ActionType
from vision_agent.verifier import (
    VisualDiffEngine,
    StateDiffEngine,
    FailureClassifier,
    RecoveryEngine,
    OutcomeVerifier,
    FailureMode,
    VerificationResult,
    RecoveryPlan,
)


def create_temp_image(color: tuple, size: tuple = (256, 256)) -> str:
    """Helper to create a temporary test image with a solid color."""
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    img = Image.new("RGB", size, color=color)
    img.save(path)
    return path


def test_visual_diff_engine():
    print("\n--- 1. Testing VisualDiffEngine ---")
    engine = VisualDiffEngine()

    # Case A: Identical images
    img1 = create_temp_image((255, 255, 255))
    score_identical, meta = engine.compute_score(img1, img1)
    assert score_identical == 0.0, f"Expected 0.0 for identical images, got {score_identical}"
    assert engine.is_unchanged(score_identical), "Identical images should be unchanged"
    print(f"  [PASS] Identical images: score={score_identical:.4f} (is_unchanged=True)")

    # Case B: Completely different images (White vs Black)
    img2 = create_temp_image((0, 0, 0))
    score_diff, meta_diff = engine.compute_score(img1, img2)
    assert score_diff > 0.05, f"Expected significant diff, got {score_diff}"
    assert engine.is_meaningfully_changed(score_diff), "Contrasting images should be meaningfully changed"
    print(f"  [PASS] Contrasting images: score={score_diff:.4f} (is_meaningfully_changed=True)")

    # Case C: Invalid path handling
    score_invalid, meta_invalid = engine.compute_score("nonexistent_1.png", "nonexistent_2.png")
    assert score_invalid == 0.0, "Invalid image paths should return 0.0"
    print("  [PASS] Handled nonexistent image paths gracefully")

    # Cleanup temp images
    for p in (img1, img2):
        if os.path.exists(p):
            os.remove(p)


def test_state_diff_engine():
    print("\n--- 2. Testing StateDiffEngine ---")
    engine = StateDiffEngine()

    elem1 = UIElement(id="btn_submit", type="button", text="Submit", bbox=BBox(x_min=10, y_min=10, x_max=60, y_max=30))
    elem2 = UIElement(id="lbl_status", type="label", text="Idle", bbox=BBox(x_min=10, y_min=40, x_max=100, y_max=60))

    before = UIState(
        application="TestApp",
        window_title="Home",
        url="http://localhost:8000/",
        elements=[elem1, elem2],
    )

    # 1. Test element text changed and new element appeared
    elem2_updated = UIElement(id="lbl_status", type="label", text="Success!", bbox=BBox(x_min=10, y_min=40, x_max=100, y_max=60))
    elem3_new = UIElement(id="btn_download", type="button", text="Download", bbox=BBox(x_min=10, y_min=70, x_max=120, y_max=100))

    after = UIState(
        application="TestApp",
        window_title="Home - Updated",
        url="http://localhost:8000/done",
        elements=[elem1, elem2_updated, elem3_new],
    )

    delta = engine.compute_delta(before, after)
    assert len(delta.appeared) == 1 and delta.appeared[0].id == "btn_download", "New element should appear"
    assert len(delta.text_changed) == 1, "Status label text should have changed"
    assert delta.url_changed, "URL should be recognized as changed"
    assert delta.title_changed, "Window title should be recognized as changed"
    assert not delta.has_modal, "Should not detect a modal here"
    print(f"  [PASS] Detected delta: {delta.summary()}")

    # 2. Test modal dialog detection
    modal_elem = UIElement(id="dlg_err", type="dialog", text="Error: Connection failed. Dismiss", bbox=BBox(x_min=50, y_min=50, x_max=300, y_max=200))
    after_with_modal = UIState(
        application="TestApp",
        elements=[elem1, modal_elem],
    )
    delta_modal = engine.compute_delta(before, after_with_modal)
    assert delta_modal.has_modal, "Should detect blocking modal"
    print("  [PASS] Correctly detected unexpected modal dialog")


def test_failure_classifier():
    print("\n--- 3. Testing FailureClassifier ---")
    visual_engine = VisualDiffEngine()
    classifier = FailureClassifier(visual_engine=visual_engine)
    state_engine = StateDiffEngine()

    elem1 = UIElement(id="input_email", type="input", text="", bbox=BBox(x_min=10, y_min=10, x_max=100, y_max=30))
    before = UIState(elements=[elem1])

    # Case 1: Modal blocked
    modal = UIElement(id="alert_1", type="alert", text="Please Confirm or Cancel", bbox=BBox(x_min=0, y_min=0, x_max=200, y_max=100))
    state_modal = UIState(elements=[elem1, modal])
    delta_modal = state_engine.compute_delta(before, state_modal)
    f_mode = classifier.classify(visual_score=0.2, delta=delta_modal, action_type="click")
    assert f_mode == FailureMode.MODAL_BLOCKED, f"Expected MODAL_BLOCKED, got {f_mode}"
    print(f"  [PASS] Classified: {f_mode.value}")

    # Case 2: Success via structural update
    elem1_typed = UIElement(id="input_email", type="input", text="user@example.com", bbox=BBox(x_min=10, y_min=10, x_max=100, y_max=30))
    state_success = UIState(elements=[elem1_typed])
    delta_success = state_engine.compute_delta(before, state_success)
    f_mode = classifier.classify(visual_score=0.15, delta=delta_success, action_type="type")
    assert f_mode == FailureMode.NONE, f"Expected NONE, got {f_mode}"
    print(f"  [PASS] Classified: {f_mode.value} (Success)")

    # Case 3: Missed Click (Screen changed visually but element state did not)
    delta_no_change = state_engine.compute_delta(before, before)
    f_mode = classifier.classify(visual_score=0.15, delta=delta_no_change, action_type="click")
    assert f_mode == FailureMode.MISSED_CLICK, f"Expected MISSED_CLICK, got {f_mode}"
    print(f"  [PASS] Classified: {f_mode.value}")

    # Case 4: Type Failed (Screen changed visually but no text changed in elements)
    f_mode = classifier.classify(visual_score=0.10, delta=delta_no_change, action_type="type")
    assert f_mode == FailureMode.TYPE_FAILED, f"Expected TYPE_FAILED, got {f_mode}"
    print(f"  [PASS] Classified: {f_mode.value}")

    # Case 5: UI Lag (Minor visual change, 0 delta)
    f_mode = classifier.classify(visual_score=0.03, delta=delta_no_change, action_type="click")
    assert f_mode == FailureMode.UI_LAG, f"Expected UI_LAG, got {f_mode}"
    print(f"  [PASS] Classified: {f_mode.value}")

    # Case 6: No Visual Change (Score ~0, 0 delta)
    f_mode = classifier.classify(visual_score=0.0, delta=delta_no_change, action_type="click")
    assert f_mode == FailureMode.NO_VISUAL_CHANGE, f"Expected NO_VISUAL_CHANGE, got {f_mode}"
    print(f"  [PASS] Classified: {f_mode.value}")


def test_recovery_engine():
    print("\n--- 4. Testing RecoveryEngine ---")
    engine = RecoveryEngine(max_retries=3)

    # 1. Recovery for MODAL_BLOCKED
    plan_modal = engine.generate_plan(FailureMode.MODAL_BLOCKED, retry_count=0)
    assert plan_modal.should_replan, "Should request replan after dismissing modal"
    assert any(step.action_type == "press" and step.params.get("key") == "escape" for step in plan_modal.steps)
    print(f"  [PASS] MODAL_BLOCKED plan: {[s.action_type for s in plan_modal.steps]}")

    # 2. Recovery for MISSED_CLICK with coordinate nudge
    plan_click = engine.generate_plan(
        FailureMode.MISSED_CLICK,
        retry_count=0,
        original_action_type="click",
        original_x=100,
        original_y=200,
    )
    click_step = next(s for s in plan_click.steps if s.action_type == "click")
    assert click_step.params["x"] == 105, f"Expected nudged coordinate 105, got {click_step.params['x']}"
    print(f"  [PASS] MISSED_CLICK plan with coordinate nudge: {click_step.params}")

    # 3. Recovery for TYPE_FAILED
    plan_type = engine.generate_plan(
        FailureMode.TYPE_FAILED,
        retry_count=0,
        original_action_type="type",
        original_text="hello world",
        original_x=50,
        original_y=50,
    )
    action_types = [s.action_type for s in plan_type.steps]
    assert "click" in action_types and "hotkey" in action_types and "type" in action_types
    print(f"  [PASS] TYPE_FAILED plan: {action_types}")

    # 4. Max retries exceeded -> Abort
    plan_abort = engine.generate_plan(FailureMode.NO_VISUAL_CHANGE, retry_count=3)
    assert plan_abort.should_abort, "Max retries exceeded must trigger should_abort=True"
    assert not plan_abort.should_replan, "Should not replan when aborting"
    assert plan_abort.failure_mode == FailureMode.MAX_RETRIES_EXCEEDED
    print(f"  [PASS] MAX_RETRIES_EXCEEDED plan: should_abort={plan_abort.should_abort}")


def test_outcome_verifier_orchestration():
    print("\n--- 5. Testing OutcomeVerifier Orchestrator ---")
    verifier = OutcomeVerifier(max_retries=3)

    elem_search = UIElement(id="input_q", type="input", text="", bbox=BBox(x_min=10, y_min=10, x_max=150, y_max=35))
    before_state = UIState(elements=[elem_search], url="https://example.com")

    # Step A: Type action that succeeds
    elem_search_typed = UIElement(id="input_q", type="input", text="Vision Agent", bbox=BBox(x_min=10, y_min=10, x_max=150, y_max=35))
    after_success = UIState(elements=[elem_search_typed], url="https://example.com")
    cmd_type = ActionCommand(action=ActionType.TYPE_TEXT, text="Vision Agent", target_id="input_q")

    res_success, recovery_none = verifier.verify_and_recover(before_state, after_success, cmd_type)
    assert res_success.passed, "Step should pass verification"
    assert res_success.failure_mode == FailureMode.NONE
    assert recovery_none is None, "No recovery plan should be returned on success"
    print("  [PASS] Step verified as SUCCESS")

    # Step B: Click action that triggers unexpected modal
    modal_elem = UIElement(id="dlg_cookie", type="dialog", text="Accept all cookies? Confirm", bbox=BBox(x_min=50, y_min=50, x_max=400, y_max=300))
    after_modal = UIState(elements=[elem_search, modal_elem], url="https://example.com")
    cmd_click = ActionCommand(action=ActionType.CLICK, x=100, y=20)

    res_fail, recovery_plan = verifier.verify_and_recover(before_state, after_modal, cmd_click)
    assert not res_fail.passed, "Step with modal should fail verification"
    assert res_fail.failure_mode == FailureMode.MODAL_BLOCKED
    assert recovery_plan is not None, "Recovery plan must be generated on failure"
    assert any(s.action_type == "press" for s in recovery_plan.steps), "Recovery should press ESC"
    print(f"  [PASS] Step failed with {res_fail.failure_mode.value}, generated {len(recovery_plan.steps)} recovery steps")

    # Step C: WAIT and FINISH commands
    cmd_wait = ActionCommand(action=ActionType.WAIT, seconds=2.0)
    res_wait = verifier.verify_step(before_state, before_state, cmd_wait)
    assert res_wait.passed, "WAIT command should verify successfully on static screen"
    print("  [PASS] WAIT command verified successfully")

    cmd_finish = ActionCommand(action=ActionType.FINISH)
    res_finish = verifier.verify_step(before_state, before_state, cmd_finish)
    assert res_finish.passed, "FINISH command should verify successfully"
    print("  [PASS] FINISH command verified successfully")


if __name__ == "__main__":
    print("==================================================================")
    print("      SIVAC Phase 4 Verification: Outcome Verifier & Recovery     ")
    print("==================================================================")
    test_visual_diff_engine()
    test_state_diff_engine()
    test_failure_classifier()
    test_recovery_engine()
    test_outcome_verifier_orchestration()
    print("\n[ALL PHASE 4 TESTS PASSED SUCCESSFULLY!]")
