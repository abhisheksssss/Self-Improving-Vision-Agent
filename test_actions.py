"""Phase 3 verification script: Tests Action Primitives, Safety Guardrails, and Executor."""

import time
from vision_agent.perception import BBox, UIElement, UIState
from vision_agent.actions import (
    ActionType,
    ActionCommand,
    ActionSafetyGuard,
    EmergencyStopMonitor,
    ActionExecutor,
)


def test_safety_guardrails():
    print("\n--- 1. Testing Safety Guardrails ---")
    guard = ActionSafetyGuard(screen_dimensions=(1920, 1080))

    # Test out-of-bounds coordinate
    oob_cmd = ActionCommand(action=ActionType.CLICK, x=2500, y=500)
    is_safe, reason = guard.validate_action(oob_cmd)
    assert not is_safe, "Out-of-bounds X should be blocked"
    print(f"  [PASS] Blocked out-of-bounds coordinate: {reason}")

    # Test dangerous app launch
    danger_app = ActionCommand(action=ActionType.APP_OPEN, app_name="format c:")
    is_safe, reason = guard.validate_action(danger_app)
    assert not is_safe, "Destructive format command should be blocked"
    print(f"  [PASS] Blocked destructive app launch: {reason}")

    # Test dangerous hotkey
    danger_key = ActionCommand(action=ActionType.HOTKEY, keys=["ctrl", "alt", "del"])
    is_safe, reason = guard.validate_action(danger_key)
    assert not is_safe, "Ctrl+Alt+Del should be blocked"
    print(f"  [PASS] Blocked dangerous hotkey: {reason}")

    # Test wait duration cap
    excessive_wait = ActionCommand(action=ActionType.WAIT, seconds=60.0)
    is_safe, reason = guard.validate_action(excessive_wait)
    assert not is_safe, "Excessive wait should be blocked"
    print(f"  [PASS] Blocked excessive wait duration: {reason}")

    # Test safe command
    safe_cmd = ActionCommand(action=ActionType.MOVE, x=500, y=500)
    is_safe, reason = guard.validate_action(safe_cmd)
    assert is_safe, "Safe move should be allowed"
    print("  [PASS] Allowed safe move action")


def test_emergency_stop():
    print("\n--- 2. Testing Emergency Stop Monitor ---")
    stop_monitor = EmergencyStopMonitor()
    executor = ActionExecutor(emergency_stop=stop_monitor)

    # Normal execution before stop
    cmd = ActionCommand(action=ActionType.WAIT, seconds=0.01)
    res = executor.execute(cmd)
    assert res.success, "Should succeed before stop"
    print("  [PASS] Normal action succeeded before emergency stop")

    # Trigger emergency stop
    stop_monitor.trigger_stop()
    res = executor.execute(cmd)
    assert not res.success, "Action must be blocked when stop is active"
    print(f"  [PASS] Action blocked after emergency stop: '{res.error}'")

    # Reset
    stop_monitor.reset()
    res = executor.execute(cmd)
    assert res.success, "Action must succeed after emergency stop reset"
    print("  [PASS] Action succeeded after reset")


def test_coordinate_resolution():
    print("\n--- 3. Testing Semantic Target ID Resolution ---")
    executor = ActionExecutor(screen_dimensions=(1920, 1080))

    # Mock UI State with a button element at (100, 200, w=60, h=40) -> center is (130, 220)
    button_elem = UIElement(
        id="btn_download_test",
        source="vlm",
        type="button",
        text="Download Report",
        bbox=BBox(x_min=100, y_min=200, x_max=160, y_max=240),
        interactive=True,
    )
    mock_state = UIState(
        screenshot_path="",
        dimensions=(1920, 1080),
        elements=[button_elem],
        application="TestApp",
        window_title="Test Window",
    )

    cmd = ActionCommand(action=ActionType.MOVE, target_id="btn_download_test")
    res = executor.execute(cmd, state=mock_state, dry_run=True)
    assert res.success, f"Action should succeed: {res.error}"
    assert cmd.x == 130 and cmd.y == 220, f"Expected center (130, 220), got ({cmd.x}, {cmd.y})"
    print(f"  [PASS] Target 'btn_download_test' resolved to center: ({cmd.x}, {cmd.y})")


def test_safe_primitive_execution():
    print("\n--- 4. Testing Action Primitive Execution & Timings ---")
    executor = ActionExecutor(screen_dimensions=(1920, 1080))

    # Test move (dry run)
    cmd_move = ActionCommand(action=ActionType.MOVE, x=300, y=300)
    res = executor.execute(cmd_move, dry_run=True)
    assert res.success
    print(f"  [PASS] Cursor Move: {res.details} ({res.execution_time_ms:.1f}ms)")

    # Test wait (real execution)
    cmd_wait = ActionCommand(action=ActionType.WAIT, seconds=0.05)
    res = executor.execute(cmd_wait)
    assert res.success
    print(f"  [PASS] Wait Action: {res.details} ({res.execution_time_ms:.1f}ms)")

    # Test finish
    cmd_finish = ActionCommand(action=ActionType.FINISH, reason="All goals verified")
    res = executor.execute(cmd_finish)
    assert res.success
    print(f"  [PASS] Finish Action: {res.details}")


def main():
    print("=" * 60)
    print("SIVAC Phase 3: Action Primitives & Safety Subsystem Test")
    print("=" * 60)

    test_safety_guardrails()
    test_emergency_stop()
    test_coordinate_resolution()
    test_safe_primitive_execution()

    print("\n" + "=" * 60)
    print("Phase 3 Action Layer Verification Complete! All Tests Passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
