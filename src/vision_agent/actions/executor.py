"""Unified Action Primitive Dispatcher and Executor for SIVAC."""

import time
import logging
from typing import Optional, Tuple
from ..perception.state import UIState, UIElement
from .schema import ActionCommand, ActionResult, ActionType
from .safety import ActionSafetyGuard, EmergencyStopMonitor
from .mouse import MouseController
from .keyboard import KeyboardController
from .browser import BrowserController
from .desktop import DesktopController

logger = logging.getLogger("sivac.actions.executor")


class ActionExecutor:
    """Dispatches validated ActionCommands to underlying OS or browser drivers."""

    def __init__(
        self,
        screen_dimensions: Tuple[int, int] = (1920, 1080),
        safety_guard: Optional[ActionSafetyGuard] = None,
        emergency_stop: Optional[EmergencyStopMonitor] = None,
    ) -> None:
        self.safety_guard = safety_guard or ActionSafetyGuard(screen_dimensions)
        self.emergency_stop = emergency_stop or EmergencyStopMonitor()
        self.mouse = MouseController()
        self.keyboard = KeyboardController()
        self.browser = BrowserController()
        self.desktop = DesktopController()

    def update_screen_dimensions(self, width: int, height: int) -> None:
        """Update display boundaries for the safety guard."""
        self.safety_guard.update_dimensions(width, height)

    def resolve_target_coordinates(
        self, cmd: ActionCommand, state: Optional[UIState]
    ) -> Optional[Tuple[int, int]]:
        """Resolve semantic target_id from UIState to screen (x, y) center coordinates."""
        if cmd.x is not None and cmd.y is not None:
            return (cmd.x, cmd.y)

        if cmd.target_id and state:
            elem = state.find_by_id(cmd.target_id) if hasattr(state, "find_by_id") else state.get_element_by_id(cmd.target_id)
            if elem and elem.bbox:
                center_x, center_y = elem.bbox.center
                cmd.x = center_x
                cmd.y = center_y
                logger.info(
                    f"Resolved target '{cmd.target_id}' -> center=({center_x}, {center_y}) text='{elem.text[:20]}'"
                )
                return (center_x, center_y)
            else:
                logger.warning(f"Target ID '{cmd.target_id}' not found in current UIState")

        return None

    def execute(
        self, cmd: ActionCommand, state: Optional[UIState] = None, dry_run: bool = False
    ) -> ActionResult:
        """Validate and dispatch an ActionCommand to the appropriate hardware primitive.

        Args:
            cmd: Structured ActionCommand
            state: Optional current UIState for target_id coordinate resolution
            dry_run: If True, validates safety and coordinates without moving hardware

        Returns:
            ActionResult with execution status, timings, and details
        """
        # 1. Emergency Stop Check
        if self.emergency_stop.is_stopped:
            return ActionResult(
                success=False,
                action=cmd.action,
                error="Execution blocked: Emergency Stop is currently active",
            )

        # 2. Resolve coordinates if target_id is provided
        if cmd.target_id:
            self.resolve_target_coordinates(cmd, state)

        # 3. Safety Guardrail Validation
        is_safe, safety_reason = self.safety_guard.validate_action(cmd)
        if not is_safe:
            logger.warning(f"Safety Violation: {safety_reason}")
            return ActionResult(
                success=False,
                action=cmd.action,
                error=f"Safety Guardrail Blocked: {safety_reason}",
            )

        start_time = time.perf_counter()

        # 4. Dry Run Mode
        if dry_run:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ActionResult(
                success=True,
                action=cmd.action,
                details=f"[DRY RUN] Validated {cmd.action} (target={cmd.target_id}, x={cmd.x}, y={cmd.y})",
                execution_time_ms=elapsed_ms,
            )

        # 5. Dispatch Action Primitive
        try:
            details = self._dispatch(cmd)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ActionResult(
                success=True,
                action=cmd.action,
                details=details,
                execution_time_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            error_str = str(e)
            # Note: FAILSAFE is disabled, but handle it gracefully if it ever fires
            if "fail-safe" in error_str.lower() or "failsafe" in error_str.lower():
                logger.warning("PyAutoGUI fail-safe triggered — continuing execution (failsafe is disabled)")
                return ActionResult(
                    success=False,
                    action=cmd.action,
                    error="PyAutoGUI FailSafe: " + error_str,
                    execution_time_ms=elapsed_ms,
                )
            logger.error(f"Action execution failed: {e}", exc_info=True)
            return ActionResult(
                success=False,
                action=cmd.action,
                error=error_str,
                execution_time_ms=elapsed_ms,
            )


    def _dispatch(self, cmd: ActionCommand) -> str:
        """Internal router mapping ActionType to the respective controller."""
        action = cmd.action

        if action == ActionType.CLICK:
            if cmd.x is None or cmd.y is None:
                raise ValueError("Click action requires valid x and y coordinates or target_id")
            self.mouse.click(cmd.x, cmd.y, button=cmd.button, clicks=cmd.clicks)
            return f"Clicked at ({cmd.x}, {cmd.y}) with button='{cmd.button}'"

        elif action == ActionType.DOUBLE_CLICK:
            if cmd.x is None or cmd.y is None:
                raise ValueError("Double click action requires valid coordinates")
            self.mouse.double_click(cmd.x, cmd.y)
            return f"Double-clicked at ({cmd.x}, {cmd.y})"

        elif action == ActionType.RIGHT_CLICK:
            if cmd.x is None or cmd.y is None:
                raise ValueError("Right click action requires valid coordinates")
            self.mouse.right_click(cmd.x, cmd.y)
            return f"Right-clicked at ({cmd.x}, {cmd.y})"

        elif action == ActionType.MOVE:
            if cmd.x is None or cmd.y is None:
                raise ValueError("Move action requires valid coordinates")
            self.mouse.move_to(cmd.x, cmd.y)
            return f"Moved cursor to ({cmd.x}, {cmd.y})"

        elif action == ActionType.SCROLL:
            amount = cmd.amount if cmd.amount is not None else -300
            self.mouse.scroll(amount, x=cmd.x, y=cmd.y)
            return f"Scrolled wheel by {amount} at ({cmd.x}, {cmd.y})"

        elif action == ActionType.TYPE_TEXT:
            if not cmd.text:
                raise ValueError("Type action requires non-empty 'text'")
            # If coordinates or target_id were provided, click first to focus input field
            if cmd.x is not None and cmd.y is not None:
                self.mouse.click(cmd.x, cmd.y)
                time.sleep(0.1)
            self.keyboard.type_text(cmd.text)
            if cmd.press_enter:
                time.sleep(0.05)
                self.keyboard.press_key("enter")
            return f"Typed text: '{cmd.text}' (press_enter={cmd.press_enter})"

        elif action == ActionType.PRESS_KEY:
            if not cmd.key:
                raise ValueError("Press key action requires 'key'")
            self.keyboard.press_key(cmd.key)
            return f"Pressed key: '{cmd.key}'"

        elif action == ActionType.HOTKEY:
            if not cmd.keys:
                raise ValueError("Hotkey action requires 'keys' list")
            self.keyboard.hotkey(cmd.keys)
            return f"Pressed hotkey combo: {cmd.keys}"

        elif action == ActionType.WAIT:
            duration = cmd.seconds if cmd.seconds is not None else 1.0
            time.sleep(duration)
            return f"Waited for {duration:.1f}s"

        elif action == ActionType.BROWSER_NAVIGATE:
            if not cmd.url:
                raise ValueError("Browser navigate requires 'url'")
            success = self.browser.open_url(cmd.url)
            return f"Navigated to URL '{cmd.url}' (opened={success})"

        elif action == ActionType.APP_OPEN:
            if not cmd.app_name:
                raise ValueError("App open requires 'app_name'")
            success = self.desktop.launch_app(cmd.app_name)
            return f"Launched application '{cmd.app_name}' (success={success})"

        elif action == ActionType.FINISH:
            return f"Task marked finished by agent. Reason: {cmd.reason or 'Goal achieved'}"

        else:
            raise NotImplementedError(f"Action '{action}' is not supported by ActionExecutor")
