"""Action Safety Subsystem for SIVAC.

Provides boundary validation, prohibited command filtering, and emergency
kill-switch monitoring before any hardware or OS primitive executes.
"""

import logging
from typing import Tuple, List, Optional
from .schema import ActionCommand, ActionType

logger = logging.getLogger("sivac.actions.safety")

# List of dangerous commands/keywords blocked by security policy
DANGEROUS_APP_PATTERNS = [
    "format",
    "diskpart",
    "regedit",
    "regdelete",
    "rmdir /s",
    "del /f",
    "powershell -enc",
]

# Blocked dangerous key combinations
DANGEROUS_HOTKEYS = [
    {"ctrl", "alt", "del"},
]


class ActionSafetyGuard:
    """Validates actions before execution against boundary and security policies."""

    def __init__(self, screen_dimensions: Tuple[int, int] = (1920, 1080)) -> None:
        self.screen_width, self.screen_height = screen_dimensions

    def update_dimensions(self, width: int, height: int) -> None:
        """Update screen boundaries dynamically if display resolution changes."""
        self.screen_width = width
        self.screen_height = height

    def validate_action(self, cmd: ActionCommand) -> Tuple[bool, Optional[str]]:
        """Validate whether an action is safe to execute.

        Returns:
            Tuple of (is_safe: bool, reason: Optional[str])
        """
        # 1. Coordinate Boundary Checks
        if cmd.x is not None:
            if cmd.x < 0 or cmd.x > self.screen_width:
                return False, f"X coordinate {cmd.x} is outside screen bounds (0 - {self.screen_width})"
        if cmd.y is not None:
            if cmd.y < 0 or cmd.y > self.screen_height:
                return False, f"Y coordinate {cmd.y} is outside screen bounds (0 - {self.screen_height})"

        # 2. Wait Duration Cap
        if cmd.action == ActionType.WAIT and cmd.seconds is not None:
            if cmd.seconds < 0 or cmd.seconds > 30.0:
                return False, f"Wait duration {cmd.seconds}s exceeds safety limit of 30 seconds"

        # 3. Dangerous Application Launch Filter
        if cmd.action == ActionType.APP_OPEN and cmd.app_name:
            app_lower = cmd.app_name.lower()
            for pattern in DANGEROUS_APP_PATTERNS:
                if pattern in app_lower:
                    return False, f"Launching '{cmd.app_name}' blocked by destructive action security policy"

        # 4. Dangerous Hotkey Detection
        if cmd.action == ActionType.HOTKEY and cmd.keys:
            keys_set = {k.lower().strip() for k in cmd.keys}
            for dangerous_combo in DANGEROUS_HOTKEYS:
                if dangerous_combo.issubset(keys_set):
                    return False, f"Hotkey combo {cmd.keys} blocked by safety policy"

        return True, None


class EmergencyStopMonitor:
    """Emergency stop monitor to immediately halt all agent execution."""

    def __init__(self) -> None:
        self.is_stopped = False

    def trigger_stop(self) -> None:
        """Manually trigger the emergency stop."""
        self.is_stopped = True
        logger.critical("EMERGENCY STOP TRIGGERED: Halting all computer interactions immediately!")

    def reset(self) -> None:
        """Reset emergency stop state."""
        self.is_stopped = False