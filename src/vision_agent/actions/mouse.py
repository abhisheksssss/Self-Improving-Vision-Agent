"""Low-level mouse control primitives using PyAutoGUI for SIVAC."""

import time
import logging
from typing import Optional
import pyautogui

# Disable PyAutoGUI fail-safe — this is an intentional automation agent.
# The fail-safe was triggering false positives on edge/corner interactions.
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.05

logger = logging.getLogger("sivac.actions.mouse")


class MouseController:
    """Low-level mouse control primitive using PyAutoGUI."""

    def click(self, x: int, y: int, button: str = "left", clicks: int = 1) -> None:
        """Click at screen coordinates (x, y)."""
        logger.info(f"Mouse Click: ({x}, {y}) | button={button} | clicks={clicks}")
        pyautogui.click(x=x, y=y, clicks=clicks, button=button, duration=0.1)

    def double_click(self, x: int, y: int) -> None:
        """Double click at screen coordinates (x, y)."""
        self.click(x, y, button="left", clicks=2)

    def right_click(self, x: int, y: int) -> None:
        """Right click at screen coordinates (x, y)."""
        self.click(x, y, button="right")

    def middle_click(self, x: int, y: int) -> None:
        """Middle click at screen coordinates (x, y)."""
        self.click(x, y, button="middle")

    def move_to(self, x: int, y: int) -> None:
        """Move cursor smoothly to (x, y)."""
        logger.info(f"Mouse Move: ({x}, {y})")
        pyautogui.moveTo(x, y, duration=0.15)

    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int) -> None:
        """Drag mouse from start to end coordinates."""
        logger.info(f"Mouse Drag: ({start_x}, {start_y}) -> ({end_x}, {end_y})")
        pyautogui.moveTo(start_x, start_y)
        pyautogui.dragTo(end_x, end_y, duration=0.3, button="left")

    def scroll(self, amount: int, x: Optional[int] = None, y: Optional[int] = None) -> None:
        """Scroll mouse wheel. Positive = up, Negative = down."""
        logger.info(f"Mouse Scroll: amount={amount} at ({x}, {y})")
        pyautogui.scroll(amount, x=x, y=y)