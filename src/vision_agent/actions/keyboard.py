"""Low-level keyboard control primitives using PyAutoGUI for SIVAC."""

import time
import logging
from typing import List
import pyautogui

# Disable the corner failsafe — this is an intentional automation agent;
# the failsafe triggers false positives when clicking near screen edges.
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.05  # Small pause between actions for stability

logger = logging.getLogger("sivac.actions.keyboard")


class KeyboardController:
    """Low-level keyboard control primitives using PyAutoGUI."""

    def type_text(self, text: str, interval: float = 0.02) -> None:
        """Type a string of text with optional interval between characters."""
        logger.info(f"Typing text: '{text}' (interval={interval}s)")
        pyautogui.write(text, interval=interval)

    def press_key(self, key: str) -> None:
        """Press a single key (e.g. 'enter', 'tab', 'backspace', 'esc', 'space')."""
        clean_key = key.lower().strip()
        logger.info(f"Keyboard Press: '{clean_key}'")
        pyautogui.press(clean_key)

    def hotkey(self, keys: List[str]) -> None:
        """Press a combination of keys simultaneously (e.g. ['ctrl', 'c'])."""
        clean_keys = [k.lower().strip() for k in keys]
        logger.info(f"Keyboard Hotkey: {'+'.join(clean_keys)}")
        pyautogui.hotkey(*clean_keys)