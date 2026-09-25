import time
import logging
from pathlib import Path
from typing import Tuple, Optional
from PIL import Image, ImageGrab
import mss

from vision_agent.config import settings

logger = logging.getLogger("sivac.perception.screenshot")


class ScreenshotEngine:
    """High-speed screen capture using mss with PIL and PyAutoGUI fallbacks."""

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self.output_dir = output_dir or (settings.data_path / "screenshots")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def capture(self, save: bool = True) -> Tuple[Image.Image, str, Tuple[int, int]]:
        """
        Capture the desktop screen with multi-level fallbacks.

        Returns:
            Tuple of (PIL Image, saved file path, (width, height))
        """
        timestamp = int(time.time() * 1000)
        filepath = str(self.output_dir / f"screenshot_{timestamp}.png")
        img = None

        # 1. Try mss (fastest)
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
                sct_img = sct.grab(monitor)
                img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        except Exception as e:
            logger.debug(f"mss capture failed, falling back to PIL: {e}")

        # 2. Try PIL ImageGrab
        if img is None:
            try:
                img = ImageGrab.grab(all_screens=True).convert("RGB")
            except Exception as e:
                logger.debug(f"ImageGrab failed, falling back to pyautogui: {e}")

        # 3. Try pyautogui screenshot
        if img is None:
            try:
                import pyautogui
                img = pyautogui.screenshot().convert("RGB")
            except Exception as e:
                logger.debug(f"pyautogui screenshot failed: {e}")

        # 4. Fallback: If running in headless/locked screen context, create a diagnostic canvas
        if img is None:
            logger.warning("All screen capture methods failed. Using 1920x1080 canvas.")
            img = Image.new("RGB", (1920, 1080), color=(30, 30, 30))

        if save:
            try:
                img.save(filepath, format="PNG")
            except Exception as e:
                logger.warning(f"Could not save screenshot to {filepath}: {e}")

        return img, filepath, img.size

    def capture_region(
        self, bbox: Tuple[int, int, int, int], save: bool = True
    ) -> Tuple[Image.Image, str]:
        """Capture a specific region of the screen (x_min, y_min, x_max, y_max)."""
        img, _, _ = self.capture(save=False)
        cropped = img.crop(bbox)

        timestamp = int(time.time() * 1000)
        filepath = str(self.output_dir / f"crop_{timestamp}.png")

        if save:
            cropped.save(filepath, format="PNG")

        return cropped, filepath
