import logging
from typing import List, Tuple, Optional
from PIL import Image

from .state import UIElement, UIState
from .screenshot import ScreenshotEngine
from .vlm_detector import VLMElemDetector
from .ocr import OCRDetector
from .dom import DOMInspector
from .ui_automation import UIAutomationInspector

logger = logging.getLogger("sivac.perception.merger")

SOURCE_PRIORITY = {"dom": 0, "uia": 1, "vlm": 2, "ocr": 3}
IOU_DEDUP_THRESHOLD = 0.5


class HybridPerceptionEngine:
    """Master perception orchestrator that fuses VLM, OCR, DOM, and UIA into a single UIState."""

    def __init__(self) -> None:
        self.screenshot = ScreenshotEngine()
        self.vlm = VLMElemDetector()
        self.ocr = OCRDetector()
        self.dom = DOMInspector()
        self.uia = UIAutomationInspector()

    def perceive(
        self,
        use_vlm: bool = True,
        use_ocr: bool = True,
        use_dom: bool = True,
        use_uia: bool = True,
        window_title: str = "",
    ) -> UIState:
        """
        Capture the current screen and extract elements from all enabled sources.

        Args:
            use_vlm: Whether to run Vision Model detection.
            use_ocr: Whether to run OCR detection.
            use_dom: Whether to run DOM inspection.
            use_uia: Whether to run Windows UI Automation.
            window_title: Optional filter for targeted window.

        Returns:
            A clean, unified, deduplicated UIState object.
        """
        img, filepath, dimensions = self.screenshot.capture(save=True)

        all_elements: List[UIElement] = []

        # 1. DOM inspection (highest priority for browser tabs)
        url = None
        if use_dom:
            try:
                url = self.dom.get_url()
                dom_elements = self.dom.detect()
                all_elements.extend(dom_elements)
            except Exception as e:
                logger.debug(f"DOM perception skipped: {e}")

        # 2. Windows UI Automation (highest priority for native desktop apps)
        if use_uia:
            try:
                uia_elements = self.uia.detect(window_title=window_title)
                all_elements.extend(uia_elements)
            except Exception as e:
                logger.debug(f"UIA perception skipped: {e}")

        # 3. VLM Visual Element Detection (for any visual layout elements)
        if use_vlm:
            try:
                vlm_elements = self.vlm.detect(img, dimensions)
                all_elements.extend(vlm_elements)
            except Exception as e:
                logger.debug(f"VLM perception skipped: {e}")

        # 4. OCR Text Detection (for text-heavy regions)
        if use_ocr:
            try:
                ocr_elements = self.ocr.detect(img, dimensions)
                all_elements.extend(ocr_elements)
            except Exception as e:
                logger.debug(f"OCR perception skipped: {e}")

        # 5. Deduplicate overlapping elements, keeping the higher-priority source
        merged_elements = self._deduplicate(all_elements)

        # 6. Detect the currently active window title and application name
        app_name, win_title = self._detect_app_context()

        # 7. Return the unified UIState
        return UIState(
            application=app_name,
            window_title=win_title,
            url=url,
            elements=merged_elements,
            screenshot_path=filepath,
            dimensions=dimensions,
        )

    def _deduplicate(self, elements: List[UIElement]) -> List[UIElement]:
        """
        Deduplicate overlapping elements using IoU (Intersection-over-Union).
        Sorts elements by source priority so more reliable sources are preserved.
        """
        sorted_elements = sorted(
            elements, key=lambda elem: SOURCE_PRIORITY.get(elem.source, 99)
        )

        kept_elements: List[UIElement] = []

        for candidate in sorted_elements:
            is_duplicate = False

            for existing in kept_elements:
                if candidate.bbox.iou(existing.bbox) > IOU_DEDUP_THRESHOLD:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept_elements.append(candidate)

        return kept_elements

    def _detect_app_context(self) -> Tuple[str, str]:
        """
        Detect active foreground window title and application name on Windows.
        Uses Win32 API to get the executable name of the foreground process,
        which is more reliable than parsing the window title text.
        """
        try:
            import ctypes
            import ctypes.wintypes

            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32

            hwnd = user32.GetForegroundWindow()

            # Get window title
            length = user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            window_title = buf.value

            # Get the process ID behind the window
            pid = ctypes.wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))

            # Open the process to read its executable name
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            hproc = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            exe_path = ""
            if hproc:
                exe_buf = ctypes.create_unicode_buffer(260)
                size = ctypes.wintypes.DWORD(260)
                if kernel32.QueryFullProcessImageNameW(hproc, 0, exe_buf, ctypes.byref(size)):
                    exe_path = exe_buf.value.lower()
                kernel32.CloseHandle(hproc)

            # Map executable name to friendly app name
            app_name = "Desktop"
            exe_lower = exe_path.lower() if exe_path else ""
            title_lower = window_title.lower()

            if "chrome.exe" in exe_lower:
                app_name = "Google Chrome"
            elif "firefox.exe" in exe_lower:
                app_name = "Mozilla Firefox"
            elif "msedge.exe" in exe_lower:
                app_name = "Microsoft Edge"
            elif "notepad.exe" in exe_lower:
                app_name = "Notepad"
            elif "notepad++.exe" in exe_lower:
                app_name = "Notepad++"
            elif "code.exe" in exe_lower:
                app_name = "Visual Studio Code"
            elif "explorer.exe" in exe_lower and "chrome" not in title_lower:
                app_name = "File Explorer"
            elif "windowsterminal.exe" in exe_lower or "powershell.exe" in exe_lower or "cmd.exe" in exe_lower or "wt.exe" in exe_lower:
                app_name = "Terminal"
            elif "python.exe" in exe_lower or "pythonw.exe" in exe_lower:
                # The agent itself is running — peek at window title for context
                app_name = "Desktop"
            elif exe_path:
                # Extract executable base name without extension
                exe_base = exe_path.rsplit("\\", 1)[-1].replace(".exe", "").title()
                app_name = exe_base if exe_base else "Desktop"
            elif window_title:
                # Last resort: clean up window title (avoid returning filenames)
                parts = [p.strip() for p in window_title.replace("—", "-").split("-")]
                # Filter out parts that look like file names (contain a dot)
                clean_parts = [p for p in parts if "." not in p and p]
                app_name = clean_parts[-1] if clean_parts else "Desktop"

            return app_name, window_title

        except Exception:
            return "Desktop", ""

