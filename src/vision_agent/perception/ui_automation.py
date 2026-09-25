import logging
from typing import List
from .state import BBox, UIElement

logger = logging.getLogger("sivac.perception.uia")


class UIAutomationInspector:
    """Extracts native controls and bounding boxes from Windows apps via UI Automation (pywinauto)."""

    def detect(self, window_title: str = "") -> List[UIElement]:
        """
        Inspect the active or targeted Windows application window.

        Args:
            window_title: Optional filter string to target a specific window by title.

        Returns:
            List of UIElement objects with source="uia".
        """
        elements = []

        try:
            from pywinauto import Desktop

            desktop = Desktop(backend="uia")
            windows = desktop.windows()

            target = None
            if window_title:
                for w in windows:
                    if window_title.lower() in w.window_text().lower():
                        target = w
                        break

            if target is None and windows:
                target = windows[0]

            if target is None:
                return elements

            children = target.descendants()
            idx = 1

            for ctrl in children:
                try:
                    rect = ctrl.rectangle()
                    text = (ctrl.window_text() or "").strip()
                    ctrl_type = ctrl.friendly_class_name()

                    if rect.width() > 3 and rect.height() > 3:
                        bbox = BBox(
                            x_min=rect.left,
                            y_min=rect.top,
                            x_max=rect.right,
                            y_max=rect.bottom,
                        )

                        elements.append(UIElement(
                            id=f"uia_{idx}",
                            type=ctrl_type.lower(),
                            text=text[:100],
                            bbox=bbox,
                            confidence=1.0,
                            source="uia",
                        ))
                        idx += 1

                except Exception:
                    continue

        except Exception as e:
            logger.debug(f"UI Automation inspection failed: {e}")

        return elements
