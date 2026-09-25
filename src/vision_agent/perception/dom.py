import asyncio
import logging
from typing import List, Optional
from .state import BBox, UIElement

logger = logging.getLogger("sivac.perception.dom")

_DOM_SELECTORS = "a, button, input, select, textarea, [role='button'], [role='link'], [role='tab'], [role='menuitem']"


class DOMInspector:
    """Extracts interactive elements and bounding boxes directly from active browser DOM via Playwright."""

    def __init__(self) -> None:
        self._page = None

    def set_page(self, page) -> None:
        """Attach an active Playwright page instance to this inspector."""
        self._page = page

    def get_url(self) -> Optional[str]:
        """Get the current URL of the active browser page."""
        if self._page:
            try:
                return self._page.url
            except Exception:
                return None
        return None

    def detect(self) -> List[UIElement]:
        """
        Synchronously query and extract interactive DOM elements from the browser page.

        Returns:
            List of UIElement objects with source="dom".
        """
        if self._page is None:
            return []

        try:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            if loop.is_running():
                return []
            else:
                return loop.run_until_complete(self._async_detect())

        except Exception as e:
            logger.debug(f"DOM inspection failed: {e}")
            return []

    async def _async_detect(self) -> List[UIElement]:
        """Async worker that evaluates bounding client rectangles and tags of DOM elements."""
        elements = []

        try:
            handles = await self._page.query_selector_all(_DOM_SELECTORS)

            for idx, handle in enumerate(handles):
                box = await handle.bounding_box()
                text = (await handle.inner_text()).strip() if hasattr(handle, "inner_text") else ""
                tag = await handle.evaluate("el => el.tagName.toLowerCase()")

                if box and box["width"] > 0 and box["height"] > 0:
                    bbox = BBox(
                        x_min=int(box["x"]),
                        y_min=int(box["y"]),
                        x_max=int(box["x"] + box["width"]),
                        y_max=int(box["y"] + box["height"]),
                    )

                    elements.append(UIElement(
                        id=f"dom_{idx + 1}",
                        type=tag,
                        text=text[:100],
                        bbox=bbox,
                        confidence=1.0,
                        source="dom",
                    ))

        except Exception as e:
            logger.debug(f"DOM element query failed: {e}")

        return elements
