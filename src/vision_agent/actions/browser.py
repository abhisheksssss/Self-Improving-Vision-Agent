"""Browser navigation and automation controller for SIVAC."""

import logging
import webbrowser
from typing import Optional

logger = logging.getLogger("sivac.actions.browser")


class BrowserController:
    """Controls browser navigation actions."""

    def __init__(self) -> None:
        self._page = None

    def set_page(self, page) -> None:
        """Set the active Playwright page instance."""
        self._page = page

    def open_url(self, url: str) -> bool:
        """Navigate to a URL using Playwright or system default browser."""
        target_url = url.strip()
        if not target_url.startswith("http://") and not target_url.startswith("https://"):
            target_url = "https://" + target_url

        logger.info(f"Opening Browser URL: {target_url}")

        if self._page:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if not loop.is_running():
                    loop.run_until_complete(self._page.goto(target_url))
                    return True
            except Exception as e:
                logger.debug(f"Playwright navigation failed, falling back to webbrowser: {e}")

        # Fallback to system default browser
        return webbrowser.open(target_url)