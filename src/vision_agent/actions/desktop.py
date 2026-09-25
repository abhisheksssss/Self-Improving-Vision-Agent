import os
import subprocess
import logging
from typing import Optional

logger = logging.getLogger("sivac.actions.desktop")

# Common Windows executable aliases — ordered by priority (full path first, then alias)
APP_SHORTCUTS = {
    "chrome": "chrome",
    "google chrome": "chrome",
    "notepad": "notepad",
    "file explorer": "explorer",
    "explorer": "explorer",
    "calculator": "calc",
    "calc": "calc",
    "cmd": "cmd",
    "terminal": "wt",
    "edge": "msedge",
    "msedge": "msedge",
    "firefox": "firefox",
    "code": "code",
    "vscode": "code",
}

# Full Windows installation paths for apps that may not be in PATH
APP_FULL_PATHS = {
    "chrome": [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%PROGRAMFILES%\Google\Chrome\Application\chrome.exe"),
    ],
    "msedge": [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ],
    "firefox": [
        r"C:\Program Files\Mozilla Firefox\firefox.exe",
        r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe",
    ],
    "code": [
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Microsoft VS Code\Code.exe"),
        r"C:\Program Files\Microsoft VS Code\Code.exe",
    ],
}


class DesktopController:
    """Launches and manages native desktop applications."""

    def _resolve_exe(self, app_name: str) -> str:
        """Resolve an app name to a full executable path if possible."""
        clean_name = app_name.lower().strip()
        exe = APP_SHORTCUTS.get(clean_name, clean_name)

        # Try full known paths first (avoids PATH dependency)
        for candidate_exe in [exe, clean_name]:
            for path in APP_FULL_PATHS.get(candidate_exe, []):
                if os.path.isfile(path):
                    logger.debug(f"Resolved '{app_name}' -> '{path}'")
                    return path

        # Fallback to alias or original name (relies on PATH or Windows associations)
        return exe

    def launch_app(self, app_name: str) -> bool:
        """Launch a Windows application by name or executable path."""
        exe = self._resolve_exe(app_name)
        logger.info(f"Launching desktop application: '{exe}'")
        try:
            subprocess.Popen(exe, shell=True)
            return True
        except Exception as e:
            logger.error(f"Failed to launch application '{app_name}': {e}")
            return False