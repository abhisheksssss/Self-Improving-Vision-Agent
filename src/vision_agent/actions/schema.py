"""Action Schemas and Result Data Models for SIVAC.

Defines the allowlisted ActionType enum, individual action parameter models,
the unified ActionCommand, and the execution ActionResult.
"""

from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Allowlisted action types that SIVAC is permitted to execute."""
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    MOVE = "move"
    DRAG = "drag"
    SCROLL = "scroll"
    TYPE_TEXT = "type"
    PRESS_KEY = "press"
    HOTKEY = "hotkey"
    WAIT = "wait"
    BROWSER_NAVIGATE = "browser_navigate"
    APP_OPEN = "app_open"
    FINISH = "finish"


class BaseAction(BaseModel):
    """Base model for all action commands."""
    action: ActionType
    reason: str = Field(default="", description="Explanation of why this action was chosen.")


class ClickAction(BaseAction):
    """Action to click on a semantic UI element or specific pixel coordinate."""
    action: ActionType = ActionType.CLICK
    target_id: Optional[str] = Field(default=None, description="ID of the UIElement from UIState (e.g. 'vlm_1', 'dom_3')")
    coordinates: Optional[Tuple[int, int]] = Field(default=None, description="Explicit (x, y) coordinates")
    x: Optional[int] = None
    y: Optional[int] = None
    button: str = Field(default="left", description="'left', 'right', or 'middle'")
    clicks: int = Field(default=1, description="1 for single, 2 for double")


class TypeAction(BaseAction):
    """Action to enter text via keyboard into the currently focused input."""
    action: ActionType = ActionType.TYPE_TEXT
    text: str = Field(description="Text to type into the focused element.")
    target_id: Optional[str] = Field(default=None, description="Optional target UIElement ID to click before typing.")
    press_enter: bool = Field(default=False, description="Whether to press Enter after typing.")


class HotkeyAction(BaseAction):
    """Action to press a key combination (e.g. ['ctrl', 'c'], ['alt', 'tab'])."""
    action: ActionType = ActionType.HOTKEY
    keys: List[str] = Field(description="List of keys to press simultaneously (e.g. ['ctrl', 'v']).")


class ScrollAction(BaseAction):
    """Action to scroll mouse wheel up (positive) or down (negative)."""
    action: ActionType = ActionType.SCROLL
    amount: int = Field(default=-300, description="Positive to scroll up, negative to scroll down.")
    x: Optional[int] = None
    y: Optional[int] = None


class WaitAction(BaseAction):
    """Action to pause and wait for UI to finish loading."""
    action: ActionType = ActionType.WAIT
    seconds: float = Field(default=1.0, ge=0.0, le=30.0, description="Number of seconds to wait (0-30s).")


class BrowserAction(BaseAction):
    """Action for browser navigation."""
    action: ActionType = ActionType.BROWSER_NAVIGATE
    url: str = Field(description="Target URL to navigate to.")


class AppAction(BaseAction):
    """Action to launch or switch to a desktop application."""
    action: ActionType = ActionType.APP_OPEN
    app_name: str = Field(description="Application executable or alias (e.g. 'notepad', 'chrome', 'calc').")


class ActionCommand(BaseModel):
    """Unified Action Command containing any of the specific action parameters."""
    action: ActionType
    target_id: Optional[str] = None
    x: Optional[int] = None
    y: Optional[int] = None
    text: Optional[str] = None
    key: Optional[str] = None
    keys: Optional[List[str]] = None
    amount: Optional[int] = None
    seconds: Optional[float] = None
    url: Optional[str] = None
    app_name: Optional[str] = None
    button: str = "left"
    clicks: int = 1
    press_enter: bool = False
    reason: str = ""


class ActionResult(BaseModel):
    """Result of an executed action returned to the LangGraph Controller."""
    success: bool
    action: ActionType
    details: str = ""
    error: Optional[str] = None
    execution_time_ms: float = 0.0