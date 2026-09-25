
"""Screenshot-Driven Vision Planner for SIVAC.

Sends the live screenshot + goal + action history to a multimodal VLM.
The model sees the actual screen and returns the next concrete action.
Falls back to the heuristic planner if VLM is unavailable.
"""

import re
import json
import base64
import io
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

from PIL import Image
from langchain_core.messages import SystemMessage, HumanMessage

from ..actions.schema import ActionCommand, ActionType
from ..perception.state import UIState
from ..model.factory import ModelFactory

logger = logging.getLogger("sivac.agent.vision_planner")


VISION_SYSTEM_PROMPT = (
    "You are SIVAC, an autonomous computer-use agent.\n"
    "You receive a screenshot of the current screen and must decide the SINGLE NEXT ACTION "
    "to make progress toward the user's goal.\n\n"
    "Rules:\n"
    "- Study the screenshot carefully before deciding.\n"
    "- Output ONLY a single valid JSON object (no markdown fences, no explanation).\n"
    "- Use pixel coordinates (x, y) based on the screenshot you see.\n"
    "- For type actions set press_enter=true if you want to press Enter after typing.\n"
    "- For click actions point to the visible center of the target element.\n"
    "- Use app_open to launch desktop applications by name (e.g. notepad, msedge).\n"
    "- Use browser_navigate to open a URL in the currently active browser.\n"
    "- Output finish only when the goal is FULLY achieved.\n\n"
    "JSON format (include only the fields you need):\n"
    "{\n"
    '  "subgoal": "What this step achieves",\n'
    '  "action": "click|double_click|right_click|type|press|hotkey|scroll|wait|browser_navigate|app_open|finish",\n'
    '  "x": 123,\n'
    '  "y": 456,\n'
    '  "text": "text to type",\n'
    '  "key": "enter",\n'
    '  "keys": ["ctrl", "a"],\n'
    '  "url": "https://...",\n'
    '  "app_name": "notepad",\n'
    '  "seconds": 1.5,\n'
    '  "press_enter": false,\n'
    '  "reason": "Why you chose this action based on what you see on screen"\n'
    "}"
)


class ScreenVisionPlanner:
    """Uses a multimodal VLM to decide the next action from a live screenshot."""

    def __init__(self, model_override=None) -> None:
        self._model = model_override
        self._cached_model = None

    def _get_model(self):
        """Lazy-load and cache the vision model."""
        if self._model is not None:
            return self._model
        if self._cached_model is not None:
            return self._cached_model
        try:
            self._cached_model = ModelFactory.get_vision_model()
            logger.info(
                "[ScreenVisionPlanner] Vision model loaded: %s",
                type(self._cached_model).__name__,
            )
            return self._cached_model
        except Exception as exc:
            logger.warning("[ScreenVisionPlanner] Vision model unavailable: %s", exc)
            return None

    def _encode_screenshot(self, screenshot_path: str) -> Optional[str]:
        """Load screenshot and return a base64-encoded PNG string (max 1280px wide)."""
        try:
            path = Path(screenshot_path)
            if not path.exists():
                return None
            img = Image.open(path).convert("RGB")
            max_w = 1280
            if img.width > max_w:
                ratio = max_w / img.width
                new_h = int(img.height * ratio)
                img = img.resize((max_w, new_h), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as exc:
            logger.warning("[ScreenVisionPlanner] Screenshot encode failed: %s", exc)
            return None

    def _build_user_text(
        self,
        goal: str,
        ui_state: Optional[UIState],
        action_history: List[Dict[str, Any]],
        step_count: int,
    ) -> str:
        """Build the text portion of the user message."""
        parts = [
            f"GOAL: {goal}",
            f"Step number: {step_count}",
        ]
        if ui_state:
            parts.append(f"Active application: {ui_state.application}")
            if ui_state.window_title:
                parts.append(f"Window title: {ui_state.window_title}")
            if ui_state.url:
                parts.append(f"Current URL: {ui_state.url}")
            w, h = ui_state.dimensions
            parts.append(f"Screenshot dimensions: {w} x {h} pixels")

        if action_history:
            parts.append("\nActions already executed (oldest first):")
            for item in action_history[-6:]:
                a = item.get("action", "?")
                t = item.get("text") or item.get("url") or item.get("app_name") or ""
                r = item.get("reason", "")
                parts.append(f"  - {a}: {t!r}  # {r[:70]}")

        parts.append(
            "\nNow look at the screenshot and output the NEXT SINGLE ACTION as JSON."
        )
        return "\n".join(parts)

    def plan(
        self,
        goal: str,
        ui_state: Optional[UIState],
        action_history: List[Dict[str, Any]],
        step_count: int,
    ) -> Optional[Tuple[str, ActionCommand]]:
        """
        Ask the vision model for the next action.

        Returns (subgoal, ActionCommand) on success, None on failure.
        """
        model = self._get_model()
        if model is None:
            return None

        screenshot_path = ui_state.screenshot_path if ui_state else None
        if not screenshot_path:
            logger.debug("[ScreenVisionPlanner] No screenshot_path -- skipping VLM call")
            return None

        b64 = self._encode_screenshot(screenshot_path)
        if not b64:
            return None

        user_text = self._build_user_text(goal, ui_state, action_history, step_count)

        try:
            response = model.invoke([
                SystemMessage(content=VISION_SYSTEM_PROMPT),
                HumanMessage(content=[
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ]),
            ])
            raw = response.content if hasattr(response, "content") else str(response)
            logger.debug("[ScreenVisionPlanner] Raw VLM response: %s", raw[:400])
            return self._parse_response(raw, ui_state)
        except Exception as exc:
            logger.warning("[ScreenVisionPlanner] VLM call failed: %s", exc)
            return None

    def _parse_response(
        self,
        raw: str,
        ui_state: Optional[UIState],
    ) -> Optional[Tuple[str, ActionCommand]]:
        """Parse the JSON action returned by the VLM."""
        cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()

        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            logger.warning(
                "[ScreenVisionPlanner] No JSON in VLM response: %s", raw[:200]
            )
            return None

        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            logger.warning(
                "[ScreenVisionPlanner] JSON parse error: %s | raw=%s", exc, raw[:200]
            )
            return None

        subgoal = data.get("subgoal", "")
        raw_action = str(data.get("action", "wait")).lower().strip()

        try:
            action_type = ActionType(raw_action)
        except ValueError:
            logger.warning(
                "[ScreenVisionPlanner] Unknown action %r -- defaulting to wait",
                raw_action,
            )
            action_type = ActionType.WAIT

        x = data.get("x")
        y = data.get("y")

        # Scale coordinates back if we downscaled the image for the VLM
        if x is not None and y is not None and ui_state:
            actual_w, _actual_h = ui_state.dimensions
            max_w = 1280
            if actual_w > max_w:
                scale = actual_w / max_w
                x = int(float(x) * scale)
                y = int(float(y) * scale)

        cmd = ActionCommand(
            action=action_type,
            x=int(x) if x is not None else None,
            y=int(y) if y is not None else None,
            text=data.get("text"),
            key=data.get("key"),
            keys=data.get("keys"),
            url=data.get("url"),
            app_name=data.get("app_name"),
            seconds=float(data.get("seconds", 1.5)),
            press_enter=bool(data.get("press_enter", False)),
            reason=data.get("reason", subgoal),
        )

        logger.info(
            "[ScreenVisionPlanner] Action decided: %s  x=%s y=%s  text=%r",
            action_type.value,
            x,
            y,
            (data.get("text") or "")[:60],
        )
        return subgoal, cmd

