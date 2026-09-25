"""Hierarchical Action Planner for SIVAC.

Planning priority:
  1. ScreenVisionPlanner — multimodal VLM sees the screenshot and decides.
  2. Text-based LLM (reasoning model) with UIState text description.
  3. Heuristic deterministic fallback.
"""

import re
import json
import logging
from typing import Optional, Tuple, Dict, Any, List

from langchain_core.messages import SystemMessage, HumanMessage

from ..actions.schema import ActionCommand, ActionType
from ..perception.state import UIState, UIElement
from ..model.factory import ModelFactory
from .state import AgentState
from .vision_planner import ScreenVisionPlanner

logger = logging.getLogger("sivac.agent.planner")

PLANNER_SYSTEM_PROMPT = """You are SIVAC's Hierarchical UI Action Planner.
Your role is to decide the immediate NEXT ACTION to achieve the user's goal on the computer screen.

Available Action Types:
- click: Click an element or coordinate (target_id, or x, y)
- double_click: Double-click an element or coordinate
- right_click: Right-click an element or coordinate
- type: Type text into the focused field (text, target_id, press_enter)
- press: Press a single key (key, e.g. 'enter', 'tab', 'escape')
- hotkey: Press simultaneous keys (keys, e.g. ['ctrl', 'c'])
- scroll: Scroll page (amount: positive=up, negative=down)
- wait: Pause for UI to load (seconds)
- browser_navigate: Navigate to a URL (url)
- app_open: Open desktop application (app_name)
- finish: Declare task complete (reason)

You MUST respond ONLY with a single valid JSON object in the following format:
{
  "subgoal": "Brief explanation of what this step accomplishes",
  "action": "click|type|press|hotkey|scroll|wait|browser_navigate|app_open|finish",
  "target_id": "optional element ID from the UI elements list",
  "x": 100,
  "y": 200,
  "text": "text to type if action=type",
  "key": "key if action=press",
  "keys": ["key1", "key2"],
  "url": "url if action=browser_navigate",
  "app_name": "app if action=app_open",
  "seconds": 1.0,
  "reason": "Detailed reason why this action was selected"
}
"""


class HierarchicalPlanner:
    """Plans structured Next Actions using reasoning LLMs and contextual guidance."""

    def __init__(self, model_override: Optional[Any] = None) -> None:
        """Initialise planner with reasoning model."""
        self._model = model_override
        self._vision_planner = ScreenVisionPlanner()

    def _get_model(self):
        if self._model == "fallback":
            return None
        if self._model:
            return self._model
        try:
            return ModelFactory.get_reasoning_model()
        except Exception as e:
            logger.warning(f"Could not initialize reasoning model: {e}")
            return None

    def plan_next_action(self, state: AgentState) -> Tuple[str, ActionCommand]:
        """Decide the next subgoal and ActionCommand for the given state.

        Planning priority:
          1. ScreenVisionPlanner (VLM sees screenshot)
          2. Text-based reasoning LLM
          3. Heuristic deterministic fallback

        Args:
            state: The current AgentState.

        Returns:
            Tuple of (planned_subgoal_string, ActionCommand).
        """
        goal = state.get("goal", "")
        ui_state: Optional[UIState] = state.get("current_ui_state")
        action_history = state.get("action_history", [])
        step_count = state.get("step_count", 0)

        # ------------------------------------------------------------------
        # TIER 1: Screenshot-based vision planning
        # The VLM sees the actual screen and decides what to do.
        # ------------------------------------------------------------------
        vision_result = self._vision_planner.plan(
            goal=goal,
            ui_state=ui_state,
            action_history=action_history,
            step_count=step_count,
        )
        if vision_result is not None:
            subgoal, cmd = vision_result
            logger.info("[Planner] Using VLM vision plan: %s -> %s", subgoal, cmd.action.value)
            return subgoal, cmd

        logger.info("[Planner] Vision planner unavailable — trying text LLM")

        # ------------------------------------------------------------------
        # TIER 2: Text-based reasoning LLM
        # ------------------------------------------------------------------
        retrieved_guidance = state.get("retrieved_guidance", "")
        prompt_content = self._build_planner_prompt(
            goal=goal,
            ui_state=ui_state,
            retrieved_guidance=retrieved_guidance,
            action_history=action_history,
            step_count=step_count,
        )

        model = self._get_model()
        if model is not None:
            try:
                response = model.invoke([
                    SystemMessage(content=PLANNER_SYSTEM_PROMPT),
                    HumanMessage(content=prompt_content),
                ])
                subgoal, cmd = self._parse_llm_response(response.content, ui_state)
                if cmd is not None:
                    return subgoal, cmd
            except Exception as e:
                logger.warning(
                    "Reasoning model invocation failed (%s), falling back to deterministic reflection",
                    e,
                )

        # ------------------------------------------------------------------
        # TIER 3: Heuristic fallback
        # ------------------------------------------------------------------
        return self._heuristic_fallback_plan(state)

    def _build_planner_prompt(
        self,
        goal: str,
        ui_state: Optional[UIState],
        retrieved_guidance: str,
        action_history: List[Dict[str, Any]],
        step_count: int,
    ) -> str:
        """Construct the prompt containing observations and memory guidance."""
        lines = [
            f"Current Goal: {goal}",
            f"Step Count: {step_count}",
        ]

        if ui_state:
            lines.append(f"Active Application: {ui_state.application}")
            lines.append(f"Window Title: {ui_state.window_title}")
            if ui_state.url:
                lines.append(f"Current URL: {ui_state.url}")

            # Top interactive elements (limit to 30 for token efficiency)
            if ui_state.elements:
                lines.append("\nDetected Interactive UI Elements:")
                for elem in ui_state.elements[:30]:
                    cx, cy = elem.bbox.center
                    text_disp = f"'{elem.text}'" if elem.text else ""
                    lines.append(f"  - [{elem.id}] {elem.type} {text_disp} at center=({cx}, {cy})")
            else:
                lines.append("\nDetected Interactive UI Elements: None detected")

        if retrieved_guidance:
            lines.append(f"\n{retrieved_guidance}")

        if action_history:
            lines.append("\nRecent Action History:")
            for item in action_history[-4:]:
                action = item.get("action", "")
                reason = item.get("reason", "")
                lines.append(f"  - Executed: {action} ({reason})")

        lines.append("\nOutput the JSON plan for the NEXT ACTION now:")
        return "\n".join(lines)

    def _parse_llm_response(
        self, text: str, ui_state: Optional[UIState]
    ) -> Tuple[str, Optional[ActionCommand]]:
        """Parse structured JSON from model response text."""
        # Find JSON object inside text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return "", None

        try:
            data = json.loads(match.group(0))
            subgoal = data.get("subgoal", "")
            raw_action = data.get("action", "wait").lower()

            # Map raw action string to ActionType
            action_type = ActionType(raw_action) if raw_action in [a.value for a in ActionType] else ActionType.WAIT

            # Resolve coordinates if target_id provided but coordinates omitted
            target_id = data.get("target_id")
            x = data.get("x")
            y = data.get("y")
            if target_id and ui_state and (x is None or y is None):
                elem = ui_state.find_by_id(target_id)
                if elem:
                    x, y = elem.bbox.center

            cmd = ActionCommand(
                action=action_type,
                target_id=target_id,
                x=x,
                y=y,
                text=data.get("text"),
                key=data.get("key"),
                keys=data.get("keys"),
                amount=data.get("amount"),
                seconds=data.get("seconds", 1.0),
                url=data.get("url"),
                app_name=data.get("app_name"),
                reason=data.get("reason", subgoal),
            )
            return subgoal, cmd
        except Exception as e:
            logger.warning(f"Error parsing planner JSON response: {e}")
            return "", None

    def _heuristic_fallback_plan(self, state: AgentState) -> Tuple[str, ActionCommand]:
        """Intelligent deterministic fallback plan when LLM is unavailable.

        Handles multi-step goals by tracking what has already been done via
        action_history and current UIState context.
        """
        goal = state.get("goal", "").lower()
        ui_state: Optional[UIState] = state.get("current_ui_state")
        history = state.get("action_history", [])

        # Helpers: what actions have we already executed?
        executed_actions = [h.get("action", "") for h in history]
        executed_apps = [h.get("app_name", "").lower() for h in history if h.get("app_name")]
        executed_urls = [h.get("url", "") for h in history if h.get("url")]

        active_app = (ui_state.application if ui_state else "Desktop").lower()
        active_title = (ui_state.window_title if ui_state else "").lower()
        current_url = (ui_state.url if ui_state else "") or ""

        # ----------------------------------------------------------------
        # PHASE 1: Open a desktop application if requested and not yet open
        # ----------------------------------------------------------------
        needs_app_open = any(w in goal for w in ("open", "launch", "start"))
        app_targets = {
            "chrome": "chrome",
            "google chrome": "chrome",
            "firefox": "firefox",
            "edge": "edge",
            "notepad": "notepad",
            "note pad": "notepad",   # handle space variant
            "notes": "notepad",      # common alias
            "notpad": "notepad",     # typo-tolerant
            "calculator": "calc",
            "calc": "calc",
            "excel": "excel",
            "word": "winword",
            "code": "code",
            "vscode": "code",
            "explorer": "explorer",
            "paint": "mspaint",
        }
        target_app = None
        for keyword, exe in app_targets.items():
            if keyword in goal:
                target_app = exe
                break

        if target_app and target_app not in executed_apps:
            # Check if app is already running by inspecting window title
            app_running = any(target_app.replace(".exe", "") in active_title or
                              target_app.replace(".exe", "") in active_app
                              for _ in [1])
            if not app_running:
                return f"Launch application '{target_app}'", ActionCommand(
                    action=ActionType.APP_OPEN,
                    app_name=target_app,
                    reason=f"Opening '{target_app}' as requested by goal",
                )

        # ----------------------------------------------------------------
        # PHASE 2: Browser navigation if a website is mentioned in goal
        # ----------------------------------------------------------------
        site_map = {
            "youtube": "https://www.youtube.com",
            "google": "https://www.google.com",
            "github": "https://www.github.com",
            "twitter": "https://www.twitter.com",
            "facebook": "https://www.facebook.com",
            "reddit": "https://www.reddit.com",
        }
        # Also handle explicit URLs
        url_match = re.search(r"https?://\S+|www\.\S+", goal)
        target_url = None
        target_site = None

        for site, url in site_map.items():
            if site in goal:
                target_url = url
                target_site = site
                break

        if url_match and not target_url:
            target_url = url_match.group(0)
            if not target_url.startswith("http"):
                target_url = "https://" + target_url

        if target_url:
            # Navigate if not already on the target URL
            already_there = target_url.rstrip("/") in current_url.rstrip("/")
            browser_open = any(b in active_app for b in ("chrome", "firefox", "edge", "google chrome", "mozilla"))

            if not already_there and target_url not in executed_urls:
                # Count how many consecutive waits we've done since launching the app
                wait_count = sum(1 for h in history if h.get("action") == "wait")
                last_action = executed_actions[-1] if executed_actions else ""

                if not browser_open and target_app in ("chrome", "firefox", "edge") and last_action == "app_open" and wait_count < 2:
                    # Give the browser 1 chance to open before proceeding
                    return "Wait for browser to open", ActionCommand(
                        action=ActionType.WAIT,
                        seconds=2.0,
                        reason="Waiting for browser to fully launch before navigating",
                    )
                # Browser is open or we've waited long enough — navigate
                return f"Navigate to {target_url}", ActionCommand(
                    action=ActionType.BROWSER_NAVIGATE,
                    url=target_url,
                    reason=f"Navigating to requested site '{target_site or target_url}'",
                )


        # ----------------------------------------------------------------
        # PHASE 3: Write/type content if goal asks to write something
        # ----------------------------------------------------------------
        write_keywords = ("write", "type", "write the", "write an", "write a", "draft")
        if any(kw in goal for kw in write_keywords):
            # Extract what to write
            write_match = re.search(
                r"(?:write|type|draft)\s+(?:the\s+|an?\s+)?(.+?)(?:\s+(?:in|on|into|there|to)\s+|$)",
                goal
            )
            content_topic = None
            if write_match:
                content_topic = write_match.group(1).strip()
            elif "about" in goal:
                about_match = re.search(r"about\s+(.+?)(?:\s+(?:in|on|there)|$)", goal)
                if about_match:
                    content_topic = "AI" if "ai" in about_match.group(1).lower() else about_match.group(1).strip()

            # Check if we've already typed content
            already_wrote = any(h.get("action") == "type" and len(str(h.get("text", ""))) > 20 for h in history)

            if content_topic and not already_wrote:
                # Generate inline fallback content
                content_map = {
                    "essay about ai": (
                        "Artificial Intelligence: The Future of Technology\n\n"
                        "Artificial Intelligence (AI) is one of the most transformative technologies "
                        "of the 21st century. It enables machines to learn, reason, and solve problems "
                        "that previously required human intelligence.\n\n"
                        "AI applications span across healthcare, education, finance, transportation, "
                        "and countless other fields. Machine learning algorithms can diagnose diseases, "
                        "optimize traffic flow, personalize learning experiences, and detect fraud with "
                        "extraordinary accuracy.\n\n"
                        "However, AI also raises important ethical questions. Issues of bias, privacy, "
                        "job displacement, and autonomous decision-making require careful consideration "
                        "from society, governments, and technologists alike.\n\n"
                        "As AI continues to evolve, collaboration between humans and intelligent systems "
                        "will define the next era of human progress. The key lies in developing AI that "
                        "is transparent, fair, and aligned with human values.\n"
                    ),
                    "ai": (
                        "Artificial Intelligence (AI) refers to computer systems that perform tasks "
                        "that typically require human intelligence, such as visual perception, speech "
                        "recognition, decision-making, and language understanding. AI is powered by "
                        "machine learning, deep learning, and neural networks.\n"
                    ),
                }

                topic_lower = (content_topic or "").lower()
                typed_content = None
                for key, val in content_map.items():
                    if key in topic_lower or topic_lower in key:
                        typed_content = val
                        break

                # Default content if no template matched
                if not typed_content:
                    typed_content = (
                        f"Topic: {content_topic.title()}\n\n"
                        f"This document covers the topic of {content_topic}. "
                        f"The subject is important and has many applications in today's world. "
                        f"Further details and analysis follow in subsequent sections.\n"
                    )

                # Find a text area / edit control in the UI elements
                if ui_state and ui_state.elements:
                    for elem in ui_state.elements:
                        elem_type = (elem.type or "").lower()
                        if elem_type in ("edit", "text", "textarea", "document", "richedit"):
                            cx, cy = elem.bbox.center
                            return f"Type content into text area", ActionCommand(
                                action=ActionType.TYPE_TEXT,
                                target_id=elem.id,
                                x=cx,
                                y=cy,
                                text=typed_content,
                                reason=f"Writing '{content_topic}' content into detected text field",
                            )

                # No specific text area found — click center of screen and type
                if ui_state and ui_state.dimensions:
                    w, h = ui_state.dimensions
                    cx, cy = w // 2, h // 2
                else:
                    cx, cy = 960, 540

                # Only type if we have an appropriate app open (notepad, word, etc.)
                writing_app_open = any(
                    app in active_app for app in ("notepad", "word", "code", "editor", "write")
                )
                if writing_app_open or (target_app in ("notepad", "winword", "code") and "app_open" in executed_actions):
                    return f"Type '{content_topic}' content", ActionCommand(
                        action=ActionType.TYPE_TEXT,
                        x=cx,
                        y=cy,
                        text=typed_content,
                        reason=f"Typing {content_topic} content into the writing application",
                    )

        # ----------------------------------------------------------------
        # PHASE 4: Perform a search if "search" is in goal
        # ----------------------------------------------------------------
        if "search" in goal or "find" in goal or "look for" in goal:
            # Extract what to search for
            search_terms_match = re.search(
                r"(?:search|find|look for|search for)\s+(?:for\s+)?(.+?)(?:\s+on\s+|\s+in\s+|$)",
                goal
            )
            search_query = None
            if search_terms_match:
                search_query = search_terms_match.group(1).strip()
                # Remove trailing site names
                for site in site_map:
                    search_query = re.sub(rf"\s+{site}\s*$", "", search_query).strip()

            if search_query:
                # Check if we've already typed a search
                already_searched = any(
                    h.get("action") == "type" and search_query.lower()[:10] in str(h.get("text", "")).lower()
                    for h in history
                )

                if not already_searched:
                    # Try to find search bar in UI elements
                    if ui_state and ui_state.elements:
                        for elem in ui_state.elements:
                            text_lower = (elem.text or "").lower()
                            elem_type = (elem.type or "").lower()
                            if elem_type in ("input", "search", "combobox", "edit") or "search" in text_lower:
                                cx, cy = elem.bbox.center
                                return f"Type search query into search bar", ActionCommand(
                                    action=ActionType.TYPE_TEXT,
                                    target_id=elem.id,
                                    x=cx,
                                    y=cy,
                                    text=search_query,
                                    press_enter=True,
                                    reason=f"Typing search query '{search_query}' into detected search field",
                                )

                    # No search field detected yet — try pressing Ctrl+L to focus address bar or wait
                    if any("navigate" in h.get("action", "") or "browser_navigate" in h.get("action", "") for h in history):
                        return "Focus browser search/address bar", ActionCommand(
                            action=ActionType.HOTKEY,
                            keys=["ctrl", "l"],
                            reason="Focusing address/search bar to type search query",
                        )

                    # Wait for page to load fully then re-perceive
                    if len(history) > 0:
                        return "Wait for page to load", ActionCommand(
                            action=ActionType.WAIT,
                            seconds=2.0,
                            reason="Waiting for page to finish loading before searching",
                        )

        # ----------------------------------------------------------------
        # PHASE 4: Click matching UI elements by keyword
        # ----------------------------------------------------------------
        if ui_state and ui_state.elements:
            clicked_targets = {h.get("target_id") for h in history if h.get("target_id")}
            keywords = [
                w for w in goal.split()
                if len(w) > 3 and w not in ("open", "click", "find", "search", "with", "then", "youtube", "google", "chrome")
            ]

            for elem in ui_state.elements:
                if elem.id in clicked_targets:
                    continue
                text_lower = (elem.text or "").lower()
                if any(kw in text_lower for kw in keywords):
                    cx, cy = elem.bbox.center
                    return f"Click matching element '{elem.text}'", ActionCommand(
                        action=ActionType.CLICK,
                        target_id=elem.id,
                        x=cx,
                        y=cy,
                        reason=f"Clicking element with text matching goal keyword",
                    )

        # ----------------------------------------------------------------
        # PHASE 5: Fallback wait if we just launched an app and screen hasn't settled
        # ----------------------------------------------------------------
        if executed_actions and executed_actions[-1] in ("app_open", "browser_navigate", "wait"):
            if len(history) < 8:  # Only wait a reasonable number of times
                return "Wait for UI to settle", ActionCommand(
                    action=ActionType.WAIT,
                    seconds=2.0,
                    reason="Waiting for application/navigation to finish loading",
                )

        # ----------------------------------------------------------------
        # PHASE 6: Declare completion only if goal steps are done
        # ----------------------------------------------------------------
        goal_steps_done = True

        # Check if we needed to open an app — did we?
        if needs_app_open and target_app and target_app not in executed_apps:
            goal_steps_done = False

        # Check if we needed to navigate — did we?
        if target_url and target_url not in executed_urls:
            goal_steps_done = False

        # Check if we needed to search — did we?
        if ("search" in goal or "find" in goal) and not any(h.get("action") == "type" for h in history):
            goal_steps_done = False

        if goal_steps_done and len(history) >= 2:
            return "Task objectives achieved", ActionCommand(
                action=ActionType.FINISH,
                reason="All identified goal phases completed successfully",
            )

        # ----------------------------------------------------------------
        # DEFAULT: Brief wait for the UI to update
        # ----------------------------------------------------------------
        return "Wait for UI to settle", ActionCommand(
            action=ActionType.WAIT,
            seconds=1.5,
            reason="Pausing for UI update before re-evaluating state",
        )
