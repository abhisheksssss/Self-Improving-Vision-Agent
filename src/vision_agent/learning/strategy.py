"""Adaptive Strategy Manager for SIVAC Phase 7.

Manages dynamic perception modality prioritization (DOM vs UIA vs VLM) based
on target application context and orchestrates feedback reinforcement loops.
"""

import logging
from typing import List, Dict, Any, Optional

from ..memory.semantic import SemanticMemoryManager

logger = logging.getLogger("sivac.learning.strategy")

# Modality Mapping Rules
BROWSER_APPS = {"chrome", "firefox", "edge", "brave", "safari", "browser"}
DESKTOP_UIA_APPS = {"notepad", "calculator", "calc", "excel", "word", "explorer", "cmd", "powershell"}


class StrategyManager:
    """Optimizes perception strategies and reinforces heuristic confidence."""

    def __init__(self, semantic_memory: Optional[SemanticMemoryManager] = None) -> None:
        """Initialise StrategyManager with semantic memory manager."""
        self.semantic = semantic_memory or SemanticMemoryManager()

    def get_preferred_modality(self, application: str, goal: str = "") -> str:
        """Determine optimal primary perception engine based on context.

        Args:
            application: Active application name.
            goal: Natural language user goal.

        Returns:
            One of: 'dom', 'uia', 'vlm', or 'hybrid'.
        """
        app_lower = application.lower().strip()
        goal_lower = goal.lower()

        # Web browser contexts favor DOM inspection
        if any(b in app_lower for b in BROWSER_APPS) or "http://" in goal_lower or "https://" in goal_lower:
            return "dom"

        # Standard Windows desktop controls favor Windows UI Automation
        if any(d in app_lower for d in DESKTOP_UIA_APPS):
            return "uia"

        # Fallback to hybrid merger combining all sources
        return "hybrid"

    def reinforce_skills_used(self, skill_ids: List[str], success: bool) -> Dict[str, float]:
        """Adjust confidence ratings for skills retrieved during task execution.

        Args:
            skill_ids: List of skill UUIDs that were injected into the planner.
            success: Whether the overall task achieved its goal.

        Returns:
            Dictionary mapping skill_id to its updated confidence score.
        """
        results: Dict[str, float] = {}
        for sid in set(skill_ids):
            updated = self.semantic.record_feedback(skill_id=sid, success=success)
            if updated:
                results[sid] = updated.confidence_score
                logger.info(
                    f"Skill [{sid}] reinforced (success={success}): "
                    f"new confidence={updated.confidence_score:.2f}"
                )
        return results
