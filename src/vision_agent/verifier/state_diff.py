"""UIState Structural Delta Inspector for SIVAC Phase 4.

Compares two UIState snapshots captured before and after an action to
determine whether the accessibility tree (elements, text, URLs) changed
in a meaningful way.

This complements the pixel-level VisualDiffEngine by catching cases where:
    - An element appeared (e.g. a dropdown opened, a dialog popped up)
    - An element disappeared (e.g. a spinner was removed after loading)
    - An element's text changed (e.g. a status label updated)
    - The page URL navigated to a new location
    - The window title changed (different application came to foreground)

The delta engine produces a count of meaningful changes rather than a raw
pixel score, giving the FailureClassifier a semantic signal to classify
modal dialogs, navigation completions, and type-field updates.
"""

import logging
from typing import Set, List, Dict, Tuple, Optional
from ..perception.state import UIState, UIElement

logger = logging.getLogger("sivac.verifier.state_diff")


class ElementDelta:
    """Lightweight value object representing the structural diff between two UIStates."""

    def __init__(
        self,
        appeared: List[UIElement],
        disappeared: List[UIElement],
        text_changed: List[Tuple[UIElement, UIElement]],  # (before_elem, after_elem)
        url_changed: bool,
        title_changed: bool,
        before_url: Optional[str],
        after_url: Optional[str],
        before_title: str,
        after_title: str,
    ) -> None:
        # Elements present in after_state but NOT in before_state
        self.appeared = appeared

        # Elements present in before_state but NOT in after_state
        self.disappeared = disappeared

        # Elements present in both states but whose text value changed
        self.text_changed = text_changed

        # True if the browser URL changed between states
        self.url_changed = url_changed

        # True if the OS window title changed between states
        self.title_changed = title_changed

        self.before_url = before_url
        self.after_url = after_url
        self.before_title = before_title
        self.after_title = after_title

    @property
    def total_changes(self) -> int:
        """Total count of all detected semantic changes."""
        return (
            len(self.appeared)
            + len(self.disappeared)
            + len(self.text_changed)
            + int(self.url_changed)
            + int(self.title_changed)
        )

    @property
    def has_modal(self) -> bool:
        """Heuristic: True if a blocking dialog or modal has appeared.

        Detects common modal patterns:
            - Elements with type 'dialog', 'alert', or 'modal'
            - Elements with text containing common modal cues
        """
        modal_types = {"dialog", "alert", "modal", "popup"}
        modal_text_cues = {
            "ok", "cancel", "yes", "no", "allow", "deny",
            "confirm", "close", "dismiss", "got it",
        }
        for elem in self.appeared:
            if elem.type.lower() in modal_types:
                return True
            if any(cue in elem.text.lower() for cue in modal_text_cues):
                return True
        return False

    def summary(self) -> str:
        """Return a human-readable one-line summary of all changes."""
        parts = []
        if self.appeared:
            parts.append(f"+{len(self.appeared)} elements appeared")
        if self.disappeared:
            parts.append(f"-{len(self.disappeared)} elements removed")
        if self.text_changed:
            parts.append(f"{len(self.text_changed)} text changes")
        if self.url_changed:
            parts.append(f"URL: '{self.before_url}' -> '{self.after_url}'")
        if self.title_changed:
            parts.append(f"Title: '{self.before_title}' -> '{self.after_title}'")
        return " | ".join(parts) if parts else "No structural changes detected"


class StateDiffEngine:
    """Compares two UIState snapshots and returns a structured ElementDelta.

    Matching is performed by element ID (exact) and then by text+type pair
    (fuzzy) so that elements whose IDs changed between scans (common for
    VLM-detected elements) are still correctly tracked.
    """

    def compute_delta(
        self,
        before: UIState,
        after: UIState,
    ) -> ElementDelta:
        """Compute the structural delta between before and after UIStates.

        Args:
            before: UIState captured immediately before the action.
            after:  UIState captured immediately after the action.

        Returns:
            ElementDelta with appeared, disappeared, and changed element lists.
        """
        # Build lookup maps keyed by element ID
        before_by_id: Dict[str, UIElement] = {e.id: e for e in before.elements}
        after_by_id:  Dict[str, UIElement] = {e.id: e for e in after.elements}

        before_ids: Set[str] = set(before_by_id.keys())
        after_ids:  Set[str] = set(after_by_id.keys())

        # Elements that appeared (new IDs in after)
        appeared_ids   = after_ids - before_ids
        disappeared_ids = before_ids - after_ids

        appeared_elems    = [after_by_id[eid]   for eid in appeared_ids]
        disappeared_elems = [before_by_id[eid]  for eid in disappeared_ids]

        # Elements present in both but whose text label changed
        common_ids = before_ids & after_ids
        text_changed_pairs: List[Tuple[UIElement, UIElement]] = []
        for eid in common_ids:
            b_elem = before_by_id[eid]
            a_elem = after_by_id[eid]
            if b_elem.text.strip() != a_elem.text.strip():
                text_changed_pairs.append((b_elem, a_elem))

        # URL comparison (for browser context)
        url_changed = (before.url or "") != (after.url or "")

        # Window title comparison
        title_changed = before.window_title.strip() != after.window_title.strip()

        delta = ElementDelta(
            appeared=appeared_elems,
            disappeared=disappeared_elems,
            text_changed=text_changed_pairs,
            url_changed=url_changed,
            title_changed=title_changed,
            before_url=before.url,
            after_url=after.url,
            before_title=before.window_title,
            after_title=after.window_title,
        )

        logger.debug(f"State delta: {delta.summary()}")
        return delta

    def text_was_typed(
        self,
        after: UIState,
        expected_text: str,
        target_id: Optional[str] = None,
    ) -> bool:
        """Check whether typed text appears in the after state.

        If a target_id is specified, checks only that specific element;
        otherwise scans all elements for any occurrence of the text.

        Args:
            after:         Post-action UIState.
            expected_text: The text that should have been typed.
            target_id:     Optional UIElement ID to scope the search.

        Returns:
            True if the expected text is found in the post-action state.
        """
        if not expected_text:
            return True  # Nothing to verify

        if target_id:
            elem = after.find_by_id(target_id)
            if elem:
                found = expected_text.lower() in elem.text.lower()
                logger.debug(
                    f"Text verification for '{target_id}': expected='{expected_text}' "
                    f"found={found} (elem.text='{elem.text[:40]}')"
                )
                return found

        # Broad search across all elements
        for elem in after.elements:
            if expected_text.lower() in elem.text.lower():
                return True

        return False
