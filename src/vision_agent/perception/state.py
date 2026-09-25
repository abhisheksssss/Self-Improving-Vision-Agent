import time
from typing import List, Tuple, Dict, Any, Literal, Optional
from pydantic import BaseModel, Field


class BBox(BaseModel):
    """
    Bounding Box — represents where an element is on screen.
    All values are in screen pixel coordinates (top-left = 0,0).
    """
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def center(self) -> Tuple[int, int]:
        """Calculate center point (x, y) for safe clicking."""
        return (self.x_min + self.x_max) // 2, (self.y_min + self.y_max) // 2

    @property
    def width(self) -> int:
        """Pixel width of the element."""
        return max(0, self.x_max - self.x_min)

    @property
    def height(self) -> int:
        """Pixel height of the element."""
        return max(0, self.y_max - self.y_min)

    @property
    def area(self) -> int:
        """Total pixel area. Used in deduplication (IoU calculation)."""
        return self.width * self.height

    def to_list(self) -> List[int]:
        """Return box as a plain list [x_min, y_min, x_max, y_max]."""
        return [self.x_min, self.y_min, self.x_max, self.y_max]

    def iou(self, other: "BBox") -> float:
        """
        Intersection over Union — measures overlap between two boxes.
        Returns 0.0 (no overlap) to 1.0 (perfect overlap).
        """
        ix_min = max(self.x_min, other.x_min)
        iy_min = max(self.y_min, other.y_min)
        ix_max = min(self.x_max, other.x_max)
        iy_max = min(self.y_max, other.y_max)

        inter = max(0, ix_max - ix_min) * max(0, iy_max - iy_min)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


class UIElement(BaseModel):
    """Represents one detected UI element on screen."""
    id: str
    type: str = "element"
    text: str = ""
    bbox: BBox
    confidence: float = 1.0
    source: Literal["vlm", "ocr", "dom", "uia"] = "vlm"
    attributes: Dict[str, Any] = Field(default_factory=dict)


class UIState(BaseModel):
    """The complete structured snapshot of the screen at one moment in time."""
    application: str = "Desktop"
    window_title: str = ""
    url: Optional[str] = None
    elements: List[UIElement] = Field(default_factory=list)
    screenshot_path: Optional[str] = None
    dimensions: Tuple[int, int] = (1920, 1080)
    timestamp: float = Field(default_factory=time.time)

    def find_by_id(self, elem_id: str) -> Optional[UIElement]:
        """Find a specific element by its ID."""
        return next((e for e in self.elements if e.id == elem_id), None)

    def get_element_by_id(self, elem_id: str) -> Optional[UIElement]:
        """Alias for find_by_id."""
        return self.find_by_id(elem_id)

    def find_by_text(self, query: str) -> List[UIElement]:
        """Find elements containing the query string (case-insensitive)."""
        q = query.lower()
        return [e for e in self.elements if q in e.text.lower()]

    def find_by_type(self, elem_type: str) -> List[UIElement]:
        """Find elements of a specific type (e.g. 'button', 'input')."""
        return [e for e in self.elements if e.type.lower() == elem_type.lower()]
