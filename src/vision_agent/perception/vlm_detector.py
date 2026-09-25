import json
import base64
import io
import re
import logging
from typing import List, Tuple
from PIL import Image
from langchain_core.messages import HumanMessage

from vision_agent.model import ModelFactory
from .state import BBox, UIElement

logger = logging.getLogger("sivac.perception.vlm")

_PROMPT = """Analyze this desktop/browser screenshot. Identify ALL visible interactive UI elements.
For each element provide its type, visible text label, and bounding box in normalized coordinates.

Bounding box format: [ymin, xmin, ymax, xmax] scaled from 0 to 1000 (0,0 = top-left, 1000,1000 = bottom-right).

Return ONLY a valid JSON array:
```json
[
  {"type": "button|input|link|icon|text|tab|menu", "text": "label or description", "bbox": [ymin, xmin, ymax, xmax]}
]
```"""


class VLMElemDetector:
    """Detects interactive UI elements in desktop/browser screenshots using VLM analysis."""

    def __init__(self) -> None:
        self._model = None  # Lazy-loaded on first detect() call

    def _get_model(self):
        """Lazy-load the vision model to avoid crashing at startup."""
        if self._model is None:
            try:
                self._model = ModelFactory.get_vision_model()
            except Exception as e:
                logger.warning(f"VLM model unavailable: {e}")
                return None
        return self._model

    def _encode_image(self, image: Image.Image) -> str:
        """Convert a PIL Image to a base64-encoded PNG string."""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def detect(self, image: Image.Image, dimensions: Tuple[int, int]) -> List[UIElement]:
        """
        Send the screenshot to the VLM and get back a list of detected UI elements.

        Args:
            image: The screenshot as a PIL Image.
            dimensions: (width, height) in pixels.

        Returns:
            List of UIElement objects with bounding boxes in screen pixel coordinates.
        """
        width, height = dimensions
        b64 = self._encode_image(image)

        model = self._get_model()
        if model is None:
            logger.debug("VLM model not available, skipping visual detection.")
            return []

        try:
            response = model.invoke([
                HumanMessage(content=[
                    {"type": "text", "text": _PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                ])
            ])
            raw_content = response.content if hasattr(response, "content") else str(response)
            return self._parse(raw_content, width, height)
        except Exception as e:
            logger.warning(f"VLM detection failed: {e}")
            return []

    def _parse(self, raw: str, width: int, height: int) -> List[UIElement]:
        """Parse the raw VLM response text into UIElement objects."""
        elements: List[UIElement] = []

        match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", raw)
        json_str = match.group(1) if match else raw.strip()

        try:
            data = json.loads(json_str)
            for idx, item in enumerate(data if isinstance(data, list) else []):
                bbox_raw = item.get("bbox", [])
                if len(bbox_raw) == 4:
                    ymin, xmin, ymax, xmax = [float(v) for v in bbox_raw]
                    scale = 1000.0 if max(ymin, xmin, ymax, xmax) <= 1000 else 1.0

                    bbox = BBox(
                        x_min=int((xmin / scale) * width),
                        y_min=int((ymin / scale) * height),
                        x_max=int((xmax / scale) * width),
                        y_max=int((ymax / scale) * height),
                    )

                    elements.append(UIElement(
                        id=f"vlm_{idx + 1}",
                        type=str(item.get("type", "element")),
                        text=str(item.get("text", "")).strip(),
                        bbox=bbox,
                        confidence=0.9,
                        source="vlm",
                    ))
        except Exception as e:
            logger.debug(f"VLM parse error: {e}")

        return elements
