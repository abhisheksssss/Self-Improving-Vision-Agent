import logging
from typing import List, Tuple
from PIL import Image

from .state import BBox, UIElement

logger = logging.getLogger("sivac.perception.ocr")


class OCRDetector:
    """Detects text regions and coordinates on screen using pytesseract OCR."""

    def __init__(self, min_confidence: float = 40.0) -> None:
        self.min_confidence = min_confidence

    def detect(self, image: Image.Image, dimensions: Tuple[int, int]) -> List[UIElement]:
        """
        Extract text elements and bounding boxes from the screenshot.

        Args:
            image: PIL Image of the desktop screenshot.
            dimensions: (width, height) in screen pixels.

        Returns:
            List of UIElement objects with source="ocr".
        """
        elements = []

        try:
            import pytesseract

            gray_image = image.convert("L")
            data = pytesseract.image_to_data(
                gray_image, output_type=pytesseract.Output.DICT
            )

            idx = 1
            total_boxes = len(data["text"])

            for i in range(total_boxes):
                text = data["text"][i].strip()
                conf = float(data["conf"][i])

                if text and conf >= self.min_confidence:
                    x = data["left"][i]
                    y = data["top"][i]
                    w = data["width"][i]
                    h = data["height"][i]

                    if w > 3 and h > 3:
                        bbox = BBox(
                            x_min=x,
                            y_min=y,
                            x_max=x + w,
                            y_max=y + h
                        )

                        elements.append(UIElement(
                            id=f"ocr_{idx}",
                            type="text",
                            text=text,
                            bbox=bbox,
                            confidence=round(conf / 100.0, 2),
                            source="ocr",
                        ))
                        idx += 1

        except Exception as e:
            logger.debug(f"OCR detection skipped or unavailable: {e}")

        return elements
