import re
from typing import List
import cv2
import easyocr


# Initialize OCR once
reader = easyocr.Reader(["en"], gpu=False)


def extract_digits_from_image(image_path: str) -> List[str]:
    """
    Use EasyOCR to detect text in an image and return only 1–2 digit numbers
    (typical sports jersey numbers).

    Args:
        image_path: Path to the input image.

    Returns:
        List of digit strings found in the image.
        Example: ["23", "7"]
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not open image: {image_path}")

    results = reader.readtext(image)

    digits = []

    for bbox, text, confidence in results:
        # Only keep 1–2 digit numbers
        matches = re.findall(r"\b\d{1,2}\b", text)
        digits.extend(matches)

    return digits