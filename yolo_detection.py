from pathlib import Path
from typing import List, Dict, Tuple
import os
import re

import cv2
from ultralytics import YOLO
import easyocr


def detect_player_boxes(
    image_path: str,
    yolo_model: YOLO,
    conf_threshold: float = 0.2,
) -> List[Tuple[int, int, int, int]]:
    """
    Detect person bounding boxes in an image.

    Returns:
        List of (x1, y1, x2, y2) boxes.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not open image: {image_path}")

    results = yolo_model.predict(source=image, conf=conf_threshold, verbose=False)

    boxes = []
    class_id_filter = 0  # person

    h, w = image.shape[:2]

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls.item())
            if cls_id != class_id_filter:
                continue

            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))

    return boxes


def extract_digits_from_boxes(
        image_path: str,
        yolo_model: YOLO,
        ocr_model: easyocr.Reader,
) -> List[str]:
    """
    Detect digits only inside the player bounding boxes.

    Args:
        image_path: Path to the input image.

    Returns:
        List of 1-2 digit strings found inside player boxes.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not open image: {image_path}")

    boxes = detect_player_boxes(image_path, yolo_model=yolo_model, conf_threshold=0.25)
    digits = []

    for x1, y1, x2, y2 in boxes:
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        results = ocr_model.readtext(crop)

        for bbox, text, confidence in results:
            matches = re.findall(r"\b\d{1,2}\b", text)
            digits.extend(matches)

    return digits


def detect_players_and_annotate(
    image_path: str,
    yolo_model: YOLO,
    conf_threshold: float = 0.2,
) -> List:
    """
    Run YOLO26 player detection and draw bounding boxes on image.

    Returns the annotated image path.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not open image: {image_path}")

    boxes = detect_player_boxes(image_path, yolo_model=yolo_model, conf_threshold=conf_threshold)

    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    os.makedirs("boxed_images", exist_ok=True)
    annotated_path = os.path.join("boxed_images", Path(image_path).name)
    cv2.imwrite(annotated_path, image)

    return annotated_path