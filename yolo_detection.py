from pathlib import Path
from typing import List, Dict
import os

import cv2
from ultralytics import YOLO


def detect_players_and_annotate(
    image_path: str,
    yolo_model_path: str = "yolo26n.pt",
    conf_threshold: float = 0.2,
) -> str:
    """
    Run YOLO26 player detection and draw bounding boxes on image.

    Returns the annotated image path.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not open image: {image_path}")

    model = YOLO(yolo_model_path)
    results = model.predict(source=image, conf=conf_threshold, verbose=False)

    # If your model includes multiple classes, set class_id_filter to player class id.
    class_id_filter = 0

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls.item())
            if class_id_filter is not None and cls_id != class_id_filter:
                continue

            conf = float(box.conf.item())
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    annotated_path = os.path.join("boxed_images", image_path.split("/")[-1])
    cv2.imwrite(annotated_path, image)

    return annotated_path