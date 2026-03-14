# from pathlib import Path
# from typing import Optional, Tuple
# import os
# import re
# from functools import lru_cache

# import cv2
# from ultralytics import YOLO
# from paddleocr import PaddleOCR


# @lru_cache(maxsize=4)
# def _get_yolo_model(yolo_model_path: str) -> YOLO:
#     """
#     Cache model instances by path so we don't repeatedly reload weights.
#     """
#     return YOLO(yolo_model_path)


# @lru_cache(maxsize=1)
# def _get_paddleocr() -> PaddleOCR:
#     """
#     Cache PaddleOCR model so it is only loaded once.
#     """
#     return PaddleOCR(
#         use_doc_orientation_classify=False,
#         use_doc_unwarping=False,
#         use_textline_orientation=False
#     )


# def _extract_best_jersey_number(crop) -> Optional[Tuple[str, float]]:
#     """
#     Run PaddleOCR on multiple processed versions of a player crop and return
#     the best detected 1-2 digit jersey number along with confidence.

#     Returns:
#         (number_string, confidence) or None
#     """
#     if crop is None or crop.size == 0:
#         return None

#     ocr = _get_paddleocr()
#     h, w = crop.shape[:2]

#     if h < 20 or w < 20:
#         return None

#     candidates = []

#     # Use a less aggressive crop so we do not cut off numbers
#     y1 = int(0.10 * h)
#     y2 = int(0.85 * h)
#     x1 = int(0.05 * w)
#     x2 = int(0.95 * w)

#     torso = crop[y1:y2, x1:x2]
#     if torso is None or torso.size == 0:
#         torso = crop

#     # Build several OCR inputs
#     resized_color = cv2.resize(
#         torso, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC
#     )

#     gray = cv2.cvtColor(resized_color, cv2.COLOR_BGR2GRAY)

#     thresh = cv2.adaptiveThreshold(
#         gray,
#         255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY,
#         11,
#         2,
#     )

#     ocr_inputs = [
#         resized_color  # often best
#         # gray,           # sometimes better
#         # thresh,         # sometimes helps, sometimes hurts
#     ]

#     def collect_from_results(results):
#         found = []

#         if isinstance(results, list):
#             for res in results:
#                 if hasattr(res, "rec_texts"):
#                     texts = getattr(res, "rec_texts", [])
#                     scores = getattr(res, "rec_scores", [1.0] * len(texts))
#                     for text, score in zip(texts, scores):
#                         text = str(text).strip()
#                         if re.fullmatch(r"\d{1,2}", text):
#                             found.append((text, float(score)))

#                 elif isinstance(res, dict):
#                     texts = res.get("rec_texts", [])
#                     scores = res.get("rec_scores", [1.0] * len(texts))
#                     for text, score in zip(texts, scores):
#                         text = str(text).strip()
#                         if re.fullmatch(r"\d{1,2}", text):
#                             found.append((text, float(score)))

#                 elif isinstance(res, list):
#                     for line in res:
#                         if (
#                             isinstance(line, list)
#                             and len(line) >= 2
#                             and isinstance(line[1], tuple)
#                             and len(line[1]) >= 2
#                         ):
#                             text, score = line[1]
#                             text = str(text).strip()
#                             if re.fullmatch(r"\d{1,2}", text):
#                                 found.append((text, float(score)))
#         return found

#     for ocr_input in ocr_inputs:
#         try:
#             results = ocr.predict(input=ocr_input)
#             candidates.extend(collect_from_results(results))
#         except Exception:
#             try:
#                 results = ocr.ocr(ocr_input, cls=False)
#                 candidates.extend(collect_from_results(results))
#             except Exception:
#                 pass

#     if not candidates:
#         return None

#     return max(candidates, key=lambda x: x[1])


# def detect_players_and_annotate(
#     image_path: str,
#     yolo_model_path: str = "yolo26n.pt",
#     conf_threshold: float = 0.3,
# ) -> str:
#     """
#     Run YOLO player detection, crop each detected player region,
#     run PaddleOCR on the crop to detect jersey numbers,
#     and draw both boxes and OCR predictions.

#     Returns the annotated image path.
#     """
#     image = cv2.imread(image_path)
#     if image is None:
#         raise FileNotFoundError(f"Could not open image: {image_path}")

#     # Keep a clean copy for cropping so drawn boxes/text do not contaminate OCR
#     image_for_crops = image.copy()

#     model = _get_yolo_model(yolo_model_path)
#     results = model.predict(source=image, conf=conf_threshold, classes=[0], verbose=False)

#     # If your model includes multiple classes, set class_id_filter to player class id.
#     class_id_filter = None

#     predicted_numbers = []

#     for r in results:
#         for box in r.boxes:
#             cls_id = int(box.cls.item())
#             if class_id_filter is not None and cls_id != class_id_filter:
#                 continue

#             conf = float(box.conf.item())
#             x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

#             # Clamp coordinates to image bounds
#             h, w = image.shape[:2]
#             x1 = max(0, min(x1, w - 1))
#             y1 = max(0, min(y1, h - 1))
#             x2 = max(0, min(x2, w - 1))
#             y2 = max(0, min(y2, h - 1))

#             if x2 <= x1 or y2 <= y1:
#                 continue

#             # Crop the detected player region from the clean image
#             crop = image_for_crops[y1:y2, x1:x2]

#             # Run PaddleOCR on the crop
#             ocr_result = _extract_best_jersey_number(crop)

#             # Default label
#             label = f"player {conf:.2f}"

#             if ocr_result is not None:
#                 jersey_number, ocr_conf = ocr_result
#                 predicted_numbers.append(
#                     {
#                         "bbox": [x1, y1, x2, y2],
#                         "player_conf": conf,
#                         "jersey_number": jersey_number,
#                         "ocr_conf": ocr_conf,
#                     }
#                 )
#                 label = f"#{jersey_number} ({ocr_conf:.2f})"

#             # Draw player box
#             cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

#             # Draw label above the box
#             cv2.putText(
#                 image,
#                 label,
#                 (x1, max(20, y1 - 10)),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7,
#                 (0, 255, 0),
#                 2,
#                 cv2.LINE_AA,
#             )

#     os.makedirs("boxed_images", exist_ok=True)
#     annotated_path = os.path.join("boxed_images", Path(image_path).name)
#     cv2.imwrite(annotated_path, image)

#     return annotated_path


# from pathlib import Path
# from typing import List, Dict, Optional, Tuple
# import os
# import re
# from functools import lru_cache

# import cv2
# from ultralytics import YOLO
# from paddleocr import PaddleOCR

# @lru_cache(maxsize=4)
# def _get_yolo_model(yolo_model_path: str) -> YOLO:
#     """
#     Cache model instances by path so we don't repeatedly reload weights.
#     """
#     return YOLO(yolo_model_path)

# @lru_cache(maxsize=1)
# def _get_paddleocr() -> PaddleOCR:
#     """
#     Cache PaddleOCR model so it is only loaded once.
#     """
#     return PaddleOCR(
#         use_doc_orientation_classify=False,
#         use_doc_unwarping=False,
#         use_textline_orientation=False
#     )

# def _extract_best_jersey_number(crop) -> Optional[Tuple[str, float]]:
#     """
#     Run PaddleOCR on a cropped player image and return the best detected
#     1-2 digit jersey number along with confidence.

#     Returns:
#         (number_string, confidence) or None
#     """
#     ocr = _get_paddleocr()

#     # PaddleOCR can return different structures depending on version.
#     # This helper tries to handle the common ones.
#     candidates = []

#     try:
#         # Newer PaddleOCR API
#         results = ocr.predict(input=crop)
#     except Exception:
#         # Older PaddleOCR API
#         results = ocr.ocr(crop, cls=False)

#     # -----------------------------
#     # Handle newer predict() output
#     # -----------------------------
#     if isinstance(results, list):
#         for res in results:
#             # Some versions return an object with a "rec_texts" field
#             if hasattr(res, "rec_texts"):
#                 texts = getattr(res, "rec_texts", [])
#                 scores = getattr(res, "rec_scores", [1.0] * len(texts))
#                 for text, score in zip(texts, scores):
#                     text = str(text).strip()
#                     match = re.fullmatch(r"\d{1,2}", text)
#                     if match:
#                         candidates.append((text, float(score)))

#             # Some versions return dictionaries
#             elif isinstance(res, dict):
#                 texts = res.get("rec_texts", [])
#                 scores = res.get("rec_scores", [1.0] * len(texts))
#                 for text, score in zip(texts, scores):
#                     text = str(text).strip()
#                     match = re.fullmatch(r"\d{1,2}", text)
#                     if match:
#                         candidates.append((text, float(score)))

#             # Older ocr() style:
#             # [[ [box], (text, conf) ], [ [box], (text, conf) ], ... ]
#             elif isinstance(res, list):
#                 for line in res:
#                     if (
#                         isinstance(line, list)
#                         and len(line) >= 2
#                         and isinstance(line[1], tuple)
#                         and len(line[1]) >= 2
#                     ):
#                         text, score = line[1]
#                         text = str(text).strip()
#                         match = re.fullmatch(r"\d{1,2}", text)
#                         if match:
#                             candidates.append((text, float(score)))

#     if not candidates:
#         return None

#     # Return highest-confidence candidate
#     return max(candidates, key=lambda x: x[1])


# def detect_players_and_annotate(
#     image_path: str,
#     yolo_model_path: str = "yolo26n.pt",
#     conf_threshold: float = 0.3,
# ) -> str:
#     """
#     Run YOLO player detection, crop each detected player region,
#     run PaddleOCR on the crop to detect jersey numbers,
#     and draw both boxes and OCR predictions.

#     Returns the annotated image path.
#     """
#     image = cv2.imread(image_path)
#     if image is None:
#         raise FileNotFoundError(f"Could not open image: {image_path}")

#     model = _get_yolo_model(yolo_model_path)
#     results = model.predict(source=image, conf=conf_threshold, classes=[0], verbose=False)

#     # If your model includes multiple classes, set class_id_filter to player class id.
#     class_id_filter = None

#     predicted_numbers = []

#     for r in results:
#         for box in r.boxes:
#             cls_id = int(box.cls.item())
#             if class_id_filter is not None and cls_id != class_id_filter:
#                 continue

#             conf = float(box.conf.item())
#             x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

#             # Clamp coordinates to image bounds
#             h, w = image.shape[:2]
#             x1 = max(0, min(x1, w - 1))
#             y1 = max(0, min(y1, h - 1))
#             x2 = max(0, min(x2, w - 1))
#             y2 = max(0, min(y2, h - 1))

#             if x2 <= x1 or y2 <= y1:
#                 continue

#             # Draw player box
#             cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

#             # Crop the detected player region
#             crop = image[y1:y2, x1:x2]

#             # Run PaddleOCR on the crop
#             ocr_result = _extract_best_jersey_number(crop)

#             label = f"player {conf:.2f}"

#             if ocr_result is not None:
#                 jersey_number, ocr_conf = ocr_result
#                 predicted_numbers.append(
#                     {
#                         "bbox": [x1, y1, x2, y2],
#                         "player_conf": conf,
#                         "jersey_number": jersey_number,
#                         "ocr_conf": ocr_conf,
#                     }
#                 )
#                 label = f"#{jersey_number} ({ocr_conf:.2f})"

#             # Draw label above the box
#             cv2.putText(
#                 image,
#                 label,
#                 (x1, max(20, y1 - 10)),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7,
#                 (0, 255, 0),
#                 2,
#                 cv2.LINE_AA,
#             )
    
#     os.makedirs("boxed_images", exist_ok=True)
#     annotated_path = os.path.join("boxed_images", Path(image_path).name)
#     cv2.imwrite(annotated_path, image)

#     return annotated_path

# def detect_players_and_annotate(
#     image_path: str,
#     yolo_model_path: str = "yolo26n.pt",
#     conf_threshold: float = 0.3,
# ) -> str:
#     """
#     Run YOLO26 player detection and draw bounding boxes on image.

#     Returns the annotated image path.
#     """
#     image = cv2.imread(image_path)
#     if image is None:
#         raise FileNotFoundError(f"Could not open image: {image_path}")

#     model = _get_yolo_model(yolo_model_path)
#     results = model.predict(source=image, conf=conf_threshold, classes=[0], verbose=False)

#     # If your model includes multiple classes, set class_id_filter to player class id.
#     class_id_filter = None

#     predicted_numbers = []

#     for r in results:
#         for box in r.boxes:
#             cls_id = int(box.cls.item())
#             if class_id_filter is not None and cls_id != class_id_filter:
#                 continue

#             conf = float(box.conf.item())
#             x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

#             cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

#             # Add Paddle OCR jersey number prediction here

#     annotated_path = os.path.join("boxed_images", image_path.split("/")[-1])
#     cv2.imwrite(annotated_path, image)

#     return annotated_path


from pathlib import Path
from typing import Optional, Tuple
import os
import re
from functools import lru_cache

import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR


@lru_cache(maxsize=4)
def _get_yolo_model(yolo_model_path: str) -> YOLO:
    return YOLO(yolo_model_path)


@lru_cache(maxsize=1)
def _get_paddleocr() -> PaddleOCR:
    return PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def _collect_digit_candidates(results):
    candidates = []

    if not isinstance(results, list):
        return candidates

    for res in results:
        if hasattr(res, "rec_texts"):
            texts = getattr(res, "rec_texts", [])
            scores = getattr(res, "rec_scores", [1.0] * len(texts))
            for text, score in zip(texts, scores):
                text = str(text).strip()
                if re.fullmatch(r"\d{1,2}", text):
                    candidates.append((text, float(score)))

        elif isinstance(res, dict):
            texts = res.get("rec_texts", [])
            scores = res.get("rec_scores", [1.0] * len(texts))
            for text, score in zip(texts, scores):
                text = str(text).strip()
                if re.fullmatch(r"\d{1,2}", text):
                    candidates.append((text, float(score)))

        elif isinstance(res, list):
            for line in res:
                if (
                    isinstance(line, list)
                    and len(line) >= 2
                    and isinstance(line[1], tuple)
                    and len(line[1]) >= 2
                ):
                    text, score = line[1]
                    text = str(text).strip()
                    if re.fullmatch(r"\d{1,2}", text):
                        candidates.append((text, float(score)))

    return candidates


def _extract_best_jersey_number(
    crop,
    resize_scale: float = 2.0,
    early_accept_conf: float = 0.80,
) -> Optional[Tuple[str, float]]:
    """
    Faster OCR:
    - only use one processed image
    - crop to likely jersey region
    - early return on strong result
    """
    if crop is None or crop.size == 0:
        return None

    h, w = crop.shape[:2]
    if h < 24 or w < 24:
        return None

    # Focus on jersey area: upper-middle torso
    y1 = int(0.15 * h)
    y2 = int(0.70 * h)
    x1 = int(0.18 * w)
    x2 = int(0.82 * w)

    torso = crop[y1:y2, x1:x2]
    if torso is None or torso.size == 0:
        torso = crop

    th, tw = torso.shape[:2]
    if th < 20 or tw < 20:
        return None

    # Resize, but not as aggressively as 3x
    if resize_scale != 1.0:
        torso = cv2.resize(
            torso,
            None,
            fx=resize_scale,
            fy=resize_scale,
            interpolation=cv2.INTER_CUBIC,
        )

    ocr = _get_paddleocr()

    try:
        results = ocr.predict(input=torso)
    except Exception:
        try:
            results = ocr.ocr(torso, cls=False)
        except Exception:
            return None

    candidates = _collect_digit_candidates(results)
    if not candidates:
        return None

    best = max(candidates, key=lambda x: x[1])

    # Early accept if good enough
    if best[1] >= early_accept_conf:
        return best

    return best


def detect_players_and_annotate(
    image_path: str,
    yolo_model_path: str = "yolo26n.pt",
    conf_threshold: float = 0.3,
    min_ocr_box_conf: float = 0.45,
    max_ocr_boxes: int = 8,
) -> str:
    """
    Faster pipeline:
    - detect players with YOLO
    - only OCR the best player boxes
    - annotate image
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not open image: {image_path}")

    image_for_crops = image.copy()
    model = _get_yolo_model(yolo_model_path)

    results = model.predict(
        source=image,
        conf=conf_threshold,
        classes=[0],
        verbose=False,
    )

    detections = []
    img_h, img_w = image.shape[:2]

    for r in results:
        for box in r.boxes:
            conf = float(box.conf.item())
            cls_id = int(box.cls.item())

            # keep only person class if needed
            if cls_id != 0:
                continue

            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

            x1 = max(0, min(x1, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            x2 = max(0, min(x2, img_w - 1))
            y2 = max(0, min(y2, img_h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            bw = x2 - x1
            bh = y2 - y1

            # Skip tiny boxes
            if bw < 30 or bh < 40:
                continue

            detections.append((conf, x1, y1, x2, y2))

    # Process strongest boxes first
    detections.sort(key=lambda d: d[0], reverse=True)

    predicted_numbers = []

    for idx, (conf, x1, y1, x2, y2) in enumerate(detections):
        crop = image_for_crops[y1:y2, x1:x2]

        label = f"player {conf:.2f}"

        # Only run OCR on top detections
        should_run_ocr = (conf >= min_ocr_box_conf) and (idx < max_ocr_boxes)

        if should_run_ocr:
            ocr_result = _extract_best_jersey_number(crop)

            if ocr_result is not None:
                jersey_number, ocr_conf = ocr_result
                predicted_numbers.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "player_conf": conf,
                        "jersey_number": jersey_number,
                        "ocr_conf": ocr_conf,
                    }
                )
                label = f"#{jersey_number} ({ocr_conf:.2f})"

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            image,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    os.makedirs("boxed_images", exist_ok=True)
    annotated_path = os.path.join("boxed_images", Path(image_path).name)
    cv2.imwrite(annotated_path, image)

    return annotated_path