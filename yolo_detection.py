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
    """
    Cache model instances by path so we don't repeatedly reload weights.
    """
    return YOLO(yolo_model_path)


@lru_cache(maxsize=1)
def _get_paddleocr() -> PaddleOCR:
    """
    Cache PaddleOCR model so it is only loaded once.
    """
    return PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    )


def _extract_best_jersey_number(crop) -> Optional[Tuple[str, float]]:
    """
    Run PaddleOCR on multiple processed versions of a player crop and return
    the best detected 1-2 digit jersey number along with confidence.

    Returns:
        (number_string, confidence) or None
    """
    if crop is None or crop.size == 0:
        return None

    ocr = _get_paddleocr()
    h, w = crop.shape[:2]

    if h < 20 or w < 20:
        return None

    candidates = []

    # Use a less aggressive crop so we do not cut off numbers
    y1 = int(0.10 * h)
    y2 = int(0.85 * h)
    x1 = int(0.05 * w)
    x2 = int(0.95 * w)

    torso = crop[y1:y2, x1:x2]
    if torso is None or torso.size == 0:
        torso = crop

    # Build several OCR inputs
    resized_color = cv2.resize(
        torso, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(resized_color, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )

    ocr_inputs = [
        resized_color  # often best
        # gray,           # sometimes better
        # thresh,         # sometimes helps, sometimes hurts
    ]

    def collect_from_results(results):
        found = []

        if isinstance(results, list):
            for res in results:
                if hasattr(res, "rec_texts"):
                    texts = getattr(res, "rec_texts", [])
                    scores = getattr(res, "rec_scores", [1.0] * len(texts))
                    for text, score in zip(texts, scores):
                        text = str(text).strip()
                        if re.fullmatch(r"\d{1,2}", text):
                            found.append((text, float(score)))

                elif isinstance(res, dict):
                    texts = res.get("rec_texts", [])
                    scores = res.get("rec_scores", [1.0] * len(texts))
                    for text, score in zip(texts, scores):
                        text = str(text).strip()
                        if re.fullmatch(r"\d{1,2}", text):
                            found.append((text, float(score)))

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
                                found.append((text, float(score)))
        return found

    for ocr_input in ocr_inputs:
        try:
            results = ocr.predict(input=ocr_input)
            candidates.extend(collect_from_results(results))
        except Exception:
            try:
                results = ocr.ocr(ocr_input, cls=False)
                candidates.extend(collect_from_results(results))
            except Exception:
                pass

    if not candidates:
        return None

    return max(candidates, key=lambda x: x[1])


def detect_players_and_annotate(
    image_path: str,
    yolo_model_path: str = "yolo26n.pt",
    conf_threshold: float = 0.3,
) -> str:
    """
    Run YOLO player detection, crop each detected player region,
    run PaddleOCR on the crop to detect jersey numbers,
    and draw both boxes and OCR predictions.

    Returns the annotated image path.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not open image: {image_path}")

    # Keep a clean copy for cropping so drawn boxes/text do not contaminate OCR
    image_for_crops = image.copy()

    model = _get_yolo_model(yolo_model_path)
    results = model.predict(source=image, conf=conf_threshold, classes=[0], verbose=False)

    # If your model includes multiple classes, set class_id_filter to player class id.
    class_id_filter = None

    predicted_numbers = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls.item())
            if class_id_filter is not None and cls_id != class_id_filter:
                continue

            conf = float(box.conf.item())
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

            # Clamp coordinates to image bounds
            h, w = image.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            # Crop the detected player region from the clean image
            crop = image_for_crops[y1:y2, x1:x2]

            # Run PaddleOCR on the crop
            ocr_result = _extract_best_jersey_number(crop)

            # Default label
            label = f"player {conf:.2f}"

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

            # Draw player box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label above the box
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

# from pathlib import Path
# from typing import List, Dict
# import os

# import cv2
# from ultralytics import YOLO


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

#     model = YOLO(yolo_model_path)
#     results = model.predict(source=image, conf=conf_threshold, verbose=False)

#     # If your model includes multiple classes, set class_id_filter to player class id.
#     class_id_filter = None

#     for r in results:
#         for box in r.boxes:
#             cls_id = int(box.cls.item())
#             if class_id_filter is not None and cls_id != class_id_filter:
#                 continue

#             conf = float(box.conf.item())
#             x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

#             cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

#     annotated_path = os.path.join("boxed_images", image_path.split("/")[-1])
#     cv2.imwrite(annotated_path, image)

#     return annotated_path

# from pathlib import Path
# from typing import List, Dict, Optional, Tuple
# import os
# import re
# from functools import lru_cache

# import cv2
# import numpy as np
# from ultralytics import YOLO
# import easyocr


# @lru_cache(maxsize=4)
# def _get_yolo_model(yolo_model_path: str) -> YOLO:
#     return YOLO(yolo_model_path)


# @lru_cache(maxsize=1)
# def _get_easyocr_reader() -> easyocr.Reader:
#     return easyocr.Reader(["en"], gpu=False)


# def _clamp_box(x1, y1, x2, y2, w, h):
#     x1 = max(0, min(int(x1), w - 1))
#     y1 = max(0, min(int(y1), h - 1))
#     x2 = max(0, min(int(x2), w))
#     y2 = max(0, min(int(y2), h))
#     return x1, y1, x2, y2


# def _preprocess_variants(crop: np.ndarray) -> List[np.ndarray]:
#     """
#     Produce several OCR-friendly versions of the same crop.
#     """
#     if crop.size == 0:
#         return []

#     gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

#     # Upscale helps OCR on small digits
#     gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

#     # Contrast enhancement
#     clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
#     enhanced = clahe.apply(gray)

#     # Denoise slightly
#     blur = cv2.GaussianBlur(enhanced, (3, 3), 0)

#     # Threshold variants
#     _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     otsu_inv = cv2.bitwise_not(otsu)

#     adaptive = cv2.adaptiveThreshold(
#         blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY, 31, 8
#     )
#     adaptive_inv = cv2.bitwise_not(adaptive)

#     return [enhanced, otsu, otsu_inv, adaptive, adaptive_inv]


# def _digit_score(
#     text: str,
#     conf: float,
#     bbox: List[List[float]],
#     region_shape: Tuple[int, int]
# ) -> float:
#     """
#     Score an OCR detection. Higher is better.
#     Prefers:
#     - 1-2 digits
#     - higher confidence
#     - centered detections
#     - larger detections (to reject tiny noise)
#     """
#     digits = re.sub(r"\D", "", text)
#     if not (1 <= len(digits) <= 2):
#         return -1e9

#     xs = [p[0] for p in bbox]
#     ys = [p[1] for p in bbox]
#     x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

#     bw = max(1.0, x2 - x1)
#     bh = max(1.0, y2 - y1)
#     area = bw * bh

#     h, w = region_shape[:2]
#     cx = (x1 + x2) / 2.0
#     cy = (y1 + y2) / 2.0

#     center_x_penalty = abs(cx - w / 2) / max(1.0, w / 2)
#     center_y_penalty = abs(cy - h / 2) / max(1.0, h / 2)

#     # Heuristic scoring
#     score = (
#         3.0 * conf
#         + 0.0005 * area
#         - 0.8 * center_x_penalty
#         - 0.4 * center_y_penalty
#     )

#     # Slight preference for 2-digit numbers
#     if len(digits) == 2:
#         score += 0.15

#     return score


# def _run_ocr_on_region(region: np.ndarray) -> Optional[Tuple[str, float]]:
#     """
#     Run OCR over multiple preprocessed variants and return the best digit string.
#     """
#     reader = _get_easyocr_reader()
#     best = None

#     for variant in _preprocess_variants(region):
#         results = reader.readtext(
#             variant,
#             detail=1,
#             allowlist="0123456789",
#             paragraph=False,
#             width_ths=0.7,
#             height_ths=0.7,
#         )

#         for bbox, text, conf in results:
#             digits = re.sub(r"\D", "", text)
#             if not (1 <= len(digits) <= 2):
#                 continue

#             score = _digit_score(digits, float(conf), bbox, variant.shape)
#             if best is None or score > best["score"]:
#                 best = {
#                     "digits": digits,
#                     "conf": float(conf),
#                     "score": score
#                 }

#     if best is None:
#         return None

#     return best["digits"], best["conf"]


# def _candidate_number_regions(player_crop: np.ndarray) -> List[np.ndarray]:
#     """
#     Generate multiple likely jersey-number subregions.
#     This is much better than OCR on the whole player crop.
#     """
#     h, w = player_crop.shape[:2]
#     regions = []

#     # Main front torso
#     boxes = [
#         (0.18, 0.20, 0.82, 0.68),
#         (0.22, 0.25, 0.78, 0.62),
#         (0.15, 0.30, 0.85, 0.72),
#         # Slightly lower for some uniforms
#         (0.20, 0.35, 0.80, 0.78),
#     ]

#     for rx1, ry1, rx2, ry2 in boxes:
#         x1 = int(w * rx1)
#         y1 = int(h * ry1)
#         x2 = int(w * rx2)
#         y2 = int(h * ry2)
#         crop = player_crop[y1:y2, x1:x2]
#         if crop.size > 0:
#             regions.append(crop)

#     return regions


# def _nms_person_boxes(boxes: List[Dict], iou_thresh: float = 0.5) -> List[Dict]:
#     if not boxes:
#         return []

#     boxes = sorted(boxes, key=lambda b: b["player_conf"], reverse=True)
#     keep = []

#     def iou(a, b):
#         ax1, ay1, ax2, ay2 = a
#         bx1, by1, bx2, by2 = b

#         ix1 = max(ax1, bx1)
#         iy1 = max(ay1, by1)
#         ix2 = min(ax2, bx2)
#         iy2 = min(ay2, by2)

#         iw = max(0, ix2 - ix1)
#         ih = max(0, iy2 - iy1)
#         inter = iw * ih

#         area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
#         area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
#         union = area_a + area_b - inter

#         return inter / union if union > 0 else 0.0

#     for candidate in boxes:
#         overlaps = any(iou(candidate["bbox"], kept["bbox"]) > iou_thresh for kept in keep)
#         if not overlaps:
#             keep.append(candidate)

#     return keep


# def detect_players_and_annotate(
#     image_path: str,
#     yolo_model_path: str = "yolo26n.pt",
#     conf_threshold: float = 0.35,
#     ocr_conf_threshold: float = 0.20,
#     class_id_filter: Optional[int] = 0,  # 0 = person for standard COCO models
# ) -> Dict:
#     """
#     Detect players, then detect likely jersey digits per player.

#     Returns:
#         {
#             "annotated_image_path": str,
#             "detections": [
#                 {
#                     "bbox": [x1, y1, x2, y2],
#                     "player_conf": float,
#                     "jersey_number": str | None,
#                     "jersey_conf": float | None
#                 }
#             ]
#         }
#     """
#     image = cv2.imread(image_path)
#     if image is None:
#         raise FileNotFoundError(f"Could not open image: {image_path}")

#     os.makedirs("boxed_images", exist_ok=True)

#     model = _get_yolo_model(yolo_model_path)
#     results = model.predict(source=image, conf=conf_threshold, verbose=False)

#     h, w = image.shape[:2]
#     vis = image.copy()
#     person_boxes = []

#     # Step 1: collect only person boxes
#     for r in results:
#         for box in r.boxes:
#             cls_id = int(box.cls.item())
#             if class_id_filter is not None and cls_id != class_id_filter:
#                 continue

#             player_conf = float(box.conf.item())
#             x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
#             x1, y1, x2, y2 = _clamp_box(x1, y1, x2, y2, w, h)

#             bw = x2 - x1
#             bh = y2 - y1

#             # Filter out boxes too small to read jersey numbers
#             if bw < 35 or bh < 80:
#                 continue

#             # Filter extremely wide boxes, which are often bad player detections
#             aspect = bw / max(1, bh)
#             if aspect > 1.2:
#                 continue

#             person_boxes.append({
#                 "bbox": [x1, y1, x2, y2],
#                 "player_conf": player_conf
#             })

#     person_boxes = _nms_person_boxes(person_boxes, iou_thresh=0.45)

#     detections = []

#     # Step 2: OCR within better jersey candidate regions
#     for det in person_boxes:
#         x1, y1, x2, y2 = det["bbox"]
#         player_crop = image[y1:y2, x1:x2]

#         best_digits = None
#         best_conf = -1.0

#         for region in _candidate_number_regions(player_crop):
#             result = _run_ocr_on_region(region)
#             if result is None:
#                 continue

#             digits, conf = result
#             if conf > best_conf:
#                 best_digits = digits
#                 best_conf = conf

#         jersey_number = None
#         jersey_conf = None
#         if best_digits is not None and best_conf >= ocr_conf_threshold:
#             jersey_number = best_digits
#             jersey_conf = best_conf

#         color = (0, 255, 0) if jersey_number is not None else (0, 165, 255)
#         label = f"person {det['player_conf']:.2f}"
#         if jersey_number is not None:
#             label += f" | #{jersey_number} ({jersey_conf:.2f})"

#         cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
#         cv2.putText(
#             vis,
#             label,
#             (x1, max(20, y1 - 8)),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.55,
#             color,
#             2,
#             cv2.LINE_AA
#         )

#         detections.append({
#             "bbox": [x1, y1, x2, y2],
#             "player_conf": det["player_conf"],
#             "jersey_number": jersey_number,
#             "jersey_conf": jersey_conf
#         })

#     annotated_path = os.path.join("boxed_images", Path(image_path).name)
#     cv2.imwrite(annotated_path, vis)

#     return annotated_path


# if __name__ == "__main__":
#     result = detect_players_and_annotate(
#         image_path="sports_image.jpg",
#         yolo_model_path="yolo26n.pt",
#         conf_threshold=0.35,
#         ocr_conf_threshold=0.20,
#         class_id_filter=0
#     )

#     print("Annotated image:", result["annotated_image_path"])
#     for d in result["detections"]:
#         print(d)