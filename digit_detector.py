from pathlib import Path
from typing import List, Dict, Optional
import re

import cv2
import easyocr


class SportsDigitDetector:
    def __init__(self, use_gpu: bool = False):
        """
        Initialize EasyOCR reader.

        Args:
            use_gpu: Set True only if EasyOCR is properly configured for GPU.
        """
        self.reader = easyocr.Reader(["en"], gpu=use_gpu)

    def _preprocess_image(self, image):
        """
        Create a few versions of the image to improve OCR robustness.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Mild denoise
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)

        # Binary threshold
        _, thresh = cv2.threshold(
            denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Inverted threshold
        thresh_inv = cv2.bitwise_not(thresh)

        return [image, gray, thresh, thresh_inv]

    def _clean_digit_text(self, text: str) -> Optional[str]:
        """
        Keep only digits and only allow 1-2 digit outputs.
        """
        cleaned = re.sub(r"\D", "", text)
        if 1 <= len(cleaned) <= 2:
            return cleaned
        return None

    def detect_digits(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        min_confidence: float = 0.25
    ) -> Dict:
        """
        Detect digits in a sports image.

        Args:
            image_path: Path to input image.
            output_path: Path to save annotated image. If None, auto-generates one.
            min_confidence: Minimum OCR confidence to keep a detection.

        Returns:
            Dict with:
              - image_path
              - annotated_image_path
              - detections: list of digit detections
        """
        image_path = str(image_path)
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not open image: {image_path}")

        vis_image = image.copy()
        variants = self._preprocess_image(image)

        all_detections = []

        for variant_index, variant in enumerate(variants):
            results = self.reader.readtext(variant)

            for bbox, text, confidence in results:
                if confidence < min_confidence:
                    continue

                cleaned = self._clean_digit_text(text)
                if not cleaned:
                    continue

                xs = [pt[0] for pt in bbox]
                ys = [pt[1] for pt in bbox]
                x1, y1 = int(min(xs)), int(min(ys))
                x2, y2 = int(max(xs)), int(max(ys))

                all_detections.append({
                    "text": cleaned,
                    "confidence": float(confidence),
                    "bbox": [x1, y1, x2, y2],
                    "variant_index": variant_index
                })

        # Deduplicate overlapping/similar detections by bbox + text
        deduped = self._deduplicate_detections(all_detections)

        # Draw on image
        for det in deduped:
            x1, y1, x2, y2 = det["bbox"]
            label = f'{det["text"]} ({det["confidence"]:.2f})'

            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                vis_image,
                label,
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        if output_path is None:
            input_path = Path(image_path)
            output_path = str(
                input_path.with_name(f"{input_path.stem}_digits_detected{input_path.suffix}")
            )

        cv2.imwrite(output_path, vis_image)

        return all_detections

    def _deduplicate_detections(self, detections: List[Dict]) -> List[Dict]:
        """
        Remove duplicate detections by preferring higher-confidence overlaps
        with the same digit text.
        """
        if not detections:
            return []

        detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)
        kept = []

        for det in detections:
            should_keep = True
            for existing in kept:
                if det["text"] == existing["text"] and self._boxes_overlap(det["bbox"], existing["bbox"]):
                    should_keep = False
                    break
            if should_keep:
                kept.append(det)

        return kept

    def _boxes_overlap(self, box_a: List[int], box_b: List[int], iou_threshold: float = 0.3) -> bool:
        """
        Check whether two boxes overlap enough to be treated as duplicates.
        """
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

        union = area_a + area_b - inter_area
        if union == 0:
            return False

        iou = inter_area / union
        return iou >= iou_threshold


if __name__ == "__main__":
    image_path = "images/_MG_3055-4-42-7-51-34-88.jpg"  # replace with your image

    detector = SportsDigitDetector(use_gpu=False)
    result = detector.detect_digits(
        image_path=image_path,
        output_path="sports_image_annotated.jpg",
        min_confidence=0.25
    )