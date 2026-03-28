import argparse
import os
from pathlib import Path
import sys
import cv2
import gc

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image

import sam3
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def build_model(directory):
    """Import SAM3 after changing into the repo directory and build the model."""
    bpe_path = Path(os.path.join(directory, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz"))
    if not bpe_path.exists():
        # Some repo layouts may keep assets at repo_root/assets
        alt_bpe_path = "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        if alt_bpe_path.exists():
            bpe_path = alt_bpe_path

    model = build_sam3_image_model(bpe_path=str(bpe_path))
    processor = Sam3Processor(model, confidence_threshold=0.5)
    return processor


def run_sam3_masks(processor, image_input, prompt: str):
    """
    Run SAM3 on either a file path, PIL image, or numpy RGB image.

    Returns:
        image_array: RGB numpy array
        masks: list of masks returned by SAM3
    """
    if isinstance(image_input, (str, Path)):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        image = Image.fromarray(image_input)
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        raise TypeError("image_input must be a path, PIL image, or numpy array")

    inference_state = processor.set_image(image)
    processor.reset_all_prompts(inference_state)
    inference_state = processor.set_text_prompt(state=inference_state, prompt=prompt)

    masks = inference_state["masks"]
    image_array = np.array(image)
    return image_array, masks


def masks_to_boxes(masks, min_area: int = 20):
    """
    Convert SAM3 masks to bounding boxes.

    Returns:
        List of (x1, y1, x2, y2)
    """
    boxes = []

    for mask in masks:
        mask_np = mask.detach().cpu().numpy().astype(np.uint8)
        mask_np = np.squeeze(mask_np, axis=0)

        ys, xs = np.where(mask_np > 0)
        if len(xs) == 0 or len(ys) == 0:
            continue

        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        if area < min_area:
            continue

        boxes.append((x1, y1, x2, y2))

    return boxes


def draw_boxes(image_rgb: np.ndarray, boxes, color=(0, 255, 0), thickness=2) -> np.ndarray:
    """
    Draw all bounding boxes in the same color.

    Args:
        image_rgb: RGB image
        boxes: list of (x1, y1, x2, y2)
        color: BGR color for OpenCV
    """
    output = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)

    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

    return output


def player_segment_image(processor, image_path: Path, prompt: str = "Athletes"):
    """Run segmentation on an image using a text prompt."""
    image = Image.open(image_path).convert("RGB")
    inference_state = processor.set_image(image)
    processor.reset_all_prompts(inference_state)
    inference_state = processor.set_text_prompt(state=inference_state, prompt=prompt)

    masks = inference_state["masks"]
    image_array = np.array(image)

    full_mask = np.zeros_like(image_array, dtype=np.float32)

    for mask in masks:
        bool_mask = mask.detach().cpu().numpy().astype(np.uint8)
        mask_squeeze = np.squeeze(bool_mask, axis=0)

        temp_mask = np.repeat(mask_squeeze[:, :, None], 3, axis=2)
        full_mask += image_array * temp_mask

    full_mask = np.clip(full_mask, 0, 255).astype(np.uint8)
    return Image.fromarray(full_mask)
    

def create_segmented_image(processor, img_path: Path) -> None:
    """Display and save the result."""
    player_result_img = player_segment_image(processor, img_path, prompt="Athletes")
    # digit_image_array, digit_masks = run_sam3_masks(processor, player_result_img, prompt="Numbers")
    # digit_boxes = masks_to_boxes(digit_masks, min_area=2)
    # result_img = draw_boxes(
    #     digit_image_array,
    #     digit_boxes,
    #     color=(0, 255, 0),
    #     thickness=2
    # )
    os.makedirs("segmented_images", exist_ok=True)
    segmented_path = os.path.join("segmented_images", Path(img_path).name)
    # cv2.imwrite(segmented_path, result_img)
    player_result_img.save(segmented_path)
    return segmented_path