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


def build_model():
    """Import SAM3 after changing into the repo directory and build the model."""
    bpe_path = Path(os.path.join("sam3", "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz"))
    if not bpe_path.exists():
        # Some repo layouts may keep assets at repo_root/assets
        alt_bpe_path = "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        if alt_bpe_path.exists():
            bpe_path = alt_bpe_path

    model = build_sam3_image_model(bpe_path=str(bpe_path))
    processor = Sam3Processor(model, confidence_threshold=0.5)
    return processor


def segment_image(image_path: Path, prompt: str = "Athletes"):
    """Run segmentation on an image using a text prompt."""
    processor = build_model()
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
    

def create_segmented_image(img_path: Path) -> None:
    """Display and save the result."""
    result_img = segment_image(img_path, prompt="Athletes")
    os.makedirs("segmented_images", exist_ok=True)
    segmented_path = os.path.join("segmented_images", Path(img_path).name)
    result_img.save(segmented_path)

    return segmented_path