"""Load and preprocess video frames for D3 (same rules as D3/data/datasets.py)."""

from pathlib import Path
from typing import List

import albumentations
import cv2
import numpy as np
import torch


def crop_center_by_percentage(image, percentage: float):
    height, width = image.shape[:2]
    if width > height:
        left_pixels = int(width * percentage)
        right_pixels = int(width * percentage)
        start_x = left_pixels
        end_x = width - right_pixels
        return image[:, start_x:end_x]
    up_pixels = int(height * percentage)
    down_pixels = int(height * percentage)
    start_y = up_pixels
    end_y = height - down_pixels
    return image[start_y:end_y, :]


def set_preprocessing(aug_type=None, aug_quality=None):
    aug_list = [albumentations.Resize(224, 224)]
    if aug_type == "Gaussian_blur":
        aug_list.append(
            albumentations.GaussianBlur(
                blur_limit=(3, 7), sigma_limit=(aug_quality, aug_quality), p=1.0
            )
        )
    if aug_type == "JEPG_compression":
        aug_list.append(
            albumentations.ImageCompression(
                quality_lower=aug_quality, quality_upper=aug_quality
            )
        )
    aug_list.append(
        albumentations.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
            p=1.0,
        )
    )
    return albumentations.Compose(aug_list)


def load_frames_from_video(video_path: Path, max_frames: int = 16) -> torch.Tensor:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    raw_frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        raw_frames.append(frame)
    cap.release()

    total = len(raw_frames)
    if total < 8:
        raise ValueError(f"Need at least 8 frames, got {total}: {video_path}")

    n_use = 8 if total < 16 else min(max_frames, 16, total)
    indices = np.linspace(0, total - 1, n_use, dtype=int)

    trans = set_preprocessing()
    chunks: List[np.ndarray] = []
    for idx in indices:
        image = crop_center_by_percentage(raw_frames[int(idx)], 0.1)
        augmented = trans(image=image)
        image = augmented["image"]
        chunks.append(image.transpose(2, 0, 1)[np.newaxis, :])

    stacked = np.concatenate(chunks, axis=0)
    return torch.tensor(stacked, dtype=torch.float32)
