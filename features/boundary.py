"""Blending-boundary scores inspired by Face X-ray (Li et al., CVPR 2020).

No pretrained Face X-ray weights are in this repo, so each frame score is a
classical inner-vs-outer ring discontinuity around the FaceMesh convex hull.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from features.face_utils import FaceMeshDetector, iter_video_frames


def empty_boundary_features() -> Dict[str, float]:
    return {
        "boundary_score_mean": np.nan,
        "boundary_score_std": np.nan,
        "boundary_score_max": np.nan,
    }


def _ring_kernel(mask: np.ndarray) -> np.ndarray:
    area = float(mask.sum())
    radius = max(3.0, np.sqrt(area / np.pi) * 0.06)
    k = int(max(3, round(radius)))
    if k % 2 == 0:
        k += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def compute_boundary_score_frame(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    min_pixels: int = 48,
) -> float:
    """Color / gradient / high-frequency gap between inner and outer face rings."""
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask_u8 = (mask > 0).astype(np.uint8)
    if int(mask_u8.sum()) < min_pixels:
        return np.nan

    kernel = _ring_kernel(mask_u8)
    eroded = cv2.erode(mask_u8, kernel)
    dilated = cv2.dilate(mask_u8, kernel)
    inner = (mask_u8 > 0) & (eroded == 0)
    outer = (dilated > 0) & (mask_u8 == 0)
    if int(inner.sum()) < min_pixels or int(outer.sum()) < min_pixels:
        return np.nan

    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    color_gap = float(np.linalg.norm(lab[inner].mean(axis=0) - lab[outer].mean(axis=0)))

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    grad_gap = abs(float(mag[inner].mean()) - float(mag[outer].mean()))

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    hf_gap = abs(float(lap[inner].var()) - float(lap[outer].var()))
    edge_energy = float(np.mean(np.abs(lap[inner | outer])))

    return float(color_gap + 0.15 * grad_gap + 0.02 * hf_gap + 0.05 * edge_energy)


def aggregate_boundary_scores(scores: Iterable[float]) -> Dict[str, float]:
    x = np.asarray(list(scores), dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return empty_boundary_features()
    return {
        "boundary_score_mean": float(np.mean(x)),
        "boundary_score_std": float(np.std(x)),
        "boundary_score_max": float(np.max(x)),
    }


def extract_boundary_features(
    video_path: Path,
    detector: Optional[FaceMeshDetector] = None,
) -> Dict[str, float]:
    """Independent video-level boundary features (testable without ArcFace)."""
    own = detector is None
    if detector is None:
        detector = FaceMeshDetector()
    scores: list[float] = []
    try:
        for frame in iter_video_frames(video_path):
            det = detector.detect(frame)
            if det is None:
                continue
            score = compute_boundary_score_frame(frame, det.mask)
            if np.isfinite(score):
                scores.append(float(score))
    finally:
        if own:
            detector.close()
    return aggregate_boundary_scores(scores)


if __name__ == "__main__":
    frame = np.zeros((240, 240, 3), dtype=np.uint8)
    frame[:] = (40, 40, 40)
    frame[60:180, 60:180] = (180, 160, 150)
    mask = np.zeros((240, 240), dtype=np.uint8)
    mask[70:170, 70:170] = 1
    score = compute_boundary_score_frame(frame, mask)
    feats = aggregate_boundary_scores([score, score + 1.0, score + 0.5])
    assert np.isfinite(score), score
    assert set(feats) == set(empty_boundary_features())
    print("boundary smoke ok", score, feats)
