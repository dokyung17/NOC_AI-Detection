"""Shared MediaPipe FaceMesh helpers (same settings as pyVHR SignalProcessing)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

# Match pyVHR/extraction/sig_processing.py
PRESENCE_THRESHOLD = 0.5
VISIBILITY_THRESHOLD = 0.5
FACEMESH_CONFIDENCE = 0.5

# MediaPipe Face Mesh → ArcFace 5-point (left eye, right eye, nose, left mouth, right mouth)
_LEFT_EYE = (33, 133, 159, 145)
_RIGHT_EYE = (362, 263, 386, 374)
_NOSE = 1
_MOUTH_LEFT = 61
_MOUTH_RIGHT = 291


@dataclass
class FaceDet:
    """One-frame face detection result."""

    xy: np.ndarray  # (468, 2) pixel x,y; invalid rows are -1
    mask: np.ndarray  # (H, W) uint8 {0, 1}
    kps5: np.ndarray  # (5, 2) ArcFace-order landmarks


def iter_video_frames(video_path: Path) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def compute_face_mask(frame_bgr: np.ndarray, xy: np.ndarray) -> Optional[np.ndarray]:
    """Convex hull of valid FaceMesh points. Reuses pyVHR's hull idea without eye/mouth holes."""
    valid = xy[(xy[:, 0] >= 0) & (xy[:, 1] >= 0)]
    if valid.shape[0] < 8:
        return None
    hull = cv2.convexHull(valid.astype(np.int32))
    mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    if int(mask.sum()) < 64:
        return None
    return mask


def five_point_kps(xy: np.ndarray) -> Optional[np.ndarray]:
    def _avg(idxs: tuple[int, ...]) -> Optional[np.ndarray]:
        pts = xy[list(idxs)]
        pts = pts[(pts[:, 0] >= 0) & (pts[:, 1] >= 0)]
        if pts.size == 0:
            return None
        return pts.mean(axis=0)

    left = _avg(_LEFT_EYE)
    right = _avg(_RIGHT_EYE)
    nose = xy[_NOSE] if xy[_NOSE, 0] >= 0 else None
    m_left = xy[_MOUTH_LEFT] if xy[_MOUTH_LEFT, 0] >= 0 else None
    m_right = xy[_MOUTH_RIGHT] if xy[_MOUTH_RIGHT, 0] >= 0 else None
    if left is None or right is None or nose is None or m_left is None or m_right is None:
        return None
    return np.stack([left, right, nose, m_left, m_right], axis=0).astype(np.float32)


class FaceMeshDetector:
    """Long-lived FaceMesh session (create once per process, not per video)."""

    def __init__(self) -> None:
        import mediapipe as mp

        self._mp = mp
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            static_image_mode=True,
            min_detection_confidence=FACEMESH_CONFIDENCE,
            min_tracking_confidence=FACEMESH_CONFIDENCE,
        )

    def close(self) -> None:
        self._mesh.close()

    def __enter__(self) -> "FaceMeshDetector":
        return self

    def __exit__(self, *args) -> bool:
        self.close()
        return False

    def detect(self, frame_bgr: np.ndarray) -> Optional[FaceDet]:
        image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        xy = np.full((468, 2), -1.0, dtype=np.float32)
        results = self._mesh.process(image)
        if not results.multi_face_landmarks:
            return None
        face_landmarks = results.multi_face_landmarks[0]
        for idx, landmark in enumerate(face_landmarks.landmark):
            if landmark.HasField("visibility") and landmark.visibility < VISIBILITY_THRESHOLD:
                continue
            if landmark.HasField("presence") and landmark.presence < PRESENCE_THRESHOLD:
                continue
            px = landmark.x * width
            py = landmark.y * height
            if 0 <= px < width and 0 <= py < height:
                xy[idx, 0] = px
                xy[idx, 1] = py
        mask = compute_face_mask(frame_bgr, xy)
        kps5 = five_point_kps(xy)
        if mask is None or kps5 is None:
            return None
        return FaceDet(xy=xy, mask=mask, kps5=kps5)


def extract_boundary_and_identity_features(
    video_path: Path,
    rec_model: object | None = None,
    detector: FaceMeshDetector | None = None,
) -> dict[str, float]:
    """Single video pass: FaceMesh → boundary scores + identity embeddings."""
    from features.boundary import (  # noqa: WPS433
        aggregate_boundary_scores,
        compute_boundary_score_frame,
        empty_boundary_features,
    )
    from features.identity import (  # noqa: WPS433
        aggregate_identity_sims,
        empty_identity_features,
        extract_face_embedding,
    )

    own_detector = detector is None
    if detector is None:
        detector = FaceMeshDetector()

    scores: list[float] = []
    embeddings: list[np.ndarray] = []
    try:
        for frame in iter_video_frames(video_path):
            try:
                det = detector.detect(frame)
            except Exception:
                continue
            if det is None:
                continue
            score = compute_boundary_score_frame(frame, det.mask)
            if np.isfinite(score):
                scores.append(float(score))
            if rec_model is not None:
                emb = extract_face_embedding(frame, det.kps5, rec_model)
                if emb is not None:
                    embeddings.append(emb)
    finally:
        if own_detector:
            detector.close()

    out = empty_boundary_features()
    out.update(empty_identity_features())
    out.update(aggregate_boundary_scores(scores))
    out.update(aggregate_identity_sims(embeddings))
    return out
