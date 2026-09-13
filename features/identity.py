"""Frame-to-frame ArcFace identity consistency features.

Weights: InsightFace buffalo_l recognition backbone
  file: w600k_r50.onnx
  architecture: ResNet-50 ArcFace (MS1M-RetinaFace / WebFace600K)
  input: 112x112 RGB, mean/std 127.5
  embedding: 512-d
  source: https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
  local cache: models/arcface/w600k_r50.onnx  (or ~/.insightface/models/buffalo_l/)
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from urllib.request import urlretrieve

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.face_utils import FaceMeshDetector, iter_video_frames

DEFAULT_ARCFACE_DIR = ROOT / "models" / "arcface"
ARCFACE_ONNX_NAME = "w600k_r50.onnx"
ARCFACE_ZIP_URL = (
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
)
ARCFACE_INPUT_SIZE = 112
ARCFACE_EMBED_DIM = 512

# Standard ArcFace 5-point template for 112x112 (InsightFace face_align.arcface_src)
_ARCFACE_SRC = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

_HOME_BUFFALO = Path.home() / ".insightface" / "models" / "buffalo_l" / ARCFACE_ONNX_NAME


class ArcFaceModel:
    """ONNX Runtime wrapper around buffalo_l w600k_r50.onnx."""

    def __init__(self, session, input_name: str, output_name: str) -> None:
        self.session = session
        self.input_name = input_name
        self.output_name = output_name

    def get_feat(self, aligned_bgr: np.ndarray) -> np.ndarray:
        blob = cv2.dnn.blobFromImages(
            [aligned_bgr],
            scalefactor=1.0 / 127.5,
            size=(ARCFACE_INPUT_SIZE, ARCFACE_INPUT_SIZE),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
        )
        out = self.session.run([self.output_name], {self.input_name: blob})[0]
        feat = np.asarray(out, dtype=np.float64).reshape(-1)
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        return feat


def empty_identity_features() -> Dict[str, float]:
    return {
        "identity_sim_mean": np.nan,
        "identity_sim_std": np.nan,
        "identity_sim_min": np.nan,
    }


def ensure_arcface_weights(model_dir: Optional[Path] = None) -> Path:
    """Return path to w600k_r50.onnx, downloading buffalo_l.zip if needed."""
    candidates = []
    if model_dir is not None:
        given = Path(model_dir)
        if given.is_file() and given.suffix.lower() == ".onnx":
            return given
        candidates.append(given / ARCFACE_ONNX_NAME)
    candidates.append(DEFAULT_ARCFACE_DIR / ARCFACE_ONNX_NAME)
    candidates.append(_HOME_BUFFALO)

    for path in candidates:
        if path.is_file() and path.suffix == ".onnx":
            return path

    dest_dir = Path(model_dir) if model_dir is not None else DEFAULT_ARCFACE_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = dest_dir / ARCFACE_ONNX_NAME
    zip_path = dest_dir / "buffalo_l.zip"
    print(f"ArcFace weights not found. Downloading {ARCFACE_ZIP_URL}")
    urlretrieve(ARCFACE_ZIP_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        rec_name = next(
            (n for n in names if Path(n).name == ARCFACE_ONNX_NAME),
            None,
        )
        if rec_name is None:
            raise FileNotFoundError(
                f"{ARCFACE_ONNX_NAME} not inside {zip_path}. Contents: {names}"
            )
        member = zf.getinfo(rec_name)
        member.filename = ARCFACE_ONNX_NAME
        zf.extract(member, dest_dir)
    zip_path.unlink(missing_ok=True)
    if not onnx_path.is_file():
        raise FileNotFoundError(f"Failed to extract ArcFace weights to {onnx_path}")
    print(f"ArcFace weights ready: {onnx_path}")
    return onnx_path


def load_arcface_model(
    device: object | str = "cpu",
    model_dir: Optional[Path] = None,
) -> ArcFaceModel:
    """Load recognition ONNX once per process. Uses CUDA EP when available."""
    import onnxruntime as ort

    onnx_path = ensure_arcface_weights(model_dir)
    device_type = getattr(device, "type", None) or str(device)
    providers: List[str] = ["CPUExecutionProvider"]
    available = ort.get_available_providers()
    if str(device_type).startswith("cuda") and "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    session = ort.InferenceSession(str(onnx_path), providers=providers)
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    print(
        f"ArcFace: {onnx_path.name} | dim={ARCFACE_EMBED_DIM} | "
        f"input={ARCFACE_INPUT_SIZE}x{ARCFACE_INPUT_SIZE} | providers={session.get_providers()}"
    )
    return ArcFaceModel(session, inputs[0].name, outputs[0].name)


def align_face(frame_bgr: np.ndarray, kps5: np.ndarray) -> Optional[np.ndarray]:
    if kps5.shape != (5, 2):
        return None
    matrix, _ = cv2.estimateAffinePartial2D(
        kps5.astype(np.float32),
        _ARCFACE_SRC,
        method=cv2.LMEDS,
    )
    if matrix is None:
        return None
    return cv2.warpAffine(
        frame_bgr,
        matrix,
        (ARCFACE_INPUT_SIZE, ARCFACE_INPUT_SIZE),
        flags=cv2.INTER_LINEAR,
        borderValue=0.0,
    )


def extract_face_embedding(
    frame_bgr: np.ndarray,
    kps5: np.ndarray,
    rec_model: ArcFaceModel,
) -> Optional[np.ndarray]:
    aligned = align_face(frame_bgr, kps5)
    if aligned is None:
        return None
    feat = rec_model.get_feat(aligned)
    if feat.size != ARCFACE_EMBED_DIM or not np.all(np.isfinite(feat)):
        return None
    return feat


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def aggregate_identity_sims(embeddings: Sequence[np.ndarray]) -> Dict[str, float]:
    if len(embeddings) < 2:
        return empty_identity_features()
    mat = np.vstack([np.asarray(e, dtype=np.float64).reshape(1, -1) for e in embeddings])
    centroid = mat.mean(axis=0)
    sims = np.array([cosine_similarity(row, centroid) for row in mat], dtype=np.float64)
    sims = sims[np.isfinite(sims)]
    if sims.size == 0:
        return empty_identity_features()
    return {
        "identity_sim_mean": float(np.mean(sims)),
        "identity_sim_std": float(np.std(sims)),
        "identity_sim_min": float(np.min(sims)),
    }


def extract_identity_features(
    video_path: Path,
    rec_model: ArcFaceModel,
    detector: Optional[FaceMeshDetector] = None,
) -> Dict[str, float]:
    """Independent video-level identity features (testable without boundary scores)."""
    own = detector is None
    if detector is None:
        detector = FaceMeshDetector()
    embeddings: list[np.ndarray] = []
    try:
        for frame in iter_video_frames(video_path):
            det = detector.detect(frame)
            if det is None:
                continue
            emb = extract_face_embedding(frame, det.kps5, rec_model)
            if emb is not None:
                embeddings.append(emb)
    finally:
        if own:
            detector.close()
    return aggregate_identity_sims(embeddings)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    a = rng.normal(size=ARCFACE_EMBED_DIM)
    b = a + 0.01 * rng.normal(size=ARCFACE_EMBED_DIM)
    c = rng.normal(size=ARCFACE_EMBED_DIM)
    assert cosine_similarity(a, a) > 0.99
    feats = aggregate_identity_sims([a, b, c])
    assert set(feats) == set(empty_identity_features())
    assert feats["identity_sim_min"] <= feats["identity_sim_mean"]
    print("identity smoke ok", feats)
