#!/usr/bin/env python3
"""
AI video detection — unified feature extraction.

rPPG (pyVHR slim) + D3 temporal + Laplacian spatial → single CSV per run.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.io import infer_label, parse_real_stems, resolve_videos  # noqa: E402
from features.rppg_metrics import extract_rppg_features, reconstruct_bvp_waveform  # noqa: E402
from pyVHR.analysis.pipeline import Pipeline  # noqa: E402
from visual.d3_model import D3_model  # noqa: E402
from visual.frames import load_frames_from_video  # noqa: E402
from visual.highfreq import highfreq_laplacian_score  # noqa: E402

# Per-process state for ProcessPoolExecutor workers.
_WORKER: Dict[str, object] = {}


def parse_roi(s: str) -> Tuple[int, int, int, int]:
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI format: x,y,width,height")
    x, y, w, h = parts
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("width and height must be positive")
    return x, y, w, h


def select_roi_interactive(video_path: Path) -> Tuple[int, int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read first frame: {video_path}")

    print("드래그로 ROI 선택 → Enter/Space=확인, c=취소")
    roi = cv2.selectROI("Select ROI (--no-face)", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    x, y, w, h = (int(v) for v in roi)
    if w == 0 or h == 0:
        raise RuntimeError("ROI not selected")
    return x, y, w, h


def resolve_rppg_roi(
    no_face: bool,
    roi_arg: Optional[Tuple[int, int, int, int]],
    full_frame: bool,
    select_roi: bool,
    videos: List[Path],
) -> Tuple[str, Optional[Tuple[int, int, int, int]]]:
    """Return (roi_approach, fixed_roi). Default: FaceMesh patches."""
    if not no_face:
        return "patches", None

    if select_roi:
        if len(videos) != 1:
            raise ValueError("--select-roi requires exactly one video")
        fixed = select_roi_interactive(videos[0])
        print(f"Selected ROI: x={fixed[0]}, y={fixed[1]}, w={fixed[2]}, h={fixed[3]}")
        return "crop", fixed
    if roi_arg is not None:
        return "crop", roi_arg
    if full_frame:
        return "crop", None

    raise ValueError("--no-face requires one of: --roi, --select-roi, --full-frame")


def extract_d3_scores(model: D3_model, video_path: Path, device: torch.device) -> tuple[float, float]:
    frames = load_frames_from_video(video_path).to(device)
    with torch.inference_mode():
        _, _, dis_std = model(frames.unsqueeze(0))
        temporal = float(dis_std.cpu().item())
        hf = highfreq_laplacian_score(frames)
    return temporal, hf


def process_video(
    pipe: Pipeline,
    model: D3_model,
    device: torch.device,
    video_path: Path,
    real_stems: set,
    roi_approach: str = "patches",
    fixed_roi: Optional[Tuple[int, int, int, int]] = None,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "video_name": video_path.name,
        "label": infer_label(video_path, real_stems),
        "success": False,
    }
    try:
        time, bpm, uncertainty, windowed_bvps, fps = pipe.run_on_video(
            str(video_path),
            roi_approach=roi_approach,
            fixed_roi=fixed_roi,
            method="cpu_POS",
            bpm_type="welch",
            post_filt=True,
            verb=False,
            return_bvp=True,
        )
        _, bvp_wave = reconstruct_bvp_waveform(windowed_bvps, time, fps)
        row.update(extract_rppg_features(bvp_wave, fps, windowed_bvps))
        temporal, hf = extract_d3_scores(model, video_path, device)
        row["d3_temporal_score"] = temporal
        row["highfreq_score"] = hf
        row["success"] = True
    except Exception as exc:
        row["error"] = str(exc)
    return row


def _init_worker(
    encoder: str,
    loss: str,
    device_str: str,
    roi_approach: str,
    fixed_roi: Optional[Tuple[int, int, int, int]],
) -> None:
    """Load Pipeline + D3 once per worker process."""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    device = torch.device(device_str)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model = D3_model(encoder_type=encoder, loss_type=loss).to(device)
    model.eval()
    _WORKER["pipe"] = Pipeline()
    _WORKER["model"] = model
    _WORKER["device"] = device
    _WORKER["roi_approach"] = roi_approach
    _WORKER["fixed_roi"] = fixed_roi


def _worker_process_video(
    payload: Tuple[int, str, frozenset],
) -> Tuple[int, Dict[str, object]]:
    idx, video_str, real_stems = payload
    row = process_video(
        _WORKER["pipe"],  # type: ignore[arg-type]
        _WORKER["model"],  # type: ignore[arg-type]
        _WORKER["device"],  # type: ignore[arg-type]
        Path(video_str),
        set(real_stems),
        roi_approach=str(_WORKER["roi_approach"]),
        fixed_roi=_WORKER["fixed_roi"],  # type: ignore[arg-type]
    )
    return idx, row


def _print_progress(done: int, total: int, row: Dict[str, object]) -> None:
    print(f"[{done}/{total}] {row.get('video_name')}")
    if row.get("success"):
        print(
            f"  rPPG pseudo_snr={row.get('pseudo_snr_db')} | "
            f"D3={row.get('d3_temporal_score'):.4f} | HF={row.get('highfreq_score'):.4f}"
        )
    else:
        print(f"  ERROR: {row.get('error')}")


def save_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    id_cols = ["video_name", "label", "success"]
    metrics = sorted({k for row in rows for k in row if k not in id_cols and k != "error"})
    fields = id_cols + metrics + (["error"] if any("error" in r for r in rows) else [])
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but PyTorch cannot access a CUDA GPU."
            )
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified rPPG + D3 feature extraction")
    parser.add_argument(
        "inputs",
        nargs="*",
        default=[str(ROOT / "data" / "videos")],
        help="Video file(s) or directory path(s). Default: data/videos",
    )
    parser.add_argument("--pattern", default="*.mp4")
    parser.add_argument("--recursive", dest="recursive", action="store_true", default=True)
    parser.add_argument("--no-recursive", dest="recursive", action="store_false")
    parser.add_argument("--save-dir", default=str(ROOT / "results"))
    parser.add_argument("--real-stems", default=None)
    parser.add_argument("--encoder", default="ResNet-18")
    parser.add_argument("--loss", default="l2", choices=["l2", "cos"])
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="D3/high-frequency device. auto selects CUDA when available.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1). Try 2-4.",
    )
    parser.add_argument(
        "--no-face",
        action="store_true",
        help="Turn OFF MediaPipe FaceMesh; extract rPPG from a fixed crop instead",
    )
    parser.add_argument(
        "--roi",
        type=parse_roi,
        default=None,
        help="Fixed crop as x,y,width,height (use with --no-face)",
    )
    parser.add_argument(
        "--full-frame",
        action="store_true",
        help="Use entire frame as crop (use with --no-face)",
    )
    parser.add_argument(
        "--select-roi",
        action="store_true",
        help="Pick crop on first frame with mouse (use with --no-face, single video)",
    )
    args = parser.parse_args()

    if args.no_face:
        roi_modes = sum([args.roi is not None, args.full_frame, args.select_roi])
        if roi_modes != 1:
            raise ValueError("--no-face requires exactly one of: --roi, --full-frame, --select-roi")
    elif args.roi or args.full_frame or args.select_roi:
        raise ValueError("--roi / --full-frame / --select-roi require --no-face")

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    videos = resolve_videos(args.inputs, args.pattern, args.recursive)
    if not videos:
        raise FileNotFoundError("No videos found.")

    roi_approach, fixed_roi = resolve_rppg_roi(
        args.no_face,
        args.roi,
        args.full_frame,
        args.select_roi,
        videos,
    )

    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    real_stems = parse_real_stems(args.real_stems)
    device = select_device(args.device)

    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Selected device: {device}")
    print(f"Workers: {args.workers}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            "CUDA memory: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )
        torch.backends.cudnn.benchmark = True
        if args.workers > 1:
            print(
                "Note: each worker loads its own D3 model on GPU; "
                "VRAM usage scales with --workers."
            )
    else:
        print("D3 and high-frequency features will run on CPU.")

    print("rPPG cpu_POS will continue to run on CPU.")
    if roi_approach == "crop":
        print(f"FaceMesh: OFF | fixed_roi: {fixed_roi or 'full_frame'}")
    else:
        print("FaceMesh: ON  | roi_approach: patches")
    print("=" * 60)

    total = len(videos)
    rows: List[Dict[str, object]] = [{} for _ in range(total)]
    ok = 0

    if args.workers == 1:
        pipe = Pipeline()
        model = D3_model(
            encoder_type=args.encoder,
            loss_type=args.loss,
        ).to(device)
        model.eval()
        for i, video in enumerate(videos, 1):
            row = process_video(
                pipe, model, device, video, real_stems, roi_approach, fixed_roi
            )
            rows[i - 1] = row
            if row.get("success"):
                ok += 1
            _print_progress(i, total, row)
    else:
        stems_frozen = frozenset(real_stems)
        payloads = [(i, str(video), stems_frozen) for i, video in enumerate(videos)]
        done = 0
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(args.encoder, args.loss, device.type, roi_approach, fixed_roi),
        ) as pool:
            futures = [pool.submit(_worker_process_video, p) for p in payloads]
            for fut in as_completed(futures):
                idx, row = fut.result()
                rows[idx] = row
                done += 1
                if row.get("success"):
                    ok += 1
                _print_progress(done, total, row)

    out = save_dir / f"unified_features_{datetime.now():%Y%m%d_%H%M%S}.csv"
    save_csv(out, rows)
    print(f"\nDone {ok}/{total} → {out}")


if __name__ == "__main__":
    main()
