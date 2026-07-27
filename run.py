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
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch

#print("[DEBUG] run.py imports completed")

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.io import infer_label, parse_real_stems, resolve_videos  # noqa: E402
from features.rppg_metrics import extract_rppg_features, reconstruct_bvp_waveform  # noqa: E402
#print("[DEBUG] importing Pipeline")
from pyVHR.analysis.pipeline import Pipeline  # noqa: E402
#print("[DEBUG] Pipeline imported")

#print("[DEBUG] importing D3_model")
from visual.d3_model import D3_model  # noqa: E402
#print("[DEBUG] D3_model imported")

#print("[DEBUG] importing frames")
from visual.frames import load_frames_from_video  # noqa: E402
#print("[DEBUG] frames imported")

#print("[DEBUG] importing highfreq")
from visual.highfreq import highfreq_laplacian_score  # noqa: E402
#print("[DEBUG] highfreq imported")


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
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "video_name": video_path.name,
        "label": infer_label(video_path, real_stems),
        "success": False,
    }
    try:
        time, bpm, uncertainty, windowed_bvps, fps = pipe.run_on_video(
            str(video_path),
            roi_approach="patches",
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


def save_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    id_cols = ["video_name", "label", "success"]
    metrics = sorted(k for row in rows for k in row if k not in id_cols and k != "error")
    fields = id_cols + metrics + (["error"] if any("error" in r for r in rows) else [])
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def main() -> None:
    #print("[DEBUG 1] main() entered")
    parser = argparse.ArgumentParser(description="Unified rPPG + D3 feature extraction")

    #print("[DEBUG 2] parser created")
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
    #print("[DEBUG 3] arguments registered")
    args = parser.parse_args()

    #print("[DEBUG 4] arguments parsed")
    #print("[DEBUG] args =", args)

    videos = resolve_videos(args.inputs, args.pattern, args.recursive)

    #print("[DEBUG 5] resolve_videos completed")
    #print("[DEBUG] video count =", len(videos))
    #print("[DEBUG] input paths =", args.inputs)

    #print("[DEBUG 6] checking video list")
    if not videos:
        raise FileNotFoundError("No videos found.")
    #print("[DEBUG 7] video list OK")

    #print("[DEBUG 8] resolving save directory")
    save_dir = Path(args.save_dir).expanduser().resolve()
    #print("[DEBUG 9] save_dir resolved:", save_dir)

    #print("[DEBUG 10] creating save directory")
    save_dir.mkdir(parents=True, exist_ok=True)
    #print("[DEBUG 11] save directory ready")

    #print("[DEBUG 12] parsing real stems")
    real_stems = parse_real_stems(args.real_stems)
    #print("[DEBUG 13] real stems parsed:", len(real_stems))

    #print("[DEBUG 14] selecting device")
    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    elif args.device == "cuda":
        #print("[DEBUG 15] checking CUDA availability")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but PyTorch cannot access a CUDA GPU."
            )
        #print("[DEBUG 16] CUDA is available")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    #print("[DEBUG 17] device selected:", device)

    #print("[DEBUG 18] printing device information")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Selected device: {device}")



    if device.type == "cuda":
        #print("[DEBUG 19] reading GPU name")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

        #print("[DEBUG 20] reading GPU properties")
        print(
            "CUDA memory: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

        # 입력 이미지 크기가 일정한 경우 연산 최적화
        #print("[DEBUG 21] enabling cudnn benchmark")
        torch.backends.cudnn.benchmark = True
    else:
        print("D3 and high-frequency features will run on CPU.")

    print("rPPG cpu_POS will continue to run on CPU.")
    print("=" * 60)

    #print("[DEBUG 22] creating pyVHR Pipeline instance")
    pipe = Pipeline()
    #print("[DEBUG 23] Pipeline instance created")

    #print("[DEBUG 24] creating D3 model")
    model = D3_model(
        encoder_type=args.encoder,
        loss_type=args.loss,
    ).to(device)
    #print("[DEBUG 25] D3 model constructed on CPU")
    #print("[DEBUG 26] moving D3 model to device:", device)
    #print("[DEBUG 27] D3 model moved to device")
    model.eval()

    #print("[DEBUG 28] model eval mode enabled")
    rows: List[Dict[str, object]] = []
    ok = 0
    #print("[DEBUG 29] starting video loop")

    for i, video in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {video.name}")

        #print(f"[DEBUG 30] processing video: {video}")
        row = process_video(pipe, model, device, video, real_stems)

        #print(f"[DEBUG 31] process_video returned: {video.name}")
        rows.append(row)
        if row.get("success"):
            ok += 1
            print(
                f"  rPPG pseudo_snr={row.get('pseudo_snr_db')} | "
                f"D3={row.get('d3_temporal_score'):.4f} | HF={row.get('highfreq_score'):.4f}"
            )
        else:
            print(f"  ERROR: {row.get('error')}")

    #print("[DEBUG 32] video loop completed")
    out = save_dir / f"unified_features_{datetime.now():%Y%m%d_%H%M%S}.csv"

    #print("[DEBUG 33] saving CSV:", out)
    save_csv(out, rows)
    #print("[DEBUG 34] CSV saved")
    print(f"\nDone {ok}/{len(videos)} → {out}")


if __name__ == "__main__":
    print("[DEBUG] calling main()")
    main()
