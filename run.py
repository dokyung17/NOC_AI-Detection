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


def extract_d3_scores(model: D3_model, video_path: Path, device: torch.device) -> tuple[float, float]:
    frames = load_frames_from_video(video_path).to(device)
    with torch.no_grad():
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
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    videos = resolve_videos(args.inputs, args.pattern, args.recursive)
    if not videos:
        raise FileNotFoundError("No videos found.")

    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    real_stems = parse_real_stems(args.real_stems)

    device = torch.device(args.device)
    pipe = Pipeline()
    model = D3_model(encoder_type=args.encoder, loss_type=args.loss).to(device)
    model.eval()

    rows: List[Dict[str, object]] = []
    ok = 0
    for i, video in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {video.name}")
        row = process_video(pipe, model, device, video, real_stems)
        rows.append(row)
        if row.get("success"):
            ok += 1
            print(
                f"  rPPG pseudo_snr={row.get('pseudo_snr_db')} | "
                f"D3={row.get('d3_temporal_score'):.4f} | HF={row.get('highfreq_score'):.4f}"
            )
        else:
            print(f"  ERROR: {row.get('error')}")

    out = save_dir / f"unified_features_{datetime.now():%Y%m%d_%H%M%S}.csv"
    save_csv(out, rows)
    print(f"\nDone {ok}/{len(videos)} → {out}")


if __name__ == "__main__":
    main()
