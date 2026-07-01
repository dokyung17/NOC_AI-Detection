"""Video path helpers."""

from pathlib import Path
from typing import List, Optional, Set


def resolve_videos(inputs: List[str], pattern: str, recursive: bool) -> List[Path]:
    videos: List[Path] = []
    seen = set()
    for raw in inputs:
        path = Path(raw).expanduser().resolve()
        if path.is_file():
            key = str(path)
            if key not in seen:
                videos.append(path)
                seen.add(key)
            continue
        if path.is_dir():
            iterator = path.rglob(pattern) if recursive else path.glob(pattern)
            for item in sorted(iterator):
                if item.is_file():
                    key = str(item.resolve())
                    if key not in seen:
                        videos.append(item.resolve())
                        seen.add(key)
    return videos


def parse_real_stems(real_stems: Optional[str]) -> Set[str]:
    if real_stems is None or str(real_stems).strip() == "":
        return set()
    return {item.strip() for item in str(real_stems).split(",") if item.strip()}


def infer_label(video_path: Path, real_stems: Set[str]) -> str:
    if real_stems:
        return "real" if video_path.stem in real_stems else "fake"
    parent = video_path.parent.name.lower()
    if parent == "real":
        return "real"
    if parent == "fake":
        return "fake"
    return ""
