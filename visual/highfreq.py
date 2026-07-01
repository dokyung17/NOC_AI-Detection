"""D3 spatial high-frequency (Laplacian) score."""

import torch


def frames_to_gray(frames: torch.Tensor) -> torch.Tensor:
    if frames.size(1) == 3:
        r, g, b = frames[:, 0:1], frames[:, 1:2], frames[:, 2:3]
        return 0.299 * r + 0.587 * g + 0.114 * b
    return frames[:, 0:1]


def laplacian_map(gray: torch.Tensor) -> torch.Tensor:
    kernel = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        dtype=gray.dtype,
        device=gray.device,
    ).view(1, 1, 3, 3)
    return torch.nn.functional.conv2d(gray, kernel, padding=1)


def highfreq_laplacian_score(frames: torch.Tensor) -> float:
    gray = frames_to_gray(frames)
    lap = laplacian_map(gray)
    return float(lap.var(unbiased=False).item())
