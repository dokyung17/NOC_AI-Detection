"""D3 spatial high-frequency (Gaussian + Laplacian) score."""

import torch
import torch.nn.functional as F


def frames_to_gray(frames: torch.Tensor) -> torch.Tensor:
    """
    frames: [T, C, H, W]
    return: [T, 1, H, W]
    """
    if frames.ndim != 4:
        raise ValueError(
            f"Expected frames shape [T,C,H,W], got {tuple(frames.shape)}"
        )

    if frames.size(1) == 3:
        r = frames[:, 0:1]
        g = frames[:, 1:2]
        b = frames[:, 2:3]

        return 0.299 * r + 0.587 * g + 0.114 * b

    return frames[:, 0:1]


def gaussian_blur(
    gray: torch.Tensor,
    kernel_size: int = 3,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    gray: [T, 1, H, W]

    Gaussian Blur를 적용해 압축 노이즈와 미세 랜덤 노이즈를 완화한다.
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd number.")

    if sigma <= 0:
        raise ValueError("sigma must be greater than 0.")

    device = gray.device
    dtype = gray.dtype

    coords = torch.arange(
        kernel_size,
        device=device,
        dtype=dtype,
    )

    coords = coords - (kernel_size - 1) / 2.0

    kernel_1d = torch.exp(
        -(coords ** 2) / (2.0 * sigma ** 2)
    )

    kernel_1d = kernel_1d / kernel_1d.sum()

    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel_2d = kernel_2d / kernel_2d.sum()

    kernel = kernel_2d.view(1, 1, kernel_size, kernel_size)

    padding = kernel_size // 2

    return F.conv2d(
        gray,
        kernel,
        padding=padding,
    )


def laplacian_map(gray: torch.Tensor) -> torch.Tensor:
    """
    gray: [T, 1, H, W]
    return: Laplacian response [T, 1, H, W]
    """
    kernel = torch.tensor(
        [
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0],
        ],
        dtype=gray.dtype,
        device=gray.device,
    ).view(1, 1, 3, 3)

    return F.conv2d(
        gray,
        kernel,
        padding=1,
    )


def highfreq_laplacian_score(
    frames: torch.Tensor,
    use_gaussian: bool = True,
    gaussian_kernel_size: int = 3,
    gaussian_sigma: float = 1.0,
) -> float:
    """
    Gaussian Blur 적용 여부를 선택할 수 있는 Laplacian 고주파 점수.

    처리 순서:
    frames
    → grayscale
    → optional Gaussian Blur
    → Laplacian
    → variance
    """
    gray = frames_to_gray(frames)

    if use_gaussian:
        gray = gaussian_blur(
            gray,
            kernel_size=gaussian_kernel_size,
            sigma=gaussian_sigma,
        )

    lap = laplacian_map(gray)

    return float(
        lap.var(unbiased=False).item()
    )