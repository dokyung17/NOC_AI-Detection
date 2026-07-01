"""rPPG detection metrics (from pyVHR run_video_cpu_metrics_core_clean)."""

from typing import Dict, Optional, Tuple

import numpy as np


def _representative_signal(win_bvp) -> Optional[np.ndarray]:
    arr = np.asarray(win_bvp)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        return arr.astype(np.float64)
    return np.nanmedian(arr, axis=0).astype(np.float64)


def reconstruct_bvp_waveform(windowed_bvps, times, fps) -> Tuple[np.ndarray, np.ndarray]:
    if len(windowed_bvps) == 0:
        return np.array([]), np.array([])

    representative = [_representative_signal(win_bvp) for win_bvp in windowed_bvps]
    first_valid = next((win for win in representative if win is not None), None)
    if first_valid is None:
        return np.array([]), np.array([])

    win_len = len(first_valid)
    win_sec = win_len / float(fps)
    starts = [max(0, int(round((t - (win_sec / 2.0)) * fps))) for t in times]
    total_len = max(s + len(sig) for s, sig in zip(starts, representative) if sig is not None)

    acc = np.zeros(total_len, dtype=np.float64)
    cnt = np.zeros(total_len, dtype=np.float64)
    for signal, start in zip(representative, starts):
        if signal is None:
            continue
        end = start + len(signal)
        acc[start:end] += signal
        cnt[start:end] += 1.0

    valid = cnt > 0
    if not np.any(valid):
        return np.array([]), np.array([])

    waveform = np.zeros_like(acc)
    waveform[valid] = acc[valid] / cnt[valid]
    t_wave = np.arange(total_len, dtype=np.float64) / float(fps)
    return t_wave[valid], waveform[valid]


def compute_fft_psd(signal_data, fps) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(signal_data, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return np.array([]), np.array([])
    x = x - np.mean(x)
    std = np.std(x)
    if std > 0:
        x = x / std
    window = np.hanning(x.size)
    xw = x * window
    freqs = np.fft.rfftfreq(xw.size, d=1.0 / float(fps))
    power = np.abs(np.fft.rfft(xw)) ** 2
    return freqs, power


def compute_pseudo_snr_db(signal_data, fps, hr_min=0.7, hr_max=4.0) -> float:
    freqs, psd = compute_fft_psd(signal_data, fps)
    if freqs.size == 0 or psd.size == 0:
        return np.nan
    mask = (freqs >= hr_min) & (freqs <= hr_max)
    if not np.any(mask):
        return np.nan
    freqs_hr = freqs[mask]
    psd_hr = psd[mask]
    if psd_hr.size == 0 or np.sum(psd_hr) <= 0:
        return np.nan
    peak_idx = int(np.argmax(psd_hr))
    peak_freq = float(freqs_hr[peak_idx])
    peak_width_hz = 0.10
    signal_mask = np.abs(freqs_hr - peak_freq) <= peak_width_hz
    signal_power = float(np.sum(psd_hr[signal_mask]))
    noise_power = float(np.sum(psd_hr[~signal_mask]))
    return float(10.0 * np.log10((signal_power + 1e-12) / (noise_power + 1e-12)))


def compute_time_signal_metrics(signal_data) -> Dict[str, float]:
    x = np.asarray(signal_data, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"absdiff_std": np.nan, "bvp_std": np.nan}
    xc = x - np.mean(x)
    diffs = np.diff(xc)
    return {
        "absdiff_std": float(np.std(np.abs(diffs))) if diffs.size > 0 else np.nan,
        "bvp_std": float(np.std(x)),
    }


def compute_windowed_bvp_consistency(windowed_bvps) -> Dict[str, float]:
    corr_values = []
    std_values = []
    for win_bvp in windowed_bvps:
        arr = np.asarray(win_bvp, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 3:
            continue
        valid_rows = []
        for row in arr:
            row = row[np.isfinite(row)]
            if row.size == arr.shape[1] and np.std(row) > 0:
                valid_rows.append(row)
        if len(valid_rows) < 2:
            continue
        mat = np.vstack(valid_rows)
        c = np.corrcoef(mat)
        upper = c[np.triu_indices_from(c, k=1)]
        upper = upper[np.isfinite(upper)]
        if upper.size > 0:
            corr_values.extend(upper.tolist())
        std_values.append(float(np.mean(np.std(mat, axis=0))))
    corr_values = np.asarray(corr_values, dtype=np.float64)
    std_values = np.asarray(std_values, dtype=np.float64)
    return {
        "patch_corr_mean": float(np.mean(corr_values)) if corr_values.size else np.nan,
        "patch_signal_std_mean": float(np.mean(std_values)) if std_values.size else np.nan,
    }


def extract_rppg_features(bvp_wave, fps, windowed_bvps) -> Dict[str, float]:
    features = {
        "absdiff_std": np.nan,
        "bvp_std": np.nan,
        "patch_corr_mean": np.nan,
        "patch_signal_std_mean": np.nan,
        "pseudo_snr_db": np.nan,
    }
    if bvp_wave is not None and len(bvp_wave) > 0:
        features.update(compute_time_signal_metrics(bvp_wave))
        features["pseudo_snr_db"] = compute_pseudo_snr_db(bvp_wave, fps)
    features.update(compute_windowed_bvp_consistency(windowed_bvps))
    return features
