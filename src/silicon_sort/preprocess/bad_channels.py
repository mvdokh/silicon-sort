"""Suggest disconnected, flat, or very noisy channels from short raw chunks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from silicon_sort.io.binary import Recording


@dataclass
class ChannelSuggestion:
    """Boolean mask plus a short reason for each flagged channel."""

    mask: np.ndarray
    reasons: dict[int, str]

    def summary(self) -> str:
        flagged = np.flatnonzero(self.mask)
        if flagged.size == 0:
            return "No bad channels suggested"
        parts = [f"{int(channel) + 1} ({self.reasons.get(int(channel), 'unusual')})" for channel in flagged]
        return "Suggested " + ", ".join(parts)


def suggest_bad_channels(
    recording: Recording,
    positions: np.ndarray | None = None,
    n_chunks: int = 6,
    chunk_seconds: float = 1.0,
    rms_high: float = 2.2,
    rms_low: float = 0.25,
    local_rms_high: float = 1.45,
    local_rms_low: float = 0.4,
    neighbor_radius_um: float = 150.0,
    rail_fraction: float = 1e-3,
    spectral_high: float = 4.0,
    spectral_low: float = 0.2,
    seed: int = 0,
) -> ChannelSuggestion:
    """Flag channels that look flat, railed, noisy, or missing high-frequency signal.

    Comparison is against the other channels, and against spatial neighbors when
    positions are available. A site about 1.5× louder than its neighbors is
    flagged. The suggestion is not applied until it is accepted.
    """
    fs = float(recording.sampling_rate)
    chunk = max(32, int(chunk_seconds * fs))
    rng = np.random.default_rng(seed)
    if recording.n_samples <= chunk:
        starts = np.array([0])
        chunk = recording.n_samples
    else:
        n_chunks = max(1, min(n_chunks, recording.n_samples // chunk))
        starts = rng.choice(recording.n_samples - chunk + 1, size=n_chunks, replace=False)

    rms_parts: list[np.ndarray] = []
    rail_parts: list[np.ndarray] = []
    spec_parts: list[np.ndarray] = []
    ac_parts: list[np.ndarray] = []
    info = np.iinfo(recording.dtype) if np.issubdtype(recording.dtype, np.integer) else None
    for start in starts:
        raw = np.array(recording.memmap()[int(start) : int(start) + chunk], copy=True)
        uv = raw.astype(np.float32) * np.float32(recording.voltage_scale_uv_per_bit)
        rms_parts.append(np.sqrt(np.mean(uv * uv, axis=0)))
        if info is None:
            rail_parts.append(np.zeros(recording.n_channels, dtype=np.float64))
        else:
            limit = 0.99 * max(abs(int(info.min)), abs(int(info.max)))
            rail_parts.append(np.mean(np.abs(raw) >= limit, axis=0))
        spec_parts.append(_high_over_mid(uv, fs))
        ac_parts.append(_lag1(uv))

    rms = np.median(np.stack(rms_parts), axis=0)
    rails = np.max(np.stack(rail_parts), axis=0)
    spectral = np.median(np.stack(spec_parts), axis=0)
    autocorr = np.median(np.stack(ac_parts), axis=0)
    rms_med = float(np.median(rms)) if rms.size else 1.0
    spec_med = float(np.median(spectral)) if spectral.size else 1.0
    local = _local_median(rms, positions, neighbor_radius_um)

    reasons: dict[int, list[str]] = {channel: [] for channel in range(recording.n_channels)}
    flat = rms < max(rms_low * rms_med, 1e-3)
    locally_flat = np.zeros(recording.n_channels, dtype=bool) if local is None else rms < local_rms_low * local
    noisy = rms > rms_high * max(rms_med, 1e-6)
    locally_noisy = np.zeros(recording.n_channels, dtype=bool) if local is None else rms > local_rms_high * local
    railed = rails > rail_fraction
    hot_spectrum = spectral > spectral_high * max(spec_med, 1e-6)
    dead_spectrum = spec_med > 1e-8
    dead_spectrum = dead_spectrum & (spectral < spectral_low * spec_med)
    smooth = _smooth_outliers(autocorr)

    for channel in range(recording.n_channels):
        tags = reasons[channel]
        if flat[channel] or locally_flat[channel]:
            tags.append("flat")
        if noisy[channel] or locally_noisy[channel]:
            tags.append("noisy")
        if railed[channel]:
            tags.append("railed")
        if hot_spectrum[channel]:
            tags.append("high-frequency noise")
        if dead_spectrum[channel]:
            tags.append("little high frequency")
        if smooth[channel] and "flat" not in tags:
            tags.append("too smooth")

    mask = np.array([bool(reasons[channel]) for channel in range(recording.n_channels)], dtype=bool)
    text = {channel: ", ".join(reasons[channel]) for channel in np.flatnonzero(mask)}
    return ChannelSuggestion(mask=mask, reasons=text)


def _local_median(rms: np.ndarray, positions: np.ndarray | None, radius_um: float) -> np.ndarray | None:
    if positions is None or len(positions) != rms.size or rms.size < 3:
        return None
    dist = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    np.fill_diagonal(dist, np.inf)
    local = np.zeros(rms.size, dtype=np.float64)
    for index in range(rms.size):
        neighbors = np.flatnonzero(dist[index] <= radius_um)
        if neighbors.size < 2:
            local[index] = np.median(rms)
        else:
            local[index] = float(np.median(rms[neighbors]))
    return local


def _smooth_outliers(autocorr: np.ndarray) -> np.ndarray:
    """Channels whose lag-1 correlation sits well above the rest of the probe."""
    if autocorr.size < 4:
        return np.zeros(autocorr.shape, dtype=bool)
    center = float(np.median(autocorr))
    mad = float(np.median(np.abs(autocorr - center)))
    margin = max(3.0 * 1.4826 * mad, 0.025)
    return autocorr > center + margin


def _lag1(uv: np.ndarray) -> np.ndarray:
    centered = uv - np.mean(uv, axis=0, keepdims=True)
    denom = np.sum(centered * centered, axis=0) + 1e-12
    return np.sum(centered[:-1] * centered[1:], axis=0) / denom


def _high_over_mid(uv: np.ndarray, fs: float) -> np.ndarray:
    nperseg = min(1024, uv.shape[0])
    if nperseg < 32:
        return np.ones(uv.shape[1], dtype=np.float64)
    freqs, power = signal.welch(uv, fs=fs, axis=0, nperseg=nperseg)
    mid = (freqs >= 300) & (freqs < 3000)
    high = freqs >= min(6000.0, 0.4 * fs)
    if not np.any(mid) or not np.any(high):
        return np.ones(uv.shape[1], dtype=np.float64)
    mid_power = np.mean(power[mid], axis=0) + 1e-12
    high_power = np.mean(power[high], axis=0)
    return high_power / mid_power
