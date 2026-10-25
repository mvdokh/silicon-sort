"""Local waveform features: neighborhood PCA, amplitude, and center of mass."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA

from silicon_sort.detect.detect import SpikeIndex
from silicon_sort.io.geometry import Probe
from silicon_sort.settings import PipelineSettings


@dataclass
class SiteFeatures:
    channel: int
    times: np.ndarray
    amplitudes: np.ndarray
    pca: np.ndarray
    center_of_mass: np.ndarray
    features: np.ndarray
    waveforms: np.ndarray
    neighbor_channels: np.ndarray


def samples_for(milliseconds: float, sampling_rate: float) -> int:
    return max(1, int(round(float(milliseconds) * 1e-3 * float(sampling_rate))))


def features_for_channel(
    data: np.ndarray,
    spikes: SpikeIndex,
    probe: Probe,
    settings: PipelineSettings,
    mask: np.ndarray,
    channel: int,
) -> SiteFeatures | None:
    """Waveforms and features for spikes whose peak channel is ``channel``."""
    clean = spikes.clean()
    chosen = np.flatnonzero(clean.channels == channel)
    if chosen.size == 0:
        return None
    fs = float(settings.sampling_rate or 1.0)
    pre = samples_for(settings.waveform_pre_ms, fs)
    post = samples_for(settings.waveform_post_ms, fs)
    neighbors = probe.neighbors(settings.feature_radius_um, mask)[channel]
    local = np.concatenate([np.array([channel], dtype=int), neighbors])
    local = np.unique(local)
    times = clean.times[chosen]
    amplitudes = clean.amplitudes[chosen]
    cube, keep = _read_cube(data, times, local, pre, post)
    if keep.size == 0:
        return None
    times = times[keep]
    amplitudes = amplitudes[keep]
    trough = cube.min(axis=1)
    weights = np.clip(-trough, 0, None)
    denom = weights.sum(axis=1, keepdims=True)
    denom[denom <= 0] = 1.0
    center = (weights @ probe.positions[local]) / denom
    flat = cube.reshape(cube.shape[0], -1)
    pca = embed_pca(flat, settings.n_pca, settings.rng_seed)
    features = np.column_stack([pca, amplitudes.reshape(-1, 1), center]).astype(np.float64)
    peak_index = int(np.flatnonzero(local == channel)[0])
    return SiteFeatures(
        channel=int(channel),
        times=times.astype(np.int64),
        amplitudes=amplitudes.astype(np.float32),
        pca=pca.astype(np.float32),
        center_of_mass=center.astype(np.float32),
        features=features,
        waveforms=cube[:, :, peak_index].astype(np.float32),
        neighbor_channels=local.astype(int),
    )


def embed_pca(flat: np.ndarray, n_components: int, seed: int) -> np.ndarray:
    n_rows = flat.shape[0]
    if n_rows == 0:
        return np.zeros((0, n_components), dtype=np.float64)
    if n_rows == 1 or min(flat.shape) < 2:
        out = np.zeros((n_rows, n_components), dtype=np.float64)
        return out
    n_comp = int(min(n_components, n_rows - 1, flat.shape[1]))
    model = PCA(n_components=n_comp, random_state=seed)
    embedded = model.fit_transform(flat)
    if n_comp < n_components:
        embedded = np.pad(embedded, ((0, 0), (0, n_components - n_comp)))
    return embedded


def _read_cube(
    data: np.ndarray,
    times: np.ndarray,
    channels: np.ndarray,
    pre: int,
    post: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_samples = data.shape[0]
    width = pre + post
    valid = (times >= pre) & (times + post <= n_samples)
    index = np.flatnonzero(valid)
    cube = np.zeros((index.size, width, channels.size), dtype=np.float32)
    for row, spike in enumerate(times[index]):
        start = int(spike) - pre
        cube[row] = np.asarray(data[start : start + width][:, channels], dtype=np.float32)
    return cube, index


def extract_windows(
    data: np.ndarray,
    times: np.ndarray,
    pre: int,
    post: int,
) -> np.ndarray:
    """Return ``(n_spikes, n_samples, n_channels)`` windows. Out-of-range spikes are zeros."""
    width = pre + post
    out = np.zeros((times.size, width, data.shape[1]), dtype=np.float32)
    n_samples = data.shape[0]
    for row, spike in enumerate(times):
        start = int(spike) - pre
        stop = start + width
        if start < 0 or stop > n_samples:
            continue
        out[row] = np.asarray(data[start:stop], dtype=np.float32)
    return out
