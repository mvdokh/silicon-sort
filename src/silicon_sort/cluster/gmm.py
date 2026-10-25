"""Per-site Gaussian mixture. Cluster count is forced or chosen by BIC."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.mixture import GaussianMixture

from silicon_sort.cluster.features import features_for_channel
from silicon_sort.detect.detect import SpikeIndex
from silicon_sort.io.geometry import Probe
from silicon_sort.settings import PipelineSettings


@dataclass
class ChannelPreview:
    channel: int
    times_s: np.ndarray
    amplitudes: np.ndarray
    pca: np.ndarray
    labels: np.ndarray
    waveforms: np.ndarray
    k_used: int
    k_auto: int


@dataclass
class SiteClustering:
    channel: int
    times: np.ndarray
    amplitudes: np.ndarray
    labels: np.ndarray
    k_used: int
    k_auto: int


def cluster_channel(
    data: np.ndarray,
    spikes: SpikeIndex,
    probe: Probe,
    settings: PipelineSettings,
    mask: np.ndarray,
    channel: int,
) -> tuple[SiteClustering | None, ChannelPreview | None]:
    features = features_for_channel(data, spikes, probe, settings, mask, channel)
    if features is None:
        return None, None
    forced = settings.forced_k(channel)
    labels, k_used, k_auto = label_features(features.features, forced, settings.k_max, settings.rng_seed + channel)
    clustering = SiteClustering(
        channel=channel,
        times=features.times,
        amplitudes=features.amplitudes,
        labels=labels.astype(np.int32),
        k_used=k_used,
        k_auto=k_auto,
    )
    fs = float(settings.sampling_rate or 1.0)
    limit = min(features.waveforms.shape[0], 120)
    preview = ChannelPreview(
        channel=channel,
        times_s=features.times[:limit] / fs,
        amplitudes=features.amplitudes[:limit],
        pca=features.pca[:limit, : min(2, features.pca.shape[1])],
        labels=labels[:limit].astype(np.int32),
        waveforms=features.waveforms[:limit],
        k_used=k_used,
        k_auto=k_auto,
    )
    return clustering, preview


def label_features(
    features: np.ndarray,
    forced_k: int | None,
    k_max: int,
    seed: int,
) -> tuple[np.ndarray, int, int]:
    """Return labels, the k that was fit, and the BIC choice (ignoring a forced k)."""
    n_rows = features.shape[0]
    if n_rows == 0:
        return np.zeros(0, dtype=np.int32), 1, 1
    scaled = _standardize(features)
    k_auto, _auto_labels = _select_bic(scaled, k_max, seed)
    if forced_k is not None and forced_k > 0:
        k_used = int(min(forced_k, max(1, n_rows)))
        labels = _fit_labels(scaled, k_used, seed)
        return labels, k_used, k_auto
    labels = _fit_labels(scaled, k_auto, seed)
    return labels, k_auto, k_auto


def _select_bic(scaled: np.ndarray, k_max: int, seed: int) -> tuple[int, np.ndarray]:
    n_rows = scaled.shape[0]
    limit = max(1, min(int(k_max), max(1, n_rows // 8)))
    best_k = 1
    best_bic = np.inf
    best_labels = np.zeros(n_rows, dtype=np.int32)
    for k in range(1, limit + 1):
        labels, bic = _fit(scaled, k, seed)
        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_labels = labels
    return best_k, best_labels


def _fit_labels(scaled: np.ndarray, k: int, seed: int) -> np.ndarray:
    labels, _bic = _fit(scaled, k, seed)
    return labels


def _fit(scaled: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, float]:
    n_rows = scaled.shape[0]
    k = int(max(1, min(k, n_rows)))
    if k == 1 or n_rows < k * 4:
        if k > 1 and n_rows < k * 4:
            k = 1
        labels = np.zeros(n_rows, dtype=np.int32)
        return labels, _bic_one(scaled)
    try:
        model = GaussianMixture(
            n_components=k,
            covariance_type="full",
            reg_covar=1e-3,
            n_init=3,
            max_iter=200,
            random_state=seed,
        )
        model.fit(scaled)
        return model.predict(scaled).astype(np.int32), float(model.bic(scaled))
    except Exception:
        return np.zeros(n_rows, dtype=np.int32), _bic_one(scaled)


def _bic_one(scaled: np.ndarray) -> float:
    if scaled.shape[0] < 2:
        return 0.0
    try:
        model = GaussianMixture(n_components=1, covariance_type="full", reg_covar=1e-3, random_state=0)
        model.fit(scaled)
        return float(model.bic(scaled))
    except Exception:
        return 0.0


def _standardize(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < 1e-6] = 1.0
    return (values - mean) / std


def cluster_all_sites(
    data: np.ndarray,
    spikes: SpikeIndex,
    probe: Probe,
    settings: PipelineSettings,
    mask: np.ndarray,
    progress=None,
) -> list[SiteClustering]:
    clean = spikes.clean()
    if clean.n_spikes == 0:
        return []
    channels = [int(ch) for ch in np.unique(clean.channels) if not mask[int(ch)]]
    found: list[SiteClustering] = []
    for index, channel in enumerate(channels):
        clustering, _preview = cluster_channel(data, spikes, probe, settings, mask, channel)
        if clustering is not None:
            found.append(clustering)
        if progress is not None:
            progress((index + 1) / max(len(channels), 1), f"Clustering channel {channel}")
    return found


# Re-export for callers that want PCA without a full site.
__all__ = [
    "ChannelPreview",
    "SiteClustering",
    "cluster_all_sites",
    "cluster_channel",
    "embed_pca",
    "label_features",
]
