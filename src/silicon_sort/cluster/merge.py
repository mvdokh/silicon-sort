"""Merge clusters across neighboring sites, then split mixtures that break the refractory period."""

from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture

from silicon_sort.cluster.features import extract_windows, samples_for
from silicon_sort.cluster.gmm import SiteClustering
from silicon_sort.io.geometry import Probe
from silicon_sort.settings import PipelineSettings


def assemble_labels(
    sites: list[SiteClustering],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack per-site results into global spike arrays and provisional cluster ids."""
    if not sites:
        empty_i = np.zeros(0, dtype=np.int64)
        empty_c = np.zeros(0, dtype=np.int32)
        empty_a = np.zeros(0, dtype=np.float32)
        return empty_i, empty_c, empty_a, np.zeros(0, dtype=np.int32)
    times = np.concatenate([site.times for site in sites])
    channels = np.concatenate([np.full(site.times.shape, site.channel, dtype=np.int32) for site in sites])
    amplitudes = np.concatenate([site.amplitudes for site in sites])
    labels = []
    next_id = 0
    peak_of: list[int] = []
    for site in sites:
        shifted = site.labels.astype(np.int32) + next_id
        labels.append(shifted)
        n_local = int(site.labels.max()) + 1 if site.labels.size else 0
        peak_of.extend([site.channel] * n_local)
        next_id += n_local
    return times, channels, amplitudes, np.concatenate(labels).astype(np.int32)


def merge_and_split(
    times: np.ndarray,
    channels: np.ndarray,
    amplitudes: np.ndarray,
    labels: np.ndarray,
    data: np.ndarray,
    probe: Probe,
    settings: PipelineSettings,
) -> tuple[np.ndarray, np.ndarray]:
    """Return remapped cluster ids and the median template for each id.

    Templates are shaped ``(n_units, n_samples, n_channels)``.
    """
    if labels.size == 0:
        fs = float(settings.sampling_rate or 1.0)
        width = samples_for(settings.waveform_pre_ms, fs) + samples_for(settings.waveform_post_ms, fs)
        return labels.astype(np.int32), np.zeros((0, width, data.shape[1]), dtype=np.float32)

    fs = float(settings.sampling_rate or 1.0)
    pre = samples_for(settings.waveform_pre_ms, fs)
    post = samples_for(settings.waveform_post_ms, fs)
    templates, peaks = _templates(data, times, labels, pre, post, channels)
    labels = _merge_neighbors(times, labels, templates, peaks, probe, settings, fs)
    templates, peaks = _templates(data, times, labels, pre, post, channels)
    labels = _split_bimodal(times, amplitudes, labels, fs, settings)
    labels = _drop_duplicate_times(times, amplitudes, labels, fs)
    labels, templates = _compact(labels, data, times, channels, pre, post)
    order = np.argsort(times, kind="mergesort")
    # Caller sorts using the returned order via times args; we sort here and the caller
    # must use the same order. Return labels aligned to the input order.
    del order
    return labels.astype(np.int32), templates


def _merge_neighbors(
    times: np.ndarray,
    labels: np.ndarray,
    templates: dict[int, np.ndarray],
    peaks: dict[int, int],
    probe: Probe,
    settings: PipelineSettings,
    fs: float,
) -> np.ndarray:
    ids = sorted(templates)
    parent = {cluster_id: cluster_id for cluster_id in ids}

    def find(cluster_id: int) -> int:
        while parent[cluster_id] != cluster_id:
            parent[cluster_id] = parent[parent[cluster_id]]
            cluster_id = parent[cluster_id]
        return cluster_id

    radius = float(settings.feature_radius_um)
    for left in range(len(ids)):
        for right in range(left + 1, len(ids)):
            a = ids[left]
            b = ids[right]
            peak_a = peaks[a]
            peak_b = peaks[b]
            if peak_a == peak_b:
                continue
            distance = float(np.linalg.norm(probe.positions[peak_a] - probe.positions[peak_b]))
            if distance > radius:
                continue
            if _cosine(templates[a], templates[b]) < settings.merge_cosine:
                continue
            if _footprint_corr(templates[a], templates[b]) < settings.merge_footprint:
                continue
            times_a = times[labels == a]
            times_b = times[labels == b]
            if not _shared_structure(times_a, times_b, fs, settings.refractory_ms):
                continue
            parent[find(b)] = find(a)
    merged = labels.copy()
    for cluster_id in ids:
        merged[labels == cluster_id] = find(cluster_id)
    return merged


def _split_bimodal(
    times: np.ndarray,
    amplitudes: np.ndarray,
    labels: np.ndarray,
    fs: float,
    settings: PipelineSettings,
) -> np.ndarray:
    if labels.size == 0:
        return labels
    updated = labels.copy()
    refractory = max(1, int(round(settings.refractory_ms * 1e-3 * fs)))
    next_id = int(labels.max()) + 1
    for cluster_id in np.unique(labels):
        index = np.flatnonzero(labels == cluster_id)
        if index.size < 40:
            continue
        ordered = np.sort(times[index])
        intervals = np.diff(ordered)
        if intervals.size == 0 or float(np.mean(intervals < refractory)) < settings.isi_split_fraction:
            continue
        amp = amplitudes[index].astype(np.float64).reshape(-1, 1)
        amp = (amp - amp.mean()) / max(float(amp.std()), 1e-6)
        try:
            one = GaussianMixture(n_components=1, reg_covar=1e-3, random_state=settings.rng_seed)
            two = GaussianMixture(n_components=2, reg_covar=1e-3, n_init=3, random_state=settings.rng_seed)
            one.fit(amp)
            two.fit(amp)
        except Exception:
            continue
        if float(two.bic(amp)) + 10 >= float(one.bic(amp)):
            continue
        part = two.predict(amp)
        if min(np.mean(part == 0), np.mean(part == 1)) < 0.1:
            continue
        updated[index[part == 1]] = next_id
        next_id += 1
    return updated


def _drop_duplicate_times(times: np.ndarray, amplitudes: np.ndarray, labels: np.ndarray, fs: float) -> np.ndarray:
    """Within a cluster, collapse events closer than 0.3 ms to the larger amplitude."""
    if labels.size == 0:
        return labels
    updated = labels.copy()
    window = max(1, int(round(0.0003 * fs)))
    for cluster_id in np.unique(labels):
        index = np.flatnonzero(updated == cluster_id)
        if index.size < 2:
            continue
        order = index[np.argsort(times[index], kind="mergesort")]
        last_time = -10**18
        last_pos = -1
        for pos in order:
            if int(times[pos]) - last_time < window and last_pos >= 0:
                if abs(float(amplitudes[pos])) <= abs(float(amplitudes[last_pos])):
                    updated[pos] = -1
                else:
                    updated[last_pos] = -1
                    last_time = int(times[pos])
                    last_pos = int(pos)
                continue
            last_time = int(times[pos])
            last_pos = int(pos)
    return updated


def _compact(
    labels: np.ndarray,
    data: np.ndarray,
    times: np.ndarray,
    channels: np.ndarray,
    pre: int,
    post: int,
) -> tuple[np.ndarray, np.ndarray]:
    keep = labels >= 0
    # Preserve input alignment: negative labels mark dropped duplicate spikes.
    surviving = sorted(int(v) for v in np.unique(labels[keep])) if np.any(keep) else []
    remap = {old: new for new, old in enumerate(surviving)}
    compacted = np.full(labels.shape, -1, dtype=np.int32)
    for old, new in remap.items():
        compacted[labels == old] = new
    width = pre + post
    templates = np.zeros((len(surviving), width, data.shape[1]), dtype=np.float32)
    for new, old in enumerate(surviving):
        chosen = np.flatnonzero(labels == old)
        templates[new] = _median_template(data, times[chosen], pre, post)
    del channels
    return compacted, templates


def _templates(
    data: np.ndarray,
    times: np.ndarray,
    labels: np.ndarray,
    pre: int,
    post: int,
    channels: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    templates: dict[int, np.ndarray] = {}
    peaks: dict[int, int] = {}
    for cluster_id in np.unique(labels):
        if cluster_id < 0:
            continue
        cluster_id = int(cluster_id)
        chosen = np.flatnonzero(labels == cluster_id)
        templates[cluster_id] = _median_template(data, times[chosen], pre, post)
        channel_values = channels[chosen]
        peaks[cluster_id] = int(np.bincount(channel_values.astype(int)).argmax()) if channel_values.size else 0
    return templates, peaks


def _median_template(data: np.ndarray, times: np.ndarray, pre: int, post: int, limit: int = 200) -> np.ndarray:
    if times.size > limit:
        times = times[:limit]
    windows = extract_windows(data, times, pre, post)
    if windows.shape[0] == 0:
        return np.zeros((pre + post, data.shape[1]), dtype=np.float32)
    return np.median(windows, axis=0).astype(np.float32)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = left.ravel().astype(np.float64)
    b = right.ravel().astype(np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _footprint_corr(left: np.ndarray, right: np.ndarray) -> float:
    a = np.max(np.abs(left), axis=0)
    b = np.max(np.abs(right), axis=0)
    if float(np.std(a)) < 1e-9 or float(np.std(b)) < 1e-9:
        return 1.0 if np.argmax(a) == np.argmax(b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _shared_structure(times_a: np.ndarray, times_b: np.ndarray, fs: float, refractory_ms: float) -> bool:
    if times_a.size == 0 or times_b.size == 0:
        return False
    if times_a.size > 2000:
        times_a = times_a[:2000]
    if times_b.size > 2000:
        times_b = times_b[:2000]
    diffs = (times_b[None, :] - times_a[:, None]) / fs * 1000.0
    diffs = diffs.ravel()
    diffs = diffs[np.abs(diffs) <= 20]
    if diffs.size < 10:
        return False
    bins = np.linspace(-20, 20, 41)
    hist, edges = np.histogram(diffs, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    refractory = np.abs(centers) <= refractory_ms
    shoulder = (np.abs(centers) > 5) & (np.abs(centers) <= 20)
    center_rate = float(np.mean(hist[refractory])) if np.any(refractory) else 0.0
    shoulder_rate = float(np.mean(hist[shoulder])) if np.any(shoulder) else 0.0
    zero = int(np.argmin(np.abs(centers)))
    # A zero-lag pile-up has no shoulder: the two clusters are the same spikes.
    coincident = float(np.mean(np.abs(diffs) <= max(refractory_ms, 0.5))) >= 0.5 and diffs.size >= 5
    dip = shoulder_rate > 0 and center_rate < 0.5 * shoulder_rate
    peak = shoulder_rate > 0 and hist[zero] > 3.0 * shoulder_rate
    return bool(dip or peak or coincident)
