"""Suggest false positives and missed spikes from each channel's own waveforms."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.signal import fftconvolve

from silicon_sort.detect.detect import _align, _ms
from silicon_sort.settings import PipelineSettings

_ARTIFACT_COSINE = 0.75
_MAX_MISSES = 24
_MAX_TEMPLATE_SPIKES = 120
Reader = Callable[[int, int], np.ndarray]


def suggest_channel(
    trace: np.ndarray | None,
    times: np.ndarray,
    artifact: np.ndarray,
    settings: PipelineSettings,
    mad: float,
    *,
    reader: Reader | None = None,
    channel: int = 0,
    n_samples: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return likely artifact times, likely missed times, and the channel template.

    Prefer ``reader`` so only spike windows and short miss excerpts are read.
    A full ``trace`` still works for tests and short arrays.
    """
    fs = float(settings.sampling_rate or 1.0)
    pre = _ms(settings.waveform_pre_ms, fs)
    post = _ms(settings.waveform_post_ms, fs)
    times = np.asarray(times, dtype=np.int64)
    artifact = np.asarray(artifact, dtype=bool)
    if n_samples <= 0:
        n_samples = 0 if trace is None else int(np.asarray(trace).shape[0])
    usable = np.flatnonzero((times >= pre) & (times + post <= n_samples))
    if times.size == 0 or usable.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), None
    chosen = usable
    if chosen.size > _MAX_TEMPLATE_SPIKES:
        rng = np.random.default_rng(int(channel) + 19)
        chosen = np.sort(rng.choice(chosen, size=_MAX_TEMPLATE_SPIKES, replace=False))
    waves = _load_waves(reader, channel, times[chosen], pre, post, trace)
    flags = artifact[chosen] if artifact.size == times.size else np.zeros(chosen.size, dtype=bool)
    template = _robust_template(waves, flags)
    if template is None:
        if artifact.size == times.size:
            return times[usable][artifact[usable]], np.zeros(0, dtype=np.int64), None
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), None
    score_idx = usable
    score_waves = waves if usable.size == chosen.size else _load_waves(reader, channel, times[usable], pre, post, trace)
    if usable.size != chosen.size:
        flags = artifact[usable] if artifact.size == times.size else np.zeros(usable.size, dtype=bool)
    else:
        flags = artifact[chosen] if artifact.size == times.size else np.zeros(chosen.size, dtype=bool)
    finite = np.isfinite(score_waves).all(axis=1)
    cosine = np.zeros(score_waves.shape[0], dtype=np.float64)
    if finite.any():
        cosine[finite] = _cosine_rows(score_waves[finite], template)
    bad = flags | (~finite) | (cosine < _ARTIFACT_COSINE)
    artifacts = times[score_idx][bad]
    kept = finite & ~flags & (cosine >= _ARTIFACT_COSINE)
    if int(kept.sum()) >= 5:
        template = np.median(score_waves[kept], axis=0).astype(np.float32)
    misses = _misses(reader, channel, n_samples, times, template, settings, mad, pre, post, trace)
    return np.asarray(artifacts, dtype=np.int64), misses, template


def align_click(
    trace: np.ndarray,
    sample: int,
    template: np.ndarray | None,
    pre: int,
    polarity: float,
    radius: int,
) -> int:
    """Move a click onto the template trough or peak inside ``radius`` samples."""
    sample = int(np.clip(sample, 0, max(trace.size - 1, 0)))
    radius = max(1, int(radius))
    if template is None or template.size < 3 or trace.size < template.size:
        return int(_align(trace, sample, radius, polarity))
    unit = np.asarray(template, dtype=np.float64)
    norm = float(np.linalg.norm(unit))
    if norm < 1e-9:
        return int(_align(trace, sample, radius, polarity))
    unit = unit / norm
    corr = fftconvolve(np.asarray(trace, dtype=np.float64), unit[::-1], mode="same")
    lag = int(template.size) // 2 - int(pre)
    center = int(np.clip(sample + lag, 1, corr.size - 2))
    lo = max(1, center - radius)
    hi = min(corr.size - 1, center + radius + 1)
    if hi <= lo:
        return int(_align(trace, sample, radius, polarity))
    peak = lo + int(np.argmax(corr[lo:hi]))
    aligned = int(np.clip(peak - lag, 0, trace.size - 1))
    return int(_align(trace, aligned, max(int(pre) // 2, 1), polarity))


def _load_waves(
    reader: Reader | None,
    channel: int,
    times: np.ndarray,
    pre: int,
    post: int,
    trace: np.ndarray | None,
) -> np.ndarray:
    length = pre + post
    waves = np.full((times.size, length), np.nan, dtype=np.float32)
    for index, sample in enumerate(times):
        start = int(sample) - pre
        stop = int(sample) + post
        if start < 0:
            continue
        if trace is not None:
            column = np.asarray(trace)
            if column.ndim == 2:
                if stop > column.shape[0] or channel >= column.shape[1]:
                    continue
                waves[index] = np.asarray(column[start:stop, channel], dtype=np.float32)
            else:
                if stop > column.shape[0]:
                    continue
                waves[index] = np.asarray(column[start:stop], dtype=np.float32)
            continue
        if reader is None:
            continue
        try:
            snippet = reader(start, stop)
        except Exception:
            continue
        if snippet.ndim == 1:
            if snippet.shape[0] == length:
                waves[index] = np.asarray(snippet, dtype=np.float32)
        elif snippet.shape[0] == length and channel < snippet.shape[1]:
            waves[index] = np.asarray(snippet[:, channel], dtype=np.float32)
    return waves


def _robust_template(waves: np.ndarray, artifact: np.ndarray) -> np.ndarray | None:
    if waves.shape[0] == 0:
        return None
    finite = np.isfinite(waves).all(axis=1)
    pool = waves[finite & ~artifact] if np.any(finite & ~artifact) else waves[finite]
    if pool.shape[0] == 0:
        return None
    return np.median(pool, axis=0).astype(np.float32)


def _cosine_rows(waves: np.ndarray, template: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(template))
    if norm < 1e-9:
        return np.zeros(waves.shape[0], dtype=np.float64)
    row_norms = np.linalg.norm(waves, axis=1)
    return waves @ template / (row_norms * norm + 1e-12)


def _misses(
    reader: Reader | None,
    channel: int,
    n_samples: int,
    times: np.ndarray,
    template: np.ndarray,
    settings: PipelineSettings,
    mad: float,
    pre: int,
    post: int,
    trace: np.ndarray | None,
) -> np.ndarray:
    if not np.isfinite(mad) or mad <= 0 or n_samples <= 0:
        return np.zeros(0, dtype=np.int64)
    fs = float(settings.sampling_rate or 1.0)
    polarity = -1.0 if settings.threshold_mad < 0 else 1.0
    dead = _ms(settings.dead_time_ms, fs)
    windows = _search_windows(times, n_samples, fs)
    found: list[int] = []
    amplitudes: list[float] = []
    occupied = np.sort(times)
    width = int(template.shape[0])
    for start, stop in windows:
        local = _read_span(reader, channel, start, stop, trace)
        if local is None or local.size < width + 2:
            continue
        hits, amps = _scan_excerpt(
            local,
            template,
            pre,
            polarity,
            float(mad),
            float(settings.threshold_mad),
            min(0.45, float(settings.template_threshold_fraction)),
            float(settings.template_min_cosine),
            dead,
            occupied,
            start,
        )
        found.extend(hits)
        amplitudes.extend(amps)
    if not found:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(np.abs(np.asarray(amplitudes)))[::-1]
    chosen = order[:_MAX_MISSES]
    return np.asarray([found[int(index)] for index in chosen], dtype=np.int64)


def _search_windows(times: np.ndarray, n_samples: int, fs: float) -> list[tuple[int, int]]:
    """Short excerpts around detections instead of a whole-file correlation."""
    pad = max(1, int(round(1.5 * fs)))
    anchors = np.asarray(times, dtype=np.int64)
    if anchors.size == 0:
        stop = min(n_samples, max(1, int(round(2.0 * fs))))
        return [(0, stop)]
    if anchors.size > 40:
        rng = np.random.default_rng(7)
        anchors = np.sort(rng.choice(anchors, size=40, replace=False))
    spans = [(max(0, int(t) - pad), min(n_samples, int(t) + pad)) for t in anchors]
    spans.sort()
    merged: list[list[int]] = []
    for start, stop in spans:
        if not merged or start > merged[-1][1]:
            merged.append([start, stop])
        else:
            merged[-1][1] = max(merged[-1][1], stop)
    return [(int(start), int(stop)) for start, stop in merged]


def _read_span(
    reader: Reader | None,
    channel: int,
    start: int,
    stop: int,
    trace: np.ndarray | None,
) -> np.ndarray | None:
    if stop <= start:
        return None
    if trace is not None:
        column = np.asarray(trace)
        if column.ndim == 2:
            if channel >= column.shape[1]:
                return None
            return np.asarray(column[start:stop, channel], dtype=np.float32)
        return np.asarray(column[start:stop], dtype=np.float32)
    if reader is None:
        return None
    try:
        snippet = reader(start, stop)
    except Exception:
        return None
    if snippet.ndim == 1:
        return np.asarray(snippet, dtype=np.float32)
    if channel >= snippet.shape[1]:
        return None
    return np.asarray(snippet[:, channel], dtype=np.float32)


def _scan_excerpt(
    local: np.ndarray,
    template: np.ndarray,
    pre: int,
    polarity: float,
    mad: float,
    threshold_mad: float,
    fraction: float,
    min_cosine: float,
    dead: int,
    occupied: np.ndarray,
    offset: int,
) -> tuple[list[int], list[float]]:
    from silicon_sort.detect.templates import _correlation_peaks, _too_close

    norm = float(np.linalg.norm(template))
    trough = float(np.min(template) if polarity < 0 else np.max(template))
    peak_amp = abs(trough)
    if norm < 1e-9 or peak_amp < 1e-9:
        return [], []
    unit = template / norm
    raw_threshold = fraction * abs(threshold_mad) * mad
    corr_threshold = raw_threshold / peak_amp * norm
    corr = fftconvolve(np.asarray(local, dtype=np.float64), unit[::-1], mode="same")
    peaks = _correlation_peaks(corr, corr_threshold, polarity)
    lag = int(template.shape[0]) // 2 - pre
    width = int(template.shape[0])
    times: list[int] = []
    amplitudes: list[float] = []
    for peak in peaks:
        global_index = offset + int(peak) - lag
        local_index = int(peak) - lag
        if local_index < pre or local_index + (width - pre) > local.size:
            continue
        aligned_local = int(_align(local, local_index, max(pre // 2, 1), polarity))
        aligned = offset + aligned_local
        if aligned_local < pre or aligned_local + (width - pre) > local.size:
            continue
        if _too_close(occupied, aligned, dead) or _too_close(np.asarray(times, dtype=np.int64), aligned, dead):
            continue
        waveform = np.asarray(local[aligned_local - pre : aligned_local - pre + width], dtype=np.float64)
        peak_value = float(waveform[pre] if pre < waveform.size else waveform[np.argmin(waveform)])
        if polarity < 0 and peak_value > -raw_threshold:
            continue
        if polarity > 0 and peak_value < raw_threshold:
            continue
        cosine = float(np.dot(waveform, template) / (np.linalg.norm(waveform) * norm + 1e-12))
        if cosine < min_cosine:
            continue
        times.append(aligned)
        amplitudes.append(peak_value)
    return times, amplitudes
