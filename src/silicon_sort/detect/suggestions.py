"""Suggest false positives and missed spikes from each channel's own waveforms."""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from silicon_sort.detect.detect import _align, _ms
from silicon_sort.detect.templates import _scan_channel
from silicon_sort.settings import PipelineSettings

_ARTIFACT_COSINE = 0.75
_MAX_MISSES = 24


def suggest_channel(
    trace: np.ndarray,
    times: np.ndarray,
    artifact: np.ndarray,
    settings: PipelineSettings,
    mad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return likely artifact times, likely missed times, and the channel template.

    The template is the median waveform of spikes that resemble each other.
    Detections that do not match it, and events already flagged as artifacts,
    are the false-positive suggestions. A lower-threshold scan of that same
    template proposes spikes the detector missed.
    """
    fs = float(settings.sampling_rate or 1.0)
    pre = _ms(settings.waveform_pre_ms, fs)
    post = _ms(settings.waveform_post_ms, fs)
    times = np.asarray(times, dtype=np.int64)
    artifact = np.asarray(artifact, dtype=bool)
    usable = np.flatnonzero((times >= pre) & (times + post <= trace.size))
    if times.size == 0 or usable.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), None
    waves = np.stack([np.asarray(trace[int(t) - pre : int(t) + post], dtype=np.float32) for t in times[usable]])
    flags = artifact[usable] if artifact.size == times.size else np.zeros(usable.size, dtype=bool)
    template = _robust_template(waves, flags)
    if template is None:
        return times[usable][flags], np.zeros(0, dtype=np.int64), None
    cosine = _cosine_rows(waves, template)
    artifacts = times[usable][flags | (cosine < _ARTIFACT_COSINE)]
    kept = ~flags & (cosine >= _ARTIFACT_COSINE)
    if int(kept.sum()) >= 5:
        template = np.median(waves[kept], axis=0).astype(np.float32)
    misses = _misses(trace, times, template, settings, mad, pre, post)
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


def _robust_template(waves: np.ndarray, artifact: np.ndarray) -> np.ndarray | None:
    if waves.shape[0] == 0:
        return None
    pool = waves[~artifact] if np.any(~artifact) else waves
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
    trace: np.ndarray,
    times: np.ndarray,
    template: np.ndarray,
    settings: PipelineSettings,
    mad: float,
    pre: int,
    post: int,
) -> np.ndarray:
    if not np.isfinite(mad) or mad <= 0:
        return np.zeros(0, dtype=np.int64)
    polarity = -1.0 if settings.threshold_mad < 0 else 1.0
    found, amplitudes = _scan_channel(
        trace,
        template,
        pre,
        polarity,
        float(mad),
        float(settings.threshold_mad),
        min(0.45, float(settings.template_threshold_fraction)),
        float(settings.template_min_cosine),
        max(pre + post, int(round(float(settings.chunk_seconds) * float(settings.sampling_rate or 1.0)))),
        np.sort(times),
        _ms(settings.dead_time_ms, float(settings.sampling_rate or 1.0)),
    )
    if not found:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(np.abs(np.asarray(amplitudes)))[::-1]
    chosen = order[:_MAX_MISSES]
    return np.asarray([found[int(index)] for index in chosen], dtype=np.int64)
