"""Optional second pass: matched filters built from first-pass waveforms."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.signal import fftconvolve

from silicon_sort.cancel import CancelToken
from silicon_sort.detect.detect import SpikeIndex, _align, _dead_time_keep, _ms, empty_index
from silicon_sort.errors import Cancelled
from silicon_sort.io.geometry import Probe
from silicon_sort.settings import PipelineSettings

Progress = Callable[[float, str], None]


def match_templates(
    data: np.ndarray,
    spikes: SpikeIndex,
    probe: Probe,
    settings: PipelineSettings,
    mask: np.ndarray,
    noise_mad: np.ndarray,
    progress: Progress | None = None,
    cancel: CancelToken | None = None,
) -> SpikeIndex:
    """Recover smaller spikes that match the median waveform of a peak channel.

    Already-detected events are left in place. New events must clear a lower
    threshold and resemble that channel's template.
    """
    del probe
    fs = float(settings.sampling_rate or 1.0)
    pre = _ms(settings.waveform_pre_ms, fs)
    post = _ms(settings.waveform_post_ms, fs)
    width = pre + post
    polarity = -1.0 if settings.threshold_mad < 0 else 1.0
    templates: dict[int, np.ndarray] = {}
    clean = spikes.clean()
    for channel in np.unique(clean.channels) if clean.n_spikes else []:
        channel = int(channel)
        if mask[channel]:
            continue
        chosen = clean.times[clean.channels == channel]
        chosen = chosen[(chosen >= pre) & (chosen + post < data.shape[0])]
        if chosen.size < 5:
            continue
        if chosen.size > 300:
            rng = np.random.default_rng(settings.rng_seed + channel)
            chosen = np.sort(rng.choice(chosen, size=300, replace=False))
        waves = np.stack([np.asarray(data[int(t) - pre : int(t) + post, channel], dtype=np.float32) for t in chosen])
        templates[channel] = np.median(waves, axis=0)

    if not templates:
        return spikes

    fs_chunk = max(width * 4, int(round(settings.chunk_seconds * fs)))
    new_times: list[int] = []
    new_channels: list[int] = []
    new_amps: list[float] = []
    occupied = _occupied(spikes, data.shape[1])
    dead = _ms(settings.dead_time_ms, fs)
    channels = list(templates)
    for index, channel in enumerate(channels):
        if cancel is not None and cancel.cancelled():
            raise Cancelled("Template matching cancelled.")
        found = _scan_channel(
            data[:, channel],
            templates[channel],
            pre,
            polarity,
            float(noise_mad[channel]),
            float(settings.threshold_mad),
            float(settings.template_threshold_fraction),
            float(settings.template_min_cosine),
            fs_chunk,
            occupied.get(channel, np.zeros(0, dtype=np.int64)),
            dead,
        )
        new_times.extend(found[0])
        new_channels.extend([channel] * len(found[0]))
        new_amps.extend(found[1])
        if progress is not None:
            progress((index + 1) / len(channels), f"Template match {index + 1}/{len(channels)}")

    if not new_times:
        return spikes
    extra = SpikeIndex(
        np.asarray(new_times, dtype=np.int64),
        np.asarray(new_channels, dtype=np.int32),
        np.asarray(new_amps, dtype=np.float32),
        np.zeros(len(new_times), dtype=bool),
    )
    merged = SpikeIndex(
        np.concatenate([spikes.times, extra.times]),
        np.concatenate([spikes.channels, extra.channels]),
        np.concatenate([spikes.amplitudes, extra.amplitudes]),
        np.concatenate([spikes.artifact, extra.artifact]),
    )
    order = np.argsort(merged.times, kind="mergesort")
    merged = SpikeIndex(
        merged.times[order],
        merged.channels[order],
        merged.amplitudes[order],
        merged.artifact[order],
    )
    keep = _dead_time_keep(
        merged.times,
        merged.channels.astype(np.int64),
        merged.artifact,
        int(data.shape[1]),
        dead,
    )
    return SpikeIndex(merged.times[keep], merged.channels[keep], merged.amplitudes[keep], merged.artifact[keep])


def _scan_channel(
    trace: np.ndarray,
    template: np.ndarray,
    pre: int,
    polarity: float,
    mad: float,
    threshold_mad: float,
    fraction: float,
    min_cosine: float,
    chunk: int,
    occupied: np.ndarray,
    dead: int,
) -> tuple[list[int], list[float]]:
    norm = float(np.linalg.norm(template))
    trough = float(np.min(template) if polarity < 0 else np.max(template))
    peak_amp = abs(trough)
    if norm < 1e-9 or peak_amp < 1e-9 or not np.isfinite(mad):
        return [], []
    unit = template / norm
    raw_threshold = fraction * abs(threshold_mad) * mad
    # Correlation of a same-shape spike whose trough equals the lower threshold.
    corr_threshold = raw_threshold / peak_amp * norm
    times: list[int] = []
    amplitudes: list[float] = []
    n_samples = trace.shape[0]
    width = template.shape[0]
    for start in range(0, n_samples, chunk):
        stop = min(n_samples, start + chunk)
        pad_left = width
        pad_right = width
        src_start = max(0, start - pad_left)
        src_stop = min(n_samples, stop + pad_right)
        snippet = np.asarray(trace[src_start:src_stop], dtype=np.float32)
        corr = fftconvolve(snippet, unit[::-1], mode="same")
        local_offset = start - src_start
        local = corr[local_offset : local_offset + (stop - start)]
        peaks = _correlation_peaks(local, corr_threshold, polarity)
        # ``same`` correlation peaks at the kernel center, not at the trough.
        lag = width // 2 - pre
        for peak in peaks:
            global_index = start + int(peak) - lag
            aligned = _align(trace, global_index, max(pre // 2, 1), polarity)
            if aligned < pre or aligned + (width - pre) > n_samples:
                continue
            if _too_close(occupied, aligned, dead) or _too_close(np.asarray(times, dtype=np.int64), aligned, dead):
                continue
            waveform = np.asarray(trace[aligned - pre : aligned - pre + width], dtype=np.float64)
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


def _correlation_peaks(corr: np.ndarray, threshold: float, polarity: float) -> np.ndarray:
    if corr.size < 3:
        return np.zeros(0, dtype=int)
    if polarity < 0:
        # A negative spike still has a positive correlation against its own template.
        hits = np.flatnonzero((corr[1:-1] >= threshold) & (corr[1:-1] >= corr[:-2]) & (corr[1:-1] > corr[2:]))
    else:
        hits = np.flatnonzero((corr[1:-1] >= threshold) & (corr[1:-1] >= corr[:-2]) & (corr[1:-1] > corr[2:]))
    return hits + 1


def _occupied(spikes: SpikeIndex, n_channels: int) -> dict[int, np.ndarray]:
    del n_channels
    clean = spikes.clean()
    out: dict[int, np.ndarray] = {}
    if clean.n_spikes == 0:
        return out
    for channel in np.unique(clean.channels):
        out[int(channel)] = np.sort(clean.times[clean.channels == channel])
    return out


def _too_close(times: np.ndarray, sample: int, dead: int) -> bool:
    if times.size == 0:
        return False
    pos = int(np.searchsorted(times, sample))
    for index in (pos - 1, pos):
        if 0 <= index < times.size and abs(int(times[index]) - sample) < dead:
            return True
    return False


def median_template_bank(
    data: np.ndarray,
    spikes: SpikeIndex,
    settings: PipelineSettings,
) -> tuple[int, dict[int, np.ndarray]]:
    """Expose per-channel median waveforms for the benchmark and previews."""
    fs = float(settings.sampling_rate or 1.0)
    pre = _ms(settings.waveform_pre_ms, fs)
    post = _ms(settings.waveform_post_ms, fs)
    clean = spikes.clean()
    bank: dict[int, np.ndarray] = {}
    if clean.n_spikes == 0:
        return pre, bank
    for channel in np.unique(clean.channels):
        channel = int(channel)
        chosen = clean.times[clean.channels == channel]
        chosen = chosen[(chosen >= pre) & (chosen + post <= data.shape[0])]
        if chosen.size == 0:
            continue
        if chosen.size > 200:
            chosen = chosen[:200]
        waves = [np.asarray(data[int(t) - pre : int(t) + post, channel], dtype=np.float32) for t in chosen]
        bank[channel] = np.median(np.stack(waves), axis=0)
    return pre, bank
