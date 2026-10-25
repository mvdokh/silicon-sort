"""Spatially and temporally aware threshold detection."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.ndimage import maximum_filter1d

from silicon_sort.cancel import CancelToken
from silicon_sort.errors import Cancelled
from silicon_sort.io.geometry import Probe
from silicon_sort.preprocess.noise import estimate_mad
from silicon_sort.preprocess.whiten import apply_whitener, estimate_whitener
from silicon_sort.settings import PipelineSettings

Progress = Callable[[float, str], None]


@dataclass
class SpikeIndex:
    times: np.ndarray
    channels: np.ndarray
    amplitudes: np.ndarray
    artifact: np.ndarray

    def clean(self) -> SpikeIndex:
        keep = ~self.artifact.astype(bool)
        return SpikeIndex(self.times[keep], self.channels[keep], self.amplitudes[keep], self.artifact[keep])

    @property
    def n_spikes(self) -> int:
        return int(self.times.size)


def empty_index() -> SpikeIndex:
    return SpikeIndex(
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.int32),
        np.zeros(0, dtype=np.float32),
        np.zeros(0, dtype=bool),
    )


def detect_traces(
    data: np.ndarray,
    probe: Probe,
    settings: PipelineSettings,
    mask: np.ndarray,
    whitener: np.ndarray | None = None,
    noise_mad: np.ndarray | None = None,
    progress: Progress | None = None,
    cancel: CancelToken | None = None,
) -> tuple[SpikeIndex, np.ndarray, np.ndarray]:
    """Detect spikes in a filtered ``(n_samples, n_channels)`` array or memmap.

    ``data`` is in microvolts after bandpass and referencing. Whitening, when
    enabled, is applied per chunk and is not written back.
    """
    fs = float(settings.sampling_rate or 1.0)
    mask = np.asarray(mask, dtype=bool) | ~probe.present
    chunk = max(1, int(round(settings.chunk_seconds * fs)))
    if noise_mad is None:
        noise_mad = estimate_mad(
            data,
            mask=mask,
            n_chunks=settings.mad_chunks,
            chunk_samples=chunk,
            seed=settings.rng_seed,
        )
    if settings.whiten:
        if whitener is None:
            whitener = estimate_whitener(
                data,
                mask,
                n_chunks=settings.whiten_chunks,
                chunk_samples=chunk,
                seed=settings.rng_seed,
            )
        det_mad = estimate_mad(
            data,
            mask=mask,
            n_chunks=settings.mad_chunks,
            chunk_samples=chunk,
            seed=settings.rng_seed + 1,
            transform=lambda rows: apply_whitener(rows, whitener, mask),
        )
    else:
        whitener = np.eye(data.shape[1], dtype=np.float32)
        det_mad = np.array(noise_mad, copy=True)

    overlap = _detection_overlap(settings, fs)
    n_samples = int(data.shape[0])
    neighbors = probe.neighbors(settings.neighbor_radius_um, mask)
    polarity = -1.0 if settings.threshold_mad < 0 else 1.0
    level = np.abs(settings.threshold_mad) * det_mad
    neighbor_level = settings.neighbor_support_fraction * level
    align = _ms(settings.align_ms, fs)
    neighbor_win = _ms(settings.neighbor_window_ms, fs)
    shape_win = _ms(settings.biphasic_window_ms, fs)
    coincidence = _ms(settings.artifact_coincidence_ms, fs)
    blank = _ms(settings.artifact_blank_ms, fs)

    times_all: list[np.ndarray] = []
    channels_all: list[np.ndarray] = []
    amps_all: list[np.ndarray] = []
    art_all: list[np.ndarray] = []
    starts = list(range(0, n_samples, chunk))
    for index, start in enumerate(starts):
        if cancel is not None and cancel.cancelled():
            raise Cancelled("Detection cancelled.")
        stop = min(n_samples, start + chunk)
        window = _read_window(data, start, stop, overlap)
        det = apply_whitener(window, whitener, mask) if settings.whiten else window
        local = _detect_window(
            amplitude=window,
            detection=det,
            start=start,
            stop=stop,
            overlap=overlap,
            n_samples=n_samples,
            mask=mask,
            neighbors=neighbors,
            polarity=polarity,
            level=level,
            neighbor_level=neighbor_level,
            align=align,
            neighbor_win=neighbor_win,
            shape_win=shape_win,
            min_neighbors=int(settings.min_neighbors),
            glitch_fraction=float(settings.glitch_energy_fraction),
            biphasic_fraction=float(settings.biphasic_fraction),
            artifact_fraction=float(settings.artifact_fraction),
            artifact_level=float(settings.artifact_mad) * noise_mad,
            coincidence=coincidence,
            blank=blank,
        )
        times_all.append(local[0])
        channels_all.append(local[1])
        amps_all.append(local[2])
        art_all.append(local[3])
        if progress is not None:
            progress((index + 1) / max(len(starts), 1), f"Detecting {index + 1}/{len(starts)}")

    if not times_all or all(part.size == 0 for part in times_all):
        spikes = empty_index()
    else:
        spikes = SpikeIndex(
            np.concatenate(times_all),
            np.concatenate(channels_all),
            np.concatenate(amps_all),
            np.concatenate(art_all),
        )
        spikes = _finalize_spikes(
            spikes,
            n_channels=data.shape[1],
            dead_samples=_ms(settings.dead_time_ms, fs),
        )
    if settings.template_match:
        from silicon_sort.detect.templates import match_templates

        spikes = match_templates(
            data,
            spikes,
            probe,
            settings,
            mask,
            noise_mad,
            progress=progress,
            cancel=cancel,
        )
    return spikes, np.asarray(noise_mad, dtype=np.float64), np.asarray(whitener, dtype=np.float32)


def _detect_window(
    amplitude: np.ndarray,
    detection: np.ndarray,
    start: int,
    stop: int,
    overlap: int,
    n_samples: int,
    mask: np.ndarray,
    neighbors: list[np.ndarray],
    polarity: float,
    level: np.ndarray,
    neighbor_level: np.ndarray,
    align: int,
    neighbor_win: int,
    shape_win: int,
    min_neighbors: int,
    glitch_fraction: float,
    biphasic_fraction: float,
    artifact_fraction: float,
    artifact_level: np.ndarray,
    coincidence: int,
    blank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    artifacts = _artifact_samples(
        amplitude,
        mask,
        artifact_level,
        artifact_fraction,
        coincidence,
        blank,
    )
    times: list[int] = []
    channels: list[int] = []
    amps: list[float] = []
    flags: list[bool] = []
    n_channels = amplitude.shape[1]
    for channel in range(n_channels):
        if mask[channel] or not np.isfinite(level[channel]):
            continue
        trace = detection[:, channel]
        if polarity < 0:
            peaks = _peaks_negative(trace, -float(level[channel]))
        else:
            peaks = _peaks_positive(trace, float(level[channel]))
        for peak in peaks:
            aligned = _align(trace, int(peak), align, polarity)
            global_index = start - overlap + aligned
            if global_index < start or global_index >= stop or global_index < 0 or global_index >= n_samples:
                continue
            is_artifact = bool(artifacts[aligned])
            if not is_artifact:
                if not _has_support(
                    detection,
                    aligned,
                    channel,
                    neighbors[channel],
                    neighbor_level,
                    neighbor_win,
                    polarity,
                    min_neighbors,
                ):
                    continue
                if not _passes_shape(
                    amplitude[:, channel],
                    aligned,
                    polarity,
                    glitch_fraction,
                    biphasic_fraction,
                    shape_win,
                ):
                    continue
            times.append(global_index)
            channels.append(channel)
            amps.append(float(amplitude[aligned, channel]))
            flags.append(is_artifact)
    return (
        np.asarray(times, dtype=np.int64),
        np.asarray(channels, dtype=np.int32),
        np.asarray(amps, dtype=np.float32),
        np.asarray(flags, dtype=bool),
    )


def _finalize_spikes(spikes: SpikeIndex, n_channels: int, dead_samples: int) -> SpikeIndex:
    if spikes.n_spikes == 0:
        return spikes
    times, channels, amplitudes, artifact = _dedup(
        spikes.times,
        spikes.channels,
        spikes.amplitudes,
        spikes.artifact.astype(bool),
        n_channels,
    )
    order = np.argsort(times, kind="mergesort")
    times = times[order]
    channels = channels[order]
    amplitudes = amplitudes[order]
    artifact = artifact[order]
    if dead_samples > 0 and times.size:
        keep = _dead_time_keep(times, channels.astype(np.int64), artifact, int(n_channels), int(dead_samples))
        times, channels, amplitudes, artifact = times[keep], channels[keep], amplitudes[keep], artifact[keep]
    return SpikeIndex(times, channels, amplitudes, artifact)


def _dedup(
    times: np.ndarray,
    channels: np.ndarray,
    amplitudes: np.ndarray,
    artifact: np.ndarray,
    n_channels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    key = times.astype(np.int64) * np.int64(n_channels) + channels.astype(np.int64)
    order = np.argsort(key, kind="mergesort")
    key = key[order]
    times = times[order]
    channels = channels[order]
    amplitudes = amplitudes[order]
    artifact = artifact[order]
    keep: list[int] = []
    cursor = 0
    n = key.size
    while cursor < n:
        end = cursor + 1
        while end < n and key[end] == key[cursor]:
            end += 1
        best = cursor + int(np.argmax(np.abs(amplitudes[cursor:end])))
        keep.append(best)
        cursor = end
    chosen = np.asarray(keep, dtype=int)
    return times[chosen], channels[chosen], amplitudes[chosen], artifact[chosen]


def _artifact_samples(
    amplitude: np.ndarray,
    mask: np.ndarray,
    level: np.ndarray,
    fraction: float,
    coincidence: int,
    blank: int,
) -> np.ndarray:
    active = np.flatnonzero(~mask & np.isfinite(level))
    if active.size == 0 or fraction <= 0:
        return np.zeros(amplitude.shape[0], dtype=bool)
    crossed = np.abs(amplitude[:, active]) > level[active]
    window = max(int(coincidence), 1)
    if window > 1:
        if window % 2 == 0:
            window += 1
        crossed = maximum_filter1d(crossed.astype(np.uint8), size=window, axis=0, mode="nearest") > 0
    count = np.sum(crossed, axis=1)
    centers = count >= fraction * active.size
    blank_win = max(int(blank), 1)
    if blank_win % 2 == 0:
        blank_win += 1
    if blank_win > 1:
        centers = maximum_filter1d(centers.astype(np.uint8), size=blank_win, mode="nearest") > 0
    return np.asarray(centers, dtype=bool)


def _has_support(
    detection: np.ndarray,
    peak: int,
    channel: int,
    neighbors: np.ndarray,
    neighbor_level: np.ndarray,
    window: int,
    polarity: float,
    min_neighbors: int,
) -> bool:
    del channel
    if neighbors.size == 0 or min_neighbors <= 0:
        return True
    start = max(0, peak - window)
    stop = min(detection.shape[0], peak + window + 1)
    snippet = detection[start:stop][:, neighbors]
    if polarity < 0:
        crossed = np.any(snippet < -neighbor_level[neighbors], axis=0)
    else:
        crossed = np.any(snippet > neighbor_level[neighbors], axis=0)
    needed = min(int(min_neighbors), int(neighbors.size))
    return int(np.sum(crossed)) >= needed


def _passes_shape(
    trace: np.ndarray,
    peak: int,
    polarity: float,
    glitch_fraction: float,
    biphasic_fraction: float,
    window: int,
) -> bool:
    start = max(0, peak - window)
    stop = min(trace.shape[0], peak + window + 1)
    waveform = np.asarray(trace[start:stop], dtype=np.float64)
    center = peak - start
    energy = float(np.dot(waveform, waveform))
    if energy <= 1e-18:
        return False
    if (float(waveform[center]) ** 2) / energy > glitch_fraction:
        return False
    others = np.delete(waveform, center)
    if others.size == 0:
        return False
    peak_value = float(waveform[center])
    if polarity < 0:
        return float(np.max(others)) >= biphasic_fraction * abs(peak_value)
    return float(np.min(others)) <= -biphasic_fraction * abs(peak_value)


def _align(trace: np.ndarray, index: int, window: int, polarity: float) -> int:
    start = max(1, index - window)
    stop = min(trace.shape[0] - 1, index + window + 1)
    if stop <= start:
        return index
    segment = trace[start:stop]
    offset = int(np.argmin(segment) if polarity < 0 else np.argmax(segment))
    return start + offset


def _read_window(data: np.ndarray, start: int, stop: int, overlap: int) -> np.ndarray:
    n_samples = data.shape[0]
    src_start = start - overlap
    src_stop = stop + overlap
    pad_left = max(0, -src_start)
    pad_right = max(0, src_stop - n_samples)
    src_start = max(0, src_start)
    src_stop = min(n_samples, src_stop)
    window = np.array(data[src_start:src_stop], dtype=np.float32, copy=True)
    if pad_left or pad_right:
        window = np.pad(window, ((pad_left, pad_right), (0, 0)), mode="edge")
    return window


def _detection_overlap(settings: PipelineSettings, fs: float) -> int:
    span = max(settings.align_ms, settings.biphasic_window_ms, settings.neighbor_window_ms, settings.artifact_blank_ms)
    return max(8, _ms(span, fs) + 2)


def _ms(milliseconds: float, fs: float) -> int:
    return max(1, int(round(float(milliseconds) * 1e-3 * float(fs))))


@njit(cache=True)
def _peaks_negative(trace: np.ndarray, threshold: float) -> np.ndarray:
    n = trace.shape[0]
    found = np.empty(n, dtype=np.int64)
    count = 0
    for index in range(1, n - 1):
        value = trace[index]
        if value < threshold and value <= trace[index - 1] and value < trace[index + 1]:
            found[count] = index
            count += 1
    return found[:count]


@njit(cache=True)
def _peaks_positive(trace: np.ndarray, threshold: float) -> np.ndarray:
    n = trace.shape[0]
    found = np.empty(n, dtype=np.int64)
    count = 0
    for index in range(1, n - 1):
        value = trace[index]
        if value > threshold and value >= trace[index - 1] and value > trace[index + 1]:
            found[count] = index
            count += 1
    return found[:count]


@njit(cache=True)
def _dead_time_keep(
    times: np.ndarray,
    channels: np.ndarray,
    artifact: np.ndarray,
    n_channels: int,
    dead_samples: int,
) -> np.ndarray:
    keep = np.ones(times.shape[0], dtype=np.bool_)
    last = np.full(n_channels, np.int64(-1000000000000000))
    for index in range(times.shape[0]):
        if artifact[index]:
            continue
        channel = channels[index]
        if times[index] - last[channel] < dead_samples:
            keep[index] = False
        else:
            last[channel] = times[index]
    return keep
