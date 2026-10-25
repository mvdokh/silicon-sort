"""Hybrid ground truth: plant known spikes into real noise and score the detector."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from silicon_sort.detect.detect import SpikeIndex, detect_traces
from silicon_sort.io.geometry import Probe
from silicon_sort.settings import PipelineSettings


@dataclass
class Score:
    precision: float
    recall: float
    tp: int
    fp: int
    fn: int


@dataclass
class BenchmarkCurve:
    amplitudes: np.ndarray
    thresholds: np.ndarray
    precision: np.ndarray
    recall: np.ndarray


def match_events(
    found_times: np.ndarray,
    found_channels: np.ndarray,
    truth_times: np.ndarray,
    truth_channels: np.ndarray,
    tolerance: int,
) -> Score:
    """Greedy one-to-one match on the same channel within ``tolerance`` samples."""
    used = np.zeros(found_times.size, dtype=bool)
    tp = 0
    for truth_time, truth_channel in zip(truth_times, truth_channels, strict=True):
        if found_times.size == 0:
            continue
        candidates = np.flatnonzero(
            (found_channels == truth_channel) & (np.abs(found_times - truth_time) <= tolerance) & ~used
        )
        if candidates.size == 0:
            continue
        best = candidates[int(np.argmin(np.abs(found_times[candidates] - truth_time)))]
        used[best] = True
        tp += 1
    fp = int(np.sum(~used)) if found_times.size else 0
    fn = int(truth_times.size - tp)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return Score(precision=precision, recall=recall, tp=tp, fp=fp, fn=fn)


def plant_waveforms(
    noise: np.ndarray,
    times: np.ndarray,
    channels: np.ndarray,
    waveform: np.ndarray,
    scales: np.ndarray,
    neighbor_channels: np.ndarray | None = None,
    neighbor_gain: float = 0.55,
) -> np.ndarray:
    """Add scaled waveforms into a copy of ``noise``. The original array is not modified."""
    hybrid = np.array(noise, dtype=np.float32, copy=True)
    pre = int(np.argmin(waveform))
    for time, channel, scale in zip(times, channels, scales, strict=True):
        start = int(time) - pre
        stop = start + waveform.size
        if start < 0 or stop > hybrid.shape[0]:
            continue
        hybrid[start:stop, int(channel)] += np.float32(scale) * waveform
        if neighbor_channels is not None:
            for neighbor in neighbor_channels:
                if 0 <= int(neighbor) < hybrid.shape[1] and int(neighbor) != int(channel):
                    hybrid[start:stop, int(neighbor)] += np.float32(scale * neighbor_gain) * waveform
    return hybrid


def sweep_precision_recall(
    noise: np.ndarray,
    waveform: np.ndarray,
    truth_times: np.ndarray,
    truth_channel: int,
    neighbor_channels: np.ndarray,
    probe: Probe,
    settings: PipelineSettings,
    mask: np.ndarray,
    amplitudes: np.ndarray,
    thresholds: np.ndarray,
    noise_mad: np.ndarray | None = None,
) -> BenchmarkCurve:
    """Plant ``waveform`` at several amplitudes and sweep the detection threshold.

    ``noise_mad`` is estimated on the unplanted noise when omitted, so injected
    spikes do not inflate the threshold.
    """
    if noise_mad is None:
        from silicon_sort.preprocess.noise import estimate_mad

        noise_mad = estimate_mad(noise, mask=mask, n_chunks=4, chunk_samples=max(1, noise.shape[0]), seed=settings.rng_seed)
    precision = np.zeros((amplitudes.size, thresholds.size), dtype=np.float64)
    recall = np.zeros_like(precision)
    for amp_index, amplitude in enumerate(amplitudes):
        hybrid = plant_waveforms(
            noise,
            truth_times,
            np.full(truth_times.shape, truth_channel, dtype=int),
            waveform,
            np.full(truth_times.shape, float(amplitude)),
            neighbor_channels=neighbor_channels,
        )
        for thr_index, threshold in enumerate(thresholds):
            tuned = _with_threshold(settings, float(threshold))
            spikes, _, _ = detect_traces(
                hybrid,
                probe,
                tuned,
                mask,
                whitener=np.eye(hybrid.shape[1], dtype=np.float32),
                noise_mad=noise_mad,
            )
            clean = spikes.clean()
            on_channel = clean.channels == truth_channel
            score = match_events(
                clean.times[on_channel],
                clean.channels[on_channel],
                truth_times,
                np.full(truth_times.shape, truth_channel, dtype=int),
                tolerance=_ms_tol(tuned),
            )
            precision[amp_index, thr_index] = score.precision
            recall[amp_index, thr_index] = score.recall
    return BenchmarkCurve(
        amplitudes=np.asarray(amplitudes, dtype=float),
        thresholds=np.asarray(thresholds, dtype=float),
        precision=precision,
        recall=recall,
    )


def benchmark_excerpt(
    data: np.ndarray,
    spikes: SpikeIndex,
    probe: Probe,
    settings: PipelineSettings,
    mask: np.ndarray,
    noise_mad: np.ndarray,
    duration_s: float = 2.0,
    amplitudes: np.ndarray | None = None,
    thresholds: np.ndarray | None = None,
    seed: int = 0,
) -> BenchmarkCurve:
    """Build a short hybrid from a filtered excerpt and an observed waveform."""
    fs = float(settings.sampling_rate or 1.0)
    n_keep = min(data.shape[0], max(1, int(duration_s * fs)))
    noise = np.array(data[:n_keep], dtype=np.float32, copy=True)
    _blank_known_spikes(
        noise,
        spikes,
        _ms_samples(settings.waveform_pre_ms, fs),
        _ms_samples(settings.waveform_post_ms, fs),
    )
    waveform, channel, neighbors = _template_or_synthetic(noise, spikes, probe, settings, mask)
    rng = np.random.default_rng(seed)
    n_events = 8
    margin = waveform.size + 10
    high = max(margin + 1, n_keep - margin)
    times = np.sort(rng.integers(margin, high, size=min(n_events, max(1, (high - margin) // 20))))
    for index in range(1, times.size):
        gap = int(0.01 * fs)
        if times[index] - times[index - 1] < gap:
            times[index] = min(high - 1, times[index - 1] + gap)
    if amplitudes is None:
        amplitudes = np.array([40.0, 80.0, 140.0])
    if thresholds is None:
        thresholds = np.array([-4.0, -5.0, -6.0, -8.0])
    return sweep_precision_recall(
        noise,
        waveform,
        times,
        channel,
        neighbors,
        probe,
        settings,
        mask,
        amplitudes,
        thresholds,
        noise_mad=noise_mad,
    )


def _blank_known_spikes(noise: np.ndarray, spikes: SpikeIndex, pre: int, post: int) -> None:
    clean = spikes.clean()
    inside = clean.times < noise.shape[0]
    for time, channel in zip(clean.times[inside], clean.channels[inside], strict=True):
        start = max(0, int(time) - pre)
        stop = min(noise.shape[0], int(time) + post)
        noise[start:stop, int(channel)] = 0.0


def _template_or_synthetic(
    noise: np.ndarray,
    spikes: SpikeIndex,
    probe: Probe,
    settings: PipelineSettings,
    mask: np.ndarray,
) -> tuple[np.ndarray, int, np.ndarray]:
    from silicon_sort.detect.templates import median_template_bank

    _pre, bank = median_template_bank(noise, spikes, settings)
    active = np.flatnonzero(~mask & probe.present)
    channel = int(active[0]) if active.size else 0
    if bank:
        channel = max(bank, key=lambda key: float(np.min(bank[key])))
        waveform = np.asarray(bank[channel], dtype=np.float32)
        trough = float(np.min(waveform))
        if trough < 0:
            waveform = waveform / abs(trough)
    else:
        fs = float(settings.sampling_rate or 30000.0)
        waveform = synthetic_biphasic(int(0.0004 * fs), int(0.0008 * fs))
    neighbors = probe.neighbors(settings.neighbor_radius_um, mask)[channel]
    return waveform.astype(np.float32), int(channel), neighbors


def synthetic_biphasic(pre: int, post: int, width: float = 3.0) -> np.ndarray:
    """Unit-trough biphasic spike, trough at sample ``pre``."""
    length = max(pre + post, 8)
    times = np.arange(length) - pre
    trough = -np.exp(-0.5 * (times / width) ** 2)
    rebound = 0.45 * np.exp(-0.5 * ((times - width * 2.5) / (width * 1.3)) ** 2)
    wave = trough + rebound
    return (wave / max(abs(float(np.min(wave))), 1e-6)).astype(np.float32)


def _with_threshold(settings: PipelineSettings, threshold: float) -> PipelineSettings:
    from silicon_sort.settings import settings_from_dict, settings_to_dict

    payload = settings_to_dict(settings)
    payload["threshold_mad"] = float(threshold)
    payload["template_match"] = False
    payload["whiten"] = False
    return settings_from_dict(payload)


def _ms_tol(settings: PipelineSettings) -> int:
    fs = float(settings.sampling_rate or 1.0)
    return max(3, int(round(0.0005 * fs)))


def _ms_samples(milliseconds: float, fs: float) -> int:
    return max(1, int(round(milliseconds * 1e-3 * fs)))
