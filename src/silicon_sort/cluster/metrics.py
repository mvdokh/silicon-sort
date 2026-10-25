"""Per-cluster firing rate, ISI violations, SNR, and waveform width."""

from __future__ import annotations

import numpy as np

from silicon_sort.cluster.features import samples_for


def compute_metrics(
    times: np.ndarray,
    channels: np.ndarray,
    amplitudes: np.ndarray,
    labels: np.ndarray,
    templates: np.ndarray,
    noise_mad: np.ndarray,
    positions: np.ndarray,
    sampling_rate: float,
    n_samples: int,
    refractory_ms: float,
) -> list[dict]:
    duration = n_samples / float(sampling_rate) if n_samples else 0.0
    refractory = max(1, int(round(refractory_ms * 1e-3 * sampling_rate)))
    rows: list[dict] = []
    valid = labels >= 0
    if not np.any(valid):
        return rows
    for cluster_id in range(int(templates.shape[0])):
        index = np.flatnonzero(labels == cluster_id)
        if index.size == 0:
            continue
        spike_times = np.sort(times[index])
        intervals = np.diff(spike_times)
        violations = float(np.mean(intervals < refractory)) if intervals.size else 0.0
        peak_channel = int(np.bincount(channels[index].astype(int)).argmax())
        mad = float(noise_mad[peak_channel]) if np.isfinite(noise_mad[peak_channel]) else np.nan
        amplitude = float(np.median(np.abs(amplitudes[index])))
        snr = amplitude / mad if mad and np.isfinite(mad) and mad > 0 else 0.0
        waveform = templates[cluster_id, :, peak_channel]
        width = _peak_to_trough_ms(waveform, sampling_rate)
        rate = index.size / duration if duration else 0.0
        if rate < 0.05 or snr < 1.5:
            quality = "noise"
        elif violations > 0.1 or snr < 3.0:
            quality = "mua"
        else:
            quality = "good"
        depth = float(positions[peak_channel, 1]) if positions.size else 0.0
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "n_spikes": int(index.size),
                "firing_rate": float(rate),
                "isi_violation_rate": float(violations),
                "snr": float(snr),
                "peak_channel": peak_channel,
                "amplitude": amplitude,
                "waveform_width_ms": float(width),
                "depth_um": depth,
                "quality": quality,
                "contam_pct": float(violations * 100.0),
            }
        )
    return rows


def _peak_to_trough_ms(waveform: np.ndarray, sampling_rate: float) -> float:
    if waveform.size == 0:
        return 0.0
    trough = int(np.argmin(waveform))
    peak = int(np.argmax(waveform))
    return abs(peak - trough) / float(sampling_rate) * 1000.0


def template_width(sampling_rate: float, pre_ms: float, post_ms: float) -> int:
    return samples_for(pre_ms, sampling_rate) + samples_for(post_ms, sampling_rate)
