"""Detection, chunk boundaries, artifacts, template matching, and the hybrid benchmark."""

from pathlib import Path

import numpy as np

from silicon_sort.detect.benchmark import (
    match_events,
    plant_waveforms,
    sweep_precision_recall,
    synthetic_biphasic,
)
from silicon_sort.detect.detect import detect_traces
from silicon_sort.detect.templates import match_templates
from silicon_sort.io.geometry import probe_from_cfg
from silicon_sort.settings import PipelineSettings


def _probe(tmp_path: Path, n_channels: int, pitch: float = 40.0):
    lines = ["line"]
    for index in range(n_channels):
        lines.append(f"{index + 1} {index + 1} 0 {index * pitch:.2f}")
    path = tmp_path / "probe.cfg"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return probe_from_cfg(path, n_channels)


def _settings(fs: float = 30000.0, **overrides) -> PipelineSettings:
    settings = PipelineSettings(
        sampling_rate=fs,
        n_channels=2,
        dtype="float32",
        whiten=False,
        chunk_seconds=1.0,
        threshold_mad=-5.0,
        neighbor_radius_um=60.0,
        artifact_fraction=1.1,
        template_match=False,
        n_workers=1,
        waveform_pre_ms=0.4,
        waveform_post_ms=0.8,
    )
    for key, value in overrides.items():
        setattr(settings, key, value)
    return settings


def _plant(data, center, wave, channel, amplitude, neighbor=None, neighbor_gain=0.6):
    pre = int(np.argmin(wave))
    start = center - pre
    # ``wave`` already troughs at -1. ``amplitude`` is the signed trough in µV.
    scale = abs(float(amplitude))
    data[start : start + wave.size, channel] += scale * wave
    if neighbor is not None:
        data[start : start + wave.size, neighbor] += scale * neighbor_gain * wave


def test_spike_on_chunk_boundary_is_detected_once(tmp_path: Path):
    fs = 30000.0
    n_samples = int(2 * fs)
    rng = np.random.default_rng(0)
    data = rng.normal(0, 6.0, size=(n_samples, 2)).astype(np.float32)
    wave = synthetic_biphasic(12, 24, width=3.0)
    boundary = int(fs)
    _plant(data, boundary, wave, 0, -250.0, neighbor=1)
    probe = _probe(tmp_path, 2)
    spikes, _, _ = detect_traces(data, probe, _settings(fs), np.zeros(2, dtype=bool))
    clean = spikes.clean()
    near = np.abs(clean.times.astype(int) - boundary) <= 15
    on_peak = clean.channels == 0
    assert int(np.sum(near & on_peak)) == 1


def test_single_sample_glitch_is_rejected(tmp_path: Path):
    rng = np.random.default_rng(1)
    data = rng.normal(0, 6.0, size=(4000, 2)).astype(np.float32)
    data[2000, 0] = -4000
    probe = _probe(tmp_path, 2)
    spikes, _, _ = detect_traces(
        data,
        probe,
        _settings(chunk_seconds=0.2),
        np.zeros(2, dtype=bool),
    )
    assert spikes.clean().n_spikes == 0


def test_common_artifact_is_excluded_from_clean_spikes(tmp_path: Path):
    fs = 10000.0
    rng = np.random.default_rng(2)
    n_channels = 8
    data = rng.normal(0, 5.0, size=(8000, n_channels)).astype(np.float32)
    wave = synthetic_biphasic(8, 16, width=2.5)
    _plant(data, 1500, wave, 0, -180.0, neighbor=1, neighbor_gain=0.7)
    for channel in range(n_channels):
        _plant(data, 5000, wave, channel, -400.0)
    probe = _probe(tmp_path, n_channels, pitch=20.0)
    settings = _settings(fs, artifact_fraction=0.4, artifact_blank_ms=1.0, chunk_seconds=0.5)
    spikes, _, _ = detect_traces(data, probe, settings, np.zeros(n_channels, dtype=bool))
    clean = spikes.clean()
    assert clean.n_spikes >= 1
    assert not np.any(np.abs(clean.times.astype(int) - 5000) <= 30)
    assert np.any(spikes.artifact) or spikes.n_spikes > clean.n_spikes


def test_template_match_recovers_a_small_copy(tmp_path: Path):
    fs = 10000.0
    rng = np.random.default_rng(3)
    data = rng.normal(0, 4.0, size=(6000, 2)).astype(np.float32)
    wave = synthetic_biphasic(8, 16, width=2.5)
    for center in (800, 1600, 2400, 3200, 4000):
        _plant(data, center, wave, 0, -160.0, neighbor=1)
    _plant(data, 5000, wave, 0, -15.0, neighbor=1, neighbor_gain=0.7)
    probe = _probe(tmp_path, 2)
    settings = _settings(fs, chunk_seconds=0.5, threshold_mad=-5.0, template_match=False)
    first, mad, _ = detect_traces(data, probe, settings, np.zeros(2, dtype=bool))
    assert not np.any(np.abs(first.clean().times.astype(int) - 5000) <= 20)
    settings.template_match = True
    settings.template_threshold_fraction = 0.35
    recovered = match_templates(data, first, probe, settings, np.zeros(2, dtype=bool), mad)
    assert np.any(np.abs(recovered.clean().times.astype(int) - 5000) <= 20)


def test_benchmark_recall_falls_toward_threshold(tmp_path: Path):
    fs = 10000.0
    rng = np.random.default_rng(4)
    noise = rng.normal(0, 8.0, size=(5000, 2)).astype(np.float32)
    wave = synthetic_biphasic(8, 16, width=2.5)
    truth = np.array([600, 1400, 2200, 3000, 4000])
    probe = _probe(tmp_path, 2)
    settings = _settings(fs, chunk_seconds=0.5)
    mad = np.array([8.0, 8.0])
    curve = sweep_precision_recall(
        noise,
        wave,
        truth,
        truth_channel=0,
        neighbor_channels=np.array([1]),
        probe=probe,
        settings=settings,
        mask=np.zeros(2, dtype=bool),
        amplitudes=np.array([40.0, 160.0]),
        thresholds=np.array([-4.0, -8.0]),
        noise_mad=mad,
    )
    large = curve.recall[1]
    small = curve.recall[0]
    assert large[0] > 0.6
    assert large[0] >= large[1]
    assert small[1] <= large[0]
    hybrid = plant_waveforms(noise, truth, np.zeros(truth.size, dtype=int), wave, np.full(truth.size, 160.0))
    assert not np.shares_memory(hybrid, noise)


def test_review_marks_drop_false_positives_and_keep_missed_spikes():
    from silicon_sort.detect.detect import SpikeIndex
    from silicon_sort.detect.review import refine_with_review

    fs = 10000.0
    n_samples = 2500
    data = np.zeros((n_samples, 1), dtype=np.float32)
    wave = synthetic_biphasic(8, 16, width=2.0)
    _plant(data, 400, wave, 0, -120.0)
    _plant(data, 900, wave, 0, -40.0)
    bump = np.hanning(24).astype(np.float32)
    data[1400 : 1400 + bump.size, 0] += bump * 80.0
    data[1800 : 1800 + bump.size, 0] += bump * 70.0
    spikes = SpikeIndex(
        np.array([400, 1400, 1800], dtype=np.int64),
        np.zeros(3, dtype=np.int32),
        np.array([-120.0, 80.0, 70.0], dtype=np.float32),
        np.zeros(3, dtype=bool),
    )
    settings = _settings(fs, waveform_pre_ms=0.8, waveform_post_ms=1.6, dead_time_ms=0.5, chunk_seconds=0.2)
    refined = refine_with_review(
        data,
        spikes,
        review_times=np.array([400, 1400], dtype=np.int64),
        review_channels=np.zeros(2, dtype=np.int32),
        review_labels=np.array([1, -1], dtype=np.int8),
        settings=settings,
        noise_mad=np.array([8.0]),
    )
    kept = set(refined.times.tolist())
    assert 400 in kept
    assert 900 in kept
    assert 1400 not in kept
    assert 1800 not in kept


def test_match_events_counts_false_positives():
    score = match_events(
        np.array([10, 50, 90]),
        np.array([0, 0, 1]),
        np.array([11, 80]),
        np.array([0, 0]),
        tolerance=5,
    )
    assert score.tp == 1
    assert score.fn == 1
    assert score.fp == 2
