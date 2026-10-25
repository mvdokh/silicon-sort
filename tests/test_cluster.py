"""Waveform features, forced cluster counts, merge, and split."""

from pathlib import Path

import numpy as np

from silicon_sort.cluster.features import extract_windows
from silicon_sort.cluster.gmm import label_features
from silicon_sort.cluster.merge import _shared_structure, merge_and_split
from silicon_sort.detect.benchmark import synthetic_biphasic
from silicon_sort.detect.detect import SpikeIndex
from silicon_sort.io.geometry import probe_from_cfg
from silicon_sort.settings import PipelineSettings


def test_extract_windows_aligns_trough():
    data = np.zeros((100, 2), dtype=np.float32)
    data[40, 0] = -5
    data[43, 0] = 2
    windows = extract_windows(data, np.array([40]), pre=5, post=8)
    assert windows.shape == (1, 13, 2)
    assert windows[0, 5, 0] == -5


def test_forced_k_overrides_bic():
    rng = np.random.default_rng(0)
    blob = rng.normal(size=(80, 3))
    labels, k_used, k_auto = label_features(blob, forced_k=1, k_max=4, seed=0)
    assert k_used == 1
    assert k_auto == 1
    assert labels.shape == (80,)
    separated = np.vstack([rng.normal(loc=-4, size=(60, 3)), rng.normal(loc=4, size=(60, 3))])
    labels, k_used, k_auto = label_features(separated, forced_k=2, k_max=4, seed=0)
    assert k_used == 2
    assert len(np.unique(labels)) == 2


def test_bic_keeps_one_cluster_for_a_single_blob():
    rng = np.random.default_rng(1)
    blob = rng.normal(size=(70, 4))
    _labels, k_used, k_auto = label_features(blob, forced_k=None, k_max=4, seed=1)
    assert k_used == 1
    assert k_auto == 1


def test_neighboring_duplicates_merge(tmp_path: Path):
    fs = 10000.0
    n_channels = 2
    data = np.zeros((5000, n_channels), dtype=np.float32)
    wave = synthetic_biphasic(8, 16)
    times = np.arange(500, 4500, 400)
    pre = int(np.argmin(wave))
    for time in times:
        start = int(time) - pre
        data[start : start + wave.size, 0] += -100 * wave
        data[start : start + wave.size, 1] += -80 * wave
    # Same events, split across the two sites with a one-sample offset on channel 1.
    times_a = times.astype(np.int64)
    times_b = times.astype(np.int64) + 1
    all_times = np.concatenate([times_a, times_b])
    channels = np.concatenate([np.zeros(times.size, dtype=np.int32), np.ones(times.size, dtype=np.int32)])
    amplitudes = np.concatenate([np.full(times.size, -100.0), np.full(times.size, -80.0)]).astype(np.float32)
    labels = np.concatenate([np.zeros(times.size, dtype=np.int32), np.ones(times.size, dtype=np.int32)])
    path = tmp_path / "probe.cfg"
    path.write_text("1 1 0 0\n2 2 0 40\n", encoding="utf-8")
    probe = probe_from_cfg(path, n_channels)
    settings = PipelineSettings(
        sampling_rate=fs,
        feature_radius_um=60,
        waveform_pre_ms=0.8,
        waveform_post_ms=1.6,
        merge_cosine=0.8,
        merge_footprint=0.5,
        refractory_ms=1.5,
    )
    merged, templates = merge_and_split(all_times, channels, amplitudes, labels, data, probe, settings)
    surviving = merged[merged >= 0]
    assert templates.shape[0] == 1
    assert np.unique(surviving).tolist() == [0]
    assert _shared_structure(times_a, times_b, fs, 1.5)


def test_bimodal_refractory_cluster_splits():
    fs = 10000.0
    rng = np.random.default_rng(2)
    # Two regular trains interleaved inside one label, with separated amplitudes.
    train_a = np.arange(0, 8000, 200)
    train_b = np.arange(10, 8000, 200)
    times = np.concatenate([train_a, train_b]).astype(np.int64)
    amplitudes = np.concatenate([np.full(train_a.size, -40.0), np.full(train_b.size, -140.0)]).astype(np.float32)
    channels = np.zeros(times.size, dtype=np.int32)
    labels = np.zeros(times.size, dtype=np.int32)
    n_samples = 9000
    data = rng.normal(0, 1, size=(n_samples, 1)).astype(np.float32)
    path_positions = np.array([[0.0, 0.0]])
    from silicon_sort.io.geometry import Probe

    probe = Probe(
        name="one",
        positions=path_positions,
        site_ids=np.array([0]),
        shanks=np.array([0]),
        present=np.array([True]),
    )
    settings = PipelineSettings(
        sampling_rate=fs,
        refractory_ms=1.5,
        isi_split_fraction=0.05,
        waveform_pre_ms=0.4,
        waveform_post_ms=0.8,
        rng_seed=2,
    )
    merged, _templates = merge_and_split(times, channels, amplitudes, labels, data, probe, settings)
    assert len(np.unique(merged[merged >= 0])) == 2
