"""Chunked filtering, referencing, whitening, and MAD."""

from pathlib import Path

import numpy as np
from scipy import signal

from silicon_sort.preprocess.filter import design_bandpass, filter_array_chunked, filter_overlap_samples
from silicon_sort.preprocess.noise import estimate_mad
from silicon_sort.preprocess.reference import reference_chunk, reference_groups
from silicon_sort.preprocess.whiten import apply_whitener, estimate_whitener, zca_from_covariance


def test_chunked_bandpass_matches_full_filter():
    rng = np.random.default_rng(0)
    fs = 30000.0
    data = rng.normal(0, 20, size=(8000, 4)).astype(np.float32)
    sos = design_bandpass(fs, 300, 6000, 3)
    overlap = filter_overlap_samples(fs, 300, sos)
    chunked = filter_array_chunked(data, sos, overlap, chunk_samples=2000, reference_mode="none")
    full = signal.sosfiltfilt(sos, data, axis=0).astype(np.float32)
    interior = slice(2 * overlap, -2 * overlap)
    scale = float(np.std(full[interior]))
    assert np.max(np.abs(chunked[interior] - full[interior])) < 1e-2 * scale


def test_common_median_ignores_masked_channel():
    chunk = np.array(
        [
            [1.0, 3.0, 100.0],
            [2.0, 4.0, -100.0],
        ],
        dtype=np.float32,
    )
    mask = np.array([False, False, True])
    groups = np.array([0, 0, -1])
    out = reference_chunk(chunk, mask, groups)
    np.testing.assert_allclose(out[:, 2], 0)
    np.testing.assert_allclose(out[:, 0], chunk[:, 0] - np.median(chunk[:, :2], axis=1))


def test_local_groups_follow_radius():
    positions = np.array([[0.0, 0.0], [10.0, 0.0], [500.0, 0.0]])
    mask = np.zeros(3, dtype=bool)
    groups = reference_groups(positions, mask, "local", local_radius_um=50, shank_gap_um=150)
    assert groups[0] == groups[1]
    assert groups[2] != groups[0]


def test_zca_removes_shared_correlation():
    rng = np.random.default_rng(1)
    shared = rng.normal(size=(5000, 1))
    private = rng.normal(scale=0.1, size=(5000, 3))
    data = np.repeat(shared, 3, axis=1) + private
    mask = np.array([False, False, True])
    whitener = estimate_whitener(data.astype(np.float32), mask, n_chunks=2, chunk_samples=2000, seed=1)
    white = apply_whitener(data[:, :].astype(np.float32), whitener, mask)
    cov = np.cov(white[:, :2], rowvar=False)
    assert abs(cov[0, 1]) < 0.15
    assert np.allclose(white[:, 2], 0)


def test_zca_matches_direct_covariance():
    cov = np.array([[2.0, 0.4], [0.4, 1.0]])
    whitener = zca_from_covariance(cov)
    whitened = whitener @ cov @ whitener.T
    np.testing.assert_allclose(whitened, np.eye(2), atol=1e-6)


def test_bad_channels_follow_neighbors_not_only_a_harsh_global_cutoff(tmp_path: Path):
    from silicon_sort.io.binary import Recording
    from silicon_sort.preprocess.bad_channels import suggest_bad_channels

    n_samples = 12000
    n_channels = 8
    rng = np.random.default_rng(4)
    uv = rng.normal(0, 25.0, size=(n_samples, n_channels))
    slow = np.convolve(rng.normal(0, 1.0, n_samples), np.ones(31) / 31.0, mode="same")
    uv[:, 6] = slow / (np.std(slow) + 1e-6) * 70.0
    uv[:, 7] *= 0.04
    path = tmp_path / "raw.dat"
    mapped = np.memmap(path, dtype=np.int16, mode="w+", shape=(n_samples, n_channels))
    mapped[:] = np.clip(np.round(uv), -32767, 32767).astype(np.int16)
    mapped.flush()
    del mapped
    recording = Recording(
        path=path,
        n_channels=n_channels,
        sampling_rate=10000.0,
        dtype=np.dtype("<i2"),
        voltage_scale_uv_per_bit=1.0,
        file_offset=0,
        n_samples=n_samples,
    )
    positions = np.column_stack([np.zeros(n_channels), np.arange(n_channels) * 50.0])
    suggestion = suggest_bad_channels(recording, positions=positions, chunk_seconds=0.4, n_chunks=3, seed=1)
    assert suggestion.mask[6]
    assert suggestion.mask[7]
    assert not np.any(suggestion.mask[:6])
    assert "7 (" in suggestion.summary()
    assert "8 (" in suggestion.summary()


def test_mad_on_gaussian():
    rng = np.random.default_rng(2)
    data = rng.normal(0, 2.0, size=(20000, 2)).astype(np.float32)
    mad = estimate_mad(data, n_chunks=4, chunk_samples=4000, seed=2)
    np.testing.assert_allclose(mad, [2.0, 2.0], rtol=0.1)
