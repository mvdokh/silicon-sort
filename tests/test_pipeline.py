"""Phy export layout and a headless synthetic session."""

import json
from pathlib import Path

import numpy as np

from silicon_sort.io.phy_export import PHY_FILES, export_phy
from silicon_sort.pipeline import Pipeline, resolve_settings
from silicon_sort.settings import PipelineSettings, load_settings, save_settings
from silicon_sort.synthetic import write_poly2_cfg, write_recording


def test_phy_folder_shapes(tmp_path: Path):
    dat = tmp_path / "amp.dat"
    dat.write_bytes(b"\x00\x00")
    settings = PipelineSettings(
        dat_path=str(dat),
        n_channels=2,
        sampling_rate=30000,
        dtype="int16",
        file_offset=0,
        voltage_scale_uv_per_bit=0.195,
    )
    times = np.array([10, 40, 5], dtype=np.int64)
    labels = np.array([0, 1, -1], dtype=np.int32)
    amplitudes = np.array([-20.0, -30.0, -5.0], dtype=np.float32)
    templates = np.zeros((2, 6, 2), dtype=np.float32)
    templates[0, 2, 0] = -20
    positions = np.array([[0.0, 0.0], [0.0, 50.0]], dtype=np.float32)
    metrics = [
        {
            "cluster_id": 0,
            "amplitude": 20.0,
            "contam_pct": 0.0,
            "quality": "good",
            "peak_channel": 0,
            "depth_um": 0.0,
            "firing_rate": 1.0,
            "n_spikes": 1,
            "snr": 5.0,
            "isi_violation_rate": 0.0,
            "waveform_width_ms": 0.3,
        },
        {
            "cluster_id": 1,
            "amplitude": 30.0,
            "contam_pct": 1.0,
            "quality": "mua",
            "peak_channel": 1,
            "depth_um": 50.0,
            "firing_rate": 2.0,
            "n_spikes": 1,
            "snr": 4.0,
            "isi_violation_rate": 0.01,
            "waveform_width_ms": 0.4,
        },
    ]
    out = export_phy(tmp_path / "phy", settings, times, labels, amplitudes, templates, positions, metrics)
    for name in PHY_FILES:
        assert (out / name).is_file()
    loaded_times = np.load(out / "spike_times.npy")
    np.testing.assert_array_equal(loaded_times, [10, 40])
    assert np.load(out / "spike_clusters.npy").dtype == np.int32
    assert np.load(out / "templates.npy").shape == (2, 6, 2)
    assert np.load(out / "amplitudes.npy").shape == (2, 1)
    assert np.load(out / "channel_positions.npy").shape == (2, 2)
    namespace: dict = {}
    exec((out / "params.py").read_text(encoding="utf-8"), namespace)
    assert namespace["n_channels_dat"] == 2
    assert namespace["sample_rate"] == 30000
    assert namespace["dtype"] == "int16"
    assert namespace["hp_filtered"] is False
    assert namespace["dat_path"] == str(dat.resolve())
    header = (out / "cluster_info.tsv").read_text(encoding="utf-8").splitlines()[0]
    assert "isi_violation_rate" in header
    saved = json.loads((out / "pipeline_settings.json").read_text(encoding="utf-8"))
    assert saved["sampling_rate"] == 30000


def test_headless_refuses_missing_metadata(tmp_path: Path):
    path = tmp_path / "settings.json"
    save_settings(PipelineSettings(dat_path=str(tmp_path / "missing.dat"), n_channels=None, sampling_rate=None), path)
    settings = load_settings(path)
    settings.dtype = ""
    try:
        resolve_settings(settings)
    except Exception as exc:
        assert "n_channels" in str(exc) or "sidecar" in str(exc).lower() or "recording" in str(exc).lower()
    else:
        raise AssertionError("missing metadata should fail")


def test_synthetic_session_exports_phy_and_reuses_cache(tmp_path: Path):
    dat = tmp_path / "session.dat"
    cfg = tmp_path / "probe.cfg"
    n_channels = 32
    fs = 30000.0
    write_poly2_cfg(cfg, n_channels)
    write_recording(dat, n_channels, fs, duration_s=1.5, seed=7)
    settings = PipelineSettings(
        dat_path=str(dat),
        cfg_path=str(cfg),
        n_channels=n_channels,
        sampling_rate=fs,
        dtype="int16",
        voltage_scale_uv_per_bit=0.195,
        output_dir=str(tmp_path / "phy"),
        n_workers=1,
        whiten=True,
        chunk_seconds=0.5,
        template_match=False,
        mad_chunks=4,
        whiten_chunks=4,
    )
    pipeline = Pipeline(settings)
    try:
        target = pipeline.run_all()
        filtered = pipeline.cache().filtered_path.stat().st_mtime_ns
        spikes = pipeline.cache().spikes_path.stat().st_mtime_ns
        settings.channel_k = {"4": 2}
        again = pipeline.run_sort()
        assert pipeline.cache().filtered_path.stat().st_mtime_ns == filtered
        assert pipeline.cache().spikes_path.stat().st_mtime_ns == spikes
        assert again.labels.size >= 1
    finally:
        pipeline.close()
    assert (target / "params.py").is_file()
    times = np.load(target / "spike_times.npy")
    assert times.dtype == np.int64
    assert times.size >= 1
    assert times.min() >= 0
    n_samples = dat.stat().st_size // (n_channels * 2)
    assert times.max() < n_samples
    templates = np.load(target / "templates.npy")
    assert templates.ndim == 3
    assert templates.shape[2] == n_channels
    namespace: dict = {}
    exec((target / "params.py").read_text(encoding="utf-8"), namespace)
    assert namespace["sample_rate"] == fs
    assert Path(namespace["dat_path"]) == dat.resolve()
    positions = np.load(target / "channel_positions.npy")
    assert positions.shape == (n_channels, 2)


def test_detection_can_skip_preprocessing(tmp_path: Path):
    dat = tmp_path / "amp.dat"
    cfg = tmp_path / "electrode.cfg"
    n_channels = 8
    fs = 10000.0
    write_poly2_cfg(cfg, n_channels)
    write_recording(dat, n_channels, fs, duration_s=0.4, seed=3)
    settings = PipelineSettings(
        dat_path=str(dat),
        cfg_path=str(cfg),
        n_channels=n_channels,
        sampling_rate=fs,
        dtype="int16",
        voltage_scale_uv_per_bit=0.195,
        preprocess=False,
        whiten=False,
        n_workers=1,
        template_match=False,
        mad_chunks=2,
        chunk_seconds=0.2,
    )
    pipeline = Pipeline(settings)
    try:
        spikes = pipeline.run_detect()
    finally:
        pipeline.close()
    assert spikes.n_spikes >= 1
    assert not pipeline.cache().filtered_path.exists()


def test_review_reuses_the_first_detection(tmp_path: Path, monkeypatch):
    dat = tmp_path / "amp.dat"
    cfg = tmp_path / "electrode.cfg"
    n_channels = 8
    fs = 10000.0
    write_poly2_cfg(cfg, n_channels)
    write_recording(dat, n_channels, fs, duration_s=0.4, seed=3)
    settings = PipelineSettings(
        dat_path=str(dat),
        cfg_path=str(cfg),
        n_channels=n_channels,
        sampling_rate=fs,
        dtype="int16",
        voltage_scale_uv_per_bit=0.195,
        preprocess=False,
        whiten=False,
        n_workers=1,
        template_match=False,
        mad_chunks=2,
        chunk_seconds=0.2,
    )
    pipeline = Pipeline(settings)
    calls = {"n": 0}
    import silicon_sort.pipeline as pipeline_module

    real = pipeline_module.detect_traces

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "detect_traces", wrapped)
    try:
        spikes = pipeline.run_detect()
        assert spikes.n_spikes >= 1
        assert calls["n"] == 1
        pipeline.review_bad = "marked"
        pipeline.review_good = "marked"
        pipeline.label_spike(int(spikes.times[0]), int(spikes.channels[0]), -1)
        again = pipeline.run_detect()
        assert calls["n"] == 1
        assert int(spikes.times[0]) not in set(again.times[again.channels == spikes.channels[0]].tolist())
        other = int(again.channels[0]) if again.n_spikes else 0
        target = (other + 1) % n_channels
        before_other = int(np.sum(again.channels == other))
        pipeline.redetect_channels([target])
        assert calls["n"] == 2
        assert int(np.sum(pipeline.spikes.channels == other)) == before_other
    finally:
        pipeline.close()
