"""Binary recording memmap and sidecar metadata."""

from pathlib import Path

import numpy as np

from silicon_sort.io.binary import hints_from_binary_json, open_recording, save_sidecar, sidecar_path
from silicon_sort.settings import PipelineSettings


def test_memmap_round_trip_reads_only_a_window(tmp_path: Path):
    path = tmp_path / "rec.dat"
    raw = np.arange(40, dtype=np.int16).reshape(10, 4)
    raw.tofile(path)
    settings = PipelineSettings(
        dat_path=str(path),
        n_channels=4,
        sampling_rate=1000,
        dtype="int16",
        voltage_scale_uv_per_bit=0.5,
    )
    save_sidecar(settings)
    recording = open_recording(settings)
    window = recording.read_uv(2, 4)
    assert window.shape == (2, 4)
    np.testing.assert_allclose(window, raw[2:4].astype(np.float32) * 0.5)
    payload = sidecar_path(path).read_text(encoding="utf-8")
    assert "1000" in payload
    recording.close()


def test_binary_json_hint(tmp_path: Path):
    dat = tmp_path / "amplifier.dat"
    dat.write_bytes(b"")
    (tmp_path / "binary.json").write_text(
        '{"kwargs": {"file_paths": ["amplifier.dat"], "num_channels": 32, '
        '"sampling_frequency": 30000, "dtype": "<i2", "gain_to_uV": 0.195, "file_offset": 0}}',
        encoding="utf-8",
    )
    hints = hints_from_binary_json(dat)
    assert hints["n_channels"] == 32
    assert hints["sampling_rate"] == 30000
    assert hints["dtype"] == "int16"
    assert hints["voltage_scale_uv_per_bit"] == 0.195
