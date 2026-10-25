"""Geometry parser tests against the sample Poly2 configuration."""

from pathlib import Path

import numpy as np
import pytest

from silicon_sort.errors import GeometryError
from silicon_sort.io.geometry import infer_shanks, parse_cfg, parse_msk, probe_from_cfg

FIXTURE = Path(__file__).parent / "fixtures" / "electrode.cfg"
MASK = Path(__file__).parent / "fixtures" / "sample.msk"


def test_sample_cfg_matches_file_order():
    probe = probe_from_cfg(FIXTURE, n_channels=32)
    assert probe.name == "poly2"
    assert probe.n_channels == 32
    assert probe.present.all()
    np.testing.assert_allclose(probe.positions[0], [21.65, 25.0])
    np.testing.assert_allclose(probe.positions[1], [21.65, 725.0])
    np.testing.assert_allclose(probe.positions[16], [-21.65, 0.0])
    np.testing.assert_array_equal(probe.channel_map(), np.arange(32))
    assert np.unique(probe.shanks).tolist() == [0]
    nearest = np.linalg.norm(probe.positions[0] - probe.positions[16])
    assert 40 < nearest < 60


def test_three_column_and_comments(tmp_path: Path):
    path = tmp_path / "probe.cfg"
    path.write_text("# header\nlinear\n1 0 0\n2 0 50\n", encoding="utf-8")
    sites = parse_cfg(path)
    assert sites.name == "linear"
    np.testing.assert_array_equal(sites.channels, [0, 1])


def test_zero_based_channels_are_kept(tmp_path: Path):
    path = tmp_path / "probe.cfg"
    path.write_text("0 0 0\n1 0 40\n", encoding="utf-8")
    sites = parse_cfg(path)
    np.testing.assert_array_equal(sites.channels, [0, 1])


def test_duplicate_channel_is_rejected(tmp_path: Path):
    path = tmp_path / "probe.cfg"
    path.write_text("1 0 0\n1 0 10\n", encoding="utf-8")
    with pytest.raises(GeometryError, match="repeats"):
        parse_cfg(path)


def test_shank_gap_splits_columns():
    shanks = infer_shanks(np.array([0.0, 10.0, 400.0, 410.0]), gap_um=150)
    assert shanks.tolist() == [0, 0, 1, 1]


def test_msk_marks_true_channels():
    mask = parse_msk(MASK, n_channels=32)
    expected = np.zeros(32, dtype=bool)
    for channel in (4, 17, 24, 25, 26, 31):
        expected[channel - 1] = True
    np.testing.assert_array_equal(mask, expected)
