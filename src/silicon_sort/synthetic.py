"""Synthetic recordings used by tests and the GUI smoke path."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from silicon_sort.detect.benchmark import synthetic_biphasic


def write_poly2_cfg(path: str | Path, n_channels: int = 32) -> None:
    """Two-column layout with about 50 µm nearest neighbors, 1-based channel ids."""
    lines = ["poly2"]
    for index in range(n_channels):
        column = index % 2
        row = index // 2
        x_um = -21.65 if column == 0 else 21.65
        y_um = row * 50.0 + (25.0 if column else 0.0)
        channel = index + 1
        lines.append(f"{channel} {channel} {x_um:.2f} {y_um:.2f}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_recording(
    path: str | Path,
    n_channels: int,
    sampling_rate: float,
    duration_s: float,
    seed: int = 0,
    scale: float = 0.195,
    units: list[dict] | None = None,
) -> np.ndarray:
    """Write an int16 recording and return the sample indices of planted spikes.

    Each unit dict may set ``channel``, ``rate_hz``, ``amplitude_uv``, and ``width``.
    """
    path = Path(path)
    n_samples = int(duration_s * sampling_rate)
    rng = np.random.default_rng(seed)
    if units is None:
        units = [
            {"channel": 4, "rate_hz": 25.0, "amplitude_uv": -180.0, "width": 3.0},
            {"channel": min(20, n_channels - 1), "rate_hz": 18.0, "amplitude_uv": -160.0, "width": 4.5},
        ]
    events: list[tuple[int, int, float, float]] = []
    for unit in units:
        channel = int(unit["channel"])
        n_events = int(unit["rate_hz"] * duration_s)
        if n_events <= 0:
            continue
        times = rng.integers(int(0.002 * sampling_rate), n_samples - int(0.002 * sampling_rate), size=n_events)
        for time in np.unique(times):
            events.append((int(time), channel, float(unit["amplitude_uv"]), float(unit.get("width", 3.0))))
    mapped = np.memmap(path, dtype=np.int16, mode="w+", shape=(n_samples, n_channels), order="C")
    chunk = max(1, int(sampling_rate))
    pre = int(0.0005 * sampling_rate)
    post = int(0.0012 * sampling_rate)
    for start in range(0, n_samples, chunk):
        stop = min(n_samples, start + chunk)
        common = rng.normal(0.0, 12.0, size=(stop - start, 1))
        private = rng.normal(0.0, 8.0, size=(stop - start, n_channels))
        uv = np.asarray(common + private, dtype=np.float32)
        for time, channel, amplitude, width in events:
            if not (start - post <= time < stop + pre):
                continue
            wave = synthetic_biphasic(pre, post, width=width)
            trough = pre
            a = time - trough
            b = a + wave.size
            left = max(a, start)
            right = min(b, stop)
            if right <= left:
                continue
            # The waveform troughs at -1, so the scale is the absolute trough depth.
            snippet = wave[left - a : left - a + (right - left)] * np.float32(abs(amplitude))
            uv[left - start : right - start, channel] += snippet
            neighbor = channel + 1 if channel + 1 < n_channels else channel - 1
            if neighbor >= 0:
                uv[left - start : right - start, neighbor] += snippet * np.float32(0.55)
        coded = np.clip(np.round(uv / np.float32(scale)), -32768, 32767).astype(np.int16)
        mapped[start:stop] = coded
    mapped.flush()
    del mapped
    return np.array([event[0] for event in events], dtype=np.int64)
