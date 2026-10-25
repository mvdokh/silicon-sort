"""Headerless interleaved binary recordings and per-file sidecar metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from silicon_sort.errors import RecordingError
from silicon_sort.settings import PipelineSettings, numpy_dtype


def sidecar_path(dat_path: str | Path) -> Path:
    """JSON stored next to the recording, named ``amplifier.dat.json``."""
    path = Path(dat_path)
    return path.with_name(path.name + ".json")


@dataclass
class Recording:
    """Memory-mapped ``(n_samples, n_channels)`` binary file."""

    path: Path
    n_channels: int
    sampling_rate: float
    dtype: np.dtype
    voltage_scale_uv_per_bit: float
    file_offset: int
    n_samples: int
    _memmap: np.memmap | None = None

    @property
    def duration_s(self) -> float:
        return self.n_samples / float(self.sampling_rate)

    def memmap(self) -> np.memmap:
        if self._memmap is None:
            self._memmap = np.memmap(
                self.path,
                dtype=self.dtype,
                mode="r",
                offset=int(self.file_offset),
                shape=(self.n_samples, self.n_channels),
                order="C",
            )
        return self._memmap

    def close(self) -> None:
        if self._memmap is not None:
            self._memmap._mmap.close()  # type: ignore[attr-defined]
            self._memmap = None

    def read_uv(self, start: int, stop: int) -> np.ndarray:
        """Return a float32 copy in microvolts. Does not load the rest of the file."""
        start = max(0, int(start))
        stop = min(self.n_samples, int(stop))
        if stop <= start:
            return np.zeros((0, self.n_channels), dtype=np.float32)
        raw = np.array(self.memmap()[start:stop], dtype=np.float32, copy=True)
        return raw * np.float32(self.voltage_scale_uv_per_bit)

    def sample_metadata(self) -> dict[str, Any]:
        stat = self.path.stat()
        return {
            "path": str(self.path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "n_channels": self.n_channels,
            "n_samples": self.n_samples,
            "sampling_rate": self.sampling_rate,
            "dtype": self.dtype.name,
            "voltage_scale_uv_per_bit": self.voltage_scale_uv_per_bit,
            "file_offset": self.file_offset,
        }


def n_samples_for(path: Path, n_channels: int, dtype: np.dtype, file_offset: int) -> int:
    nbytes = path.stat().st_size - int(file_offset)
    if nbytes < 0:
        raise RecordingError(f"File offset {file_offset} is past the end of {path.name}.")
    width = int(n_channels) * int(dtype.itemsize)
    if width <= 0:
        raise RecordingError("n_channels must be positive.")
    if nbytes % width != 0:
        raise RecordingError(
            f"{path.name} has {nbytes} bytes after the offset, which is not divisible by "
            f"{n_channels} channels of {dtype.name} ({width} bytes per sample)."
        )
    return nbytes // width


def open_recording(settings: PipelineSettings) -> Recording:
    settings.require_recording_meta()
    path = Path(settings.dat_path)
    if not path.is_file():
        raise RecordingError(f"Recording not found: {path}")
    dtype = settings.resolved_dtype()
    n_channels = int(settings.n_channels)  # type: ignore[arg-type]
    n_samples = n_samples_for(path, n_channels, dtype, int(settings.file_offset))
    return Recording(
        path=path,
        n_channels=n_channels,
        sampling_rate=float(settings.sampling_rate),  # type: ignore[arg-type]
        dtype=dtype,
        voltage_scale_uv_per_bit=float(settings.voltage_scale_uv_per_bit),
        file_offset=int(settings.file_offset),
        n_samples=n_samples,
    )


def save_sidecar(settings: PipelineSettings, path: str | Path | None = None) -> Path:
    """Remember channel count, rate, dtype, scale, and mask beside the recording."""
    settings.require_recording_meta()
    target = Path(path) if path is not None else sidecar_path(settings.dat_path)
    payload = {
        "n_channels": int(settings.n_channels),  # type: ignore[arg-type]
        "sampling_rate": float(settings.sampling_rate),  # type: ignore[arg-type]
        "dtype": numpy_dtype(settings.dtype).name,
        "voltage_scale_uv_per_bit": float(settings.voltage_scale_uv_per_bit),
        "file_offset": int(settings.file_offset),
        "mask": [int(v) for v in settings.mask],
        "dat_path": str(Path(settings.dat_path).resolve()) if settings.dat_path else "",
    }
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def load_sidecar(dat_path: str | Path) -> dict[str, Any]:
    path = sidecar_path(dat_path)
    if not path.is_file():
        raise RecordingError(f"No sidecar metadata at {path}.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RecordingError(f"{path} is not a JSON object.")
    for key in ("n_channels", "sampling_rate", "dtype"):
        if payload.get(key) in (None, ""):
            raise RecordingError(f"Sidecar {path} is missing {key}.")
    return payload


def apply_sidecar(settings: PipelineSettings, dat_path: str | Path | None = None) -> bool:
    """Fill empty recording fields from the sidecar. Returns True if a file was read."""
    path = Path(dat_path or settings.dat_path)
    side = sidecar_path(path)
    if not side.is_file():
        return False
    payload = load_sidecar(path)
    settings.n_channels = int(payload["n_channels"])
    settings.sampling_rate = float(payload["sampling_rate"])
    settings.dtype = str(payload["dtype"])
    settings.voltage_scale_uv_per_bit = float(payload.get("voltage_scale_uv_per_bit", 0.195))
    settings.file_offset = int(payload.get("file_offset", 0))
    if "mask" in payload and not settings.mask:
        settings.mask = [int(v) for v in payload.get("mask", [])]
    return True


def hints_from_binary_json(dat_path: str | Path) -> dict[str, Any]:
    """Read a sibling SpikeInterface ``binary.json`` when it describes this file."""
    path = Path(dat_path)
    meta = path.parent / "binary.json"
    if not meta.is_file():
        return {}
    try:
        payload = json.loads(meta.read_text(encoding="utf-8"))
        kwargs = payload.get("kwargs", {})
    except (OSError, json.JSONDecodeError, AttributeError):
        return {}
    names = [Path(str(item)).name for item in kwargs.get("file_paths", [])]
    if names and path.name not in names:
        return {}
    hints: dict[str, Any] = {}
    if kwargs.get("num_channels"):
        hints["n_channels"] = int(kwargs["num_channels"])
    if kwargs.get("sampling_frequency"):
        hints["sampling_rate"] = float(kwargs["sampling_frequency"])
    if kwargs.get("dtype"):
        hints["dtype"] = numpy_dtype(str(kwargs["dtype"])).name
    if kwargs.get("gain_to_uV") is not None:
        hints["voltage_scale_uv_per_bit"] = float(kwargs["gain_to_uV"])
    if kwargs.get("file_offset") is not None:
        hints["file_offset"] = int(kwargs["file_offset"])
    return hints


class ScaledView:
    """Float32 microvolt view of an integer memmap. Slices stay on disk until read."""

    def __init__(self, data: np.ndarray, scale: float) -> None:
        self._data = data
        self._scale = np.float32(scale)
        self.shape = data.shape
        self.dtype = np.dtype(np.float32)

    def __getitem__(self, item):
        return np.asarray(self._data[item], dtype=np.float32) * self._scale


def read_padded(
    memmap: np.ndarray,
    start: int,
    stop: int,
    overlap: int,
    scale: float,
) -> np.ndarray:
    """Read ``[start, stop)`` plus ``overlap`` samples on each side, edge-padded."""
    n_samples = int(memmap.shape[0])
    src_start = start - overlap
    src_stop = stop + overlap
    pad_left = max(0, -src_start)
    pad_right = max(0, src_stop - n_samples)
    src_start = max(0, src_start)
    src_stop = min(n_samples, src_stop)
    chunk = np.array(memmap[src_start:src_stop], dtype=np.float32, copy=True)
    chunk *= np.float32(scale)
    if pad_left or pad_right:
        chunk = np.pad(chunk, ((pad_left, pad_right), (0, 0)), mode="edge")
    return chunk
