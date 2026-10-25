"""Pipeline settings, JSON round-trip, and cache hashes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


DTYPE_MAP: dict[str, np.dtype] = {
    "int16": np.dtype("<i2"),
    "int32": np.dtype("<i4"),
    "uint16": np.dtype("<u2"),
    "float32": np.dtype("<f4"),
    "float64": np.dtype("<f8"),
}


def numpy_dtype(name: str) -> np.dtype:
    key = name.strip().lower()
    if key in {"<i2", "i2"}:
        key = "int16"
    elif key in {"<i4", "i4"}:
        key = "int32"
    elif key in {"<u2", "u2"}:
        key = "uint16"
    elif key in {"<f4", "f4", "float"}:
        key = "float32"
    elif key in {"<f8", "f8", "double"}:
        key = "float64"
    if key not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype {name!r}. Use one of {sorted(DTYPE_MAP)}.")
    return DTYPE_MAP[key]


@dataclass
class PipelineSettings:
    """Every knob the GUI and the headless runner share."""

    dat_path: str = ""
    cfg_path: str = ""
    output_dir: str = ""

    n_channels: int | None = None
    sampling_rate: float | None = None
    dtype: str = "int16"
    voltage_scale_uv_per_bit: float = 0.195
    file_offset: int = 0
    mask: list[int] = field(default_factory=list)

    highpass_hz: float = 300.0
    lowpass_hz: float = 6000.0
    filter_order: int = 3
    reference_mode: str = "global"
    local_radius_um: float = 100.0
    shank_gap_um: float = 150.0
    preprocess: bool = True
    whiten: bool = True
    chunk_seconds: float = 1.0
    n_workers: int = 0
    mad_chunks: int = 12
    whiten_chunks: int = 12

    threshold_mad: float = -5.0
    neighbor_radius_um: float = 60.0
    neighbor_support_fraction: float = 0.5
    neighbor_window_ms: float = 0.3
    min_neighbors: int = 1
    dead_time_ms: float = 0.7
    align_ms: float = 0.4
    glitch_energy_fraction: float = 0.8
    biphasic_fraction: float = 0.2
    biphasic_window_ms: float = 1.0
    artifact_fraction: float = 0.4
    artifact_mad: float = 8.0
    artifact_coincidence_ms: float = 0.5
    artifact_blank_ms: float = 1.0
    template_match: bool = False
    template_threshold_fraction: float = 0.6
    template_min_cosine: float = 0.8

    n_pca: int = 3
    feature_radius_um: float = 100.0
    waveform_pre_ms: float = 0.6
    waveform_post_ms: float = 1.2
    k_max: int = 4
    channel_k: dict[str, int] = field(default_factory=dict)
    merge_cosine: float = 0.9
    merge_footprint: float = 0.8
    refractory_ms: float = 1.5
    isi_split_fraction: float = 0.1
    rng_seed: int = 0

    def resolved_dtype(self) -> np.dtype:
        return numpy_dtype(self.dtype)

    def mask_array(self, n_channels: int) -> np.ndarray:
        mask = np.zeros(n_channels, dtype=bool)
        for index in self.mask:
            if 0 <= int(index) < n_channels:
                mask[int(index)] = True
        return mask

    def set_mask(self, mask: np.ndarray) -> None:
        self.mask = [int(i) for i in np.flatnonzero(np.asarray(mask, dtype=bool))]

    def forced_k(self, channel: int) -> int | None:
        value = self.channel_k.get(str(channel))
        if value is None:
            return None
        value = int(value)
        return value if value > 0 else None

    def require_recording_meta(self) -> None:
        missing = [
            name
            for name, value in (
                ("n_channels", self.n_channels),
                ("sampling_rate", self.sampling_rate),
                ("dtype", self.dtype),
            )
            if value is None or value == ""
        ]
        if missing:
            raise ValueError(
                "Recording metadata is incomplete ("
                + ", ".join(missing)
                + "). Set n_channels, sampling rate, and dtype in the GUI or the settings JSON."
            )
        if int(self.n_channels) < 1:
            raise ValueError("n_channels must be positive.")
        if float(self.sampling_rate) <= 0:
            raise ValueError("sampling_rate must be positive.")


def settings_to_dict(settings: PipelineSettings) -> dict[str, Any]:
    payload = asdict(settings)
    payload["channel_k"] = {str(k): int(v) for k, v in settings.channel_k.items()}
    payload["mask"] = [int(v) for v in settings.mask]
    return payload


def settings_from_dict(payload: dict[str, Any]) -> PipelineSettings:
    known = {f.name for f in PipelineSettings.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    data = {key: value for key, value in payload.items() if key in known}
    if "channel_k" in data and data["channel_k"] is not None:
        data["channel_k"] = {str(k): int(v) for k, v in data["channel_k"].items()}
    if data.get("mask") is None:
        data["mask"] = []
    return PipelineSettings(**data)


def save_settings(settings: PipelineSettings, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings_to_dict(settings), indent=2) + "\n", encoding="utf-8")


def load_settings(path: str | Path) -> PipelineSettings:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a settings object.")
    return settings_from_dict(payload)


def stable_hash(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"Cannot hash value of type {type(value)!r}")


def worker_count(requested: int) -> int:
    if requested and requested > 0:
        return int(requested)
    import os

    return max(1, os.cpu_count() or 1)
