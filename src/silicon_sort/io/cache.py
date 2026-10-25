"""On-disk cache for filtered traces, spike indices, and stage manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from silicon_sort.settings import stable_hash


def cache_dir_for(dat_path: str | Path) -> Path:
    path = Path(dat_path)
    return path.with_name(path.stem + ".silicon-sort")


@dataclass
class SessionCache:
    root: Path

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def filtered_path(self) -> Path:
        return self.root / "filtered.dat"

    @property
    def spikes_path(self) -> Path:
        return self.root / "spikes.npz"

    @property
    def noise_path(self) -> Path:
        return self.root / "noise_mad.npy"

    @property
    def whitener_path(self) -> Path:
        return self.root / "whitener.npy"

    @property
    def sort_path(self) -> Path:
        return self.root / "sort.npz"

    @property
    def metrics_path(self) -> Path:
        return self.root / "metrics.json"

    @property
    def recording_json(self) -> Path:
        return self.root / "recording.json"

    def load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.is_file():
            return {}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def save_manifest(self, manifest: dict[str, Any]) -> None:
        self.manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    def stage_hash(self, stage: str) -> str | None:
        entry = self.load_manifest().get(stage)
        if not isinstance(entry, dict):
            return None
        value = entry.get("sha")
        return str(value) if value else None

    def valid(self, stage: str, sha: str) -> bool:
        return self.stage_hash(stage) == sha and self._stage_files_exist(stage)

    def mark(self, stage: str, sha: str, **extra: Any) -> None:
        manifest = self.load_manifest()
        manifest[stage] = {"sha": sha, **extra}
        self.save_manifest(manifest)

    def invalidate_from(self, stage: str) -> None:
        order = ["preprocess", "detect", "sort"]
        if stage not in order:
            return
        manifest = self.load_manifest()
        for name in order[order.index(stage) :]:
            manifest.pop(name, None)
        self.save_manifest(manifest)

    def _stage_files_exist(self, stage: str) -> bool:
        if stage == "preprocess":
            return self.filtered_path.is_file()
        if stage == "detect":
            return self.spikes_path.is_file() and self.noise_path.is_file()
        if stage == "sort":
            return self.sort_path.is_file() and self.metrics_path.is_file()
        return False

    def open_filtered(self, n_samples: int, n_channels: int, mode: str = "r") -> np.memmap:
        return np.memmap(
            self.filtered_path,
            dtype=np.float32,
            mode=mode,
            shape=(int(n_samples), int(n_channels)),
            order="C",
        )

    def create_filtered(self, n_samples: int, n_channels: int) -> np.memmap:
        mapped = np.memmap(
            self.filtered_path,
            dtype=np.float32,
            mode="w+",
            shape=(int(n_samples), int(n_channels)),
            order="C",
        )
        mapped.flush()
        return mapped


def hash_payload(payload: dict[str, Any]) -> str:
    return stable_hash(payload)
