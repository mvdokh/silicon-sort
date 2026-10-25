"""Tolerant parser for probe ``.cfg`` geometry and optional ``.msk`` masks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from silicon_sort.errors import GeometryError


@dataclass
class Probe:
    """Electrode geometry in recording-file channel order."""

    name: str
    positions: np.ndarray
    site_ids: np.ndarray
    shanks: np.ndarray
    present: np.ndarray

    @property
    def n_channels(self) -> int:
        return int(self.positions.shape[0])

    def channel_map(self) -> np.ndarray:
        return np.arange(self.n_channels, dtype=np.int32)

    def neighbors(self, radius_um: float, mask: np.ndarray | None = None) -> list[np.ndarray]:
        """For each channel, other unmasked channels inside ``radius_um``."""
        active = np.ones(self.n_channels, dtype=bool) if mask is None else ~np.asarray(mask, dtype=bool)
        active &= self.present
        dist = pairwise_distances(self.positions)
        out: list[np.ndarray] = []
        for index in range(self.n_channels):
            if not active[index]:
                out.append(np.zeros(0, dtype=int))
                continue
            hits = np.flatnonzero((dist[index] <= radius_um) & (dist[index] > 0) & active)
            out.append(hits.astype(int, copy=False))
        return out


@dataclass
class CfgSites:
    name: str
    site_ids: np.ndarray
    channels: np.ndarray
    positions: np.ndarray


def parse_cfg(path: str | Path) -> CfgSites:
    """Parse a geometry file.

    A name line such as ``poly2`` is optional. Data lines are either
    ``site channel x y`` or ``channel x y``. Channel ids that start at 1 are
    converted to 0-based file indices. ``#`` comments and blank lines are ignored.
    """
    path = Path(path)
    if not path.is_file():
        raise GeometryError(f"Geometry file not found: {path}")
    name = path.stem
    site_ids: list[int] = []
    channels: list[int] = []
    positions: list[tuple[float, float]] = []
    saw_numeric = False
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        nums = _maybe_floats(parts)
        if nums is None:
            if saw_numeric:
                raise GeometryError(f"{path.name}:{line_number}: expected numbers, got {raw!r}.")
            name = parts[0]
            continue
        saw_numeric = True
        if len(nums) == 3:
            channel, x_um, y_um = nums
            site = channel
        elif len(nums) == 4:
            site, channel, x_um, y_um = nums
        else:
            raise GeometryError(
                f"{path.name}:{line_number}: expected 3 or 4 numbers (channel x y, or site channel x y)."
            )
        if not np.isfinite([x_um, y_um]).all():
            raise GeometryError(f"{path.name}:{line_number}: coordinates must be finite.")
        site_ids.append(int(round(site)))
        channels.append(int(round(channel)))
        positions.append((float(x_um), float(y_um)))
    if not channels:
        raise GeometryError(f"{path.name} does not contain any electrode rows.")
    channel_arr = np.asarray(channels, dtype=int)
    site_arr = np.asarray(site_ids, dtype=int)
    if channel_arr.min() >= 1:
        channel_arr = channel_arr - 1
        if site_arr.min() >= 1:
            site_arr = site_arr - 1
    if np.unique(channel_arr).size != channel_arr.size:
        raise GeometryError(f"{path.name} repeats a channel id.")
    if channel_arr.min() < 0:
        raise GeometryError(f"{path.name} has a negative channel index.")
    return CfgSites(
        name=name,
        site_ids=site_arr,
        channels=channel_arr,
        positions=np.asarray(positions, dtype=np.float64),
    )


def probe_from_cfg(path: str | Path, n_channels: int, shank_gap_um: float = 150.0) -> Probe:
    sites = parse_cfg(path)
    if int(sites.channels.max()) >= int(n_channels):
        raise GeometryError(
            f"{Path(path).name} refers to channel {int(sites.channels.max())}, "
            f"but the recording has {n_channels} channels."
        )
    positions = np.zeros((n_channels, 2), dtype=np.float64)
    site_ids = np.arange(n_channels, dtype=int)
    present = np.zeros(n_channels, dtype=bool)
    positions[sites.channels] = sites.positions
    site_ids[sites.channels] = sites.site_ids
    present[sites.channels] = True
    shanks = np.zeros(n_channels, dtype=int)
    if present.any():
        shanks[present] = infer_shanks(positions[present, 0], shank_gap_um)
    return Probe(
        name=sites.name,
        positions=positions,
        site_ids=site_ids,
        shanks=shanks,
        present=present,
    )


def infer_shanks(x_um: np.ndarray, gap_um: float = 150.0) -> np.ndarray:
    """Split a probe into shanks where successive x coordinates jump by more than ``gap_um``."""
    x_um = np.asarray(x_um, dtype=float)
    shanks = np.zeros(x_um.shape[0], dtype=int)
    if x_um.size == 0:
        return shanks
    order = np.argsort(x_um, kind="mergesort")
    sid = 0
    shanks[order[0]] = 0
    for previous, current in zip(order[:-1], order[1:]):
        if x_um[current] - x_um[previous] > gap_um:
            sid += 1
        shanks[current] = sid
    return shanks


def parse_msk(path: str | Path, n_channels: int) -> np.ndarray:
    """Parse a 1-based ``channel T/F`` mask. ``T`` means the channel is excluded."""
    mask = np.zeros(n_channels, dtype=bool)
    path = Path(path)
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            raise GeometryError(f"{path.name}:{line_number}: expected 'channel T/F'.")
        channel = int(parts[0])
        flag = parts[1].upper()
        if flag not in {"T", "F"}:
            raise GeometryError(f"{path.name}:{line_number}: flag must be T or F.")
        index = channel - 1 if channel >= 1 else channel
        if not 0 <= index < n_channels:
            raise GeometryError(f"{path.name}:{line_number}: channel {channel} is out of range.")
        mask[index] = flag == "T"
    return mask


def pairwise_distances(positions: np.ndarray) -> np.ndarray:
    delta = positions[:, None, :] - positions[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=2))


def connected_groups(positions: np.ndarray, radius_um: float, active: np.ndarray) -> np.ndarray:
    """Label radius-connected components. Inactive channels get -1."""
    n_channels = positions.shape[0]
    parent = np.arange(n_channels, dtype=int)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    indices = np.flatnonzero(active)
    dist = pairwise_distances(positions)
    for left in range(indices.size):
        for right in range(left + 1, indices.size):
            i = int(indices[left])
            j = int(indices[right])
            if dist[i, j] <= radius_um:
                parent[find(i)] = find(j)
    labels = np.full(n_channels, -1, dtype=int)
    next_id = 0
    roots: dict[int, int] = {}
    for index in indices:
        root = find(int(index))
        if root not in roots:
            roots[root] = next_id
            next_id += 1
        labels[int(index)] = roots[root]
    return labels


def _maybe_floats(parts: list[str]) -> list[float] | None:
    try:
        return [float(part) for part in parts]
    except ValueError:
        return None
