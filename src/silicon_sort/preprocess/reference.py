"""Common median reference across all channels, a shank, or a local group."""

from __future__ import annotations

import numpy as np

from silicon_sort.io.geometry import connected_groups, infer_shanks


def reference_groups(
    positions: np.ndarray,
    mask: np.ndarray,
    mode: str,
    local_radius_um: float,
    shank_gap_um: float,
) -> np.ndarray | None:
    """Group labels for referencing. ``None`` means one global group of unmasked channels."""
    active = ~np.asarray(mask, dtype=bool)
    if mode == "global":
        groups = np.full(mask.shape[0], -1, dtype=int)
        groups[active] = 0
        return groups
    if mode == "shank":
        groups = np.full(mask.shape[0], -1, dtype=int)
        if active.any():
            groups[active] = infer_shanks(positions[active, 0], shank_gap_um)
        return groups
    if mode == "local":
        return connected_groups(positions, local_radius_um, active)
    raise ValueError(f"Unknown reference mode {mode!r}. Use global, shank, or local.")


def reference_chunk(chunk: np.ndarray, mask: np.ndarray, groups: np.ndarray | None) -> np.ndarray:
    """Subtract the median of each group at every sample. Masked channels are zeroed.

    Groups of a single channel are left unchanged, because the median would erase them.
    """
    out = np.array(chunk, dtype=np.float32, copy=True)
    mask = np.asarray(mask, dtype=bool)
    if groups is None:
        groups = np.zeros(out.shape[1], dtype=int)
        groups[mask] = -1
    for group_id in np.unique(groups):
        if group_id < 0:
            continue
        columns = np.flatnonzero(groups == group_id)
        if columns.size < 2:
            continue
        median = np.median(out[:, columns], axis=1, keepdims=True)
        out[:, columns] -= median.astype(np.float32, copy=False)
    out[:, mask] = 0
    return out
