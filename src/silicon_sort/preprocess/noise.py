"""Robust noise scale from random chunks, without sorting the whole file."""

from __future__ import annotations

import numpy as np


def estimate_mad(
    data: np.ndarray,
    mask: np.ndarray | None = None,
    n_chunks: int = 12,
    chunk_samples: int = 30000,
    seed: int = 0,
    max_rows: int = 200_000,
    transform=None,
) -> np.ndarray:
    """Per-channel ``median(|x|) / 0.6745`` on a subsample of chunks.

    Masked channels are reported as infinity so they cannot cross a threshold.
    """
    n_samples, n_channels = data.shape
    if mask is None:
        mask = np.zeros(n_channels, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    rows = _collect_rows(data, n_chunks, chunk_samples, seed, max_rows)
    if transform is not None and rows.shape[0]:
        rows = np.asarray(transform(rows), dtype=np.float32)
    if rows.shape[0] == 0:
        mad = np.ones(n_channels, dtype=np.float64)
    else:
        med = np.median(np.abs(rows), axis=0)
        mad = med / 0.67448975
        mad = np.maximum(mad, 1e-6)
    mad = np.asarray(mad, dtype=np.float64)
    mad[mask] = np.inf
    return mad


def _collect_rows(
    data: np.ndarray,
    n_chunks: int,
    chunk_samples: int,
    seed: int,
    max_rows: int,
) -> np.ndarray:
    n_samples = int(data.shape[0])
    if n_samples == 0:
        return np.zeros((0, data.shape[1]), dtype=np.float32)
    rng = np.random.default_rng(seed)
    chunk_samples = max(1, min(int(chunk_samples), n_samples))
    if n_samples <= chunk_samples:
        rows = np.asarray(data, dtype=np.float32)
    else:
        n_possible = max(1, n_samples // chunk_samples)
        n_chunks = max(1, min(int(n_chunks), n_possible))
        starts = rng.choice(n_samples - chunk_samples + 1, size=n_chunks, replace=False)
        rows = np.concatenate(
            [np.asarray(data[int(start) : int(start) + chunk_samples], dtype=np.float32) for start in starts],
            axis=0,
        )
    if rows.shape[0] > max_rows:
        pick = rng.choice(rows.shape[0], size=max_rows, replace=False)
        rows = rows[pick]
    return rows
