"""Spatial ZCA whitening estimated on random chunks and applied on the fly."""

from __future__ import annotations

import numpy as np

from silicon_sort.backend import apply_matrix


def estimate_whitener(
    data: np.ndarray,
    mask: np.ndarray,
    n_chunks: int = 12,
    chunk_samples: int = 30000,
    seed: int = 0,
    eigenvalue_floor: float = 1e-3,
) -> np.ndarray:
    """Return a full ``(n_channels, n_channels)`` ZCA matrix.

    Masked channels stay at zero in the input and are left as an identity diagonal
    so they are not mixed back into the recording.
    """
    n_samples, n_channels = data.shape
    mask = np.asarray(mask, dtype=bool)
    active = np.flatnonzero(~mask)
    whitener = np.eye(n_channels, dtype=np.float32)
    if active.size < 2 or n_samples < 8:
        return whitener
    samples = _sample_rows(data, n_chunks, chunk_samples, seed)
    block = samples[:, active]
    block = block - block.mean(axis=0, keepdims=True)
    cov = (block.T @ block) / max(block.shape[0] - 1, 1)
    whitener[np.ix_(active, active)] = zca_from_covariance(cov, eigenvalue_floor).astype(np.float32)
    return whitener


def zca_from_covariance(cov: np.ndarray, eigenvalue_floor: float = 1e-3) -> np.ndarray:
    cov = np.asarray(cov, dtype=np.float64)
    cov = 0.5 * (cov + cov.T)
    evals, evecs = np.linalg.eigh(cov)
    positive = evals[evals > 0]
    scale = float(np.median(positive)) if positive.size else 1.0
    floor = float(eigenvalue_floor) * max(scale, 1e-12)
    evals = np.maximum(evals, floor)
    inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    return inv_sqrt


def apply_whitener(chunk: np.ndarray, whitener: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    out = apply_matrix(np.asarray(chunk, dtype=np.float32), np.asarray(whitener, dtype=np.float32))
    if mask is not None:
        out[:, np.asarray(mask, dtype=bool)] = 0
    return out


def _sample_rows(data: np.ndarray, n_chunks: int, chunk_samples: int, seed: int) -> np.ndarray:
    n_samples = data.shape[0]
    rng = np.random.default_rng(seed)
    if n_samples <= chunk_samples:
        return np.asarray(data, dtype=np.float32)
    n_chunks = max(1, min(int(n_chunks), n_samples // max(chunk_samples, 1) + 1))
    starts = rng.integers(0, max(n_samples - chunk_samples, 1), size=n_chunks)
    parts = [np.asarray(data[int(start) : int(start) + chunk_samples], dtype=np.float32) for start in starts]
    return np.concatenate(parts, axis=0)
