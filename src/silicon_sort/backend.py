"""Optional CUDA matmul. The CPU path is always correct on its own."""

from __future__ import annotations

import numpy as np


def _cupy():
    try:
        import cupy as cp
    except Exception:
        return None
    try:
        if int(cp.cuda.runtime.getDeviceCount()) < 1:
            return None
    except Exception:
        return None
    return cp


def cuda_available() -> bool:
    return _cupy() is not None


def apply_matrix(data: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Return ``data @ matrix``. Large chunks use CuPy when a CUDA device exists."""
    if data.size == 0:
        return np.zeros((data.shape[0], matrix.shape[1]), dtype=np.float32)
    cp = _cupy() if data.shape[0] >= 200_000 and data.shape[1] >= 8 else None
    if cp is not None:
        out = cp.asnumpy(cp.asarray(data, dtype=cp.float32) @ cp.asarray(matrix, dtype=cp.float32))
        return np.asarray(out, dtype=np.float32)
    return np.asarray(data @ matrix, dtype=np.float32)
