"""Chunked zero-phase bandpass with overlap so chunk edges stay faithful."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy import signal

from silicon_sort.cancel import CancelToken
from silicon_sort.errors import Cancelled
from silicon_sort.preprocess.reference import reference_chunk
from silicon_sort.settings import worker_count


Progress = Callable[[float, str], None]


def design_bandpass(sampling_rate: float, highpass_hz: float, lowpass_hz: float, order: int) -> np.ndarray:
    nyquist = 0.5 * float(sampling_rate)
    low = float(highpass_hz)
    high = float(lowpass_hz)
    if not 0 < low < high < nyquist:
        raise ValueError(
            f"Bandpass {low:g}–{high:g} Hz is outside the range valid for sampling rate {sampling_rate:g}."
        )
    return signal.butter(int(order), [low, high], btype="bandpass", fs=float(sampling_rate), output="sos")


def filter_overlap_samples(sampling_rate: float, highpass_hz: float, sos: np.ndarray) -> int:
    """About three highpass cycles, and at least the filtfilt padding length."""
    cycles = int(np.ceil(3.0 * float(sampling_rate) / max(float(highpass_hz), 1.0)))
    n_sections = int(sos.shape[0])
    padlen = 3 * (2 * n_sections + 1)
    return int(max(cycles, padlen, 32))


def iter_cores(n_samples: int, chunk_samples: int) -> list[tuple[int, int]]:
    cores: list[tuple[int, int]] = []
    start = 0
    chunk_samples = max(1, int(chunk_samples))
    while start < n_samples:
        stop = min(n_samples, start + chunk_samples)
        cores.append((start, stop))
        start = stop
    return cores


def filter_array_chunked(
    data: np.ndarray,
    sos: np.ndarray,
    overlap: int,
    chunk_samples: int,
    mask: np.ndarray | None = None,
    reference_mode: str = "none",
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """Filter an in-memory array with the same overlap rule as the file path."""
    n_samples, n_channels = data.shape
    out = np.zeros((n_samples, n_channels), dtype=np.float32)
    if mask is None:
        mask = np.zeros(n_channels, dtype=bool)
    for start, stop in iter_cores(n_samples, chunk_samples):
        out[start:stop] = _filter_core(data, start, stop, sos, overlap, mask, reference_mode, groups)
    return out


def _filter_core(
    data: np.ndarray,
    start: int,
    stop: int,
    sos: np.ndarray,
    overlap: int,
    mask: np.ndarray,
    reference_mode: str,
    groups: np.ndarray | None,
) -> np.ndarray:
    padded = _pad_array(data, start, stop, overlap)
    filtered = _sosfiltfilt(padded, sos)
    core = np.asarray(filtered[overlap : overlap + (stop - start)], dtype=np.float32)
    if reference_mode != "none":
        core = reference_chunk(core, mask, groups)
    else:
        core = core.copy()
        core[:, mask] = 0
    return core


def _pad_array(data: np.ndarray, start: int, stop: int, overlap: int) -> np.ndarray:
    n_samples = data.shape[0]
    src_start = start - overlap
    src_stop = stop + overlap
    pad_left = max(0, -src_start)
    pad_right = max(0, src_stop - n_samples)
    src_start = max(0, src_start)
    src_stop = min(n_samples, src_stop)
    chunk = np.array(data[src_start:src_stop], dtype=np.float32, copy=True)
    if pad_left or pad_right:
        chunk = np.pad(chunk, ((pad_left, pad_right), (0, 0)), mode="edge")
    return chunk


def _sosfiltfilt(chunk: np.ndarray, sos: np.ndarray) -> np.ndarray:
    if chunk.shape[0] < 8:
        return chunk.copy()
    n_sections = int(sos.shape[0])
    default_pad = 3 * (2 * n_sections + 1)
    padlen = min(default_pad, chunk.shape[0] - 1)
    if padlen < 1:
        return signal.sosfilt(sos, chunk, axis=0).astype(np.float32, copy=False)
    filtered = signal.sosfiltfilt(sos, chunk, axis=0, padlen=padlen)
    return np.asarray(filtered, dtype=np.float32)


def filter_recording_to_memmap(
    dat_path: str,
    n_samples: int,
    n_channels: int,
    dtype: np.dtype,
    file_offset: int,
    scale: float,
    out_path: str,
    sos: np.ndarray,
    overlap: int,
    chunk_samples: int,
    mask: np.ndarray,
    reference_mode: str,
    groups: np.ndarray | None,
    n_workers: int,
    progress: Progress | None = None,
    cancel: CancelToken | None = None,
) -> None:
    """Filter a recording in chunks and write a float32 memmap. Workers never share the GUI."""
    cores = iter_cores(n_samples, chunk_samples)
    if not cores:
        return
    tasks = [
        {
            "dat_path": dat_path,
            "n_samples": n_samples,
            "n_channels": n_channels,
            "dtype": np.dtype(dtype).str,
            "file_offset": int(file_offset),
            "scale": float(scale),
            "start": start,
            "stop": stop,
            "sos": np.asarray(sos, dtype=np.float64),
            "overlap": int(overlap),
            "mask": np.asarray(mask, dtype=bool),
            "reference_mode": reference_mode,
            "groups": None if groups is None else np.asarray(groups, dtype=np.int32),
        }
        for start, stop in cores
    ]
    workers = 1 if len(tasks) == 1 else worker_count(n_workers)
    done = 0
    if workers == 1:
        for task in tasks:
            _raise_if_cancelled(cancel)
            start, stop, core = filter_chunk_task(task)
            _write_core(out_path, n_samples, n_channels, start, stop, core)
            done += 1
            _report(progress, done / len(tasks), f"Filtering {done}/{len(tasks)}")
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(filter_chunk_task, task) for task in tasks]
        try:
            for future in as_completed(futures):
                _raise_if_cancelled(cancel)
                start, stop, core = future.result()
                _write_core(out_path, n_samples, n_channels, start, stop, core)
                done += 1
                _report(progress, done / len(tasks), f"Filtering {done}/{len(tasks)}")
        except Cancelled:
            for future in futures:
                future.cancel()
            raise


def filter_chunk_task(task: dict) -> tuple[int, int, np.ndarray]:
    """Process-pool entry point. Reopens the recording and returns one core chunk."""
    mapped = np.memmap(
        task["dat_path"],
        dtype=np.dtype(task["dtype"]),
        mode="r",
        offset=int(task["file_offset"]),
        shape=(int(task["n_samples"]), int(task["n_channels"])),
        order="C",
    )
    start = int(task["start"])
    stop = int(task["stop"])
    overlap = int(task["overlap"])
    padded = _read_scaled(mapped, start, stop, overlap, float(task["scale"]))
    filtered = _sosfiltfilt(padded, task["sos"])
    core = np.asarray(filtered[overlap : overlap + (stop - start)], dtype=np.float32)
    core = reference_chunk(core, task["mask"], task["groups"]) if task["reference_mode"] != "none" else core
    if task["reference_mode"] == "none":
        core = core.copy()
        core[:, task["mask"]] = 0
    del mapped
    return start, stop, core


def _read_scaled(mapped: np.ndarray, start: int, stop: int, overlap: int, scale: float) -> np.ndarray:
    n_samples = mapped.shape[0]
    src_start = start - overlap
    src_stop = stop + overlap
    pad_left = max(0, -src_start)
    pad_right = max(0, src_stop - n_samples)
    src_start = max(0, src_start)
    src_stop = min(n_samples, src_stop)
    chunk = np.array(mapped[src_start:src_stop], dtype=np.float32, copy=True)
    chunk *= np.float32(scale)
    if pad_left or pad_right:
        chunk = np.pad(chunk, ((pad_left, pad_right), (0, 0)), mode="edge")
    return chunk


def _write_core(
    out_path: str,
    n_samples: int,
    n_channels: int,
    start: int,
    stop: int,
    core: np.ndarray,
) -> None:
    mapped = np.memmap(
        out_path,
        dtype=np.float32,
        mode="r+",
        shape=(n_samples, n_channels),
        order="C",
    )
    mapped[start:stop] = core
    mapped.flush()
    del mapped


def _raise_if_cancelled(cancel: CancelToken | None) -> None:
    if cancel is not None and cancel.cancelled():
        raise Cancelled("Filtering cancelled.")


def _report(progress: Progress | None, fraction: float, message: str) -> None:
    if progress is not None:
        progress(float(fraction), message)
