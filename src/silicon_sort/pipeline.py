"""Headless stage runner shared by the CLI and the GUI."""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from silicon_sort.cancel import CancelToken
from silicon_sort.cluster.gmm import ChannelPreview, cluster_all_sites, cluster_channel
from silicon_sort.cluster.merge import assemble_labels, merge_and_split
from silicon_sort.cluster.metrics import compute_metrics
from silicon_sort.detect.detect import SpikeIndex, detect_traces, empty_index
from silicon_sort.detect.review import refine_with_review, upsert_label
from silicon_sort.errors import Cancelled, RecordingError
from silicon_sort.io.binary import Recording, ScaledView, apply_sidecar, load_sidecar, open_recording, save_sidecar
from silicon_sort.io.cache import SessionCache, cache_dir_for, hash_payload
from silicon_sort.io.geometry import Probe, probe_from_cfg
from silicon_sort.io.phy_export import export_phy
from silicon_sort.preprocess.filter import (
    design_bandpass,
    filter_overlap_samples,
    filter_recording_to_memmap,
)
from silicon_sort.preprocess.reference import reference_groups
from silicon_sort.settings import PipelineSettings, save_settings

Progress = Callable[[float, str], None]


@dataclass
class SortResult:
    times: np.ndarray
    channels: np.ndarray
    amplitudes: np.ndarray
    labels: np.ndarray
    templates: np.ndarray
    metrics: list[dict[str, Any]]
    noise_mad: np.ndarray


class Pipeline:
    """Run preprocess, detection, autosort, and Phy export with a disk cache."""

    def __init__(self, settings: PipelineSettings):
        self.settings = settings
        self.recording: Recording | None = None
        self.probe: Probe | None = None
        self.spikes: SpikeIndex | None = None
        self.noise_mad: np.ndarray | None = None
        self.whitener: np.ndarray | None = None
        self.result: SortResult | None = None
        self.review_times = np.zeros(0, dtype=np.int64)
        self.review_channels = np.zeros(0, dtype=np.int32)
        self.review_labels = np.zeros(0, dtype=np.int8)
        self.review_bad = "channel"
        self.review_good = "channel"

    def label_spike(self, time: int, channel: int, label: int) -> None:
        """Remember one event as good (1), bad (-1), or clear it (0)."""
        recording = self.recording
        rate = float(self.settings.sampling_rate or (recording.sampling_rate if recording is not None else 30000.0))
        tolerance = max(1, int(round(0.0004 * rate)))
        self.review_times, self.review_channels, self.review_labels = upsert_label(
            self.review_times,
            self.review_channels,
            self.review_labels,
            int(time),
            int(channel),
            int(label),
            tolerance,
        )

    def cache(self) -> SessionCache:
        if not self.settings.dat_path:
            raise RecordingError("No recording path is set.")
        return SessionCache(cache_dir_for(self.settings.dat_path))

    def ensure_meta(self) -> None:
        """Load a sidecar when the settings JSON does not already carry recording metadata."""
        if self.settings.n_channels and self.settings.sampling_rate and self.settings.dtype:
            return
        if not self.settings.dat_path:
            raise RecordingError("No recording path is set.")
        apply_sidecar(self.settings)
        self.settings.require_recording_meta()

    def open(self) -> tuple[Recording, Probe]:
        self.ensure_meta()
        if self.recording is not None:
            self.recording.close()
        self.recording = open_recording(self.settings)
        self.probe = probe_from_cfg(
            self.settings.cfg_path,
            self.recording.n_channels,
            self.settings.shank_gap_um,
        )
        return self.recording, self.probe

    def close(self) -> None:
        if self.recording is not None:
            self.recording.close()
            self.recording = None

    def run_preprocess(
        self,
        progress: Progress | None = None,
        cancel: CancelToken | None = None,
        force: bool = False,
    ) -> bool:
        """Filter and reference the recording. Returns True when the cache was reused."""
        recording, probe = self.open()
        cache = self.cache()
        mask = self._mask(recording.n_channels)
        sha = hash_payload(self._preprocess_payload(recording, probe, mask))
        if cache.valid("preprocess", sha) and not force:
            if progress is not None:
                progress(1.0, "Preprocessing is cached")
            return True
        _check(cancel)
        sos = design_bandpass(
            recording.sampling_rate,
            self.settings.highpass_hz,
            self.settings.lowpass_hz,
            self.settings.filter_order,
        )
        overlap = filter_overlap_samples(recording.sampling_rate, self.settings.highpass_hz, sos)
        groups = reference_groups(
            probe.positions,
            mask,
            self.settings.reference_mode,
            self.settings.local_radius_um,
            self.settings.shank_gap_um,
        )
        chunk = max(1, int(round(self.settings.chunk_seconds * recording.sampling_rate)))
        mapped = cache.create_filtered(recording.n_samples, recording.n_channels)
        mapped.flush()
        del mapped
        filter_recording_to_memmap(
            dat_path=str(recording.path),
            n_samples=recording.n_samples,
            n_channels=recording.n_channels,
            dtype=recording.dtype,
            file_offset=recording.file_offset,
            scale=recording.voltage_scale_uv_per_bit,
            out_path=str(cache.filtered_path),
            sos=sos,
            overlap=overlap,
            chunk_samples=chunk,
            mask=mask,
            reference_mode=self.settings.reference_mode,
            groups=groups,
            n_workers=self.settings.n_workers,
            progress=progress,
            cancel=cancel,
        )
        cache.mark(
            "preprocess",
            sha,
            n_samples=recording.n_samples,
            n_channels=recording.n_channels,
        )
        cache.invalidate_from("detect")
        side = save_sidecar(self.settings)
        cache.recording_json.write_text(side.read_text(encoding="utf-8"), encoding="utf-8")
        save_settings(self.settings, cache.root / "pipeline_settings.json")
        if progress is not None:
            progress(1.0, "Preprocessing finished")
        return False

    def run_detect(
        self,
        progress: Progress | None = None,
        cancel: CancelToken | None = None,
        force: bool = False,
    ) -> SpikeIndex:
        if self.settings.preprocess:
            self.run_preprocess(progress=progress, cancel=cancel, force=False)
        else:
            self.open()
        recording, probe = self.recording, self.probe
        assert recording is not None and probe is not None
        cache = self.cache()
        mask = self._mask(recording.n_channels)
        signal_payload = self._signal_payload(recording, probe, mask)
        detect_payload = self._detect_payload()
        base_sha = hash_payload({"signal": signal_payload, "detect": detect_payload})
        sha = hash_payload(
            {
                "signal": signal_payload,
                "detect": detect_payload,
                "review": self._review_payload(),
            }
        )
        if cache.valid("detect", sha) and not force:
            self._load_detection(cache)
            if progress is not None:
                progress(1.0, "Detection is cached")
            assert self.spikes is not None
            return self.spikes
        _check(cancel)
        with self.open_signal() as signal:
            if cache.valid("detect_base", base_sha) and not force:
                if progress is not None:
                    progress(0.2, "Reusing the first detection")
                spikes = _spike_index(cache.spikes_base_path)
                noise_mad = np.load(cache.noise_path)
                whitener = np.load(cache.whitener_path) if cache.whitener_path.is_file() else None
            else:
                spikes, noise_mad, whitener = detect_traces(
                    signal,
                    probe,
                    self.settings,
                    mask,
                    progress=progress,
                    cancel=cancel,
                )
                _save_spikes(cache.spikes_base_path, spikes)
                cache.mark("detect_base", base_sha, n_spikes=spikes.n_spikes)
            if np.any(self.review_labels):
                if progress is not None:
                    progress(0.35, "Applying good and bad examples")
                spikes = refine_with_review(
                    signal,
                    spikes,
                    self.review_times,
                    self.review_channels,
                    self.review_labels,
                    self.settings,
                    noise_mad,
                    progress=progress,
                    cancel=cancel,
                    bad_scope=self.review_bad,
                    good_scope=self.review_good,
                )
        np.savez(
            cache.spikes_path,
            times=spikes.times,
            channels=spikes.channels,
            amplitudes=spikes.amplitudes,
            artifact=spikes.artifact,
        )
        np.save(cache.noise_path, noise_mad)
        if whitener is None:
            whitener = np.eye(recording.n_channels, dtype=np.float32)
        np.save(cache.whitener_path, whitener)
        cache.mark("detect", sha, n_spikes=spikes.n_spikes)
        cache.invalidate_from("sort")
        self.spikes = spikes
        self.noise_mad = noise_mad
        self.whitener = whitener
        if progress is not None:
            progress(1.0, f"Detected {spikes.clean().n_spikes} spikes")
        return spikes

    def preview_channel(
        self,
        channel: int,
        progress: Progress | None = None,
        cancel: CancelToken | None = None,
    ) -> ChannelPreview | None:
        self.run_detect(progress=progress, cancel=cancel)
        recording = self.recording
        probe = self.probe
        assert recording is not None and probe is not None and self.spikes is not None
        with self.open_signal() as signal:
            _clustering, preview = cluster_channel(
                signal,
                self.spikes,
                probe,
                self.settings,
                self._mask(recording.n_channels),
                int(channel),
            )
        return preview

    def run_sort(
        self,
        progress: Progress | None = None,
        cancel: CancelToken | None = None,
        force: bool = False,
    ) -> SortResult:
        self.run_detect(progress=progress, cancel=cancel, force=False)
        recording = self.recording
        probe = self.probe
        assert recording is not None and probe is not None and self.spikes is not None and self.noise_mad is not None
        cache = self.cache()
        mask = self._mask(recording.n_channels)
        sha = hash_payload(
            {
                "detect": {
                    "signal": self._signal_payload(recording, probe, mask),
                    "detect": self._detect_payload(),
                    "review": self._review_payload(),
                },
                "sort": self._sort_payload(),
            }
        )
        if cache.valid("sort", sha) and not force:
            self.result = _load_sort(cache, self.noise_mad)
            if progress is not None:
                progress(1.0, "Autosort is cached")
            return self.result
        _check(cancel)
        with self.open_signal() as signal:
            sites = cluster_all_sites(
                signal,
                self.spikes,
                probe,
                self.settings,
                mask,
                progress=progress,
            )
            times, channels, amplitudes, labels = assemble_labels(sites)
            _check(cancel)
            labels, templates = merge_and_split(
                times,
                channels,
                amplitudes,
                labels,
                signal,
                probe,
                self.settings,
            )
        if times.size:
            order = np.argsort(times, kind="mergesort")
            times = times[order]
            channels = channels[order]
            amplitudes = amplitudes[order]
            labels = labels[order]
        metrics = compute_metrics(
            times,
            channels,
            amplitudes,
            labels,
            templates,
            self.noise_mad,
            probe.positions,
            recording.sampling_rate,
            recording.n_samples,
            self.settings.refractory_ms,
        )
        result = SortResult(
            times=times.astype(np.int64),
            channels=channels.astype(np.int32),
            amplitudes=amplitudes.astype(np.float32),
            labels=labels.astype(np.int32),
            templates=np.asarray(templates, dtype=np.float32),
            metrics=metrics,
            noise_mad=np.asarray(self.noise_mad, dtype=np.float64),
        )
        np.savez(
            cache.sort_path,
            times=result.times,
            channels=result.channels,
            amplitudes=result.amplitudes,
            labels=result.labels,
            templates=result.templates,
        )
        cache.metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        cache.mark("sort", sha, n_clusters=len(metrics))
        save_settings(self.settings, cache.root / "pipeline_settings.json")
        self.result = result
        if progress is not None:
            progress(1.0, f"Autosort kept {len(metrics)} clusters")
        return result

    def run_export(
        self,
        output_dir: str | Path | None = None,
        progress: Progress | None = None,
        cancel: CancelToken | None = None,
    ) -> Path:
        result = self.run_sort(progress=progress, cancel=cancel)
        probe = self.probe
        assert probe is not None
        target = Path(output_dir or self.settings.output_dir or (self.cache().root / "phy"))
        export_phy(
            target,
            self.settings,
            result.times,
            result.labels,
            result.amplitudes,
            result.templates,
            probe.positions,
            result.metrics,
        )
        if progress is not None:
            progress(1.0, f"Wrote Phy folder {target}")
        return target

    def run_all(
        self,
        output_dir: str | Path | None = None,
        progress: Progress | None = None,
        cancel: CancelToken | None = None,
    ) -> Path:
        return self.run_export(output_dir=output_dir, progress=progress, cancel=cancel)

    def _mask(self, n_channels: int) -> np.ndarray:
        mask = self.settings.mask_array(n_channels)
        if self.probe is not None:
            mask = mask | ~self.probe.present
        return mask

    @contextmanager
    def open_signal(self):
        """Yield the trace detection and clustering should read.

        Preprocessing on yields the cached filtered memmap. Off yields the raw
        file scaled to microvolts, without bandpass or referencing.
        """
        recording = self.recording
        assert recording is not None
        if self.settings.preprocess:
            mapped = self.cache().open_filtered(recording.n_samples, recording.n_channels, mode="r")
            try:
                yield mapped
            finally:
                del mapped
        else:
            yield ScaledView(recording.memmap(), recording.voltage_scale_uv_per_bit)

    def _signal_payload(self, recording: Recording, probe: Probe, mask: np.ndarray) -> dict[str, Any]:
        if not self.settings.preprocess:
            stat = recording.path.stat()
            return {
                "preprocess": False,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "n_channels": recording.n_channels,
                "sampling_rate": recording.sampling_rate,
                "dtype": recording.dtype.str,
                "scale": recording.voltage_scale_uv_per_bit,
                "offset": recording.file_offset,
                "mask": np.flatnonzero(mask).tolist(),
            }
        payload = self._preprocess_payload(recording, probe, mask)
        payload["preprocess"] = True
        return payload

    def _preprocess_payload(self, recording: Recording, probe: Probe, mask: np.ndarray) -> dict[str, Any]:
        stat = recording.path.stat()
        return {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "n_channels": recording.n_channels,
            "sampling_rate": recording.sampling_rate,
            "dtype": recording.dtype.str,
            "scale": recording.voltage_scale_uv_per_bit,
            "offset": recording.file_offset,
            "mask": np.flatnonzero(mask).tolist(),
            "highpass_hz": self.settings.highpass_hz,
            "lowpass_hz": self.settings.lowpass_hz,
            "filter_order": self.settings.filter_order,
            "reference_mode": self.settings.reference_mode,
            "local_radius_um": self.settings.local_radius_um,
            "shank_gap_um": self.settings.shank_gap_um,
            "chunk_seconds": self.settings.chunk_seconds,
            "positions": np.round(probe.positions, 3).tolist(),
        }

    def _review_payload(self) -> dict[str, Any]:
        marks: list[list[int]] = []
        if self.review_labels.size:
            order = np.lexsort((self.review_times, self.review_channels, self.review_labels))
            marks = [
                [int(self.review_times[index]), int(self.review_channels[index]), int(self.review_labels[index])]
                for index in order
            ]
        return {"marks": marks, "bad": self.review_bad, "good": self.review_good}

    def _detect_payload(self) -> dict[str, Any]:
        keys = (
            "whiten",
            "mad_chunks",
            "whiten_chunks",
            "rng_seed",
            "threshold_mad",
            "neighbor_radius_um",
            "neighbor_support_fraction",
            "neighbor_window_ms",
            "min_neighbors",
            "dead_time_ms",
            "align_ms",
            "glitch_energy_fraction",
            "biphasic_fraction",
            "biphasic_window_ms",
            "artifact_fraction",
            "artifact_mad",
            "artifact_coincidence_ms",
            "artifact_blank_ms",
            "template_match",
            "template_threshold_fraction",
            "template_min_cosine",
            "waveform_pre_ms",
            "waveform_post_ms",
        )
        return {key: getattr(self.settings, key) for key in keys}

    def _sort_payload(self) -> dict[str, Any]:
        keys = (
            "n_pca",
            "feature_radius_um",
            "waveform_pre_ms",
            "waveform_post_ms",
            "k_max",
            "channel_k",
            "merge_cosine",
            "merge_footprint",
            "refractory_ms",
            "isi_split_fraction",
            "rng_seed",
        )
        return {key: getattr(self.settings, key) for key in keys}

    def _load_detection(self, cache: SessionCache) -> None:
        self.spikes = _spike_index(cache.spikes_path)
        self.noise_mad = np.load(cache.noise_path)
        if cache.whitener_path.is_file():
            self.whitener = np.load(cache.whitener_path)
        else:
            self.whitener = None


def _save_spikes(path: Path, spikes: SpikeIndex) -> None:
    np.savez(
        path,
        times=spikes.times,
        channels=spikes.channels,
        amplitudes=spikes.amplitudes,
        artifact=spikes.artifact,
    )


def _spike_index(path: Path) -> SpikeIndex:
    payload = np.load(path)
    return SpikeIndex(
        payload["times"].astype(np.int64),
        payload["channels"].astype(np.int32),
        payload["amplitudes"].astype(np.float32),
        payload["artifact"].astype(bool),
    )


def resolve_settings(settings: PipelineSettings) -> PipelineSettings:
    """Fill recording metadata from the sidecar. Refuse to run when it is still incomplete."""
    if settings.n_channels and settings.sampling_rate and settings.dtype:
        settings.require_recording_meta()
        return settings
    if not settings.dat_path or not Path(settings.dat_path).is_file():
        raise RecordingError("Settings are missing the recording path.")
    try:
        payload = load_sidecar(settings.dat_path)
    except RecordingError as exc:
        raise RecordingError(
            "Headless run needs n_channels, sampling_rate, and dtype in the settings JSON or the sidecar."
        ) from exc
    settings.n_channels = int(payload["n_channels"])
    settings.sampling_rate = float(payload["sampling_rate"])
    settings.dtype = str(payload["dtype"])
    settings.voltage_scale_uv_per_bit = float(payload.get("voltage_scale_uv_per_bit", settings.voltage_scale_uv_per_bit))
    settings.file_offset = int(payload.get("file_offset", settings.file_offset))
    if not settings.mask and payload.get("mask"):
        settings.mask = [int(v) for v in payload["mask"]]
    settings.require_recording_meta()
    return settings


def _load_sort(cache: SessionCache, noise_mad: np.ndarray) -> SortResult:
    payload = np.load(cache.sort_path)
    metrics = json.loads(cache.metrics_path.read_text(encoding="utf-8"))
    return SortResult(
        times=payload["times"].astype(np.int64),
        channels=payload["channels"].astype(np.int32),
        amplitudes=payload["amplitudes"].astype(np.float32),
        labels=payload["labels"].astype(np.int32),
        templates=payload["templates"].astype(np.float32),
        metrics=metrics,
        noise_mad=np.asarray(noise_mad, dtype=np.float64),
    )


def _check(cancel: CancelToken | None) -> None:
    if cancel is not None and cancel.cancelled():
        raise Cancelled("Cancelled.")


def empty_spikes() -> SpikeIndex:
    return empty_index()
