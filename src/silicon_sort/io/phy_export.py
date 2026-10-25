"""Write a Phy folder and the pipeline settings used to produce it."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from silicon_sort.settings import PipelineSettings, save_settings


PHY_FILES = (
    "spike_times.npy",
    "spike_clusters.npy",
    "spike_templates.npy",
    "templates.npy",
    "amplitudes.npy",
    "channel_map.npy",
    "channel_positions.npy",
    "params.py",
    "cluster_info.tsv",
)


def export_phy(
    output_dir: str | Path,
    settings: PipelineSettings,
    times: np.ndarray,
    labels: np.ndarray,
    amplitudes: np.ndarray,
    templates: np.ndarray,
    positions: np.ndarray,
    metrics: list[dict],
) -> Path:
    """Write the Phy arrays. ``labels`` of -1 are dropped. Times stay in samples."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    keep = labels >= 0
    times = np.asarray(times, dtype=np.int64)[keep]
    labels = np.asarray(labels, dtype=np.int32)[keep]
    amplitudes = np.asarray(amplitudes, dtype=np.float32)[keep]
    order = np.argsort(times, kind="mergesort")
    times = times[order]
    labels = labels[order]
    amplitudes = np.abs(amplitudes[order])
    n_channels = int(positions.shape[0])
    n_samples = int(templates.shape[1]) if templates.ndim == 3 else 0
    if templates.size == 0:
        templates = np.zeros((0, n_samples, n_channels), dtype=np.float32)
    templates = np.asarray(templates, dtype=np.float32)
    np.save(output / "spike_times.npy", times)
    np.save(output / "spike_clusters.npy", labels)
    np.save(output / "spike_templates.npy", labels.copy())
    np.save(output / "templates.npy", templates)
    np.save(output / "amplitudes.npy", amplitudes.reshape(-1, 1))
    np.save(output / "channel_map.npy", np.arange(n_channels, dtype=np.int32))
    np.save(output / "channel_positions.npy", np.asarray(positions, dtype=np.float32))
    (output / "params.py").write_text(_params_py(settings, n_channels), encoding="utf-8")
    (output / "cluster_info.tsv").write_text(_cluster_info(metrics), encoding="utf-8")
    save_settings(settings, output / "pipeline_settings.json")
    return output


def _params_py(settings: PipelineSettings, n_channels: int) -> str:
    dat_path = str(Path(settings.dat_path).resolve()) if settings.dat_path else ""
    lines = [
        f"dat_path = {dat_path!r}",
        f"n_channels_dat = {n_channels}",
        f"dtype = {settings.resolved_dtype().name!r}",
        f"offset = {int(settings.file_offset)}",
        f"sample_rate = {float(settings.sampling_rate or 0.0)}",
        "hp_filtered = False",
        "",
    ]
    return "\n".join(lines)


def _cluster_info(metrics: list[dict]) -> str:
    header = [
        "cluster_id",
        "Amplitude",
        "ContamPct",
        "KSLabel",
        "amp",
        "ch",
        "depth",
        "fr",
        "n_spikes",
        "snr",
        "isi_violation_rate",
        "waveform_width_ms",
    ]
    rows = ["\t".join(header)]
    for row in metrics:
        values = [
            str(int(row["cluster_id"])),
            f"{float(row['amplitude']):.6g}",
            f"{float(row['contam_pct']):.6g}",
            str(row["quality"]),
            f"{float(row['amplitude']):.6g}",
            str(int(row["peak_channel"])),
            f"{float(row['depth_um']):.6g}",
            f"{float(row['firing_rate']):.6g}",
            str(int(row["n_spikes"])),
            f"{float(row['snr']):.6g}",
            f"{float(row['isi_violation_rate']):.6g}",
            f"{float(row['waveform_width_ms']):.6g}",
        ]
        rows.append("\t".join(values))
    return "\n".join(rows) + "\n"
