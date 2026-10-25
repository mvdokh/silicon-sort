# Command line

```bash
silicon-sort gui [--dat PATH] [--cfg PATH]
silicon-sort run SETTINGS.json [--output DIR]
```

`gui` opens the window. `--dat` and `--cfg` preload those paths. If a recording is given, the traces are shown as soon as the window is up.

`run` loads a settings JSON, runs preprocess (if enabled), detection, clustering, merge, and Phy export, and prints the output directory. `--output` overrides `output_dir` in the file. Progress goes to stderr. The run refuses to start when channel count, sampling rate, or dtype is missing from both the JSON and the sidecar.

Save a JSON from the File menu after a GUI session, or write one. Unknown keys are ignored. Missing keys keep the defaults below.

## Settings

| Key | Default | Meaning |
| --- | --- | --- |
| `dat_path` | `""` | Recording path. |
| `cfg_path` | `""` | Geometry path. Required for a full run, because detection needs the probe. |
| `output_dir` | `""` | Phy folder. |
| `n_channels` | unset | Channel count in the binary. |
| `sampling_rate` | unset | Hertz. |
| `dtype` | `int16` | Sample type. |
| `voltage_scale_uv_per_bit` | `0.195` | Microvolts per integer step. |
| `file_offset` | `0` | Bytes skipped before sample 0. |
| `mask` | `[]` | 0-based channels excluded from detection and from the reference. |
| `preprocess` | `true` | Bandpass and reference before detection. The GUI turns this off for a new window. |
| `highpass_hz` | `300` | Highpass cutoff. |
| `lowpass_hz` | `6000` | Lowpass cutoff. |
| `filter_order` | `3` | Butterworth order. |
| `reference_mode` | `global` | `global`, `shank`, or `local`. |
| `local_radius_um` | `100` | Radius for local referencing. |
| `shank_gap_um` | `150` | Gap in `x` that starts a new shank. |
| `whiten` | `true` | ZCA during detection. Not written as a filtered file. |
| `chunk_seconds` | `1` | Chunk length for filter and detection. |
| `n_workers` | `0` | Process count. 0 uses every CPU. |
| `mad_chunks` | `12` | Chunks used to estimate MAD. |
| `whiten_chunks` | `12` | Chunks used to estimate the whitener. |
| `threshold_mad` | `-5` | Detection threshold. Sign is polarity. |
| `neighbor_radius_um` | `60` | Neighbor search radius. |
| `neighbor_support_fraction` | `0.5` | Neighbor threshold as a fraction of the detection threshold. |
| `neighbor_window_ms` | `0.3` | Time window for a neighbor crossing. |
| `min_neighbors` | `1` | Required agreeing neighbors. |
| `dead_time_ms` | `0.7` | Per-site refractory used at detection. |
| `align_ms` | `0.4` | Search around a crossing for the peak. |
| `glitch_energy_fraction` | `0.8` | Drop spikes dominated by one sample. |
| `biphasic_fraction` | `0.2` | Minimum opposite lobe, relative to the peak. |
| `biphasic_window_ms` | `1` | Where that lobe is measured. |
| `artifact_fraction` | `0.4` | Fraction of channels in a coincident artifact. |
| `artifact_mad` | `8` | Size of that coincident event. |
| `artifact_coincidence_ms` | `0.5` | Coincidence window. |
| `artifact_blank_ms` | `1` | Blank after an artifact. |
| `template_match` | `false` | Second-pass template scan. |
| `template_threshold_fraction` | `0.6` | Threshold used by that scan, relative to `threshold_mad`. |
| `template_min_cosine` | `0.8` | Shape match required to add a spike. |
| `n_pca` | `3` | Principal components kept per neighborhood. |
| `feature_radius_um` | `100` | Sites included in a spike's features, and the merge radius. |
| `waveform_pre_ms` | `0.6` | Waveform samples before the peak. |
| `waveform_post_ms` | `1.2` | Waveform samples after the peak. |
| `k_max` | `4` | Largest mixture size BIC may choose. |
| `channel_k` | `{}` | Forced `k` per channel. Keys are channel indexes as strings. `0` means Auto. |
| `merge_cosine` | `0.9` | Template cosine required to merge neighbors. |
| `merge_footprint` | `0.8` | Footprint correlation required to merge. |
| `refractory_ms` | `1.5` | Refractory interval for merge and split. |
| `isi_split_fraction` | `0.1` | Fraction of refractory violations that makes a cluster a split candidate. |
| `rng_seed` | `0` | Seed for chunk sampling and mixture fits. |

## View keys

These are window shortcuts, not settings.

| Key | Action |
| --- | --- |
| Left / right | Scroll time by half a window. |
| Up / down | Move one channel toward the top or the bottom of the stack. |
| `[` / `]` | Zoom the time window. |
| `M` | Toggle the mask on the selected channel. |
| Drag up / down | Move through channels. Shift+scroll does the same. |
| Wheel | Scroll time. Ctrl+wheel zooms time. |

## Tests

```bash
pytest
```

The suite uses synthetic recordings. It does not need the sample session on disk.
