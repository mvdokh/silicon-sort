# Autosort

Autosort clusters every peak channel, then merges and splits, then can write a Phy folder.

| Control | Default | What it does |
| --- | --- | --- |
| Merge cosine | 0.9 | Minimum cosine between two templates before neighboring sites can merge. |
| Merge footprint | 0.8 | Minimum correlation of the spatial footprints of those templates. |
| Refractory ms | 1.5 | Interval used both for the merge check and for the split check. |
| ISI split | 0.1 | A cluster is a split candidate when at least this fraction of its intervals fall inside the refractory period. |

**Run autosort** needs a detection. It reuses the cached spike index when the detection settings have not changed.

**Export Phy** asks for a folder and writes:

| File | Contents |
| --- | --- |
| `spike_times.npy` | Spike times in samples. |
| `spike_clusters.npy` | Cluster id of each spike. Ids below 0 are omitted. |
| `spike_templates.npy` | Same ids, one template per cluster. |
| `templates.npy` | Median waveforms, shaped `(n_units, n_samples, n_channels)`, in microvolts. |
| `amplitudes.npy` | Spike amplitudes in microvolts. |
| `channel_map.npy` | Channels kept in the export. |
| `channel_positions.npy` | `x`, `y` in micrometers. |
| `params.py` | Path, channel count, dtype, offset, and sample rate for Phy. |
| `cluster_info.tsv` | Per-cluster amplitude, contamination, label, channel, depth, rate, count, SNR, ISI violation rate, and waveform width. |
| `pipeline_settings.json` | The settings that produced the folder. |

`params.py` points at the raw recording and sets `hp_filtered = False`. Open the folder in Phy for curation.

## Merge

Two clusters merge only when their peak channels are different, the sites are within `feature_radius_um`, both the template cosine and the footprint correlation clear the cutoffs, and the spike trains share structure inside the refractory window. Same-site clusters are not merged here. They were already separate components of that channel's mixture.

## Split

A cluster with at least 40 spikes is split when too many inter-spike intervals fall inside the refractory period and a two-component mixture on amplitude beats a single component by BIC. Both pieces have to hold at least 10% of the spikes. After that, spikes that land on the same sample are reduced to one, and the templates are rebuilt.
