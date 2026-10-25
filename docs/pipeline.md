# Pipeline

The GUI and `silicon-sort run` call the same pipeline. Stages are hashed. When the inputs of a stage have not changed, the cache is reused and later stages are not repeated.

```mermaid
flowchart LR
  dat[BinaryFile] --> open[OpenMemmap]
  open --> pre[OptionalPreprocess]
  pre --> det[Detect]
  det --> rev[ApplyReviewMarks]
  rev --> clu[ClusterEachChannel]
  clu --> merge[MergeAndSplit]
  merge --> phy[PhyFolder]
```

## Reading the file

The binary is a memory map of shape `(n_samples, n_channels)`, C order, little-endian. Integer files are scaled to microvolts on read by `voltage_scale_uv_per_bit`. A read copies only the requested window.

If `n_channels`, `sampling_rate`, and `dtype` are missing from the settings, the pipeline looks for `yourfile.dat.json` beside the recording.

The cache directory is `<stem>.silicon-sort` next to the binary. It holds `filtered.dat` when preprocessing has been run, `spikes.npz`, `noise_mad.npy`, `whitener.npy`, a sort archive, and `manifest.json`. The manifest stores a hash per stage. Changing the recording bytes, the mask, the filter, or the threshold invalidates detection. Changing only the merge cutoffs reuses the spike index.

## Preprocess

The dataclass default is `preprocess: true`, so a headless run filters. The GUI sets it to false on a new window, so scrolling and detection stay on the raw microvolt view until you opt in.

When preprocessing is on:

1. Design a 3rd-order Butterworth bandpass and apply it with `sosfiltfilt` in chunks. Chunks overlap so the edges match.
2. Subtract a common median. `global` uses every unmasked channel. `shank` splits columns of contacts whose `x` differs by more than `shank_gap_um`. `local` uses groups inside `local_radius_um`.
3. Write float32 microvolts to `filtered.dat`. Masked channels are zero.

Whitening is separate. A ZCA matrix is estimated from random chunks and multiplied in during detection and clustering. It is stored as `whitener.npy` and is not a second full-length file. With whitening off, that matrix is the identity.

## Detection

Noise scale per channel:

```
MAD = median(|x|) / 0.6745
```

The constant converts a median absolute deviation into a Gaussian standard deviation. The estimate uses `mad_chunks` random chunks so a few spikes do not pull the threshold up. Masked channels get an infinite value.

The threshold is `threshold_mad * MAD`. The sign of `threshold_mad` is the polarity: negative looks for troughs. Each crossing is aligned to the extreme inside `align_ms`. A spike is kept when at least `min_neighbors` contacts inside `neighbor_radius_um` also cross `neighbor_support_fraction` of the threshold inside `neighbor_window_ms`. Sites with an empty neighborhood are kept. The same site then waits `dead_time_ms`.

Single-sample glitches are dropped when too much of the waveform energy sits in one sample (`glitch_energy_fraction`). A spike with no opposite lobe inside `biphasic_window_ms` can be dropped when that lobe is smaller than `biphasic_fraction` of the peak.

An artifact is a sample where at least `artifact_fraction` of channels exceed `artifact_mad` times their MAD inside `artifact_coincidence_ms`. Those events are blanked for `artifact_blank_ms` and excluded from clustering. They can still be reviewed.

If `template_match` is on, a median template is built per channel from the spikes just found and scanned at a lower threshold. New events must also match the template shape. See [Detect](detect.md).

## Review marks

Marks live with the session, not in the spike file. A bad mark removes detections within 0.4 ms on that channel, and any other spike on the channel whose waveform cosine to the bad median is at least 0.9, unless that spike is as close to a good template. A good mark is inserted if detection missed it, and the good template is scanned at the template-match fraction to recover similar spikes.

The hash of the detection stage includes the marks, so **Detect again** does not reuse a spike index that ignored them.

## Clustering

Each peak channel is clustered on its own. The feature vector is a PCA of the waveforms on the channels inside `feature_radius_um`, plus amplitude and a center of mass. A full-covariance Gaussian mixture is fit for each `k` from 1 to `k_max` (also capped by the spike count). The lowest BIC wins, unless `channel_k` forces a count for that site.

Autosort then walks pairs of clusters on different sites. They merge when the sites are close enough, template cosine and footprint correlation both pass, and the two spike trains share structure on the refractory scale. A cluster that violates `refractory_ms` on at least `isi_split_fraction` of its intervals, and whose amplitudes prefer two components, is split. Details and the Phy files are in [Autosort](autosort.md).

## Workers

`n_workers` of 0 means all CPUs. The GUI and the tests use one worker where a process pool would be spawned from a thread. Filtering chunks can use a process pool; that pool is not started from a daemon thread. Progress callbacks update the bar. Cancel stops the stage at the next chunk.
