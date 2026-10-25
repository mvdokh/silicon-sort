# Clusters

This stage fits one channel so you can see whether the automatic cluster count is reasonable before autosort runs on the whole probe.

| Control | What it does |
| --- | --- |
| Channel | Peak channel to preview. The list follows the current display order. |
| Clusters | `Auto`, or a forced count from 1 to 4 for this channel only. |
| BIC maximum | Largest `k` Auto is allowed to consider. The menu runs from 1 to 6. The settings default is 4. |
| Preview channel | Fits this channel and draws the three plots. |

## What the preview shows

- Waveforms of the spikes assigned to each component, colored by cluster.
- The same spikes in the first two principal components.
- Amplitude against time.

`Auto` fits a full-covariance Gaussian mixture for `k` from 1 up to the BIC maximum, and up to one eighth of the spike count, and keeps the `k` with the lowest BIC. Forcing a count skips that choice for this channel and stores it in `channel_k` in the settings. Autosort honors that force when it clusters the same site later.

Features for a spike are a local PCA of the waveform neighborhood, the spike amplitude, and an amplitude-weighted center of mass across nearby sites. The neighborhood radius is `feature_radius_um` (100 µm). The waveform window is 0.6 ms before the peak and 1.2 ms after.

Preview does not write the Phy folder and does not merge across channels. That happens in autosort.
