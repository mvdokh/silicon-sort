# Silicon Sort

Desktop spike sorter for low-channel brainstem silicon probes. It bandpasses and references a raw binary recording in chunks, detects spikes with a spatial neighborhood rule, clusters each site with a Gaussian mixture, and writes a Phy folder. Curation stays in Phy.

The heavy work runs off the GUI thread. Filtered data and the spike index are cached beside the recording, so later stages do not repeat earlier ones. Nothing in the pipeline loads the whole recording into memory.

## Install

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[gui,dev]"
```

The sorter runs on CPU. A CUDA device is used for the whitening multiply only when CuPy imports and sees a GPU. This Mac has no CUDA, so that path stays unused.

## Open the app

```bash
silicon-sort gui
```

The window has four stages, left to right:

1. **Load.** Choose the raw `.dat` and the probe `.cfg`. Set channel count, sampling rate, dtype, and microvolts per bit. The app remembers those in `yourfile.dat.json`. Show the probe and scroll the raw traces. Mask bad channels by clicking a site or a trace, or press `M`. Suggest bad channels, then accept or override the suggestion. Run preprocessing.
2. **Detect.** Set the threshold, neighborhood, dead time, and artifact rule. Run detection. The plots show the filtered traces, an amplitude histogram, and firing rate over time. Run benchmark plants known waveforms into a short copy of the noise and draws precision and recall against threshold.
3. **Clusters.** Pick a channel and a cluster count. Auto uses BIC. Preview shows waveforms, a PCA scatter, and amplitude over time.
4. **Autosort.** Cluster every site from those settings, merge neighbors whose templates and correlograms match, split refractory violations that are bimodal, and export Phy.

Left and right arrows scroll the traces. `[` and `]` change the time window. Parameters live in a dock you can move.

## Sample session

The Intan session this was built against is a Neuronexus A1x32-Poly2:

| Field | Value |
| --- | --- |
| File | `amplifier.dat` (raw wideband, not the already-whitened `amplifier_cmr_zca.dat`) |
| Channels | 32 |
| Sampling rate | 30000 Hz |
| Dtype | int16 |
| Scale | 0.195 µV/bit |
| Geometry | `electrode.cfg`, 1-based `site channel x y` |

Defaults in the GUI match that file: bandpass 300–6000 Hz, global common median reference, whitening on, threshold −5 MAD, neighbor radius 60 µm.

`electrode.cfg` looks like this:

```
poly2
1 1 21.65 25
2 2 21.65 725
```

Channel numbers start at 1 and follow the order of samples in the binary file. A line with three numbers is `channel x y`.

The filtered cache is float32, about twice the size of an int16 recording. The 3.4 GB example needs about 6.8 GB next to the file, in a folder named `amplifier.silicon-sort`.

## Headless run

Save settings from the File menu, or write the same JSON yourself. A run refuses to start when channel count, sampling rate, or dtype is missing from both the JSON and the sidecar.

```bash
silicon-sort run pipeline_settings.json --output phy_out
```

The Phy folder contains `spike_times.npy`, `spike_clusters.npy`, `spike_templates.npy`, `templates.npy`, `amplitudes.npy`, `channel_map.npy`, `channel_positions.npy`, `params.py`, `cluster_info.tsv`, and `pipeline_settings.json`. Times are in samples. Templates are in microvolts. `params.py` points at the raw file.

## Tests

```bash
pytest
```

Tests use synthetic memmaps. They cover the geometry parser, chunked filtering, chunk-boundary detection, artifact blanking, waveform windows, clustering, and the Phy files.

## How a recording moves through the stages

Preprocessing writes one float32 memmap: zero-phase Butterworth bandpass, then a common median reference (global, per shank, or per local group). Masked channels are left out of the median and set to zero. Whitening is a spatial ZCA matrix estimated on random chunks and applied while detecting, so it does not get a second full-file copy.

Detection thresholds the whitened trace, asks for a neighbor inside the radius, aligns to the peak, enforces a per-site dead time, and drops single-sample glitches and waveforms with no opposite lobe. A coincident deflection on many channels is flagged as an artifact and kept out of clustering. An optional second pass matches the median waveform of each channel to pick up smaller spikes.

Clustering is per peak channel. Features are a local PCA of the waveform neighborhood, the amplitude, and an amplitude-weighted center of mass. A Gaussian mixture is fit for k = 1..4, and BIC picks k unless you force it. Clusters on neighboring sites merge when their templates and footprints agree and their cross-correlogram shows a shared refractory dip or the same spikes counted twice. A cluster that breaks the refractory period and is bimodal in amplitude is split.

Changing only the cluster count or the merge settings reuses the filtered file and the spike index.
