<p align="center">
  <img src="spike-sorter-icon-v6.svg" alt="Spike Sorter" width="160">
</p>

# Silicon Sort

Desktop spike sorter for low-channel brainstem silicon probes. It reads a raw binary recording without loading the whole file, optionally filters it, detects spikes, clusters each site, and writes a Phy folder. Manual curation stays in Phy.

A longer description of the pipeline, every setting, and the cache is in the [documentation](https://mvdokh.github.io/silicon-sort/).

## Quick start

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[gui]"
silicon-sort gui
```

To open a session immediately:

```bash
silicon-sort gui --dat amplifier.dat --cfg electrode.cfg
```

The window fills the screen. Parameters sit on the left. The log is a short strip under that column.

## 1 Load

The parameters panel has two steps, **Preprocess** and **Mask**.

**Preprocess.** Choose the recording and the probe geometry. A `.dat` with no geometry is drawn in file order, channels 1 through 32 (or however many channels the file has). Loading the `.cfg` reorders the traces by electrode depth. Set channel count, sampling rate, dtype, microvolts per bit, and a byte offset if the file has a header. The GUI leaves **Preprocessing** off, so detection uses the raw recording in microvolts. Turn it on to bandpass and common-median reference into a cache first. Highpass, lowpass, reference (`global`, `shank`, or `local`), local radius, and whitening are on this page. **Run preprocessing** is optional and only matters when preprocessing is on.

**Mask.** Click a trace to select that channel. Shift-click or Ctrl-click adds more. **Mask selected** adds them to the mask. **Suggest bad channels** flags flat, railed, or noisy sites, and **Accept suggestions** adds those to the mask without clearing channels you already chose. **Clear mask** and **Import .msk** are here too. `M` toggles the selected channel. Clicking a site on the probe toggles that channel. Masked channels are left out of detection.

For an Intan RHD2132 / Neuronexus A1x32-Poly2 recording the usual values are 32 channels, 30000 Hz, int16, and 0.195 µV/bit. `electrode.cfg` is a name line, then `site channel x y` in micrometers, with channel numbers starting at 1.

## 2 Detect

**Threshold MAD** is the detection line in noise units. Negative values look for downward spikes. The default is −5, about five robust standard deviations below zero. **Neighbor radius**, **Neighbor fraction**, and **Min neighbors** require a nearby site to cross a lower threshold at the same time. **Dead time** blocks a second spike on the same site for that many milliseconds. **Artifact fraction** and **Artifact MAD** mark coincident deflections, such as movement, and keep them out of clustering. **Template match** is an optional second pass: each channel's median waveform is scanned at 60% of the threshold, and new spikes are added only when they match that shape. It does not remove spikes from the first pass.

After detection, the trace shows a box over each spike for the full height of its channel. The panel on the right is the waveform review, the amplitude histogram, and the benchmark plot.

**Suggest** builds a template from the spikes already found on the current channel. Detections that do not match it are listed as suggested artifacts. A lower-threshold scan of the same template lists likely misses. Mark each one **Good** or **Bad**, then **Detect again**. Bad marks drop that spike and other spikes on the channel with nearly the same shape. Good marks are kept, and similar missed spikes are searched for. Turn on **Click a trace to add a missed spike** and click the trace; the mark snaps to that channel's template so the click does not have to land on the trough.

**Run benchmark** plants known waveforms into a short copy of the noise and draws precision and recall against threshold. Each run stays on the plot. The title is the precision at your current threshold and how it moved from the previous run.

## 3 Clusters

Pick a peak channel. **Clusters** is Auto, or a forced count from 1 to 4. **BIC maximum** is the largest count Auto may choose. **Preview channel** fits that site only and shows waveforms, a PCA scatter, and amplitude against time. Use this to decide the count before autosort.

## 4 Autosort

Clusters every site, then merges neighbors whose templates and spatial footprints agree and whose spike times share structure. **Merge cosine** and **Merge footprint** are those cutoffs. **Refractory ms** and **ISI split** split a cluster that breaks the refractory period and is bimodal in amplitude. **Export Phy** writes the folder Phy reads. Times stay in samples. Templates are in microvolts.

## Moving around the traces

Drag the traces up or down to move through channels. Shift+scroll does the same. The horizontal bar, the mouse wheel, and the left and right arrows move through time. Ctrl+scroll, or `[` and `]`, changes the time window. **Window** is how much time is on screen. **µV / channel** is the vertical distance between baselines. **Spacing** spreads the channels apart: 1 fits every channel, and a higher value shows fewer rows while keeping the channel that was at the top of the view at the top.

## Headless run

Save settings from the File menu, then:

```bash
silicon-sort run pipeline_settings.json --output phy_out
```

A headless run preprocesses by default. The GUI does not, until you turn preprocessing on. The full settings list is in the [command-line docs](https://mvdokh.github.io/silicon-sort/cli/).

## Tests

```bash
pytest
```
