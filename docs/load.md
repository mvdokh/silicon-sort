# Load

The first stage is two steps in the parameters panel: **Preprocess**, then **Mask**. The traces and the probe stay on screen while you switch.

Choosing a `.dat` opens it immediately. You do not press a separate show button.

## Recording

| Control | What it sets |
| --- | --- |
| Recording | Headerless interleaved binary, samples by channels, C order. |
| Geometry | Probe file. Empty until you choose one. |
| Channels | Number of channels stored in the file. Default 32. |
| Sampling rate | Hertz. Default 30000. |
| Dtype | `int16`, `int32`, `uint16`, `float32`, or `float64`. Intan amplifier files are int16. |
| µV / bit | Scale from integer samples to microvolts. RHD2132 is 0.195. |
| Offset | Bytes to skip before the first sample. |

A sidecar named `yourfile.dat.json`, if it sits next to the binary, fills these fields when you browse.

### Channel order

With no geometry, the display order is the file order. Channel 1 is the first row of that order and channel 32 is the last. Loading a `.cfg` replaces that order with electrode depth: the site with the largest `y` is at the top of the stack. Sites missing from the cfg stay in the recording but are not drawn as present probe contacts.

## Filter

Preprocessing is **off** when you open the GUI. Detection, review, and clustering then read the raw file scaled to microvolts. A headless `silicon-sort run` still preprocesses unless the settings JSON says otherwise.

| Control | What it sets |
| --- | --- |
| Preprocessing | `off` uses the raw recording. `on` bandpasses and references into the cache, and later stages read that cache. |
| Highpass Hz | Butterworth highpass. Default 300. Used only when preprocessing is on. |
| Lowpass Hz | Butterworth lowpass. Default 6000. |
| Reference | `global` subtracts one median across channels. `shank` splits on gaps in `x` wider than the shank gap (150 µm). `local` groups sites inside the local radius. |
| Local radius | Micrometers for local-group referencing. Default 100. |
| Whiten | `on` estimates a spatial ZCA matrix and applies it during detection. The whitened signal is not written as a second file. |

**Run preprocessing** builds or reuses the filtered cache. It does nothing useful while preprocessing is off, because detection will not read the cache.

The filter is a 3rd-order Butterworth, applied zero-phase in overlapping chunks. Masked channels are excluded from the median and written as zeros.

## Bad channels

| Control | What it does |
| --- | --- |
| Suggest bad channels | Scores a few raw chunks for flat, railed, noisy, or spectrally odd channels. Nothing is masked until you accept. |
| Accept suggestions | Adds the suggestion to the current mask. Channels you already masked stay masked. |
| Mask selected | Adds the channels selected on the traces. |
| Clear mask | Clears every masked channel. |
| Import .msk | Reads a 1-based channel list. `T` excludes that channel. |

Click a trace to select one channel. Shift-click or Ctrl-click adds or removes a channel from the selection. `M` toggles the mask on the selected channel. Clicking a probe site toggles that site directly.

The suggestion compares each channel with the global median and with neighbors inside 150 µm. It looks at RMS, the fraction of samples near the rails, high-frequency content, and lag-1 autocorrelation. A channel can be noisy next to its neighbors without being an outlier of the whole probe.

Masked channel indexes in the settings JSON are 0-based. The numbers on the traces and in the log are 1-based.
