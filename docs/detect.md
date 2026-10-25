# Detect

Detection thresholds each channel in units of its own noise, asks a neighbor to agree, and writes a spike index. The view is the traces, a waveform review, an amplitude histogram for the selected channel, and a precision/recall plot.

## Parameters

| Control | Default | What it does |
| --- | --- | --- |
| Threshold MAD | −5 | Detection line in robust noise units. The sign chooses polarity. Negative detects downward spikes, which matches Intan. |
| Neighbor radius µm | 60 | A contact inside this radius can support the spike. |
| Neighbor fraction | 0.5 | The neighbor must cross this fraction of the detection threshold. |
| Min neighbors | 1 | How many neighbors must agree. A site with no neighbors is kept. |
| Dead time ms | 0.7 | After a spike, that site cannot emit another until this interval has passed. |
| Artifact fraction | 0.4 | Fraction of channels that must cross together before the sample is movement or EMG. |
| Artifact MAD | 8 | How large that coincident event must be, in the same MAD units. |
| Template match | off | Second pass. See below. |

**Run detection** uses the raw recording when preprocessing is off, and the filtered cache when it is on. Good and bad marks from the review are applied on that run.

### What MAD means

For each channel the noise scale is

```
MAD = median(|voltage|) / 0.6745
```

estimated on a handful of chunks, not on the whole file. Dividing by 0.6745 puts the median absolute value on the standard-deviation scale of Gaussian noise, without letting large spikes inflate it. A threshold of −5 is a crossing at −5 times that scale. Masked channels are given an infinite MAD so they never cross.

### Template match

Turn this on for a second pass after the threshold detector.

For every channel that already has at least five spikes, the pass builds one template: the median waveform of those spikes, using at most 300 of them. Artifact-flagged spikes and masked channels are left out. The template is then correlated along the channel.

A new spike is added only when all of these hold:

- Its shape matches the template with cosine at least 0.8 (`template_min_cosine`).
- Its trough reaches 60% of the detection threshold (`template_threshold_fraction`). At −5 MAD that is about −3 MAD.
- It is outside the dead time of a spike the first pass already found.

Spikes from the first pass stay. This pass only adds matches, and it is slower because it scans each eligible channel.

## Review

The waveform panel draws every spike on the current channel, not a short window around the selection. The selected waveform is white. Good marks are green and bad marks are red.

| Control | What it does |
| --- | --- |
| Previous / next channel | Moves in display order and centers the trace on that channel's first event. |
| Previous / next event | Moves the waveform and centers the trace on that time. The spike on the trace is gold. |
| Suggest | Scores the current channel. |
| Suggested artifacts | Detections that do not match the channel template, plus events already flagged as artifacts. |
| Suggested misses | Times where the same template matches below the detection threshold and no spike was kept. |
| Good | Keep this spike. On the next detection, similar missed shapes are searched for. |
| Bad | Drop this spike. Other spikes on this channel with nearly the same shape (cosine at least 0.9 to the bad median) are dropped too, unless they are closer to a good template. |
| Clear | Remove the mark. |
| Click a trace to add a missed spike | The click is aligned to the channel template, then stored as a good spike. |
| Detect again | Runs detection with the marks. |

**Suggest** does not train a network. It takes the median waveform of the spikes that resemble each other, flags detections whose cosine to that waveform is below 0.75, and scans the channel at a lower threshold (45% of the detection threshold, and not above the template-match fraction) for up to 24 misses.

## Benchmark

**Run benchmark** copies about two seconds of the signal, blanks spikes that were already detected, and plants a known waveform at several amplitudes. It then sweeps the threshold and records precision and recall.

The plot under the histogram draws one precision curve per run and the recall of the latest run. The title is the precision at your current threshold, and the change from the previous run when there is one. The gold line is that threshold. This score is on the planted excerpt. It is a check that the detector settings still recover a known shape, not a count of your review marks.

## Traces

Each spike is a translucent box from the start of the waveform window to the end (0.6 ms before the peak and 1.2 ms after, unless those settings change) and from the bottom of the channel row to the top. The focused spike is gold.
