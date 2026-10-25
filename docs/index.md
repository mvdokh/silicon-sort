<p align="center">
  <img src="assets/spike-sorter-icon-v6.svg" alt="Spike Sorter" width="180">
</p>

# Silicon Sort

Silicon Sort is a desktop spike sorter for low-channel brainstem recordings, built around a Neuronexus A1x32-Poly2 probe and an Intan RHD2132 headstage. It reads a headerless binary file as a memory map, detects spikes, clusters each site, and writes a folder Phy can open. Sorting decisions that need a person stay in Phy.

The recording is never loaded as one array. Filtering, when you ask for it, is written beside the file in chunks. Detection and clustering read windows from that map.

## What it does

1. Open the binary and the probe geometry, and mask channels that should not be sorted.
2. Detect spikes with a robust noise threshold and a neighbor rule. Review false positives and misses, then detect again.
3. Preview how many clusters a single channel wants.
4. Cluster every channel, merge neighbors that are the same unit, split mixtures that violate the refractory period, and export Phy.

## What it leaves alone

Phy is the curation tool. Silicon Sort does not relabel clusters by hand, and it does not claim a unit is a cell. The export is the handoff: spike times in samples, templates in microvolts, and a `cluster_info.tsv` of simple metrics.

## Where to go next

- [Quick start](quickstart.md) if you want a recording on screen.
- [Load](load.md), [Detect](detect.md), [Clusters](clusters.md), and [Autosort](autosort.md) for each control.
- [Pipeline](pipeline.md) for how the stages work.
- [Command line](cli.md) for a run with no window.
