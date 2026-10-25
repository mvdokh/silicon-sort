"""Spike detection."""

from silicon_sort.detect.benchmark import BenchmarkCurve, match_events, sweep_precision_recall
from silicon_sort.detect.detect import SpikeIndex, detect_traces

__all__ = [
    "BenchmarkCurve",
    "SpikeIndex",
    "detect_traces",
    "match_events",
    "sweep_precision_recall",
]
