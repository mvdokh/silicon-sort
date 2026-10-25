"""Preprocessing stages."""

from silicon_sort.preprocess.bad_channels import suggest_bad_channels
from silicon_sort.preprocess.filter import design_bandpass, filter_array_chunked
from silicon_sort.preprocess.noise import estimate_mad
from silicon_sort.preprocess.reference import reference_chunk, reference_groups
from silicon_sort.preprocess.whiten import apply_whitener, estimate_whitener

__all__ = [
    "apply_whitener",
    "design_bandpass",
    "estimate_mad",
    "estimate_whitener",
    "filter_array_chunked",
    "reference_chunk",
    "reference_groups",
    "suggest_bad_channels",
]
