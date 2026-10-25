"""Apply hand labels so a second detection pass drops false positives and recovers misses."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from silicon_sort.cancel import CancelToken
from silicon_sort.detect.detect import SpikeIndex, _finalize_spikes, _ms
from silicon_sort.detect.templates import _scan_channel
from silicon_sort.errors import Cancelled
from silicon_sort.settings import PipelineSettings

Progress = Callable[[float, str], None]

_REJECT_COSINE = 0.9


def upsert_label(
    times: np.ndarray,
    channels: np.ndarray,
    labels: np.ndarray,
    time: int,
    channel: int,
    label: int,
    tolerance: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Insert, replace, or clear one mark. ``label`` 1 is good, -1 is bad, 0 clears it."""
    times = np.asarray(times, dtype=np.int64).copy()
    channels = np.asarray(channels, dtype=np.int32).copy()
    labels = np.asarray(labels, dtype=np.int8).copy()
    same = np.flatnonzero((channels == int(channel)) & (np.abs(times - int(time)) <= int(tolerance)))
    if int(label) == 0:
        if same.size:
            keep = np.ones(times.size, dtype=bool)
            keep[same] = False
            return times[keep], channels[keep], labels[keep]
        return times, channels, labels
    if same.size:
        labels[same] = np.int8(label)
        times[same[0]] = np.int64(time)
        if same.size > 1:
            keep = np.ones(times.size, dtype=bool)
            keep[same[1:]] = False
            return times[keep], channels[keep], labels[keep]
        return times, channels, labels
    return (
        np.concatenate([times, np.array([time], dtype=np.int64)]),
        np.concatenate([channels, np.array([channel], dtype=np.int32)]),
        np.concatenate([labels, np.array([label], dtype=np.int8)]),
    )


def refine_with_review(
    data: np.ndarray,
    spikes: SpikeIndex,
    review_times: np.ndarray,
    review_channels: np.ndarray,
    review_labels: np.ndarray,
    settings: PipelineSettings,
    noise_mad: np.ndarray,
    progress: Progress | None = None,
    cancel: CancelToken | None = None,
    bad_scope: str = "channel",
    good_scope: str = "channel",
) -> SpikeIndex:
    """Drop events like the bad marks, keep the good marks, and find similar missed spikes.

    ``bad_scope`` and ``good_scope`` are ``marked``, ``channel``, or ``all``.
    Marked applies each example only to itself. Channel also matches the same
    shape on that example's channel. All matches the shape on every channel.
    """
    review_times = np.asarray(review_times, dtype=np.int64)
    review_channels = np.asarray(review_channels, dtype=np.int32)
    review_labels = np.asarray(review_labels, dtype=np.int8)
    if review_labels.size == 0 or not np.any(review_labels):
        return spikes

    fs = float(settings.sampling_rate or 1.0)
    pre = _ms(settings.waveform_pre_ms, fs)
    post = _ms(settings.waveform_post_ms, fs)
    tolerance = max(1, _ms(0.4, fs))
    polarity = -1.0 if settings.threshold_mad < 0 else 1.0
    good_template, bad_template = _templates(
        data, review_times, review_channels, review_labels, pre, post
    )

    times = np.asarray(spikes.times, dtype=np.int64)
    channels = np.asarray(spikes.channels, dtype=np.int32)
    amplitudes = np.asarray(spikes.amplitudes, dtype=np.float32)
    artifact = np.asarray(spikes.artifact, dtype=bool)
    if times.size:
        bad_hit = _near_label(times, channels, review_times, review_channels, review_labels, -1, tolerance)
        keep = ~bad_hit
        times, channels, amplitudes, artifact = times[keep], channels[keep], amplitudes[keep], artifact[keep]

    extra_t, extra_c, extra_a = _missing_good(
        data,
        times,
        channels,
        review_times,
        review_channels,
        review_labels,
        pre,
        tolerance,
        polarity,
    )
    if extra_t.size:
        times = np.concatenate([times, extra_t])
        channels = np.concatenate([channels, extra_c])
        amplitudes = np.concatenate([amplitudes, extra_a])
        artifact = np.concatenate([artifact, np.zeros(extra_t.size, dtype=bool)])

    found_t: list[int] = []
    found_c: list[int] = []
    found_a: list[float] = []
    pairs = _good_pairs(good_template, noise_mad, good_scope)
    for index, (channel, template) in enumerate(pairs):
        if cancel is not None and cancel.cancelled():
            raise Cancelled("Detection cancelled.")
        if not np.isfinite(noise_mad[channel]):
            continue
        occupied = np.sort(times[channels == channel]) if times.size else np.zeros(0, dtype=np.int64)
        new_times, new_amps = _scan_channel(
            data[:, channel],
            template,
            pre,
            polarity,
            float(noise_mad[channel]),
            float(settings.threshold_mad),
            float(settings.template_threshold_fraction),
            float(settings.template_min_cosine),
            max(pre + post, int(round(max(settings.chunk_seconds, 30.0) * fs))),
            occupied,
            _ms(settings.dead_time_ms, fs),
        )
        found_t.extend(new_times)
        found_c.extend([channel] * len(new_times))
        found_a.extend(new_amps)
        if progress is not None:
            progress((index + 1) / max(len(pairs), 1), f"Review template {index + 1}/{len(pairs)}")

    if found_t:
        times = np.concatenate([times, np.asarray(found_t, dtype=np.int64)])
        channels = np.concatenate([channels, np.asarray(found_c, dtype=np.int32)])
        amplitudes = np.concatenate([amplitudes, np.asarray(found_a, dtype=np.float32)])
        artifact = np.concatenate([artifact, np.zeros(len(found_t), dtype=bool)])

    merged = SpikeIndex(times, channels, amplitudes, artifact)
    merged = _finalize_spikes(merged, n_channels=int(data.shape[1]), dead_samples=_ms(settings.dead_time_ms, fs))
    if merged.n_spikes == 0:
        return merged

    protected = _near_label(
        merged.times,
        merged.channels,
        review_times,
        review_channels,
        review_labels,
        1,
        tolerance,
    )
    rejected = _near_label(
        merged.times,
        merged.channels,
        review_times,
        review_channels,
        review_labels,
        -1,
        tolerance,
    )
    if bad_scope == "marked" or not bad_template:
        shaped = np.zeros(merged.n_spikes, dtype=bool)
    else:
        shaped = _matches_bad_shape(
            data,
            merged,
            bad_template,
            good_template,
            pre,
            post,
            protected,
            scope=bad_scope,
        )
    keep = protected | ~(rejected | shaped)
    return SpikeIndex(merged.times[keep], merged.channels[keep], merged.amplitudes[keep], merged.artifact[keep])


def _templates(
    data: np.ndarray,
    times: np.ndarray,
    channels: np.ndarray,
    labels: np.ndarray,
    pre: int,
    post: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    good: dict[int, list[np.ndarray]] = {}
    bad: dict[int, list[np.ndarray]] = {}
    for time, channel, label in zip(times, channels, labels, strict=True):
        if int(label) == 0:
            continue
        wave = _waveform(data, int(time), int(channel), pre, post)
        if wave is None:
            continue
        bucket = good if int(label) > 0 else bad
        bucket.setdefault(int(channel), []).append(wave)
    return _median_bank(good), _median_bank(bad)


def _good_pairs(
    good_template: dict[int, np.ndarray],
    noise_mad: np.ndarray,
    scope: str,
) -> list[tuple[int, np.ndarray]]:
    if scope == "marked" or not good_template:
        return []
    if scope == "all":
        eligible = [channel for channel in range(noise_mad.size) if np.isfinite(noise_mad[channel])]
        return [(channel, template) for template in good_template.values() for channel in eligible]
    return [(channel, template) for channel, template in good_template.items()]


def _median_bank(groups: dict[int, list[np.ndarray]]) -> dict[int, np.ndarray]:
    bank: dict[int, np.ndarray] = {}
    for channel, waves in groups.items():
        chosen = waves[:200]
        bank[channel] = np.median(np.stack(chosen), axis=0).astype(np.float32)
    return bank


def _missing_good(
    data: np.ndarray,
    times: np.ndarray,
    channels: np.ndarray,
    review_times: np.ndarray,
    review_channels: np.ndarray,
    review_labels: np.ndarray,
    pre: int,
    tolerance: int,
    polarity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    add_t: list[int] = []
    add_c: list[int] = []
    add_a: list[float] = []
    good = np.flatnonzero(review_labels > 0)
    for index in good:
        channel = int(review_channels[index])
        time = int(review_times[index])
        if times.size and np.any((channels == channel) & (np.abs(times - time) <= tolerance)):
            continue
        if time < 0 or time >= data.shape[0]:
            continue
        add_t.append(time)
        add_c.append(channel)
        add_a.append(float(data[time, channel]))
    del pre, polarity
    if not add_t:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)
    return (
        np.asarray(add_t, dtype=np.int64),
        np.asarray(add_c, dtype=np.int32),
        np.asarray(add_a, dtype=np.float32),
    )


def _matches_bad_shape(
    data: np.ndarray,
    spikes: SpikeIndex,
    bad_template: dict[int, np.ndarray],
    good_template: dict[int, np.ndarray],
    pre: int,
    post: int,
    protected: np.ndarray,
    scope: str = "channel",
) -> np.ndarray:
    rejected = np.zeros(spikes.n_spikes, dtype=bool)
    every_bad = list(bad_template.values())
    every_good = list(good_template.values())
    channels = range(int(data.shape[1])) if scope == "all" else sorted(bad_template)
    width = pre + post
    for channel in channels:
        if scope == "all":
            candidates = every_bad
            goods = every_good
        else:
            template = bad_template.get(int(channel))
            candidates = [] if template is None else [template]
            good = good_template.get(int(channel))
            goods = [] if good is None else [good]
        if not candidates:
            continue
        chosen = np.flatnonzero((spikes.channels == channel) & ~protected)
        if chosen.size == 0:
            continue
        times = spikes.times[chosen]
        order = np.argsort(times)
        chosen = chosen[order]
        times = times[order]
        block = max(width + 1, 60000)
        cursor = 0
        while cursor < times.size:
            end = cursor
            while end + 1 < times.size and int(times[end + 1]) < int(times[cursor]) + block:
                end += 1
            start = int(times[cursor]) - pre
            stop = int(times[end]) + post
            if start < 0 or stop > data.shape[0]:
                cursor = end + 1
                continue
            column = np.asarray(data[start:stop], dtype=np.float32)[:, channel]
            for index, time in zip(chosen[cursor : end + 1], times[cursor : end + 1], strict=True):
                local = int(time) - start - pre
                wave = column[local : local + width]
                if wave.size != width:
                    continue
                cosine_bad = max(_cosine(wave, template) for template in candidates)
                if cosine_bad < _REJECT_COSINE:
                    continue
                if any(cosine_bad <= _cosine(wave, template) + 0.02 for template in goods):
                    continue
                rejected[index] = True
            cursor = end + 1
    return rejected


def _near_label(
    times: np.ndarray,
    channels: np.ndarray,
    review_times: np.ndarray,
    review_channels: np.ndarray,
    review_labels: np.ndarray,
    label: int,
    tolerance: int,
) -> np.ndarray:
    hit = np.zeros(times.shape[0], dtype=bool)
    wanted = np.flatnonzero(review_labels == label)
    if times.size == 0 or wanted.size == 0:
        return hit
    for index in wanted:
        channel = int(review_channels[index])
        time = int(review_times[index])
        hit |= (channels == channel) & (np.abs(times - time) <= tolerance)
    return hit


def _waveform(data: np.ndarray, time: int, channel: int, pre: int, post: int) -> np.ndarray | None:
    start = time - pre
    stop = time + post
    if start < 0 or stop > data.shape[0] or stop <= start:
        return None
    return np.asarray(data[start:stop, channel], dtype=np.float32)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-12))
