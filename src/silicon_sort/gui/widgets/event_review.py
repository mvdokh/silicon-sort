"""Per-channel review of detected spikes before a second detection pass."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from silicon_sort.detect.suggestions import align_click, suggest_channel
from silicon_sort.gui.widgets.parameters import combo


class EventReview(QWidget):
    """Step through channels, mark events good or bad, and add spikes the detector missed."""

    channelChanged = Signal(int)

    def __init__(self, window) -> None:
        super().__init__()
        self.window = window
        self.channel = 0
        self.selected = 0
        self._events: list[tuple[int, int, float]] = []
        self._waves = np.zeros((0, 0), dtype=np.float32)
        self._wave_key: tuple | None = None
        self._artifact_times: dict[int, np.ndarray] = {}
        self._miss_times: dict[int, np.ndarray] = {}
        self._templates: dict[int, np.ndarray] = {}
        self.setMinimumHeight(210)

        self.title = QLabel("Run detection, then mark events on each channel")
        self.title.setWordWrap(True)
        self.prev_channel = QPushButton("Previous channel")
        self.next_channel = QPushButton("Next channel")
        self.prev_event = QPushButton("Previous event")
        self.next_event = QPushButton("Next event")
        self.good_button = QPushButton("Good")
        self.bad_button = QPushButton("Bad")
        self.clear_button = QPushButton("Clear")
        self.apply_button = QPushButton("Detect again")
        self.add_box = QCheckBox("Click a trace to add a missed spike")
        self.mode = combo(
            ["All events", "Suggested artifacts", "Suggested misses"],
            "All events",
            "Suggested artifacts are detections that do not match this channel's template. Suggested misses are template matches the detector skipped.",
        )
        self.suggest_button = QPushButton("Suggest")
        self.suggest_button.setToolTip("Score this channel. Artifacts are poor matches to its template. Misses are similar spikes below the detection threshold.")
        self.status = QLabel("")
        self.status.setWordWrap(True)
        for button, tip in (
            (self.good_button, "Keep this spike. Similar spikes that were missed are added on the next detection."),
            (self.bad_button, "Drop this spike. Other spikes on this channel with the same shape are dropped too."),
            (self.clear_button, "Remove the good or bad mark from this event."),
            (
                self.apply_button,
                "Run detection again using these marks, so the same false positives and missed shapes are handled across the file.",
            ),
            (
                self.add_box,
                "Turn this on, then click a trace. That time is stored as a good spike on the channel you clicked.",
            ),
        ):
            button.setToolTip(tip)
        self.good_button.setStyleSheet("color: #98c379;")
        self.bad_button.setStyleSheet("color: #e06c75;")

        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Time", units="ms")
        self.plot.setLabel("left", "µV")
        self.plot.setMinimumHeight(140)
        self.plot.showGrid(x=False, y=True, alpha=0.15)
        self.plot.hideButtons()
        self.plot.setMenuEnabled(False)
        self.plot.getPlotItem().layout.setContentsMargins(0, 0, 0, 0)
        self._curves: list[pg.PlotDataItem] = []
        self.plot.scene().sigMouseClicked.connect(self._on_click)

        navigation = QHBoxLayout()
        for widget in (self.prev_channel, self.next_channel, self.prev_event, self.next_event):
            navigation.addWidget(widget)
        review_row = QHBoxLayout()
        review_row.addWidget(self.mode, 1)
        review_row.addWidget(self.suggest_button)
        actions = QHBoxLayout()
        for widget in (self.good_button, self.bad_button, self.clear_button, self.add_box, self.apply_button):
            actions.addWidget(widget)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(self.title)
        layout.addLayout(navigation)
        layout.addLayout(review_row)
        layout.addWidget(self.plot, 1)
        layout.addLayout(actions)
        layout.addWidget(self.status)

        self.prev_channel.clicked.connect(lambda: self._step_channel(-1))
        self.next_channel.clicked.connect(lambda: self._step_channel(1))
        self.prev_event.clicked.connect(lambda: self._step_event(-1))
        self.next_event.clicked.connect(lambda: self._step_event(1))
        self.good_button.clicked.connect(lambda: self._mark(1))
        self.bad_button.clicked.connect(lambda: self._mark(-1))
        self.clear_button.clicked.connect(lambda: self._mark(0))
        self.apply_button.clicked.connect(lambda: self.window.detect.run_detection())
        self.suggest_button.clicked.connect(self.suggest)
        self.mode.currentTextChanged.connect(lambda _text: self.refresh())

    def show_channel(self, channel: int, announce: bool = False) -> None:
        self.channel = int(channel)
        self.selected = 0
        self.refresh()
        if announce:
            self.channelChanged.emit(self.channel)

    def consider_click(self, channel: int, sample: int) -> None:
        if not self.add_box.isChecked():
            return
        self.channel = int(channel)
        aligned = self._align_sample(int(channel), int(sample))
        self.window.pipeline.label_spike(aligned, int(channel), 1)
        self.refresh()
        self._select_time(aligned)
        self.window.detect.sync_review_marks()

    def suggest(self) -> None:
        spikes = self.window.pipeline.spikes
        recording = self.window.pipeline.recording
        if spikes is None or recording is None:
            self.window.log("Run detection before asking for suggestions.")
            return
        channel = int(self.channel)
        mad = self.window.pipeline.noise_mad
        settings = self.window.settings

        def work(token, progress):
            del token
            progress(0.2, f"Scoring channel {channel + 1}")
            with self.window.pipeline.open_signal() as signal:
                trace = np.asarray(signal[:, channel], dtype=np.float32)
            chosen = np.flatnonzero(spikes.channels == channel)
            noise = 1.0 if mad is None or channel >= mad.size or not np.isfinite(mad[channel]) else float(mad[channel])
            return suggest_channel(
                trace,
                spikes.times[chosen],
                spikes.artifact[chosen],
                settings,
                noise,
            )

        def done(result) -> None:
            artifacts, misses, template = result
            self._artifact_times[channel] = np.asarray(artifacts, dtype=np.int64)
            self._miss_times[channel] = np.asarray(misses, dtype=np.int64)
            if template is not None:
                self._templates[channel] = np.asarray(template, dtype=np.float32)
            self.mode.blockSignals(True)
            self.mode.setCurrentText("Suggested artifacts")
            self.mode.blockSignals(False)
            self.channel = channel
            self.selected = 0
            self.refresh()
            self.window.log(
                f"Channel {channel + 1}: {artifacts.size} suggested artifacts, {misses.size} suggested misses"
            )

        self.window.run_task(work, done)

    def refresh(self) -> None:
        for curve in self._curves:
            self.plot.removeItem(curve)
        self._curves.clear()
        self._events = self._collect()
        recording = self.window.pipeline.recording
        fs = 1.0 if recording is None else float(recording.sampling_rate)
        pre, post = self._window(fs)
        key = (self.channel, pre, post, tuple(sample for sample, _label, _amplitude in self._events))
        if key != self._wave_key:
            self._wave_key = key
            self._waves = self._load_waves(self._events, pre, post)
        if not self._events:
            mode = self._review_mode()
            if mode == "artifacts":
                self.title.setText(f"Channel {self.channel + 1}: no suggested artifacts")
                self.status.setText("Press Suggest. Mark the ones that are bad, then detect again.")
            elif mode == "misses":
                self.title.setText(f"Channel {self.channel + 1}: no suggested misses")
                self.status.setText("Press Suggest, or turn on the checkbox and click a missed spike. The click snaps to this channel's template.")
            else:
                self.title.setText(f"Channel {self.channel + 1}: no spikes yet")
                self.status.setText("Mark a missed spike by turning on the checkbox and clicking the trace.")
            return
        self.selected = int(np.clip(self.selected, 0, len(self._events) - 1))
        time_ms = (np.arange(pre + post) - pre) / fs * 1000.0
        labels = np.array([label for _sample, label, _amplitude in self._events], dtype=int)
        chosen = np.arange(len(self._events))
        for label, color in ((0, "#8b93a1"), (1, "#98c379"), (-1, "#e06c75")):
            if label == 0:
                group = chosen[(labels == 0) & (chosen != self.selected)]
            elif label > 0:
                group = chosen[(labels > 0) & (chosen != self.selected)]
            else:
                group = chosen[(labels < 0) & (chosen != self.selected)]
            curve = self._stacked_curve(time_ms, group, color, 1.0)
            if curve is not None:
                self._curves.append(curve)
        selected_wave = self._waves[self.selected]
        if selected_wave.size and np.isfinite(selected_wave).all():
            self._curves.append(self.plot.plot(time_ms, selected_wave, pen=pg.mkPen("#f2f4f8", width=2.4)))
        if time_ms.size:
            self.plot.setXRange(float(time_ms[0]), float(time_ms[-1]), padding=0.02)
        sample, label, _amplitude = self._events[self.selected]
        kind = {"artifacts": "suggested artifact", "misses": "suggested miss"}.get(self._review_mode(), "event")
        self.title.setText(
            f"Channel {self.channel + 1} · {kind} {self.selected + 1} of {len(self._events)} · {_label_name(label)}"
        )
        when = 0.0 if recording is None else sample / fs
        good = int(np.sum(labels > 0))
        bad = int(np.sum(labels < 0))
        self.status.setText(f"At {when:.3f} s. This channel has {good} good and {bad} bad marks.")
        self._follow_trace(int(sample))

    def sync_marks(self) -> None:
        self.refresh()

    def _collect(self) -> list[tuple[int, int, float]]:
        pipeline = self.window.pipeline
        found: dict[int, tuple[int, float]] = {}
        spikes = pipeline.spikes
        if spikes is not None and spikes.n_spikes:
            on_channel = spikes.channels == self.channel
            if self._review_mode() != "artifacts":
                on_channel = on_channel & ~spikes.artifact
            chosen = np.flatnonzero(on_channel)
            for index in chosen:
                found[int(spikes.times[index])] = (0, float(spikes.amplitudes[index]))
        tolerance = self._tolerance()
        for time, channel, label in zip(pipeline.review_times, pipeline.review_channels, pipeline.review_labels, strict=True):
            if int(channel) != self.channel or int(label) == 0:
                continue
            match = None
            for existing in found:
                if abs(existing - int(time)) <= tolerance:
                    match = existing
                    break
            if match is None:
                found[int(time)] = (int(label), float("nan"))
            else:
                amplitude = found.pop(match)[1]
                found[int(time)] = (int(label), amplitude)
        events = [(time, found[time][0], found[time][1]) for time in sorted(found)]
        return self._filter_mode(events)

    def _review_mode(self) -> str:
        text = self.mode.currentText()
        if text == "Suggested artifacts":
            return "artifacts"
        if text == "Suggested misses":
            return "misses"
        return "all"

    def _filter_mode(self, events: list[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
        mode = self._review_mode()
        if mode == "all":
            return events
        wanted = self._artifact_times.get(self.channel) if mode == "artifacts" else self._miss_times.get(self.channel)
        if wanted is None or wanted.size == 0:
            return []
        tolerance = self._tolerance()
        kept = [event for event in events if np.any(np.abs(wanted - event[0]) <= tolerance)]
        if mode != "misses":
            return kept
        present = np.array([event[0] for event in kept], dtype=np.int64)
        for time in wanted:
            if present.size and np.any(np.abs(present - int(time)) <= tolerance):
                continue
            kept.append((int(time), self._label_at(int(time)), float("nan")))
        kept.sort(key=lambda event: event[0])
        return kept

    def _label_at(self, sample: int) -> int:
        pipeline = self.window.pipeline
        if pipeline.review_labels.size == 0:
            return 0
        tolerance = self._tolerance()
        hits = np.flatnonzero(
            (pipeline.review_channels == self.channel) & (np.abs(pipeline.review_times - sample) <= tolerance)
        )
        if hits.size == 0:
            return 0
        return int(pipeline.review_labels[hits[-1]])

    def _align_sample(self, channel: int, sample: int) -> int:
        recording = self.window.pipeline.recording
        if recording is None:
            return int(sample)
        fs = float(recording.sampling_rate)
        pre, post = self._window(fs)
        radius = max(pre, int(round(0.001 * fs)))
        start = max(0, int(sample) - radius - pre)
        stop = min(recording.n_samples, int(sample) + radius + post)
        if stop <= start:
            return int(sample)
        try:
            snippet = np.asarray(self.window.signal_reader(start, stop)[:, channel], dtype=np.float32)
        except Exception:
            return int(sample)
        template = self._templates.get(channel)
        if template is None:
            template = self._template_from_spikes(channel, pre, post)
        polarity = -1.0 if self.window.settings.threshold_mad < 0 else 1.0
        local = align_click(snippet, int(sample) - start, template, pre, polarity, radius)
        return int(start + local)

    def _template_from_spikes(self, channel: int, pre: int, post: int) -> np.ndarray | None:
        spikes = self.window.pipeline.spikes
        recording = self.window.pipeline.recording
        if spikes is None or recording is None or spikes.n_spikes == 0:
            return None
        times = spikes.times[(spikes.channels == channel) & ~spikes.artifact]
        times = times[(times >= pre) & (times + post <= recording.n_samples)]
        if times.size == 0:
            return None
        if times.size > 40:
            times = times[:40]
        waves = []
        for time in times:
            try:
                snippet = self.window.signal_reader(int(time) - pre, int(time) + post)
            except Exception:
                continue
            if snippet.shape[0] == pre + post and channel < snippet.shape[1]:
                waves.append(np.asarray(snippet[:, channel], dtype=np.float32))
        if not waves:
            return None
        template = np.median(np.stack(waves), axis=0).astype(np.float32)
        self._templates[channel] = template
        return template

    def _window(self, sampling_rate: float) -> tuple[int, int]:
        pre = max(1, int(round(float(self.window.settings.waveform_pre_ms) * 1e-3 * sampling_rate)))
        post = max(1, int(round(float(self.window.settings.waveform_post_ms) * 1e-3 * sampling_rate)))
        return pre, post

    def _load_waves(self, events: list[tuple[int, int, float]], pre: int, post: int) -> np.ndarray:
        length = pre + post
        waves = np.full((len(events), length), np.nan, dtype=np.float32)
        recording = self.window.pipeline.recording
        if recording is None or not events:
            return waves
        times = np.array([sample for sample, _label, _amplitude in events], dtype=np.int64)
        order = np.argsort(times, kind="mergesort")
        span = max(length, int(round(float(recording.sampling_rate))))
        group: list[int] = []
        group_start = 0
        group_stop = 0

        def flush() -> None:
            nonlocal group_start, group_stop
            if not group:
                return
            try:
                snippet = self.window.signal_reader(group_start, group_stop)
            except Exception:
                group.clear()
                return
            if self.channel >= snippet.shape[1]:
                group.clear()
                return
            for index in group:
                local = int(times[index]) - group_start
                if local < pre or local + post > snippet.shape[0]:
                    continue
                waves[index] = snippet[local - pre : local + post, self.channel]
            group.clear()

        for index in order:
            sample = int(times[int(index)])
            start = sample - pre
            stop = sample + post
            if start < 0 or stop > recording.n_samples:
                continue
            if group and stop - group_start > span:
                flush()
            if not group:
                group_start = start
                group_stop = stop
            else:
                group_stop = max(group_stop, stop)
            group.append(int(index))
        flush()
        return waves

    def _stacked_curve(self, time_ms: np.ndarray, indices: np.ndarray, color: str, width: float):
        if indices.size == 0 or time_ms.size == 0 or self._waves.size == 0:
            return None
        length = int(time_ms.size)
        block = length + 1
        x = np.full(int(indices.size) * block, np.nan, dtype=np.float64)
        y = np.full(int(indices.size) * block, np.nan, dtype=np.float64)
        for slot, index in enumerate(indices):
            wave = self._waves[int(index)]
            if wave.size != length or not np.isfinite(wave).all():
                continue
            offset = slot * block
            x[offset : offset + length] = time_ms
            y[offset : offset + length] = wave
        if not np.isfinite(y).any():
            return None
        return self.plot.plot(x, y, pen=pg.mkPen(color, width=width), connect="finite")

    def _follow_trace(self, sample: int) -> None:
        trace = self.window.detect.trace
        if trace.reader is None or trace.n_samples <= 0:
            return
        trace.focus_spike(self.channel, int(sample))

    def _mark(self, label: int) -> None:
        if not self._events:
            return
        sample, _label, _amplitude = self._events[self.selected]
        self.window.pipeline.label_spike(int(sample), self.channel, int(label))
        self.refresh()
        self.window.detect.sync_review_marks()

    def _step_event(self, step: int) -> None:
        if not self._events:
            return
        self.selected = int(np.clip(self.selected + step, 0, len(self._events) - 1))
        self.refresh()

    def _step_channel(self, step: int) -> None:
        order = np.asarray(self.window.display_order, dtype=int)
        if order.size == 0:
            count = 0 if self.window.pipeline.recording is None else self.window.pipeline.recording.n_channels
            order = np.arange(count)
        if order.size == 0:
            return
        hits = np.flatnonzero(order == self.channel)
        position = 0 if hits.size == 0 else int(hits[0])
        self.show_channel(int(order[(position + step) % order.size]), announce=True)

    def _select_time(self, sample: int) -> None:
        if not self._events:
            return
        times = np.array([event[0] for event in self._events])
        self.selected = int(np.argmin(np.abs(times - sample)))
        self.refresh()

    def _on_click(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self._waves.size == 0:
            return
        point = self.plot.plotItem.vb.mapSceneToView(event.scenePos())
        self.pick_waveform(float(point.x()), float(point.y()))

    def pick_waveform(self, time_ms: float, voltage: float) -> None:
        if self._waves.size == 0:
            return
        recording = self.window.pipeline.recording
        fs = 1.0 if recording is None else float(recording.sampling_rate)
        pre, _post = self._window(fs)
        sample = int(np.clip(round(time_ms * fs / 1000.0 + pre), 0, self._waves.shape[1] - 1))
        finite = np.isfinite(self._waves).all(axis=1)
        if not finite.any():
            return
        errors = np.abs(self._waves[:, sample] - float(voltage))
        errors[~finite] = np.inf
        picked = int(np.argmin(errors))
        if not np.isfinite(errors[picked]):
            return
        self.selected = picked
        self.refresh()

    def _tolerance(self) -> int:
        recording = self.window.pipeline.recording
        rate = 30000.0 if recording is None else float(recording.sampling_rate)
        return max(1, int(round(0.0004 * rate)))


def _label_color(label: int) -> str:
    if label > 0:
        return "#98c379"
    if label < 0:
        return "#e06c75"
    return "#8b93a1"


def _label_name(label: int) -> str:
    if label > 0:
        return "good"
    if label < 0:
        return "bad"
    return "unlabeled"
