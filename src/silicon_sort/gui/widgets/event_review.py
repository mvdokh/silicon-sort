"""Per-channel review of detected spikes before a second detection pass."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget


class EventReview(QWidget):
    """Step through channels, mark events good or bad, and add spikes the detector missed."""

    channelChanged = Signal(int)

    def __init__(self, window) -> None:
        super().__init__()
        self.window = window
        self.channel = 0
        self.selected = 0
        self._events: list[tuple[int, int, float]] = []
        self._waves: list[np.ndarray] = []
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
        actions = QHBoxLayout()
        for widget in (self.good_button, self.bad_button, self.clear_button, self.add_box, self.apply_button):
            actions.addWidget(widget)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(self.title)
        layout.addLayout(navigation)
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
        self.window.pipeline.label_spike(int(sample), int(channel), 1)
        self.refresh()
        self._select_time(int(sample))
        self.window.detect.sync_review_marks()

    def refresh(self) -> None:
        for curve in self._curves:
            self.plot.removeItem(curve)
        self._curves.clear()
        self._events = self._collect()
        self._waves = []
        if not self._events:
            self.title.setText(f"Channel {self.channel + 1}: no spikes yet")
            self.status.setText("Mark a missed spike by turning on the checkbox and clicking the trace.")
            return
        self.selected = int(np.clip(self.selected, 0, len(self._events) - 1))
        recording = self.window.pipeline.recording
        fs = 1.0 if recording is None else float(recording.sampling_rate)
        pre_ms = float(self.window.settings.waveform_pre_ms)
        post_ms = float(self.window.settings.waveform_post_ms)
        pre = max(1, int(round(pre_ms * 1e-3 * fs)))
        post = max(1, int(round(post_ms * 1e-3 * fs)))
        start = max(0, self.selected - 12)
        stop = min(len(self._events), start + 24)
        start = max(0, stop - 24)
        visible = list(range(start, stop))
        time_ms = (np.arange(pre + post) - pre) / fs * 1000.0
        drawn: list[tuple[int, np.ndarray]] = []
        for index in visible:
            sample, _label, _amplitude = self._events[index]
            wave = self._read_wave(int(sample), pre, post)
            if wave is None:
                continue
            drawn.append((index, wave))
            self._waves.append(wave)
        self._visible_index = [index for index, _wave in drawn]
        for index, wave in drawn:
            _sample, label, _amplitude = self._events[index]
            selected = index == self.selected
            color = "#f2f4f8" if selected else _label_color(label)
            width = 2.4 if selected else 1.0
            curve = self.plot.plot(time_ms, wave, pen=pg.mkPen(color, width=width))
            self._curves.append(curve)
        if time_ms.size:
            self.plot.setXRange(float(time_ms[0]), float(time_ms[-1]), padding=0.02)
        sample, label, _amplitude = self._events[self.selected]
        self.title.setText(
            f"Channel {self.channel + 1} · event {self.selected + 1} of {len(self._events)} · {_label_name(label)}"
        )
        when = 0.0 if recording is None else sample / fs
        good = sum(1 for _t, lab, _a in self._events if lab > 0)
        bad = sum(1 for _t, lab, _a in self._events if lab < 0)
        self.status.setText(f"At {when:.3f} s. This channel has {good} good and {bad} bad marks.")

    def sync_marks(self) -> None:
        self.refresh()

    def _collect(self) -> list[tuple[int, int, float]]:
        pipeline = self.window.pipeline
        found: dict[int, tuple[int, float]] = {}
        spikes = pipeline.spikes
        if spikes is not None and spikes.n_spikes:
            chosen = np.flatnonzero((spikes.channels == self.channel) & ~spikes.artifact)
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
        return [(time, found[time][0], found[time][1]) for time in sorted(found)]

    def _read_wave(self, sample: int, pre: int, post: int) -> np.ndarray | None:
        recording = self.window.pipeline.recording
        if recording is None:
            return None
        start = sample - pre
        stop = sample + post
        if start < 0 or stop > recording.n_samples:
            return None
        try:
            snippet = self.window.signal_reader(start, stop)
        except Exception:
            return None
        if snippet.shape[0] != pre + post or self.channel >= snippet.shape[1]:
            return None
        return np.asarray(snippet[:, self.channel], dtype=np.float32)

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
        if event.button() != Qt.MouseButton.LeftButton or not self._waves:
            return
        point = self.plot.plotItem.vb.mapSceneToView(event.scenePos())
        recording = self.window.pipeline.recording
        fs = 1.0 if recording is None else float(recording.sampling_rate)
        pre = max(1, int(round(float(self.window.settings.waveform_pre_ms) * 1e-3 * fs)))
        sample = int(np.clip(round(float(point.x()) * fs / 1000.0 + pre), 0, self._waves[0].size - 1))
        errors = [abs(float(wave[sample]) - float(point.y())) for wave in self._waves]
        picked = int(np.argmin(errors))
        if picked < len(self._visible_index):
            self.selected = self._visible_index[picked]
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
