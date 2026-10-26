"""Detection parameters, quality-control plots, and the hybrid benchmark."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from silicon_sort.detect.benchmark import benchmark_excerpt
from silicon_sort.gui.widgets.event_review import EventReview
from silicon_sort.gui.widgets.parameters import combo, double_box, int_box
from silicon_sort.gui.widgets.probe_waveforms import ProbeWaveformMap
from silicon_sort.gui.widgets.qc_plots import HistogramPlot, PRPlot
from silicon_sort.gui.widgets.trace_view import TraceView


class DetectTab(QWidget):
    def __init__(self, window) -> None:
        super().__init__()
        self.window = window
        self.form = QWidget()
        layout = QVBoxLayout(self.form)
        layout.setContentsMargins(8, 8, 8, 8)

        self.multi_step = QPushButton("1  Multi")
        self.single_step = QPushButton("2  Single")
        self.proximity_step = QPushButton("3  Proximity")
        for button in (self.multi_step, self.single_step, self.proximity_step):
            button.setCheckable(True)
        self.multi_step.setChecked(True)
        steps = QButtonGroup(self.form)
        steps.setExclusive(True)
        steps.addButton(self.multi_step, 0)
        steps.addButton(self.single_step, 1)
        steps.addButton(self.proximity_step, 2)
        step_row = QHBoxLayout()
        step_row.addWidget(self.multi_step, 1)
        step_row.addWidget(self.single_step, 1)
        step_row.addWidget(self.proximity_step, 1)
        layout.addLayout(step_row)

        self.threshold = double_box(
            -5,
            -20,
            20,
            0.5,
            "Threshold in MAD units. Negative values detect downward spikes, which matches Intan polarity.",
            decimals=1,
        )
        self.radius = double_box(60, 0, 500, 5, "A neighbor inside this radius must also cross a lower threshold.")
        self.support = double_box(0.5, 0.1, 1, 0.05, "Neighbor threshold as a fraction of the detection threshold.", decimals=2)
        self.min_neighbors = int_box(1, 0, 8, "How many neighbors must agree. Sites with no neighbors are kept.")
        self.dead = double_box(0.7, 0.1, 5, 0.1, "Per-site dead time in milliseconds.", decimals=2)
        self.artifact_fraction = double_box(
            0.4,
            0.05,
            1,
            0.05,
            "Fraction of channels that must fire together before a sample is treated as movement or EMG.",
            decimals=2,
        )
        self.artifact_mad = double_box(8, 2, 50, 1, "Amplitude, in MAD, of that coincident event.", decimals=1)
        self.template_match = combo(
            ["off", "on"],
            "off",
            "Second pass. Match first-pass waveforms to recover smaller spikes. Slower.",
        )
        detect_form = QFormLayout()
        detect_form.addRow("Threshold MAD", self.threshold)
        detect_form.addRow("Neighbor radius µm", self.radius)
        detect_form.addRow("Neighbor fraction", self.support)
        detect_form.addRow("Min neighbors", self.min_neighbors)
        detect_form.addRow("Dead time ms", self.dead)
        detect_form.addRow("Artifact fraction", self.artifact_fraction)
        detect_form.addRow("Artifact MAD", self.artifact_mad)
        detect_form.addRow("Template match", self.template_match)
        self.run_button = QPushButton("Run detection")
        self.run_button.setToolTip(
            "Detect spikes on every channel. Uses the raw recording when preprocessing is off, and the filtered "
            "cache when it is on. Good and bad marks from the review panel are applied on this run."
        )
        self.benchmark_button = QPushButton("Run benchmark")
        self.benchmark_button.setToolTip(
            "Plant known waveforms into a short copy of the filtered noise and plot precision and recall."
        )
        self.precision = PRPlot()
        self.precision.setMinimumHeight(150)
        self._benchmark_history: list = []

        multi_page = QWidget()
        multi_layout = QVBoxLayout(multi_page)
        multi_layout.setContentsMargins(0, 0, 0, 0)
        detect_box = _group("Detection", detect_form)
        detect_box.layout().addWidget(self.run_button)
        detect_box.layout().addWidget(self.benchmark_button)
        multi_layout.addWidget(detect_box)
        multi_layout.addWidget(self.precision, 1)

        single_note = QLabel(
            "Review one channel at a time. Detect again re-runs detection only for the channel you are viewing."
        )
        single_note.setWordWrap(True)
        self.single_redetect = QPushButton("Detect again (this channel)")
        self.single_redetect.setToolTip("Re-detect spikes on the selected channel only. Other channels stay as they are.")
        single_page = QWidget()
        single_layout = QVBoxLayout(single_page)
        single_layout.setContentsMargins(0, 0, 0, 0)
        single_layout.addWidget(single_note)
        single_layout.addWidget(_group("Redetect", _vstack(self.single_redetect)))
        single_layout.addStretch(1)

        self.proximity_radius = int_box(
            1,
            0,
            8,
            "How many channels above and below the center to show. Missing neighbors leave an empty row.",
        )
        self.proximity_redetect_scope = combo(
            ["Center channel", "All channels in view"],
            "Center channel",
            "Detect again either re-detects only the center channel, or every real channel shown in the proximity view.",
        )
        proximity_form = QFormLayout()
        proximity_form.addRow("Nearby ±", self.proximity_radius)
        proximity_form.addRow("Detect again", self.proximity_redetect_scope)
        self.proximity_redetect = QPushButton("Detect again")
        self.proximity_redetect.setToolTip("Re-detect using the scope above. Empty slots are skipped.")
        proximity_note = QLabel(
            "The selected channel is the center. Rows above and below are the next sites along the probe; "
            "empty rows appear when there is no neighbor in that direction."
        )
        proximity_note.setWordWrap(True)
        proximity_page = QWidget()
        proximity_layout = QVBoxLayout(proximity_page)
        proximity_layout.setContentsMargins(0, 0, 0, 0)
        proximity_layout.addWidget(proximity_note)
        proximity_box = _group("Proximity", proximity_form)
        proximity_box.layout().addWidget(self.proximity_redetect)
        proximity_layout.addWidget(proximity_box)
        proximity_layout.addStretch(1)

        self.steps = QStackedWidget()
        self.steps.addWidget(multi_page)
        self.steps.addWidget(single_page)
        self.steps.addWidget(proximity_page)
        steps.idClicked.connect(self._on_step)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(self.steps)
        layout.addWidget(scroll, 1)

        # Compat aliases used by older tests and EventReview wiring.
        self.view_mode = _ViewModeProxy(self)
        self.proximity_count = self.proximity_radius

        self.trace = TraceView()
        self.probe_waves = ProbeWaveformMap()
        self.probe_waves.setMinimumWidth(100)
        self.review = EventReview(window)
        self.histogram = HistogramPlot()
        self.histogram.setMinimumWidth(140)
        self.side = QSplitter(Qt.Orientation.Vertical)
        self.side.setHandleWidth(2)
        self.side.setChildrenCollapsible(False)
        self.side.addWidget(self.review)
        self.side.addWidget(self.histogram)
        self.side.setStretchFactor(0, 4)
        self.side.setStretchFactor(1, 1)
        self.side.setSizes([520, 140])
        self.side.setMinimumWidth(220)

        # Size to the visible page only — otherwise the review panel keeps Multi wide.
        self.right = _CurrentPageStack()
        self.right.addWidget(self.probe_waves)
        self.right.addWidget(self.side)

        self.view = QSplitter(Qt.Orientation.Horizontal)
        self.view.setHandleWidth(2)
        self.view.setChildrenCollapsible(False)
        self.view.addWidget(self.trace)
        self.view.addWidget(self.right)
        self.view.setStretchFactor(0, 1)
        self.view.setStretchFactor(1, 0)
        self.view.setSizes([1100, 200])
        self._view_mode: str | None = None
        self._probe_width_min = 100
        self._probe_width_default = 200
        self._probe_width_max = 300

        self.run_button.clicked.connect(self.run_detection)
        self.benchmark_button.clicked.connect(self.run_benchmark)
        self.single_redetect.clicked.connect(self.redetect_current_scope)
        self.proximity_redetect.clicked.connect(self.redetect_current_scope)
        self.trace.channelSelected.connect(self._channel)
        self.trace.timePicked.connect(self.review.consider_click)
        self.trace.maskRequested.connect(window.toggle_mask)
        self.probe_waves.channelSelected.connect(self._channel)
        self.probe_waves.channelActivated.connect(self._open_single)
        self.review.channelChanged.connect(self._from_review)
        self.proximity_radius.valueChanged.connect(lambda _value: self._apply_view_mode())
        self._channel_id = 0
        self._apply_view_mode()

    def view_mode_text(self) -> str:
        index = self.steps.currentIndex()
        return ("Multi channel", "Single channel", "Proximity")[index]

    def collect(self) -> None:
        settings = self.window.settings
        settings.threshold_mad = float(self.threshold.value())
        settings.neighbor_radius_um = float(self.radius.value())
        settings.neighbor_support_fraction = float(self.support.value())
        settings.min_neighbors = int(self.min_neighbors.value())
        settings.dead_time_ms = float(self.dead.value())
        settings.artifact_fraction = float(self.artifact_fraction.value())
        settings.artifact_mad = float(self.artifact_mad.value())
        settings.template_match = self.template_match.currentText() == "on"

    def sync_recording(self) -> None:
        """Show the loaded recording on Detect before spikes exist."""
        recording = self.window.pipeline.recording
        if recording is None:
            return
        self.trace.set_source(
            self.window.signal_reader,
            recording.n_samples,
            recording.sampling_rate,
            recording.n_channels,
            self.window.display_order,
            self.window.mask_array(),
        )
        if self.window.pipeline.spikes is None:
            self.trace.set_spikes(None, None)
            self.trace.set_thresholds(None)
            self.probe_waves.clear_waveforms()
        self._sync_probe_map()
        self._apply_view_mode()

    def run_detection(self) -> None:
        self.window.load.collect()
        self.collect()
        self.review.apply_examples()

        def work(token, progress):
            return self.window.pipeline.run_detect(progress=progress, cancel=token)

        self.window.run_task(work, lambda _spikes: self.show_detection())

    def redetect_current_scope(self) -> None:
        """Re-detect the single channel or the proximity center / all shown channels."""
        self.window.load.collect()
        self.collect()
        self.review.apply_examples()
        channels = self._redetect_channels()
        if not channels:
            self.window.log("Select a real channel before redetecting.")
            return

        def work(token, progress):
            return self.window.pipeline.redetect_channels(channels, progress=progress, cancel=token)

        self.window.run_task(work, lambda _spikes: self.show_detection())

    def _redetect_channels(self) -> list[int]:
        mode = self.view_mode_text()
        center = int(self._channel_id)
        if mode == "Single channel":
            return [center] if center >= 0 else []
        if mode == "Proximity":
            if self.proximity_redetect_scope.currentText() == "Center channel":
                return [center] if center >= 0 else []
            return [int(channel) for channel in self._proximity_channels(center) if int(channel) >= 0]
        return []

    def show_detection(self) -> None:
        recording = self.window.pipeline.recording
        spikes = self.window.pipeline.spikes
        if recording is None or spikes is None:
            return
        if self.trace.reader is None:
            self.sync_recording()
        mad = self.window.pipeline.noise_mad
        if mad is not None:
            self.trace.set_thresholds(np.abs(self.window.settings.threshold_mad) * mad)
        self.trace.set_spikes(
            spikes.times,
            spikes.channels,
            spikes.artifact,
            pre_s=float(self.window.settings.waveform_pre_ms) * 1e-3,
            post_s=float(self.window.settings.waveform_post_ms) * 1e-3,
        )
        self.sync_review_marks()
        self._fill_probe_waveforms()
        channel = self._channel_id if 0 <= self._channel_id < recording.n_channels else 0
        self.review.show_channel(channel)
        self._channel(channel)
        self._apply_view_mode()
        clean = spikes.clean()
        self.window.log(f"{clean.n_spikes} spikes, {int(spikes.artifact.sum())} artifact-flagged")

    def _on_step(self, index: int) -> None:
        self.steps.setCurrentIndex(index)
        self._apply_view_mode()

    def _channel(self, channel: int) -> None:
        if int(channel) < 0:
            return
        self._channel_id = int(channel)
        self.window.selected = int(channel)
        self.probe_waves.set_selected(int(channel))
        spikes = self.window.pipeline.spikes
        if spikes is not None:
            values = spikes.amplitudes[(spikes.channels == channel) & ~spikes.artifact]
            self.histogram.set_values(values)
        mode = self.view_mode_text()
        if mode == "Proximity":
            self._configure_trace_order(self._proximity_channels(int(channel)))
        elif mode == "Single channel":
            self._configure_trace_order([int(channel)])
        else:
            self.trace.reveal(int(channel))

    def _from_review(self, channel: int) -> None:
        if int(channel) < 0:
            return
        self._channel_id = int(channel)
        self.window.selected = int(channel)
        self.probe_waves.set_selected(int(channel))
        self.trace.reveal(int(channel))
        spikes = self.window.pipeline.spikes
        if spikes is None:
            return
        values = spikes.amplitudes[(spikes.channels == channel) & ~spikes.artifact]
        self.histogram.set_values(values)
        mode = self.view_mode_text()
        if mode == "Proximity":
            self._configure_trace_order(self._proximity_channels(int(channel)))
        elif mode == "Single channel":
            self._configure_trace_order([int(channel)])

    def _open_single(self, channel: int) -> None:
        self.single_step.setChecked(True)
        self.steps.setCurrentIndex(1)
        self._channel_id = int(channel)
        self._apply_view_mode()
        self._channel(int(channel))
        self.review.show_channel(int(channel), announce=False)

    def _apply_view_mode(self, _text: str = "") -> None:
        mode = self.view_mode_text()
        switched = mode != self._view_mode
        self._view_mode = mode
        channel = int(self._channel_id)
        if mode == "Multi channel":
            self.right.setCurrentWidget(self.probe_waves)
            self.right.setMinimumWidth(self._probe_width_min)
            self.right.setMaximumWidth(self._probe_width_max)
            self.right.updateGeometry()
            if switched:
                self._set_right_width(self._probe_width_default)
            self._configure_all_channels()
            self._sync_probe_map()
            if self.window.pipeline.spikes is not None:
                self._fill_probe_waveforms()
            return
        self.right.setMaximumWidth(16777215)
        self.right.setMinimumWidth(220)
        self.right.setCurrentWidget(self.side)
        self.right.updateGeometry()
        if switched:
            self._set_right_width(300)
        if mode == "Single channel":
            self._configure_trace_order([channel] if channel >= 0 else [])
        else:
            self._configure_trace_order(self._proximity_channels(channel))
        if self.window.pipeline.spikes is not None and channel >= 0:
            self.review.show_channel(channel, announce=False)

    def _set_right_width(self, width: int) -> None:
        total = sum(self.view.sizes())
        if total <= 0:
            total = 1250
        right = int(np.clip(width, self.right.minimumWidth(), max(self.right.maximumWidth(), width)))
        self.view.setSizes([max(total - right, 1), right])

    def _configure_all_channels(self) -> None:
        recording = self.window.pipeline.recording
        if recording is None:
            return
        order = np.asarray(self.window.display_order, dtype=int)
        if order.size == 0:
            order = np.arange(recording.n_channels, dtype=int)
        self.trace.order = order
        if order.size:
            selected = int(self._channel_id)
            self.trace.selected = selected if selected in set(order.tolist()) else int(order[len(order) // 2])
        self.trace.channel_gap = 1
        self.trace.channel_offset = 0
        self.trace._reveal_row(self.trace.selected)
        self.trace.refresh()

    def _configure_trace_order(self, channels: list[int]) -> None:
        recording = self.window.pipeline.recording
        if recording is None or not channels:
            return
        order = np.asarray(channels, dtype=int)
        self.trace.order = order
        real = [int(channel) for channel in channels if int(channel) >= 0]
        if self._channel_id in real:
            self.trace.selected = int(self._channel_id)
        elif real:
            self.trace.selected = int(real[0])
        self.trace.channel_gap = 1
        self.trace.channel_offset = 0
        if self.trace.selected >= 0:
            self.trace._reveal_row(self.trace.selected)
        self.trace.refresh()

    def _proximity_channels(self, channel: int) -> list[int]:
        """Center channel with ``+/-`` neighbors along the probe; missing sides are ``-1``."""
        radius = int(self.proximity_radius.value())
        slots = [-1] * (2 * radius + 1)
        slots[radius] = int(channel)
        if radius == 0:
            return slots
        probe = self.window.pipeline.probe
        recording = self.window.pipeline.recording
        n_channels = 0 if recording is None else recording.n_channels
        if n_channels <= 0 or channel < 0:
            return slots
        mask = self.window.mask_array()
        if probe is not None and probe.positions.shape[0] == n_channels:
            active = np.flatnonzero(probe.present & ~mask) if mask.size == n_channels else np.flatnonzero(probe.present)
            if active.size == 0:
                return slots
            # Top of the probe (high y) first so empty slots appear above edge sites.
            order = active[np.argsort(-probe.positions[active, 1], kind="mergesort")]
        else:
            order = np.asarray(
                [int(item) for item in self.window.display_order if 0 <= int(item) < n_channels],
                dtype=int,
            )
            if order.size == 0:
                order = np.arange(n_channels, dtype=int)
            if mask.size == n_channels:
                order = order[~mask[order]]
        hits = np.flatnonzero(order == int(channel))
        if hits.size == 0:
            return slots
        index = int(hits[0])
        for step in range(1, radius + 1):
            above = index - step
            below = index + step
            if above >= 0:
                slots[radius - step] = int(order[above])
            if below < order.size:
                slots[radius + step] = int(order[below])
        return slots

    def _sync_probe_map(self) -> None:
        probe = self.window.pipeline.probe
        recording = self.window.pipeline.recording
        if probe is not None:
            self.probe_waves.set_probe(
                probe.positions,
                probe.present,
                self.window.mask_array(),
                self._channel_id,
            )
            return
        if recording is None:
            self.probe_waves.set_probe(np.zeros((0, 2)), selected=0)
            return
        n = recording.n_channels
        cols = max(1, int(np.ceil(np.sqrt(n))))
        xs = np.arange(n) % cols
        ys = np.arange(n) // cols
        positions = np.column_stack([xs * 40.0, ys * 40.0]).astype(float)
        self.probe_waves.set_probe(positions, np.ones(n, dtype=bool), self.window.mask_array(), self._channel_id)

    def _fill_probe_waveforms(self) -> None:
        spikes = self.window.pipeline.spikes
        recording = self.window.pipeline.recording
        if spikes is None or recording is None:
            self.probe_waves.clear_waveforms()
            return
        pre = max(1, int(round(float(self.window.settings.waveform_pre_ms) * 1e-3 * recording.sampling_rate)))
        post = max(1, int(round(float(self.window.settings.waveform_post_ms) * 1e-3 * recording.sampling_rate)))
        length = pre + post
        bank: dict[int, np.ndarray] = {}
        clean = spikes.clean()
        for channel in np.unique(clean.channels):
            times = clean.times[clean.channels == channel]
            times = times[(times >= pre) & (times + post <= recording.n_samples)]
            if times.size == 0:
                continue
            if times.size > 100:
                rng = np.random.default_rng(int(channel) + 11)
                times = np.sort(rng.choice(times, size=100, replace=False))
            waves = np.full((times.size, length), np.nan, dtype=np.float32)
            for index, sample in enumerate(times):
                try:
                    snippet = self.window.signal_reader(int(sample) - pre, int(sample) + post)
                except Exception:
                    continue
                if snippet.shape[0] == length and int(channel) < snippet.shape[1]:
                    waves[index] = snippet[:, int(channel)]
            bank[int(channel)] = waves
        self.probe_waves.set_waveforms(bank, pre, post)

    def sync_review_marks(self) -> None:
        pipeline = self.window.pipeline
        self.trace.set_review(pipeline.review_times, pipeline.review_channels, pipeline.review_labels)

    def run_benchmark(self) -> None:
        self.window.load.collect()
        self.collect()

        def work(token, progress):
            del token
            progress(0.05, "Scoring a one-second excerpt")
            pipeline = self.window.pipeline
            if pipeline.recording is None or pipeline.probe is None:
                pipeline.open()
            recording = pipeline.recording
            probe = pipeline.probe
            if recording is None or probe is None:
                raise RuntimeError("Load a recording and a geometry file before the benchmark.")
            n_keep = min(recording.n_samples, max(1, int(recording.sampling_rate)))
            excerpt = np.asarray(self.window.signal_reader(0, n_keep), dtype=np.float32)
            spikes = pipeline.spikes
            mad = pipeline.noise_mad
            if spikes is None or mad is None:
                from silicon_sort.detect.detect import detect_traces

                tuned = self.window.settings
                spikes, mad, _whitener = detect_traces(
                    excerpt,
                    probe,
                    tuned,
                    self.window.mask_array(),
                    progress=progress,
                )
            return benchmark_excerpt(
                excerpt,
                spikes,
                probe,
                self.window.settings,
                self.window.mask_array(),
                mad,
                duration_s=n_keep / recording.sampling_rate,
            )

        self.window.run_task(work, self._show_curve)

    def _show_curve(self, curve) -> None:
        self._benchmark_history.append(curve)
        self.precision.set_history(
            [(run.thresholds, run.precision, run.recall) for run in self._benchmark_history],
            operating=abs(float(self.window.settings.threshold_mad)),
        )
        best = int(np.argmax(curve.recall[-1])) if curve.recall.size else 0
        if curve.recall.size:
            self.window.log(
                f"Benchmark at {curve.amplitudes[-1]:.0f} µV: "
                f"recall {curve.recall[-1, best]:.2f}, precision {curve.precision[-1, best]:.2f}"
            )


class _ViewModeProxy:
    """Small shim so existing tests can setCurrentText / read currentText."""

    def __init__(self, tab: DetectTab) -> None:
        self._tab = tab
        self._blocked = False

    def currentText(self) -> str:
        return self._tab.view_mode_text()

    def setCurrentText(self, text: str) -> None:
        mapping = {"Multi channel": 0, "Single channel": 1, "Proximity": 2}
        index = mapping.get(str(text), 0)
        buttons = (self._tab.multi_step, self._tab.single_step, self._tab.proximity_step)
        buttons[index].setChecked(True)
        self._tab.steps.setCurrentIndex(index)
        if not self._blocked:
            self._tab._apply_view_mode()

    def blockSignals(self, blocked: bool) -> None:
        self._blocked = bool(blocked)

    def currentTextChanged(self):  # pragma: no cover - unused Qt-like hook
        return None


class _CurrentPageStack(QStackedWidget):
    """Report size hints for the visible page so hidden review width does not leak."""

    def sizeHint(self):
        page = self.currentWidget()
        return page.sizeHint() if page is not None else super().sizeHint()

    def minimumSizeHint(self):
        page = self.currentWidget()
        return page.minimumSizeHint() if page is not None else super().minimumSizeHint()


def _group(title: str, body: QFormLayout | QVBoxLayout) -> QGroupBox:
    box = QGroupBox(title)
    box.setLayout(body)
    return box


def _vstack(*widgets: QWidget) -> QVBoxLayout:
    layout = QVBoxLayout()
    for widget in widgets:
        layout.addWidget(widget)
    return layout
