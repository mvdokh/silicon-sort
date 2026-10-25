"""Detection parameters, quality-control plots, and the hybrid benchmark."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFormLayout, QPushButton, QSplitter, QVBoxLayout, QWidget

from silicon_sort.detect.benchmark import benchmark_excerpt
from silicon_sort.gui.widgets.event_review import EventReview
from silicon_sort.gui.widgets.parameters import combo, double_box, int_box
from silicon_sort.gui.widgets.qc_plots import HistogramPlot
from silicon_sort.gui.widgets.trace_view import TraceView


class DetectTab(QWidget):
    def __init__(self, window) -> None:
        super().__init__()
        self.window = window
        self.form = QWidget()
        layout = QVBoxLayout(self.form)
        form = QFormLayout()
        layout.addLayout(form)
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
        form.addRow("Threshold MAD", self.threshold)
        form.addRow("Neighbor radius µm", self.radius)
        form.addRow("Neighbor fraction", self.support)
        form.addRow("Min neighbors", self.min_neighbors)
        form.addRow("Dead time ms", self.dead)
        form.addRow("Artifact fraction", self.artifact_fraction)
        form.addRow("Artifact MAD", self.artifact_mad)
        form.addRow("Template match", self.template_match)
        self.run_button = QPushButton("Run detection")
        self.run_button.setToolTip(
            "Detect spikes. Uses the raw recording when preprocessing is off, and the filtered cache when it is on. "
            "Good and bad marks from the review panel are applied on this run."
        )
        self.benchmark_button = QPushButton("Run benchmark")
        self.benchmark_button.setToolTip(
            "Plant known waveforms into a short copy of the filtered noise and plot precision and recall."
        )
        layout.addWidget(self.run_button)
        layout.addWidget(self.benchmark_button)
        layout.addStretch(1)

        self.trace = TraceView()
        self.review = EventReview(window)
        self.histogram = HistogramPlot()
        self.histogram.setMinimumWidth(260)
        side = QSplitter(Qt.Orientation.Vertical)
        side.addWidget(self.review)
        side.addWidget(self.histogram)
        side.setStretchFactor(0, 3)
        side.setStretchFactor(1, 2)
        side.setSizes([420, 240])
        self.view = QSplitter(Qt.Orientation.Horizontal)
        self.view.addWidget(self.trace)
        self.view.addWidget(side)
        self.view.setStretchFactor(0, 3)
        self.view.setStretchFactor(1, 0)
        self.view.setSizes([900, 360])

        self.run_button.clicked.connect(self.run_detection)
        self.benchmark_button.clicked.connect(self.run_benchmark)
        self.trace.channelSelected.connect(self._channel)
        self.trace.timePicked.connect(self.review.consider_click)
        self.trace.maskRequested.connect(window.toggle_mask)
        self.review.channelChanged.connect(self._from_review)
        self._channel_id = 0

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

    def run_detection(self) -> None:
        self.window.load.collect()
        self.collect()

        def work(token, progress):
            return self.window.pipeline.run_detect(progress=progress, cancel=token)

        self.window.run_task(work, lambda _spikes: self.show_detection())

    def show_detection(self) -> None:
        recording = self.window.pipeline.recording
        spikes = self.window.pipeline.spikes
        if recording is None or spikes is None:
            return
        self.trace.set_source(
            self.window.signal_reader,
            recording.n_samples,
            recording.sampling_rate,
            recording.n_channels,
            self.window.display_order,
            self.window.mask_array(),
        )
        mad = self.window.pipeline.noise_mad
        if mad is not None:
            self.trace.set_thresholds(np.abs(self.window.settings.threshold_mad) * mad)
        self.trace.set_spikes(spikes.times, spikes.channels, spikes.artifact)
        self.sync_review_marks()
        self.review.show_channel(self._channel_id if self._channel_id < recording.n_channels else 0)
        clean = spikes.clean()
        self._channel(self._channel_id if self._channel_id < recording.n_channels else 0)
        self.window.log(f"{clean.n_spikes} spikes, {int(spikes.artifact.sum())} artifact-flagged")

    def _channel(self, channel: int) -> None:
        self._channel_id = int(channel)
        spikes = self.window.pipeline.spikes
        if spikes is None:
            return
        values = spikes.amplitudes[(spikes.channels == channel) & ~spikes.artifact]
        self.histogram.set_values(values)
        if self.review.channel != int(channel):
            self.review.show_channel(int(channel))

    def _from_review(self, channel: int) -> None:
        self._channel_id = int(channel)
        self.trace.reveal(int(channel))
        spikes = self.window.pipeline.spikes
        if spikes is None:
            return
        values = spikes.amplitudes[(spikes.channels == channel) & ~spikes.artifact]
        self.histogram.set_values(values)

    def sync_review_marks(self) -> None:
        pipeline = self.window.pipeline
        self.trace.set_review(pipeline.review_times, pipeline.review_channels, pipeline.review_labels)

    def run_benchmark(self) -> None:
        self.window.load.collect()
        self.collect()

        def work(token, progress):
            progress(0.05, "Preparing a hybrid excerpt")
            self.window.pipeline.run_detect(progress=progress, cancel=token)
            recording = self.window.pipeline.recording
            probe = self.window.pipeline.probe
            spikes = self.window.pipeline.spikes
            mad = self.window.pipeline.noise_mad
            assert recording is not None and probe is not None and spikes is not None and mad is not None
            n_keep = min(recording.n_samples, int(2.0 * recording.sampling_rate))
            excerpt = self.window.signal_reader(0, n_keep)
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
        best = int(np.argmax(curve.recall[-1])) if curve.recall.size else 0
        if curve.recall.size:
            self.window.log(
                f"Benchmark at {curve.amplitudes[-1]:.0f} µV: "
                f"recall {curve.recall[-1, best]:.2f}, precision {curve.precision[-1, best]:.2f}"
            )
