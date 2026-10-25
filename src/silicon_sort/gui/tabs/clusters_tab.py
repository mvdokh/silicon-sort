"""Per-channel cluster count, waveform, PCA, and amplitude previews."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QFormLayout, QPushButton, QSplitter, QVBoxLayout, QWidget

from silicon_sort.gui.widgets.qc_plots import ScatterPreview, WaveformPreview


class ClustersTab(QWidget):
    def __init__(self, window) -> None:
        super().__init__()
        self.window = window
        self.form = QWidget()
        layout = QVBoxLayout(self.form)
        form = QFormLayout()
        layout.addLayout(form)
        self.channel = QComboBox()
        self.channel.setToolTip("Peak channel to preview. Each site can hold more than one unit.")
        self.k = QComboBox()
        self.k.addItem("Auto", 0)
        for value in range(1, 5):
            self.k.addItem(str(value), value)
        self.k.setToolTip("Forced cluster count for this channel. Auto chooses k by BIC, from 1 up to k max.")
        self.k_max = QComboBox()
        for value in range(1, 7):
            self.k_max.addItem(str(value), value)
        self.k_max.setCurrentIndex(3)
        self.k_max.setToolTip("Largest k BIC is allowed to pick when the channel is set to Auto.")
        form.addRow("Channel", self.channel)
        form.addRow("Clusters", self.k)
        form.addRow("BIC maximum", self.k_max)
        self.preview_button = QPushButton("Preview channel")
        self.preview_button.setToolTip("Fit this channel only, so you can judge the cluster count before autosort.")
        layout.addWidget(self.preview_button)
        layout.addStretch(1)

        self.waves = WaveformPreview()
        self.pca = ScatterPreview("PC 1", "PC 2")
        self.amplitude = ScatterPreview("Time (s)", "Amplitude (µV)")
        self.view = QSplitter()
        self.view.addWidget(self.waves)
        column = QSplitter(Qt.Orientation.Vertical)
        column.addWidget(self.pca)
        column.addWidget(self.amplitude)
        self.view.addWidget(column)

        self.preview_button.clicked.connect(self.preview)
        self.k.currentIndexChanged.connect(self._store_k)
        self.k_max.currentIndexChanged.connect(self._store_k)

    def set_channels(self, channels: list[int]) -> None:
        current = self.channel.currentData()
        self.channel.blockSignals(True)
        self.channel.clear()
        for channel in channels:
            self.channel.addItem(str(channel), channel)
        if current is not None:
            index = self.channel.findData(current)
            if index >= 0:
                self.channel.setCurrentIndex(index)
        self.channel.blockSignals(False)

    def collect(self) -> None:
        settings = self.window.settings
        settings.k_max = int(self.k_max.currentData())
        channel = self.channel.currentData()
        if channel is None:
            return
        forced = int(self.k.currentData())
        if forced > 0:
            settings.channel_k[str(int(channel))] = forced
        else:
            settings.channel_k.pop(str(int(channel)), None)

    def _store_k(self) -> None:
        self.collect()

    def preview(self) -> None:
        self.window.load.collect()
        self.window.detect.collect()
        self.collect()
        channel = self.channel.currentData()
        if channel is None:
            self.window.log("Load a recording before previewing clusters.")
            return

        def work(token, progress):
            progress(0.1, f"Previewing channel {channel}")
            return self.window.pipeline.preview_channel(int(channel), progress=progress, cancel=token)

        self.window.run_task(work, self._show)

    def _show(self, preview) -> None:
        if preview is None:
            self.window.log("No spikes on that channel.")
            return
        self.waves.set_waveforms(preview.waveforms, preview.labels)
        if preview.pca.shape[1] >= 2:
            self.pca.set_points(preview.pca[:, 0], preview.pca[:, 1], preview.labels)
        else:
            self.pca.set_points(preview.pca[:, 0], np.zeros(preview.pca.shape[0]), preview.labels)
        self.amplitude.set_points(preview.times_s, preview.amplitudes, preview.labels)
        self.window.log(f"Channel {preview.channel}: k={preview.k_used} (BIC suggested {preview.k_auto})")

