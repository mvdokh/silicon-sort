"""Scrolling multichannel trace viewer with peak-preserving decimation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton, QScrollBar, QSpinBox, QVBoxLayout, QWidget

Reader = Callable[[int, int], np.ndarray]


class TraceView(QWidget):
    """Stacked traces for one time window of a memmap.

    The scrollbar, mouse wheel, and left/right arrows move that window through
    the recording. Each row keeps its own color and channel number.
    """

    maskRequested = Signal(int)
    channelSelected = Signal(int)
    timePicked = Signal(int, int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.plot = _TracePlot(self)
        self.plot.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.plot.showGrid(x=True, y=False, alpha=0.12)
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.setLabel("left", "Channel")
        self.plot.setMenuEnabled(False)
        self.plot.hideButtons()
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.getAxis("left").setWidth(46)
        self.bar = QScrollBar(Qt.Orientation.Horizontal)
        self.bar.setToolTip("Scroll through the recording. The mouse wheel does the same. Hold Ctrl and scroll to zoom.")
        self.time_box = QDoubleSpinBox()
        self.time_box.setRange(5, 5000)
        self.time_box.setDecimals(0)
        self.time_box.setSingleStep(10)
        self.time_box.setSuffix(" ms")
        self.time_box.setValue(100)
        self.time_box.setToolTip("How much time is on screen. Smaller windows zoom in.")
        self.scale_box = QDoubleSpinBox()
        self.scale_box.setRange(10, 8000)
        self.scale_box.setDecimals(0)
        self.scale_box.setSingleStep(20)
        self.scale_box.setSuffix(" µV")
        self.scale_box.setValue(200)
        self.scale_box.setToolTip(
            "Vertical distance between channels. Lower this to make spikes taller."
        )
        self.auto_button = QPushButton("Auto")
        self.auto_button.setToolTip("Fit the vertical scale to the noise on screen.")
        self.gap_box = QSpinBox()
        self.gap_box.setRange(1, 128)
        self.gap_box.setValue(1)
        self.gap_box.setToolTip(
            "Channel spacing. 1 fits every channel on screen. Raise it to spread them apart, "
            "then use the bar on the right or Shift+scroll to move through one channel at a time."
        )
        self.channel_bar = QScrollBar(Qt.Orientation.Vertical)
        self.channel_bar.setToolTip("Scroll through channels. Shift+scroll does the same.")
        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.addWidget(QLabel("Window"))
        controls.addWidget(self.time_box)
        controls.addWidget(QLabel("µV / channel"))
        controls.addWidget(self.scale_box)
        controls.addWidget(self.auto_button)
        controls.addWidget(QLabel("Spacing"))
        controls.addWidget(self.gap_box)
        controls.addStretch(1)
        plot_row = QHBoxLayout()
        plot_row.setContentsMargins(0, 0, 0, 0)
        plot_row.setSpacing(2)
        plot_row.addWidget(self.plot, 1)
        plot_row.addWidget(self.channel_bar)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addLayout(controls)
        layout.addLayout(plot_row, 1)
        layout.addWidget(self.bar)

        self.reader: Reader | None = None
        self.n_samples = 0
        self.sampling_rate = 1.0
        self.n_channels = 0
        self.order = np.zeros(0, dtype=int)
        self.mask = np.zeros(0, dtype=bool)
        self.selected = 0
        self.start_s = 0.0
        self.window_s = 0.1
        self.spikes_t: np.ndarray | None = None
        self.spikes_c: np.ndarray | None = None
        self.spikes_artifact: np.ndarray | None = None
        self.threshold_uv: np.ndarray | None = None
        self.spacing = 200.0
        self.channel_gap = 1
        self.channel_offset = 0
        self._manual_scale = False
        self.review_times = np.zeros(0, dtype=np.int64)
        self.review_channels = np.zeros(0, dtype=np.int32)
        self.review_labels = np.zeros(0, dtype=np.int8)
        self.row_labels: list[str] = []
        self._curves: list[pg.PlotDataItem] = []
        self._marks = pg.ScatterPlotItem(size=6, pxMode=True)
        self.plot.addItem(self._marks)
        self.plot.scene().sigMouseClicked.connect(self._on_click)
        self.bar.valueChanged.connect(self._on_bar)
        self.channel_bar.valueChanged.connect(self._on_channel_bar)
        self.time_box.valueChanged.connect(self._on_time)
        self.scale_box.valueChanged.connect(self._on_scale)
        self.gap_box.valueChanged.connect(self._on_gap)
        self.auto_button.clicked.connect(self._on_auto_scale)
        self._updating_bar = False

    def set_source(
        self,
        reader: Reader,
        n_samples: int,
        sampling_rate: float,
        n_channels: int,
        order: np.ndarray | None = None,
        mask: np.ndarray | None = None,
    ) -> None:
        self.reader = reader
        self.n_samples = int(n_samples)
        self.sampling_rate = float(sampling_rate)
        self.n_channels = int(n_channels)
        self.order = np.arange(n_channels) if order is None else np.asarray(order, dtype=int)
        self.mask = np.zeros(n_channels, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        self.selected = int(self.order[len(self.order) // 2]) if self.order.size else 0
        self.start_s = 0.0
        self.refresh()

    def set_mask(self, mask: np.ndarray) -> None:
        self.mask = np.asarray(mask, dtype=bool)
        self.refresh()

    def set_spikes(
        self,
        times: np.ndarray | None,
        channels: np.ndarray | None,
        artifact: np.ndarray | None = None,
    ) -> None:
        if times is None or channels is None or times.size == 0:
            self.spikes_t = None
            self.spikes_c = None
            self.spikes_artifact = None
        else:
            self.spikes_t = np.asarray(times)
            self.spikes_c = np.asarray(channels)
            self.spikes_artifact = (
                np.zeros(times.shape, dtype=bool) if artifact is None else np.asarray(artifact, dtype=bool)
            )
        self.refresh()

    def set_review(self, times: np.ndarray, channels: np.ndarray, labels: np.ndarray) -> None:
        self.review_times = np.asarray(times, dtype=np.int64)
        self.review_channels = np.asarray(channels, dtype=np.int32)
        self.review_labels = np.asarray(labels, dtype=np.int8)
        self.refresh()

    def set_thresholds(self, threshold_uv: np.ndarray | None) -> None:
        self.threshold_uv = None if threshold_uv is None else np.asarray(threshold_uv, dtype=float)
        self.refresh()

    def reveal(self, channel: int) -> None:
        """Select a channel and scroll the stack until that row is on screen."""
        self.selected = int(channel)
        rows = [int(item) for item in self.order]
        if self.selected in rows:
            row = rows.index(self.selected)
            shown = self._shown_rows()
            if row < self.channel_offset or row >= self.channel_offset + shown:
                self.channel_offset = int(np.clip(row, 0, self._max_channel_offset()))
        self.refresh()

    def scroll_channels(self, step: int) -> None:
        self.channel_offset = int(np.clip(self.channel_offset + int(step), 0, self._max_channel_offset()))
        self.refresh()

    def scroll(self, fraction: float) -> None:
        self.start_s = float(np.clip(self.start_s + fraction * self.window_s, 0, self._max_start()))
        self.refresh()

    def zoom(self, factor: float) -> None:
        self.window_s = float(np.clip(self.window_s * factor, 0.005, 5.0))
        self.start_s = float(np.clip(self.start_s, 0, self._max_start()))
        self.refresh()

    def refresh(self) -> None:
        for curve in self._curves:
            self.plot.removeItem(curve)
        self._curves.clear()
        self._marks.setData([])
        self._sync_bar()
        if self.reader is None or self.n_channels == 0 or self.n_samples == 0:
            self.row_labels = []
            return
        start = int(self.start_s * self.sampling_rate)
        stop = min(self.n_samples, start + max(2, int(self.window_s * self.sampling_rate)))
        if stop <= start:
            return
        data = self.reader(start, stop)
        time, decimated = _minmax_decimate(data, target=1500, t0=start / self.sampling_rate, fs=self.sampling_rate)
        if not self._manual_scale:
            scale = np.median(np.std(decimated, axis=0))
            self.spacing = float(max(8 * scale, 50.0))
        self._sync_controls()
        row_of = {int(channel): row for row, channel in enumerate(self.order)}
        self.row_labels = []
        ticks: list[tuple[float, str]] = []
        for row, channel in enumerate(self.order):
            channel = int(channel)
            label = str(channel + 1)
            self.row_labels.append(label)
            offset = row * self.spacing
            ticks.append((offset, label))
            color = _channel_color(channel, masked=bool(self.mask[channel]), selected=channel == self.selected)
            width = 2.4 if channel == self.selected else 1.15
            curve = self.plot.plot(time, decimated[:, channel] + offset, pen=pg.mkPen(color, width=width))
            self._curves.append(curve)
            if self.threshold_uv is not None and np.isfinite(self.threshold_uv[channel]):
                level = offset - abs(float(self.threshold_uv[channel]))
                line = self.plot.plot(
                    time,
                    np.full(time.shape, level),
                    pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DashLine),
                )
                self._curves.append(line)
        self.plot.getAxis("left").setTicks([ticks])
        if time.size:
            self.plot.setXRange(float(time[0]), float(time[-1]), padding=0.01)
        rows = max(len(self.order), 1)
        shown = self._shown_rows()
        self.channel_offset = int(np.clip(self.channel_offset, 0, max(0, rows - shown)))
        self.plot.setYRange(
            (self.channel_offset - 0.55) * self.spacing,
            (self.channel_offset + shown - 0.45) * self.spacing,
            padding=0,
        )
        self._sync_channel_bar(rows, shown)
        if self.spikes_t is not None and self.spikes_c is not None:
            in_window = (self.spikes_t >= start) & (self.spikes_t < stop)
            xs = []
            ys = []
            brushes = []
            artifacts = (
                self.spikes_artifact[in_window]
                if self.spikes_artifact is not None
                else np.zeros(int(in_window.sum()), dtype=bool)
            )
            for sample, channel, artifact in zip(self.spikes_t[in_window], self.spikes_c[in_window], artifacts, strict=True):
                if int(channel) not in row_of:
                    continue
                xs.append(float(sample) / self.sampling_rate)
                ys.append(row_of[int(channel)] * self.spacing)
                brushes.append(pg.mkBrush(_spike_color(bool(artifact), self._review_label(int(channel), int(sample)))))
            if xs:
                self._marks.setData(x=np.array(xs), y=np.array(ys), brush=brushes, pen=None)

    def _on_time(self, value: float) -> None:
        self.window_s = float(np.clip(value / 1000.0, 0.005, 5.0))
        self.start_s = float(np.clip(self.start_s, 0, self._max_start()))
        self.refresh()

    def _on_scale(self, value: float) -> None:
        self._manual_scale = True
        self.spacing = float(value)
        self.refresh()

    def _on_gap(self, value: int) -> None:
        self.channel_gap = max(1, int(value))
        self.channel_offset = int(np.clip(self.channel_offset, 0, self._max_channel_offset()))
        self.refresh()

    def _on_auto_scale(self) -> None:
        self._manual_scale = False
        self.refresh()

    def _sync_controls(self) -> None:
        self.time_box.blockSignals(True)
        self.scale_box.blockSignals(True)
        self.time_box.setValue(self.window_s * 1000.0)
        self.scale_box.setValue(self.spacing)
        self.time_box.blockSignals(False)
        self.scale_box.blockSignals(False)

    def _review_label(self, channel: int, sample: int) -> int:
        if self.review_labels.size == 0:
            return 0
        tolerance = max(1, int(round(0.0004 * self.sampling_rate)))
        same = (self.review_channels == channel) & (np.abs(self.review_times - sample) <= tolerance)
        hits = np.flatnonzero(same)
        if hits.size == 0:
            return 0
        nearest = hits[int(np.argmin(np.abs(self.review_times[hits] - sample)))]
        return int(self.review_labels[nearest])

    def _wheel(self, event) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.zoom(0.8 if delta > 0 else 1.25)
        elif event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self.scroll_channels(-1 if delta > 0 else 1)
        else:
            self.scroll(-0.2 if delta > 0 else 0.2)
        event.accept()

    def _on_bar(self, value: int) -> None:
        if self._updating_bar:
            return
        self.start_s = float(value) / 1000.0
        self.refresh()

    def _sync_bar(self) -> None:
        duration_ms = 0.0 if self.sampling_rate <= 0 else self.n_samples / self.sampling_rate * 1000.0
        window_ms = self.window_s * 1000.0
        maximum = max(0, int(round(duration_ms - window_ms)))
        self._updating_bar = True
        self.bar.setRange(0, maximum)
        self.bar.setPageStep(max(1, int(window_ms)))
        self.bar.setSingleStep(max(1, int(window_ms / 20)))
        self.bar.setValue(int(np.clip(round(self.start_s * 1000.0), 0, maximum)))
        self._updating_bar = False

    def _on_channel_bar(self, value: int) -> None:
        if self._updating_bar:
            return
        self.channel_offset = int(value)
        self.refresh()

    def _sync_channel_bar(self, rows: int, shown: int) -> None:
        maximum = max(0, rows - shown)
        self._updating_bar = True
        self.channel_bar.setRange(0, maximum)
        self.channel_bar.setPageStep(max(1, shown))
        self.channel_bar.setSingleStep(1)
        self.channel_bar.setValue(int(np.clip(self.channel_offset, 0, maximum)))
        self._updating_bar = False

    def _shown_rows(self) -> int:
        rows = max(len(self.order), 1)
        return max(1, int(np.ceil(rows / max(1, self.channel_gap))))

    def _max_channel_offset(self) -> int:
        return max(0, len(self.order) - self._shown_rows())

    def _on_click(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self.spacing <= 0 or self.order.size == 0:
            return
        point = self.plot.plotItem.vb.mapSceneToView(event.scenePos())
        row = int(np.clip(round(point.y() / self.spacing), 0, self.order.size - 1))
        self.selected = int(self.order[row])
        sample = int(np.clip(round(float(point.x()) * self.sampling_rate), 0, max(self.n_samples - 1, 0)))
        self.channelSelected.emit(self.selected)
        self.timePicked.emit(self.selected, sample)
        self.refresh()

    def _max_start(self) -> float:
        if self.sampling_rate <= 0:
            return 0.0
        return max(0.0, self.n_samples / self.sampling_rate - self.window_s)


class _TracePlot(pg.PlotWidget):
    def __init__(self, owner: TraceView) -> None:
        super().__init__()
        self.owner = owner

    def wheelEvent(self, event) -> None:  # noqa: N802
        self.owner._wheel(event)


def _spike_color(artifact: bool, review: int) -> str:
    if review > 0:
        return "#98c379"
    if review < 0 or artifact:
        return "#e06c75"
    return "#f2f4f8"


def _channel_color(channel: int, masked: bool, selected: bool) -> QColor:
    hue = int((channel * 137.508) % 360)
    if masked:
        return QColor.fromHsv(hue, 50, 90)
    if selected:
        return QColor.fromHsv(hue, 190, 255)
    return QColor.fromHsv(hue, 170, 220)


def _minmax_decimate(data: np.ndarray, target: int, t0: float, fs: float) -> tuple[np.ndarray, np.ndarray]:
    n_samples = data.shape[0]
    if n_samples <= target:
        time = t0 + np.arange(n_samples) / fs
        return time, data
    bin_size = max(1, n_samples // target)
    usable = (n_samples // bin_size) * bin_size
    folded = data[:usable].reshape(-1, bin_size, data.shape[1])
    low = folded.min(axis=1)
    high = folded.max(axis=1)
    stacked = np.empty((low.shape[0] * 2, data.shape[1]), dtype=np.float32)
    stacked[0::2] = low
    stacked[1::2] = high
    edges = t0 + (np.arange(low.shape[0]) * bin_size) / fs
    time = np.empty(stacked.shape[0], dtype=float)
    time[0::2] = edges
    time[1::2] = edges + 0.5 * bin_size / fs
    return time, stacked
