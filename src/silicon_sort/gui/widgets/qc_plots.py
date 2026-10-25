"""Quality-control plots for detection and clustering."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget


class HistogramPlot(pg.PlotWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        _relax(self)
        self.setLabel("bottom", "Amplitude", units="µV")
        self.setLabel("left", "Count")
        self.item = pg.BarGraphItem(x=[], height=[], width=1)
        self.addItem(self.item)

    def set_values(self, values: np.ndarray) -> None:
        if values.size == 0:
            self.item.setOpts(x=[], height=[], width=1)
            return
        counts, edges = np.histogram(np.abs(values), bins=40)
        centers = 0.5 * (edges[:-1] + edges[1:])
        width = float(edges[1] - edges[0]) if edges.size > 1 else 1.0
        self.item.setOpts(x=centers, height=counts, width=width * 0.9, brush=pg.mkBrush("#6ea8fe"))


class RatePlot(pg.PlotWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        _relax(self)
        self.setLabel("bottom", "Time", units="s")
        self.setLabel("left", "Spikes / bin")
        self.curve = self.plot(pen=pg.mkPen("#98c379", width=2))

    def set_times(self, times_s: np.ndarray, duration_s: float, bin_s: float = 1.0) -> None:
        if duration_s <= 0:
            self.curve.setData([], [])
            return
        bins = max(1, int(np.ceil(duration_s / bin_s)))
        counts, edges = np.histogram(times_s, bins=bins, range=(0, bins * bin_s))
        centers = 0.5 * (edges[:-1] + edges[1:])
        self.curve.setData(centers, counts)


class ScatterPreview(pg.PlotWidget):
    def __init__(self, xlabel: str, ylabel: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setLabel("bottom", xlabel)
        self.setLabel("left", ylabel)
        self.scatter = pg.ScatterPlotItem(size=7, pxMode=True)
        self.addItem(self.scatter)

    def set_points(self, x: np.ndarray, y: np.ndarray, labels: np.ndarray | None = None) -> None:
        if x.size == 0:
            self.scatter.setData([])
            return
        brushes = [_brush(0 if labels is None else int(label)) for label in (np.zeros(x.size, dtype=int) if labels is None else labels)]
        self.scatter.setData(x=x, y=y, brush=brushes, pen=None)


class WaveformPreview(pg.PlotWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setLabel("bottom", "Sample")
        self.setLabel("left", "µV")
        self._curves: list[pg.PlotDataItem] = []

    def set_waveforms(self, waveforms: np.ndarray, labels: np.ndarray) -> None:
        for curve in self._curves:
            self.removeItem(curve)
        self._curves.clear()
        if waveforms.size == 0:
            return
        time = np.arange(waveforms.shape[1])
        step = max(1, waveforms.shape[0] // 40)
        for row in range(0, waveforms.shape[0], step):
            curve = self.plot(time, waveforms[row], pen=pg.mkPen(_color(int(labels[row])), width=1))
            self._curves.append(curve)


class PRPlot(pg.PlotWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        _relax(self)
        self.setLabel("bottom", "Threshold (MAD)")
        self.setLabel("left", "Precision / recall")
        self.addLegend(offset=(4, 4))
        self._curves: list[pg.PlotDataItem] = []

    def set_curve(self, thresholds: np.ndarray, precision: np.ndarray, recall: np.ndarray) -> None:
        self.set_history([(thresholds, precision, recall)])

    def set_history(self, runs: list[tuple[np.ndarray, np.ndarray, np.ndarray]], operating: float | None = None) -> None:
        for curve in self._curves:
            self.removeItem(curve)
        for item in list(self.items()):
            if isinstance(item, pg.InfiniteLine):
                self.removeItem(item)
        self._curves.clear()
        if not runs:
            self.setTitle("")
            return
        for index, (thresholds, precision, _recall) in enumerate(runs):
            if thresholds.size == 0 or precision.size == 0:
                continue
            latest = index == len(runs) - 1
            color = "#6ea8fe" if latest else "#5c6370"
            width = 2.4 if latest else 1.2
            name = f"Run {index + 1}" if latest or len(runs) > 1 else "Precision"
            self._curves.append(
                self.plot(np.abs(thresholds), precision[-1], pen=pg.mkPen(color, width=width), name=name)
            )
        thresholds, _precision, recall = runs[-1]
        if thresholds.size and recall.size:
            self._curves.append(
                self.plot(np.abs(thresholds), recall[-1], pen=pg.mkPen("#98c379", width=2, style=Qt.PenStyle.DashLine), name="Recall")
            )
        title = ""
        if operating is not None and thresholds.size and runs[-1][1].size:
            row = runs[-1][1][-1]
            nearest = int(np.argmin(np.abs(np.abs(thresholds) - abs(float(operating)))))
            title = f"Precision {float(row[nearest]):.2f} at {abs(float(thresholds[nearest])):.1f} MAD"
            if len(runs) > 1 and runs[-2][1].size:
                previous = runs[-2][1][-1]
                earlier = int(np.argmin(np.abs(np.abs(runs[-2][0]) - abs(float(operating)))))
                delta = float(row[nearest]) - float(previous[earlier])
                title += f" ({delta:+.2f})"
            self.addItem(pg.InfiniteLine(pos=abs(float(operating)), angle=90, pen=pg.mkPen("#e5c07b", width=1)))
        self.setTitle(title)


def _relax(plot: pg.PlotWidget) -> None:
    plot.setMinimumHeight(220)
    plot.setMinimumWidth(180)
    item = plot.getPlotItem()
    item.hideButtons()
    item.setMenuEnabled(False)
    item.layout.setContentsMargins(0, 0, 0, 0)
    plot.getAxis("left").setWidth(40)
    plot.getAxis("bottom").setHeight(32)


_COLORS = ("#6ea8fe", "#e5c07b", "#c678dd", "#e06c75", "#56b6c2", "#98c379")


def _color(label: int) -> str:
    return _COLORS[label % len(_COLORS)]


def _brush(label: int):
    return pg.mkBrush(_color(label))
