"""Quality-control plots for detection and clustering."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
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
        for curve in self._curves:
            self.removeItem(curve)
        self._curves.clear()
        if thresholds.size == 0:
            return
        # Plot the largest amplitude row, which is the last one.
        self._curves.append(self.plot(np.abs(thresholds), precision[-1], pen=pg.mkPen("#6ea8fe", width=2), name="Precision"))
        self._curves.append(self.plot(np.abs(thresholds), recall[-1], pen=pg.mkPen("#98c379", width=2), name="Recall"))


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
