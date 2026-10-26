"""Probe map with per-site spike waveforms instead of dots."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget

from silicon_sort.gui.widgets.plot_style import hide_axes, rest_range, show_axes, style_plot
from silicon_sort.gui.widgets.trace_view import _channel_color

_MAX_WAVES = 100


class ProbeWaveformMap(pg.PlotWidget):
    """Draw each channel at its probe site, with up to 100 random waveforms."""

    channelSelected = Signal(int)
    channelActivated = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        style_plot(self)
        self.setLabel("bottom", "x", units="µm")
        self.setLabel("left", "y", units="µm")
        hide_axes(self)
        self.setAspectLocked(False)
        self._curves: list[pg.PlotDataItem] = []
        self._markers = pg.ScatterPlotItem(pxMode=True, size=10)
        self.addItem(self._markers)
        self.positions = np.zeros((0, 2), dtype=float)
        self.present = np.zeros(0, dtype=bool)
        self.mask = np.zeros(0, dtype=bool)
        self.selected = 0
        self._waves: dict[int, np.ndarray] = {}
        self._pre = 0
        self._post = 0
        self._span_x = 40.0
        self._span_y = 40.0
        self.scene().sigMouseClicked.connect(self._on_click)

    def set_probe(
        self,
        positions: np.ndarray,
        present: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        selected: int = 0,
    ) -> None:
        self.positions = np.asarray(positions, dtype=float)
        n = self.positions.shape[0]
        self.present = np.ones(n, dtype=bool) if present is None else np.asarray(present, dtype=bool)
        self.mask = np.zeros(n, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        self.selected = int(selected)
        if n:
            xs = self.positions[self.present, 0] if self.present.any() else self.positions[:, 0]
            ys = self.positions[self.present, 1] if self.present.any() else self.positions[:, 1]
            self._span_x = max(18.0, float(np.ptp(xs)) / max(np.sqrt(n), 1.0) * 0.45)
            self._span_y = max(18.0, float(np.ptp(ys)) / max(np.sqrt(n), 1.0) * 0.45)
            show_axes(self)
            self.setXRange(float(xs.min()) - self._span_x, float(xs.max()) + self._span_x, padding=0.02)
            self.setYRange(float(ys.min()) - self._span_y, float(ys.max()) + self._span_y, padding=0.02)
        else:
            rest_range(self)
        self.refresh()

    def set_waveforms(
        self,
        waves_by_channel: dict[int, np.ndarray],
        pre: int,
        post: int,
    ) -> None:
        """``waves_by_channel`` maps channel -> (n_waves, n_samples) float arrays."""
        self._waves = {int(channel): np.asarray(waves, dtype=np.float32) for channel, waves in waves_by_channel.items()}
        self._pre = int(pre)
        self._post = int(post)
        self.refresh()

    def clear_waveforms(self) -> None:
        self._waves = {}
        self.refresh()

    def set_selected(self, channel: int) -> None:
        self.selected = int(channel)
        self.refresh()

    def set_mask(self, mask: np.ndarray) -> None:
        self.mask = np.asarray(mask, dtype=bool)
        self.refresh()

    def refresh(self) -> None:
        for curve in self._curves:
            self.removeItem(curve)
        self._curves.clear()
        n = self.positions.shape[0]
        if n == 0:
            self._markers.setData([])
            return
        marker_x = []
        marker_y = []
        brushes = []
        for channel in range(n):
            if not self.present[channel]:
                continue
            x0, y0 = self.positions[channel]
            waves = self._waves.get(channel)
            if waves is None or waves.size == 0:
                marker_x.append(x0)
                marker_y.append(y0)
                brushes.append(
                    pg.mkBrush(_channel_color(channel, masked=bool(self.mask[channel]), selected=channel == self.selected))
                )
                continue
            drawn = self._draw_site(channel, float(x0), float(y0), waves)
            if drawn is not None:
                self._curves.append(drawn)
        self._markers.setData(
            x=marker_x,
            y=marker_y,
            brush=brushes,
            pen=pg.mkPen("#111"),
        )

    def _draw_site(self, channel: int, x0: float, y0: float, waves: np.ndarray) -> pg.PlotDataItem | None:
        if waves.ndim == 1:
            waves = waves[None, :]
        count = min(waves.shape[0], _MAX_WAVES)
        if count <= 0:
            return None
        if waves.shape[0] > count:
            rng = np.random.default_rng(channel + 17)
            pick = rng.choice(waves.shape[0], size=count, replace=False)
            waves = waves[pick]
        length = waves.shape[1]
        if length < 2:
            return None
        amp = float(np.nanpercentile(np.abs(waves), 95)) if np.isfinite(waves).any() else 1.0
        amp = max(amp, 1.0)
        xs = np.linspace(-0.5 * self._span_x, 0.5 * self._span_x, length)
        block = length + 1
        x = np.full(count * block, np.nan, dtype=np.float64)
        y = np.full(count * block, np.nan, dtype=np.float64)
        for index, wave in enumerate(waves):
            offset = index * block
            x[offset : offset + length] = x0 + xs
            y[offset : offset + length] = y0 + (wave / amp) * (0.4 * self._span_y)
        color = _channel_color(channel, masked=bool(self.mask[channel]), selected=channel == self.selected)
        width = 1.6 if channel == self.selected else 0.9
        return self.plot(x, y, pen=pg.mkPen(color, width=width), connect="finite")

    def channel_at(self, x_um: float, y_um: float, max_um: float | None = None) -> int | None:
        if self.positions.size == 0:
            return None
        limit = float(max_um if max_um is not None else max(self._span_x, self._span_y))
        active = np.flatnonzero(self.present) if self.present.size else np.arange(self.positions.shape[0])
        if active.size == 0:
            return None
        distance = np.linalg.norm(self.positions[active] - np.array([x_um, y_um]), axis=1)
        index = int(np.argmin(distance))
        if distance[index] <= limit:
            return int(active[index])
        return None

    def _on_click(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        point = self.plotItem.vb.mapSceneToView(event.scenePos())
        channel = self.channel_at(point.x(), point.y())
        if channel is None:
            return
        self.selected = channel
        self.refresh()
        self.channelSelected.emit(channel)
        if event.double():
            self.channelActivated.emit(channel)
