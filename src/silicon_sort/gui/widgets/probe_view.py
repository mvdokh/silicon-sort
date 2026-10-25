"""Probe map. Click a site to mask or unmask it."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget


class ProbeMap(pg.PlotWidget):
    channelClicked = Signal(int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setAspectLocked(False)
        self.setLabel("bottom", "x", units="µm")
        self.setLabel("left", "y", units="µm")
        self.showGrid(x=False, y=False)
        self.hideButtons()
        self.setMenuEnabled(False)
        self.getPlotItem().layout.setContentsMargins(0, 0, 0, 0)
        self.getAxis("left").setWidth(36)
        self.getAxis("bottom").setHeight(28)
        self.scatter = pg.ScatterPlotItem(pxMode=True, size=14)
        self.addItem(self.scatter)
        self.positions = np.zeros((0, 2), dtype=float)
        self.scene().sigMouseClicked.connect(self._on_click)

    def set_state(
        self,
        positions: np.ndarray,
        mask: np.ndarray,
        suggested: np.ndarray | None = None,
        selected: int | None = None,
    ) -> None:
        self.positions = np.asarray(positions, dtype=float)
        n_channels = self.positions.shape[0]
        if suggested is None:
            suggested = np.zeros(n_channels, dtype=bool)
        brushes = []
        for index in range(n_channels):
            if mask[index]:
                brushes.append(pg.mkBrush("#e06c75"))
            elif suggested[index]:
                brushes.append(pg.mkBrush("#e5c07b"))
            elif index == selected:
                brushes.append(pg.mkBrush("#ffffff"))
            else:
                brushes.append(pg.mkBrush("#6ea8fe"))
        self.scatter.setData(x=self.positions[:, 0], y=self.positions[:, 1], brush=brushes, pen=pg.mkPen("#111"))
        if n_channels:
            self.setXRange(float(self.positions[:, 0].min()) - 40, float(self.positions[:, 0].max()) + 40)
            self.setYRange(float(self.positions[:, 1].min()) - 40, float(self.positions[:, 1].max()) + 40)

    def channel_at(self, x_um: float, y_um: float, max_um: float = 40.0) -> int | None:
        if self.positions.size == 0:
            return None
        distance = np.linalg.norm(self.positions - np.array([x_um, y_um]), axis=1)
        index = int(np.argmin(distance))
        if distance[index] <= max_um:
            return index
        return None

    def _on_click(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        point = self.plotItem.vb.mapSceneToView(event.scenePos())
        channel = self.channel_at(point.x(), point.y())
        if channel is not None:
            self.channelClicked.emit(channel)
