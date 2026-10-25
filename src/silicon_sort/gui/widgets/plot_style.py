"""Shared plot axes: tick marks and numbers, no spine or grid."""

from __future__ import annotations

import pyqtgraph as pg

_TICK = "#d7dbe2"


class _Axis(pg.AxisItem):
    """Leave a gap between the axis title and the tick numbers."""

    def _updateWidth(self) -> None:
        super()._updateWidth()
        if self.orientation != "left" or self.fixedWidth is not None or not self.label.isVisible():
            return
        extra = int(round(self.label.boundingRect().height() * 0.45 + 10))
        width = int(self.minimumWidth() + extra)
        self.setMinimumWidth(width)
        self.setMaximumWidth(width)

    def _updateHeight(self) -> None:
        super()._updateHeight()
        if self.orientation != "bottom" or self.fixedHeight is not None or not self.label.isVisible():
            return
        extra = int(round(self.label.boundingRect().height() * 0.45 + 4))
        height = int(self.minimumHeight() + extra)
        self.setMinimumHeight(height)
        self.setMaximumHeight(height)


def style_plot(plot: pg.PlotWidget) -> None:
    """Ticks and numbers only, held on a small range until data arrives."""
    item = plot.getPlotItem()
    item.setAxisItems({"left": _Axis("left"), "bottom": _Axis("bottom")})
    item.hideButtons()
    item.setMenuEnabled(False)
    item.showGrid(x=False, y=False)
    item.hideAxis("top")
    item.hideAxis("right")
    item.layout.setContentsMargins(0, 0, 0, 0)
    item.getViewBox().setBorder(pg.mkPen(None))
    item.getViewBox().setMouseEnabled(x=False, y=False)
    for name in ("left", "bottom"):
        axis = item.getAxis(name)
        axis.setPen(None)
        axis.setTextPen(_TICK)
        axis.setTickPen(_TICK)
        axis.setStyle(tickTextOffset=4, autoExpandTextSpace=True, autoReduceTextSpace=False)
    item.disableAutoRange()
    rest_range(plot)


def rest_range(plot: pg.PlotWidget) -> None:
    plot.setXRange(0, 1, padding=0)
    plot.setYRange(0, 1, padding=0)
