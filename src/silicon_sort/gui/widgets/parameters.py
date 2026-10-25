"""Small helpers for labelled parameter widgets."""

from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QFormLayout, QSpinBox


def double_box(
    value: float,
    minimum: float,
    maximum: float,
    step: float,
    tip: str,
    decimals: int = 2,
) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(minimum, maximum)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setValue(value)
    box.setToolTip(tip)
    return box


def int_box(value: int, minimum: int, maximum: int, tip: str) -> QSpinBox:
    box = QSpinBox()
    box.setRange(minimum, maximum)
    box.setValue(value)
    box.setToolTip(tip)
    return box


def combo(options: list[str], current: str, tip: str) -> QComboBox:
    box = QComboBox()
    box.addItems(options)
    index = options.index(current) if current in options else 0
    box.setCurrentIndex(index)
    box.setToolTip(tip)
    return box


def add_row(form: QFormLayout, label: str, widget, tip: str | None = None) -> None:
    if tip:
        widget.setToolTip(tip)
    form.addRow(label, widget)
