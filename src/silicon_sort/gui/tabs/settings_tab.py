"""Appearance: VS Code color themes and UI text size."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from silicon_sort.gui.theme import current_font_px, current_theme, theme_names


class SettingsTab(QWidget):
    def __init__(self, window) -> None:
        super().__init__()
        self.window = window
        self.form = QWidget()
        layout = QVBoxLayout(self.form)
        layout.setContentsMargins(8, 8, 8, 8)
        form = QFormLayout()
        layout.addLayout(form)

        self.theme = QComboBox()
        self.theme.addItems(theme_names())
        self.theme.setCurrentText(current_theme().name)
        self.theme.setToolTip("VS Code-inspired color themes for the window and the plots.")
        form.addRow("Color theme", self.theme)

        self.font_slider = QSlider(Qt.Orientation.Horizontal)
        self.font_slider.setRange(10, 22)
        self.font_slider.setValue(current_font_px())
        self.font_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.font_slider.setTickInterval(2)
        self.font_slider.setToolTip("Size of text in the window, including the parameters and the log.")
        self.font_label = QLabel(f"{current_font_px()} px")
        font_row = QWidget()
        font_layout = QVBoxLayout(font_row)
        font_layout.setContentsMargins(0, 0, 0, 0)
        font_layout.setSpacing(4)
        font_layout.addWidget(self.font_slider)
        font_layout.addWidget(self.font_label)
        form.addRow("Text size", font_row)
        layout.addStretch(1)

        self.view = QWidget()
        view_layout = QVBoxLayout(self.view)
        view_layout.setContentsMargins(24, 24, 24, 24)
        self.preview = QLabel(
            "Appearance\n\n"
            "Choose a color theme and a text size. The choice is kept for the next launch."
        )
        self.preview.setWordWrap(True)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        view_layout.addWidget(self.preview)
        view_layout.addStretch(1)

        self.theme.currentTextChanged.connect(self._apply)
        self.font_slider.valueChanged.connect(self._on_font)

    def collect(self) -> None:
        return

    def _on_font(self, value: int) -> None:
        self.font_label.setText(f"{int(value)} px")
        self._apply()

    def _apply(self, _text: str = "") -> None:
        self.window.apply_appearance(self.theme.currentText(), int(self.font_slider.value()))
