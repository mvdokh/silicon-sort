"""VS Code-inspired themes, font size, and pyqtgraph defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass

import pyqtgraph as pg
from PySide6.QtCore import QSettings
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

_ORG = "silicon-sort"
_APP = "SiliconSort"


@dataclass(frozen=True)
class Theme:
    name: str
    base: str
    panel: str
    field: str
    text: str
    muted: str
    accent: str
    border: str
    button: str
    button_hover: str
    checked: str
    disabled: str
    plot_bg: str
    plot_fg: str
    highlight_text: str


THEMES: dict[str, Theme] = {
    "Dark+": Theme(
        name="Dark+",
        base="#1e1e1e",
        panel="#252526",
        field="#3c3c3c",
        text="#d4d4d4",
        muted="#9d9d9d",
        accent="#007acc",
        border="#3e3e42",
        button="#2d2d2d",
        button_hover="#3a3a3a",
        checked="#094771",
        disabled="#6a6a6a",
        plot_bg="#1e1e1e",
        plot_fg="#cccccc",
        highlight_text="#ffffff",
    ),
    "Light+": Theme(
        name="Light+",
        base="#ffffff",
        panel="#f3f3f3",
        field="#ffffff",
        text="#333333",
        muted="#6e6e6e",
        accent="#005fb8",
        border="#cecece",
        button="#e8e8e8",
        button_hover="#dcdcdc",
        checked="#cce8ff",
        disabled="#a0a0a0",
        plot_bg="#ffffff",
        plot_fg="#333333",
        highlight_text="#ffffff",
    ),
    "One Dark Pro": Theme(
        name="One Dark Pro",
        base="#282c34",
        panel="#21252b",
        field="#1e2227",
        text="#abb2bf",
        muted="#5c6370",
        accent="#61afef",
        border="#3b4048",
        button="#2c313a",
        button_hover="#353b45",
        checked="#2c3a4f",
        disabled="#636d83",
        plot_bg="#282c34",
        plot_fg="#abb2bf",
        highlight_text="#282c34",
    ),
    "Solarized Dark": Theme(
        name="Solarized Dark",
        base="#002b36",
        panel="#073642",
        field="#002b36",
        text="#839496",
        muted="#586e75",
        accent="#268bd2",
        border="#094553",
        button="#073642",
        button_hover="#0a4a57",
        checked="#094553",
        disabled="#586e75",
        plot_bg="#002b36",
        plot_fg="#93a1a1",
        highlight_text="#fdf6e3",
    ),
    "Solarized Light": Theme(
        name="Solarized Light",
        base="#fdf6e3",
        panel="#eee8d5",
        field="#fdf6e3",
        text="#657b83",
        muted="#93a1a1",
        accent="#268bd2",
        border="#ddd6c1",
        button="#eee8d5",
        button_hover="#e4ddc9",
        checked="#d6ebe8",
        disabled="#93a1a1",
        plot_bg="#fdf6e3",
        plot_fg="#657b83",
        highlight_text="#fdf6e3",
    ),
    "Monokai": Theme(
        name="Monokai",
        base="#272822",
        panel="#1e1f1c",
        field="#3e3d32",
        text="#f8f8f2",
        muted="#75715e",
        accent="#a6e22e",
        border="#3e3d32",
        button="#3e3d32",
        button_hover="#49483e",
        checked="#3e4a2d",
        disabled="#75715e",
        plot_bg="#272822",
        plot_fg="#f8f8f2",
        highlight_text="#272822",
    ),
    "GitHub Dark": Theme(
        name="GitHub Dark",
        base="#0d1117",
        panel="#161b22",
        field="#0d1117",
        text="#e6edf3",
        muted="#8b949e",
        accent="#2f81f7",
        border="#30363d",
        button="#21262d",
        button_hover="#30363d",
        checked="#1f3a5f",
        disabled="#6e7681",
        plot_bg="#0d1117",
        plot_fg="#e6edf3",
        highlight_text="#ffffff",
    ),
}

DEFAULT_THEME = "One Dark Pro"
DEFAULT_FONT_PX = 11
_MIN_FONT = 10
_MAX_FONT = 22

_current_theme = DEFAULT_THEME
_current_font_px = DEFAULT_FONT_PX


def theme_names() -> list[str]:
    return list(THEMES)


def current_theme() -> Theme:
    return THEMES.get(_current_theme, THEMES[DEFAULT_THEME])


def current_font_px() -> int:
    return int(_current_font_px)


_PREFS_VERSION = 2


def load_preferences() -> tuple[str, int]:
    store = QSettings(_ORG, _APP)
    try:
        version = int(store.value("prefs_version", 0))
    except (TypeError, ValueError):
        version = 0
    if version < _PREFS_VERSION:
        return DEFAULT_THEME, DEFAULT_FONT_PX
    name = str(store.value("theme", DEFAULT_THEME))
    if name not in THEMES:
        name = DEFAULT_THEME
    try:
        size = int(store.value("font_px", DEFAULT_FONT_PX))
    except (TypeError, ValueError):
        size = DEFAULT_FONT_PX
    return name, int(np_clip(size, _MIN_FONT, _MAX_FONT))


def save_preferences(theme_name: str, font_px: int) -> None:
    store = QSettings(_ORG, _APP)
    store.setValue("prefs_version", _PREFS_VERSION)
    store.setValue("theme", theme_name)
    store.setValue("font_px", int(font_px))


def np_clip(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def apply_theme(app: QApplication, theme_name: str | None = None, font_px: int | None = None) -> Theme:
    """Apply a named VS Code-style theme and optional UI font size."""
    global _current_theme, _current_font_px
    saved_name, saved_size = load_preferences()
    name = theme_name if theme_name is not None else saved_name
    size = font_px if font_px is not None else saved_size
    theme = THEMES.get(name, THEMES[DEFAULT_THEME])
    font_px = np_clip(size, _MIN_FONT, _MAX_FONT)
    _current_theme = theme.name
    _current_font_px = font_px
    save_preferences(theme.name, font_px)

    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(theme.base))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(theme.text))
    palette.setColor(QPalette.ColorRole.Base, QColor(theme.field))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(theme.panel))
    palette.setColor(QPalette.ColorRole.Text, QColor(theme.text))
    palette.setColor(QPalette.ColorRole.Button, QColor(theme.button))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(theme.text))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(theme.accent))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(theme.highlight_text))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(theme.panel))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(theme.text))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(theme.muted))
    app.setPalette(palette)
    font = QFont(app.font())
    font.setPointSize(font_px)
    app.setFont(font)
    app.setStyleSheet(_stylesheet(theme, font_px))
    _configure_pyqtgraph(theme)
    return theme


def apply_dark_theme(app: QApplication) -> Theme:
    """Backward-compatible entry point used by launch and tests."""
    return apply_theme(app)


def _configure_pyqtgraph(theme: Theme) -> None:
    offscreen = os.environ.get("QT_QPA_PLATFORM") == "offscreen"
    try:
        pg.setConfigOptions(useOpenGL=not offscreen, enableExperimental=not offscreen)
    except Exception:
        pg.setConfigOptions(useOpenGL=False)
    pg.setConfigOption("background", theme.plot_bg)
    pg.setConfigOption("foreground", theme.plot_fg)
    pg.setConfigOption("antialias", False)


def _stylesheet(theme: Theme, font_px: int) -> str:
    return f"""
        QMainWindow, QWidget {{ font-size: {font_px}px; }}
        QGroupBox {{
            font-weight: 600; border: 1px solid {theme.border}; margin-top: 10px; padding: 8px;
        }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}
        QPushButton {{
            background: {theme.button}; border: 1px solid {theme.border}; border-radius: 4px; padding: 6px 10px;
        }}
        QPushButton:hover {{ background: {theme.button_hover}; }}
        QPushButton:checked {{
            background: {theme.checked}; color: {theme.text}; border: 1px solid {theme.accent};
        }}
        QPushButton:disabled {{ color: {theme.disabled}; }}
        QLineEdit, QComboBox {{
            background: {theme.field}; border: 1px solid {theme.border}; border-radius: 4px; padding: 3px 6px;
        }}
        QSpinBox, QDoubleSpinBox {{
            background: {theme.field}; border: 1px solid {theme.border}; border-radius: 0px; padding: 3px 6px;
        }}
        QSlider::groove:horizontal {{
            height: 6px; background: {theme.border}; border-radius: 3px;
        }}
        QSlider::handle:horizontal {{
            width: 14px; margin: -5px 0; background: {theme.accent}; border-radius: 7px;
        }}
        QTabWidget::pane {{ border: none; margin: 0; padding: 0; }}
        QTabBar::tab {{ padding: 4px 12px; }}
        QTabBar::tab:selected {{ color: {theme.text}; border-bottom: 2px solid {theme.accent}; }}
        QSplitter::handle:horizontal {{ width: 2px; background: {theme.base}; }}
        QSplitter::handle:vertical {{ height: 2px; background: {theme.base}; }}
        QMainWindow::separator {{ width: 2px; height: 2px; background: {theme.base}; }}
        QHeaderView::section {{ background: {theme.panel}; padding: 4px; border: none; }}
        QDockWidget::title {{ background: {theme.panel}; padding: 6px; }}
        QPlainTextEdit {{
            background: {theme.field}; border: 1px solid {theme.border}; color: {theme.text};
        }}
    """
