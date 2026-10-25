"""Dark Fusion theme and pyqtgraph defaults."""

from __future__ import annotations

import os

import pyqtgraph as pg
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


def apply_dark_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    palette = QPalette()
    base = QColor("#1b1d22")
    panel = QColor("#252830")
    text = QColor("#e6e8ee")
    muted = QColor("#a8b0bd")
    accent = QColor("#6ea8fe")
    palette.setColor(QPalette.ColorRole.Window, base)
    palette.setColor(QPalette.ColorRole.WindowText, text)
    palette.setColor(QPalette.ColorRole.Base, QColor("#14161b"))
    palette.setColor(QPalette.ColorRole.AlternateBase, panel)
    palette.setColor(QPalette.ColorRole.Text, text)
    palette.setColor(QPalette.ColorRole.Button, panel)
    palette.setColor(QPalette.ColorRole.ButtonText, text)
    palette.setColor(QPalette.ColorRole.Highlight, accent)
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#0d1117"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, panel)
    palette.setColor(QPalette.ColorRole.ToolTipText, text)
    palette.setColor(QPalette.ColorRole.PlaceholderText, muted)
    app.setPalette(palette)
    app.setStyleSheet(
        """
        QMainWindow, QWidget { font-size: 13px; }
        QGroupBox { font-weight: 600; border: 1px solid #343844; margin-top: 10px; padding: 8px; }
        QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
        QPushButton {
            background: #2e3440; border: 1px solid #465064; border-radius: 4px; padding: 6px 10px;
        }
        QPushButton:hover { background: #3a4252; }
        QPushButton:checked { background: #243044; color: #ffffff; border: 1px solid #6ea8fe; }
        QPushButton:disabled { color: #808898; }
        QLineEdit, QComboBox {
            background: #14161b; border: 1px solid #3a4150; border-radius: 4px; padding: 3px 6px;
        }
        QSpinBox, QDoubleSpinBox {
            background: #14161b; border: 1px solid #3a4150; border-radius: 0px; padding: 3px 6px;
        }
        QTabWidget::pane { border: none; margin: 0; padding: 0; }
        QTabBar::tab { padding: 4px 12px; }
        QTabBar::tab:selected { color: #ffffff; border-bottom: 2px solid #6ea8fe; }
        QSplitter::handle:horizontal { width: 2px; background: #1b1d22; }
        QSplitter::handle:vertical { height: 2px; background: #1b1d22; }
        QMainWindow::separator { width: 2px; height: 2px; background: #1b1d22; }
        QHeaderView::section { background: #252830; padding: 4px; border: none; }
        QDockWidget::title { background: #252830; padding: 6px; }
        """
    )
    offscreen = os.environ.get("QT_QPA_PLATFORM") == "offscreen"
    try:
        pg.setConfigOptions(useOpenGL=not offscreen, enableExperimental=not offscreen)
    except Exception:
        pg.setConfigOptions(useOpenGL=False)
    pg.setConfigOption("background", "#1b1d22")
    pg.setConfigOption("foreground", "#d7dbe2")
    pg.setConfigOption("antialias", False)
