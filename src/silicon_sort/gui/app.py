"""Launch the desktop app."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from silicon_sort.gui.main_window import MainWindow, preload
from silicon_sort.gui.theme import apply_dark_theme


def launch(dat_path: str = "", cfg_path: str = "") -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    apply_dark_theme(app)
    window = MainWindow(dat_path, cfg_path)
    preload(window, dat_path, cfg_path)
    window.showMaximized()
    window.fit_layout()
    return app.exec()
