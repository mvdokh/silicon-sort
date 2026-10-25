"""Application icon loaded from the bundled SVG."""

from __future__ import annotations

from pathlib import Path

import PySide6.QtSvg  # noqa: F401  (registers the SVG icon plugin)
from PySide6.QtGui import QIcon


def icon_path() -> Path:
    return Path(__file__).resolve().parent / "assets" / "spike-sorter-icon-v6.svg"


def app_icon() -> QIcon:
    return QIcon(str(icon_path()))
