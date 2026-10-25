"""Background workers so the window stays responsive."""

from __future__ import annotations

import traceback
from collections.abc import Callable

from PySide6.QtCore import QThread, Signal

from silicon_sort.cancel import CancelToken
from silicon_sort.errors import Cancelled

Work = Callable[[CancelToken, Callable[[float, str], None]], object]


class FunctionWorker(QThread):
    """Run a pipeline function off the GUI thread."""

    progressed = Signal(float, str)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, fn: Work):
        super().__init__()
        self._fn = fn
        self.token = CancelToken()

    def cancel(self) -> None:
        self.token.cancel()

    def run(self) -> None:
        try:
            result = self._fn(self.token, self._progress)
            self.succeeded.emit(True if result is None else result)
        except Cancelled:
            self.failed.emit("Cancelled")
        except Exception:
            self.failed.emit(traceback.format_exc())

    def _progress(self, fraction: float, message: str) -> None:
        self.progressed.emit(float(fraction), str(message))
