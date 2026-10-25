"""Cancellation flag shared by the CLI, GUI, and pipeline."""

from __future__ import annotations

import threading


class CancelToken:
    """Thread-safe flag checked between chunks."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def cancelled(self) -> bool:
        return self._event.is_set()
