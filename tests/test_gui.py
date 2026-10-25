"""Offscreen checks for the desktop window."""

from __future__ import annotations

import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_OPENGL", "software")

from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from silicon_sort.gui.main_window import MainWindow
from silicon_sort.gui.theme import apply_dark_theme
from silicon_sort.synthetic import write_poly2_cfg, write_recording


def _app() -> QApplication:
    application = QApplication.instance() or QApplication([])
    apply_dark_theme(application)
    return application


def _wait(window: MainWindow, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    while window._worker is not None and window._worker.isRunning():
        QApplication.processEvents()
        modal = QApplication.activeModalWidget()
        if modal is not None:
            modal.close()
        if time.time() > deadline:
            raise AssertionError(window.log_view.toPlainText() or "stage timed out")
        time.sleep(0.01)
    QApplication.processEvents()


def test_window_loads_masks_and_stays_responsive(tmp_path: Path):
    _app()
    dat = tmp_path / "session.dat"
    cfg = tmp_path / "probe.cfg"
    write_poly2_cfg(cfg, n_channels=8)
    write_recording(dat, n_channels=8, sampling_rate=30000, duration_s=0.25, seed=1)
    window = MainWindow(str(dat), str(cfg))
    window.load.n_channels.setValue(8)
    window.load.sampling_rate.setValue(30000)
    window.load.scale.setValue(0.195)
    window.show()
    QApplication.processEvents()
    window.load.show_recording()
    QApplication.processEvents()

    assert window.load.view.widget(0) is window.load.trace
    window.load.view.setSizes([1100, 200])
    QApplication.processEvents()
    assert window.load.probe.width() < window.load.trace.width()
    assert window.load.probe.width() <= 230
    window.load.trace.scale_box.setValue(80)
    assert window.load.trace.spacing == 80
    window.load.trace.time_box.setValue(40)
    assert abs(window.load.trace.window_s - 0.04) < 1e-6
    window.load.trace.time_box.setValue(100)
    window.load.trace.gap_box.setValue(8)
    assert window.load.trace._shown_rows() == 1
    window.load.trace.channel_bar.setValue(3)
    assert window.load.trace.channel_offset == 3
    window.load.trace.gap_box.setValue(1)
    assert window.detect.view.orientation() == Qt.Orientation.Horizontal
    assert window.detect.view.widget(0) is window.detect.trace
    side = window.detect.view.widget(1)
    assert side.widget(0) is window.detect.review
    assert side.widget(1) is window.detect.histogram
    assert window.detect.review.good_button.text() == "Good"
    assert set(window.load.trace.row_labels) == {str(channel) for channel in range(1, 9)}
    pens = [curve.opts["pen"].color().name() for curve in window.load.trace._curves]
    assert len(set(pens)) == 8
    bar = window.load.trace.bar
    assert bar.maximum() > 0
    bar.setValue(bar.maximum())
    assert window.load.trace.start_s > 0
    window.load.trace.scroll(-1.0)

    probe = window.load.probe
    site = probe.channel_at(float(probe.positions[3, 0]), float(probe.positions[3, 1]))
    assert site == 3
    window.toggle_mask(site)
    assert window.load.trace.mask[3]
    window.tabs.setCurrentIndex(1)
    window.refresh_masks()
    assert window.detect.trace.mask[3]
    window.tabs.setCurrentIndex(0)

    before = window.load.trace.start_s
    QTest.keyClick(window, Qt.Key.Key_Right)
    QApplication.processEvents()
    assert window.load.trace.start_s > before
    selected = int(window.load.trace.selected)
    masked_before = selected in window.settings.mask
    QTest.keyClick(window, Qt.Key.Key_M)
    QApplication.processEvents()
    assert (selected in window.settings.mask) is not masked_before

    ticks: list[bool] = []
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(threading.current_thread() is threading.main_thread()))
    timer.start(20)
    result: dict[str, object] = {}

    def work(token, progress):
        progress(0.4, "sleeping")
        time.sleep(0.2)
        progress(1.0, "awake")
        return threading.current_thread() is not threading.main_thread()

    window.run_task(work, lambda value: result.update(off_thread=value))
    _wait(window)
    timer.stop()
    assert result["off_thread"] is True
    assert any(ticks), "the GUI thread did not process timers while work ran"

    window.load.run_preprocess()
    _wait(window)
    assert window.pipeline.cache().filtered_path.is_file()
    assert 3 in window.settings.mask
    window.close()
