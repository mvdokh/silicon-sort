"""Offscreen checks for the desktop window."""

from __future__ import annotations

import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_OPENGL", "software")

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from silicon_sort.detect.benchmark import BenchmarkCurve
from silicon_sort.detect.detect import SpikeIndex
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


def test_window_icon_loads():
    from silicon_sort.gui.icon import app_icon, icon_path

    _app()
    assert icon_path().is_file()
    icon = app_icon()
    assert not icon.isNull()
    window = MainWindow()
    assert not window.windowIcon().isNull()
    window.close()


def test_empty_plots_show_ticks_without_axis_lines():
    from PySide6.QtCore import Qt

    _app()
    window = MainWindow()
    window.show()
    QApplication.processEvents()
    plots = (
        window.load.probe,
        window.load.trace.plot,
        window.detect.trace.plot,
        window.detect.histogram,
        window.detect.precision,
        window.detect.review.plot,
        window.clusters.waves,
        window.clusters.pca,
    )
    for plot in plots:
        left = plot.getAxis("left")
        bottom = plot.getAxis("bottom")
        assert left.pen().style() == Qt.PenStyle.NoPen
        assert bottom.pen().style() == Qt.PenStyle.NoPen
        assert left.tickPen().style() != Qt.PenStyle.NoPen
        assert not plot.getAxis("top").isVisible()
        assert not plot.getAxis("right").isVisible()
        x_range, y_range = plot.viewRange()
        assert abs(x_range[1] - x_range[0]) < 5
        assert abs(y_range[1] - y_range[0]) < 5
    y_axis = window.load.probe.getAxis("left")
    QApplication.processEvents()
    assert y_axis.width() > y_axis.label.boundingRect().height() + 24
    window.close()


def test_load_parameters_are_split_into_load_preprocess_and_mask():
    _app()
    window = MainWindow()
    window.show()
    QApplication.processEvents()

    def page_of(widget) -> int:
        page = widget
        while page is not None and window.load.steps.indexOf(page) < 0:
            page = page.parentWidget()
        return window.load.steps.indexOf(page)

    assert window.load.steps.currentIndex() == 0
    assert window.load.load_step.isChecked()
    assert page_of(window.load.dat) == 0
    assert page_of(window.load.cfg) == 0
    assert page_of(window.load.run_button) == 1
    assert page_of(window.load.highpass) == 1
    assert page_of(window.load.suggest_button) == 2
    assert page_of(window.load.mask_selected_button) == 2
    window.load.preprocess_step.click()
    QApplication.processEvents()
    assert window.load.steps.currentIndex() == 1
    assert window.load.preprocess_step.isChecked()
    window.load.mask_step.click()
    QApplication.processEvents()
    assert window.load.steps.currentIndex() == 2
    assert window.load.mask_step.isChecked()
    window.load.load_step.click()
    QApplication.processEvents()
    assert window.load.steps.currentIndex() == 0
    window.close()


def test_benchmark_plot_keeps_each_run():
    _app()
    window = MainWindow()
    first = BenchmarkCurve(
        amplitudes=np.array([80.0]),
        thresholds=np.array([-4.0, -6.0]),
        precision=np.array([[0.40, 0.70]]),
        recall=np.array([[0.90, 0.50]]),
    )
    second = BenchmarkCurve(
        amplitudes=np.array([80.0]),
        thresholds=np.array([-4.0, -6.0]),
        precision=np.array([[0.55, 0.86]]),
        recall=np.array([[0.88, 0.48]]),
    )
    window.settings.threshold_mad = -6.0
    window.detect._show_curve(first)
    window.detect._show_curve(second)
    assert len(window.detect.precision._curves) == 3
    assert "0.86" in window.detect.precision.plotItem.titleLabel.text
    assert "+0.16" in window.detect.precision.plotItem.titleLabel.text
    window.close()


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
    assert window.load.trace.channel_offset == 7
    assert window.load.trace.channel_bar.value() == 0
    window.load.trace.channel_bar.setValue(window.load.trace.channel_bar.maximum())
    assert window.load.trace.channel_offset == 0
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


def test_dat_stays_in_file_order_until_cfg(tmp_path: Path):
    _app()
    dat = tmp_path / "session.dat"
    cfg = tmp_path / "probe.cfg"
    write_recording(dat, n_channels=8, sampling_rate=30000, duration_s=0.05, seed=1)
    cfg.write_text(
        "reversed\n" + "\n".join(f"{index + 1} {index + 1} 0.00 {70 - index * 10:.2f}" for index in range(8)) + "\n",
        encoding="utf-8",
    )
    window = MainWindow()
    window.load.n_channels.setValue(8)
    window.load.sampling_rate.setValue(30000)
    window.load.dat.setText(str(dat))
    window.load.show_recording()
    QApplication.processEvents()
    assert list(window.display_order) == list(range(8))
    assert window.pipeline.probe is None

    window.load.cfg.setText(str(cfg))
    window.load.show_recording()
    QApplication.processEvents()
    assert list(window.display_order) == list(range(7, -1, -1))
    window.close()


def test_review_draws_every_waveform_and_seeks_the_trace(tmp_path: Path):
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
    window.load.show_recording()
    times = np.linspace(800, 6800, 30).astype(np.int64)
    window.pipeline.spikes = SpikeIndex(
        times,
        np.zeros(times.size, dtype=np.int32),
        np.full(times.size, -80.0, dtype=np.float32),
        np.zeros(times.size, dtype=bool),
    )
    window.detect.show_detection()
    review = window.detect.review
    trace = window.detect.trace
    assert review._waves.shape[0] == 30
    assert np.isfinite(review._waves).all()
    assert len(review._curves) == 2

    review._step_event(1)
    sample = int(review._events[1][0])
    center = sample / trace.sampling_rate
    assert review.selected == 1
    assert trace.focused_sample == sample
    assert trace.focused_channel == 0
    assert trace.start_s <= center <= trace.start_s + trace.window_s

    review._step_event(-1)
    assert trace.focused_sample == int(review._events[0][0])

    target = 17
    pre = max(1, int(round(0.6e-3 * trace.sampling_rate)))
    column = int(np.argmax(np.ptp(review._waves, axis=0)))
    gap = np.abs(review._waves[:, column] - review._waves[target, column])
    gap[target] = np.inf
    if float(gap.min()) <= 1e-4:
        column = pre
    time_ms = (column - pre) / trace.sampling_rate * 1000.0
    review.pick_waveform(time_ms, float(review._waves[target, column]))
    picked = int(review._events[review.selected][0])
    center = picked / trace.sampling_rate
    assert review.selected == target
    assert trace.focused_sample == picked
    assert trace.start_s <= center <= trace.start_s + trace.window_s
    boxes = trace._boxes.boxes
    assert boxes.shape[0] > 0
    assert np.allclose(boxes[:, 2], 0.0006 + 0.0012)
    assert np.allclose(boxes[:, 3], trace.spacing)

    late = int(times[22])
    review.show_channel(1)
    window.detect._channel(0)
    review.consider_click(0, late + 3)
    assert review.channel == 0
    assert int(review._events[review.selected][0]) == late
    assert trace.focused_sample == late

    review.click_action.setCurrentText("Mark bad")
    review.consider_click(0, int(times[4]))
    assert review._label_at(int(times[4])) == -1
    review.bad_apply.setCurrentText("Marked only")
    review.good_apply.setCurrentText("Similar on every channel")
    review.apply_examples()
    assert window.pipeline.review_bad == "marked"
    assert window.pipeline.review_good == "all"

    from silicon_sort.gui.widgets.trace_view import ADDED_MISS_COLOR

    review.click_action.setCurrentText("Add missed spike")
    review.consider_click(0, 200)
    added = review._events[review.selected]
    assert not np.isfinite(added[2])
    assert ADDED_MISS_COLOR in trace._boxes.colors
    pen = review._curves[-1].opts["pen"]
    assert pen.color().name().lower() == ADDED_MISS_COLOR
    assert window.detect.precision.parent() is window.detect.form
    assert window.detect.view.widget(1).indexOf(window.detect.precision) < 0
    window.close()


def test_selected_channels_stay_in_the_mask(tmp_path: Path):
    _app()
    dat = tmp_path / "session.dat"
    cfg = tmp_path / "probe.cfg"
    write_poly2_cfg(cfg, n_channels=8)
    write_recording(dat, n_channels=8, sampling_rate=30000, duration_s=0.05, seed=1)
    window = MainWindow(str(dat), str(cfg))
    window.load.n_channels.setValue(8)
    window.load.sampling_rate.setValue(30000)
    window.load.scale.setValue(0.195)
    window.load.show_recording()
    window.pick_channel(1, False)
    window.pick_channel(4, True)
    window.mask_selected()
    assert set(window.settings.mask) == {1, 4}
    window.suggested = np.zeros(8, dtype=bool)
    window.suggested[2] = True
    window.accept_suggestions()
    assert set(window.settings.mask) == {1, 2, 4}
    window.pick_channel(6, False)
    window.mask_selected()
    assert set(window.settings.mask) == {1, 2, 4, 6}
    window.close()
