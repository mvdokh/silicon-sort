"""Main window: four stages, dockable parameters, and a shared session."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QStatusBar,
    QTabWidget,
)
from PySide6.QtWidgets import QDockWidget

from silicon_sort.gui.tabs.autosort_tab import AutosortTab
from silicon_sort.gui.tabs.clusters_tab import ClustersTab
from silicon_sort.gui.tabs.detect_tab import DetectTab
from silicon_sort.gui.tabs.load_tab import LoadTab
from silicon_sort.gui.workers import FunctionWorker
from silicon_sort.pipeline import Pipeline
from silicon_sort.settings import PipelineSettings, load_settings, save_settings


class MainWindow(QMainWindow):
    def __init__(self, dat_path: str = "", cfg_path: str = "") -> None:
        super().__init__()
        self.setWindowTitle("Silicon Sort")
        self.resize(1280, 800)
        self.setDockOptions(
            QMainWindow.DockOption.AnimatedDocks
            | QMainWindow.DockOption.AllowNestedDocks
            | QMainWindow.DockOption.AllowTabbedDocks
        )
        self.settings = PipelineSettings(dat_path=dat_path, cfg_path=cfg_path)
        self.settings.preprocess = False
        self.pipeline = Pipeline(self.settings)
        self.suggested = np.zeros(0, dtype=bool)
        self.display_order = np.zeros(0, dtype=int)
        self.selected = 0
        self.picked: set[int] = set()
        self._worker: FunctionWorker | None = None
        self._filtered = None

        self.load = LoadTab(self)
        self.detect = DetectTab(self)
        self.clusters = ClustersTab(self)
        self.autosort = AutosortTab(self)
        self._stages = (self.load, self.detect, self.clusters, self.autosort)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setContentsMargins(0, 0, 0, 0)
        self.tabs.addTab(self.load.view, "1  Load")
        self.tabs.addTab(self.detect.view, "2  Detect")
        self.tabs.addTab(self.clusters.view, "3  Clusters")
        self.tabs.addTab(self.autosort.view, "4  Autosort")
        self.tabs.currentChanged.connect(self._stage_changed)
        self.setCentralWidget(self.tabs)

        self.stack = QStackedWidget()
        for stage in self._stages:
            self.stack.addWidget(stage.form)
        self._parameters = QDockWidget("Parameters", self)
        self._parameters.setWidget(self.stack)
        self._parameters.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self._parameters.setMinimumWidth(300)
        self._parameters.setMaximumWidth(420)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._parameters)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(400)
        self._log = QDockWidget("Log", self)
        self._log.setWidget(self.log_view)
        self._log.setMaximumHeight(140)
        self.splitDockWidget(self._parameters, self._log, Qt.Orientation.Vertical)

        self._build_menu()
        self._build_status()
        self._shortcuts()
        if dat_path:
            self.load.dat.setText(dat_path)
        if cfg_path:
            self.load.cfg.setText(cfg_path)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._close_filtered()
        self.pipeline.close()
        super().closeEvent(event)

    def collect_all(self) -> None:
        self.load.collect()
        self.detect.collect()
        self.clusters.collect()
        self.autosort.collect()

    def mask_array(self) -> np.ndarray:
        recording = self.pipeline.recording
        n_channels = recording.n_channels if recording is not None else int(self.settings.n_channels or 0)
        return self.settings.mask_array(n_channels)

    def raw_reader(self, start: int, stop: int) -> np.ndarray:
        recording = self.pipeline.recording
        if recording is None:
            self.pipeline.open()
            recording = self.pipeline.recording
        assert recording is not None
        return recording.read_uv(start, stop)

    def signal_reader(self, start: int, stop: int) -> np.ndarray:
        if self.settings.preprocess:
            return self.filtered_reader(start, stop)
        return self.raw_reader(start, stop)

    def filtered_reader(self, start: int, stop: int) -> np.ndarray:
        recording = self.pipeline.recording
        if recording is None:
            self.pipeline.open()
            recording = self.pipeline.recording
        assert recording is not None
        mapped = self._filtered
        shape = (recording.n_samples, recording.n_channels)
        if mapped is None or tuple(mapped.shape) != shape:
            self._close_filtered()
            self._filtered = self.pipeline.cache().open_filtered(*shape, mode="r")
            mapped = self._filtered
        return np.array(mapped[int(start) : int(stop)], dtype=np.float32, copy=True)

    def remember_file_order(self, n_channels: int) -> None:
        self.display_order = np.arange(int(n_channels), dtype=int)
        self.clusters.set_channels(list(range(int(n_channels))))
        if n_channels:
            self.selected = int(n_channels) // 2

    def remember_geometry(self, probe) -> None:
        self.display_order = np.argsort(probe.positions[:, 1], kind="mergesort")
        visible = [int(channel) for channel in self.display_order if probe.present[int(channel)]]
        self.clusters.set_channels(visible)
        if visible:
            self.selected = visible[len(visible) // 2]

    def refresh_probe(self) -> None:
        probe = self.pipeline.probe
        if probe is None:
            return
        suggested = self.suggested
        if suggested.shape != (probe.n_channels,):
            suggested = np.zeros(probe.n_channels, dtype=bool)
        self.load.probe.set_state(probe.positions, self.mask_array(), suggested, self.selected)

    def refresh_masks(self) -> None:
        mask = self.mask_array()
        self.load.trace.set_mask(mask)
        self.detect.trace.set_mask(mask)
        self.refresh_probe()

    def toggle_mask(self, channel: int) -> None:
        recording = self.pipeline.recording
        if recording is None:
            return
        mask = self.mask_array()
        channel = int(channel)
        if not 0 <= channel < mask.size:
            return
        mask[channel] = not mask[channel]
        self.settings.set_mask(mask)
        self.selected = channel
        self.refresh_masks()
        state = "masked" if mask[channel] else "included"
        self.log(f"Channel {channel} {state}")

    def accept_suggestions(self) -> None:
        if self.suggested.size == 0:
            self.log("No suggestion yet.")
            return
        mask = self.mask_array()
        suggested = np.zeros(mask.shape, dtype=bool)
        count = min(mask.size, int(self.suggested.size))
        suggested[:count] = np.asarray(self.suggested[:count], dtype=bool)
        self.settings.set_mask(mask | suggested)
        self.refresh_masks()
        self.log(f"Mask now has {int((mask | suggested).sum())} channels")

    def pick_channel(self, channel: int, additive: bool) -> None:
        channel = int(channel)
        if additive:
            if channel in self.picked:
                self.picked.discard(channel)
            else:
                self.picked.add(channel)
        else:
            self.picked = {channel}
        self.selected = channel if channel in self.picked else (next(iter(self.picked)) if self.picked else channel)
        self.load.trace.selected = self.selected
        self.load.trace.set_highlight(self.picked)
        self.refresh_probe()

    def mask_selected(self) -> None:
        if not self.picked:
            self.log("Select channels on the traces first. Shift-click or Ctrl-click adds another.")
            return
        recording = self.pipeline.recording
        if recording is None:
            self.log("Load a recording before masking channels.")
            return
        mask = self.mask_array()
        labels = []
        for channel in sorted(self.picked):
            if 0 <= channel < mask.size:
                mask[channel] = True
                labels.append(str(channel + 1))
        self.settings.set_mask(mask)
        self.refresh_masks()
        self.log(f"Masked channels {', '.join(labels)}")

    def clear_mask(self) -> None:
        self.settings.mask = []
        self.refresh_masks()
        self.log("Mask cleared")

    def select_channel(self, channel: int) -> None:
        self.selected = int(channel)
        self.refresh_probe()
        if self.tabs.currentIndex() == 1 and self.pipeline.spikes is not None:
            self.detect._channel(self.selected)

    def active_trace(self):
        if self.tabs.currentIndex() == 1:
            return self.detect.trace
        return self.load.trace

    def log(self, message: str) -> None:
        self.log_view.appendPlainText(message)
        self.statusBar().showMessage(message, 8000)

    def run_task(self, fn, on_success) -> None:
        if self._worker is not None and self._worker.isRunning():
            self.log("A stage is already running.")
            return
        self._close_filtered()
        self.collect_all()
        self._set_busy(True)
        self.progress.setValue(0)
        worker = FunctionWorker(fn)
        worker.progressed.connect(self._on_progress)
        worker.succeeded.connect(lambda result: self._finish(on_success, result))
        worker.failed.connect(self._fail)
        self._worker = worker
        worker.start()

    def cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.log("Cancelling…")

    def _finish(self, on_success, result) -> None:
        self._set_busy(False)
        self.progress.setValue(1000)
        on_success(result)

    def _fail(self, message: str) -> None:
        self._set_busy(False)
        if message == "Cancelled":
            self.log("Cancelled")
            return
        self.log(message.strip().splitlines()[-1])
        QMessageBox.warning(self, "Stage failed", message[-2500:])

    def _on_progress(self, fraction: float, message: str) -> None:
        self.progress.setValue(int(np.clip(fraction, 0, 1) * 1000))
        self.statusBar().showMessage(message, 4000)

    def _set_busy(self, busy: bool) -> None:
        for button in (
            self.load.run_button,
            self.load.suggest_button,
            self.detect.run_button,
            self.detect.benchmark_button,
            self.detect.review.suggest_button,
            self.clusters.preview_button,
            self.autosort.run_button,
            self.autosort.export_button,
        ):
            button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)

    def fit_layout(self) -> None:
        self.resizeDocks([self._parameters], [360], Qt.Orientation.Horizontal)
        self.resizeDocks([self._parameters, self._log], [900, 120], Qt.Orientation.Vertical)

    def _stage_changed(self, index: int) -> None:
        self.stack.setCurrentIndex(index)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        open_action = QAction("Open settings…", self)
        open_action.triggered.connect(self._open_settings)
        save_action = QAction("Save settings…", self)
        save_action.triggered.connect(self._save_settings)
        file_menu.addAction(open_action)
        file_menu.addAction(save_action)

    def _build_status(self) -> None:
        status = QStatusBar()
        self.setStatusBar(status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedWidth(180)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel)
        status.addPermanentWidget(self.progress)
        status.addPermanentWidget(self.cancel_button)

    def _shortcuts(self) -> None:
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, lambda: self.active_trace().scroll(0.5))
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, lambda: self.active_trace().scroll(-0.5))
        QShortcut(QKeySequence(Qt.Key.Key_Up), self, lambda: self.active_trace().scroll_channels(1))
        QShortcut(QKeySequence(Qt.Key.Key_Down), self, lambda: self.active_trace().scroll_channels(-1))
        QShortcut(QKeySequence(Qt.Key.Key_BracketRight), self, lambda: self.active_trace().zoom(1.25))
        QShortcut(QKeySequence(Qt.Key.Key_BracketLeft), self, lambda: self.active_trace().zoom(1 / 1.25))
        QShortcut(QKeySequence(Qt.Key.Key_M), self, lambda: self.toggle_mask(self.active_trace().selected))

    def _open_settings(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Settings", "", "JSON (*.json)")
        if not path:
            return
        self.settings = load_settings(path)
        self._close_filtered()
        self.pipeline.close()
        self.pipeline = Pipeline(self.settings)
        self._fill_forms()
        self.log(f"Loaded settings {path}")

    def _save_settings(self) -> None:
        self.collect_all()
        path, _ = QFileDialog.getSaveFileName(self, "Settings", "pipeline_settings.json", "JSON (*.json)")
        if not path:
            return
        save_settings(self.settings, path)
        self.log(f"Saved settings {path}")

    def _fill_forms(self) -> None:
        settings = self.settings
        self.load.dat.setText(settings.dat_path)
        self.load.cfg.setText(settings.cfg_path)
        if settings.n_channels:
            self.load.n_channels.setValue(int(settings.n_channels))
        if settings.sampling_rate:
            self.load.sampling_rate.setValue(float(settings.sampling_rate))
        self.load.dtype.setCurrentText(settings.dtype)
        self.load.scale.setValue(float(settings.voltage_scale_uv_per_bit))
        self.load.offset.setValue(int(settings.file_offset))
        self.load.highpass.setValue(float(settings.highpass_hz))
        self.load.lowpass.setValue(float(settings.lowpass_hz))
        self.load.reference.setCurrentText(settings.reference_mode)
        self.load.local_radius.setValue(float(settings.local_radius_um))
        self.load.preprocess.setCurrentText("on" if settings.preprocess else "off")
        self.load.whiten.setCurrentText("on" if settings.whiten else "off")
        self.detect.threshold.setValue(float(settings.threshold_mad))
        self.detect.radius.setValue(float(settings.neighbor_radius_um))
        self.detect.support.setValue(float(settings.neighbor_support_fraction))
        self.detect.min_neighbors.setValue(int(settings.min_neighbors))
        self.detect.dead.setValue(float(settings.dead_time_ms))
        self.detect.artifact_fraction.setValue(float(settings.artifact_fraction))
        self.detect.artifact_mad.setValue(float(settings.artifact_mad))
        self.detect.template_match.setCurrentText("on" if settings.template_match else "off")
        self.autosort.merge_cosine.setValue(float(settings.merge_cosine))
        self.autosort.merge_footprint.setValue(float(settings.merge_footprint))
        self.autosort.refractory.setValue(float(settings.refractory_ms))
        self.autosort.isi_split.setValue(float(settings.isi_split_fraction))

    def _close_filtered(self) -> None:
        mapped = self._filtered
        self._filtered = None
        if mapped is not None:
            del mapped


def preload(window: MainWindow, dat_path: str, cfg_path: str) -> None:
    if dat_path:
        window.load.dat.setText(dat_path)
    if cfg_path:
        window.load.cfg.setText(cfg_path)
    if dat_path:
        window.load.show_recording()
