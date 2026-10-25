"""Load a recording, confirm geometry, and mask channels."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from silicon_sort.errors import RecordingError
from silicon_sort.gui.widgets.parameters import combo, double_box, int_box
from silicon_sort.gui.widgets.probe_view import ProbeMap
from silicon_sort.gui.widgets.trace_view import TraceView
from silicon_sort.io.binary import hints_from_binary_json, load_sidecar
from silicon_sort.io.geometry import parse_msk
from silicon_sort.preprocess.bad_channels import suggest_bad_channels


class LoadTab(QWidget):
    def __init__(self, window) -> None:
        super().__init__()
        self.window = window
        self.form = QWidget()
        layout = QVBoxLayout(self.form)
        form = QFormLayout()
        layout.addLayout(form)
        self.dat = QLineEdit()
        self.dat.setToolTip("Headerless interleaved binary, samples by channels.")
        self.cfg = QLineEdit()
        self.cfg.setToolTip("Probe geometry: a name line, then site channel x y in micrometers.")
        form.addRow("Recording", _browse_row(self.dat, self._browse_dat))
        form.addRow("Geometry", _browse_row(self.cfg, self._browse_cfg))
        self.n_channels = int_box(32, 1, 1024, "Number of channels stored in the binary file.")
        self.sampling_rate = double_box(30000, 1000, 100000, 1000, "Sampling rate in Hz.", decimals=1)
        self.dtype = combo(
            ["int16", "int32", "uint16", "float32", "float64"],
            "int16",
            "Sample dtype. Intan amplifier files are int16.",
        )
        self.scale = double_box(0.195, 0.0, 1000, 0.001, "Microvolts per bit. Intan RHD2132 is 0.195.", decimals=4)
        self.offset = int_box(0, 0, 10**9, "Bytes to skip before the first sample.")
        form.addRow("Channels", self.n_channels)
        form.addRow("Sampling rate", self.sampling_rate)
        form.addRow("Dtype", self.dtype)
        form.addRow("µV / bit", self.scale)
        form.addRow("Offset", self.offset)
        self.preprocess = combo(
            ["off", "on"],
            "off",
            "Off keeps the raw recording for viewing, detection, and clustering. On bandpasses and applies the common median reference first. You can leave this off and still scroll the raw traces.",
        )
        form.addRow("Preprocessing", self.preprocess)
        self.highpass = double_box(300, 0.1, 10000, 50, "Highpass cutoff in Hz, used only when preprocessing is on.")
        self.lowpass = double_box(6000, 100, 20000, 100, "Lowpass cutoff in Hz.")
        self.reference = combo(
            ["global", "shank", "local"],
            "global",
            "Common median reference. Shanks split on x gaps. Local groups use the radius below.",
        )
        self.local_radius = double_box(100, 10, 500, 10, "Radius in µm for local-group referencing.")
        self.whiten = combo(["on", "off"], "on", "ZCA whitening is applied during detection and is not written to the cache.")
        form.addRow("Highpass Hz", self.highpass)
        form.addRow("Lowpass Hz", self.lowpass)
        form.addRow("Reference", self.reference)
        form.addRow("Local radius", self.local_radius)
        form.addRow("Whiten", self.whiten)

        self.show_button = QPushButton("Show recording")
        self.show_button.setToolTip("Open the memmap, draw the probe, and show raw traces.")
        self.suggest_button = QPushButton("Suggest bad channels")
        self.suggest_button.setToolTip("Flag flat, railed, or very noisy channels from a few raw chunks. Nothing is masked until you accept.")
        self.accept_button = QPushButton("Accept suggestions")
        self.accept_button.setToolTip("Replace the current mask with the suggestion. Click sites afterwards to override.")
        self.clear_button = QPushButton("Clear mask")
        self.mask_button = QPushButton("Import .msk")
        self.mask_button.setToolTip("Import a 1-based channel T/F mask. T excludes the channel.")
        self.run_button = QPushButton("Run preprocessing")
        self.run_button.setToolTip("Optional. Bandpass and common-median reference the recording into the cache. Detection uses it only when preprocessing is on.")
        for button in (
            self.show_button,
            self.suggest_button,
            self.accept_button,
            self.clear_button,
            self.mask_button,
            self.run_button,
        ):
            layout.addWidget(button)
        layout.addStretch(1)

        self.probe = ProbeMap()
        self.probe.setMinimumWidth(150)
        self.probe.setMaximumWidth(230)
        self.trace = TraceView()
        self.view = QSplitter()
        self.view.addWidget(self.trace)
        self.view.addWidget(self.probe)
        self.view.setStretchFactor(0, 1)
        self.view.setStretchFactor(1, 0)
        self.view.setSizes([1100, 200])

        self.show_button.clicked.connect(self.show_recording)
        self.suggest_button.clicked.connect(self.suggest)
        self.accept_button.clicked.connect(window.accept_suggestions)
        self.clear_button.clicked.connect(window.clear_mask)
        self.mask_button.clicked.connect(self.import_mask)
        self.run_button.clicked.connect(self.run_preprocess)
        self.probe.channelClicked.connect(window.toggle_mask)
        self.trace.maskRequested.connect(window.toggle_mask)
        self.trace.channelSelected.connect(window.select_channel)

    def collect(self) -> None:
        settings = self.window.settings
        settings.dat_path = self.dat.text().strip()
        settings.cfg_path = self.cfg.text().strip()
        settings.n_channels = int(self.n_channels.value())
        settings.sampling_rate = float(self.sampling_rate.value())
        settings.dtype = self.dtype.currentText()
        settings.voltage_scale_uv_per_bit = float(self.scale.value())
        settings.file_offset = int(self.offset.value())
        settings.preprocess = self.preprocess.currentText() == "on"
        settings.highpass_hz = float(self.highpass.value())
        settings.lowpass_hz = float(self.lowpass.value())
        settings.reference_mode = self.reference.currentText()
        settings.local_radius_um = float(self.local_radius.value())
        settings.whiten = self.whiten.currentText() == "on"

    def apply_hints(self, hints: dict) -> None:
        if "n_channels" in hints:
            self.n_channels.setValue(int(hints["n_channels"]))
        if "sampling_rate" in hints:
            self.sampling_rate.setValue(float(hints["sampling_rate"]))
        if "dtype" in hints:
            self.dtype.setCurrentText(str(hints["dtype"]))
        if "voltage_scale_uv_per_bit" in hints:
            self.scale.setValue(float(hints["voltage_scale_uv_per_bit"]))
        if "file_offset" in hints:
            self.offset.setValue(int(hints["file_offset"]))
        if hints.get("mask"):
            self.window.settings.mask = [int(v) for v in hints["mask"]]

    def _browse_dat(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Recording", "", "Binary (*.dat *.bin);;All files (*)")
        if not path:
            return
        self.dat.setText(path)
        hints = hints_from_binary_json(path)
        try:
            hints.update(load_sidecar(path))
        except RecordingError:
            pass
        self.apply_hints(hints)
        self.window.log(f"Recording {path}")

    def _browse_cfg(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Geometry", "", "Config (*.cfg);;All files (*)")
        if path:
            self.cfg.setText(path)

    def show_recording(self) -> None:
        self.collect()
        try:
            recording, probe = self.window.pipeline.open()
        except Exception as exc:
            self.window.log(str(exc))
            return
        self.window.remember_geometry(probe)
        self.trace.set_source(
            self.window.raw_reader,
            recording.n_samples,
            recording.sampling_rate,
            recording.n_channels,
            self.window.display_order,
            self.window.mask_array(),
        )
        self.window.refresh_probe()
        self.window.log(f"{recording.n_samples / recording.sampling_rate:.1f} s, {recording.n_channels} channels")

    def suggest(self) -> None:
        self.collect()

        def work(token, progress):
            progress(0.1, "Scanning chunks for bad channels")
            recording, _probe = self.window.pipeline.open()
            probe = self.window.pipeline.probe
            positions = None if probe is None else probe.positions
            suggestion = suggest_bad_channels(recording, positions=positions, seed=self.window.settings.rng_seed)
            progress(1.0, suggestion.summary())
            return suggestion

        self.window.run_task(work, self._suggested)

    def _suggested(self, suggestion) -> None:
        self.window.suggested = suggestion.mask
        self.window.refresh_probe()
        self.window.log(suggestion.summary())

    def import_mask(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Channel mask", "", "Mask (*.msk);;All files (*)")
        if not path:
            return
        self.collect()
        mask = parse_msk(path, int(self.n_channels.value()))
        self.window.settings.set_mask(mask)
        self.window.refresh_masks()

    def run_preprocess(self) -> None:
        self.collect()

        def work(token, progress):
            self.window.pipeline.run_preprocess(progress=progress, cancel=token)
            return True

        self.window.run_task(work, lambda _result: self.window.log("Preprocessing cached"))


def _browse_row(edit: QLineEdit, slot) -> QWidget:
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    button = QPushButton("Browse")
    button.clicked.connect(slot)
    layout.addWidget(edit, 1)
    layout.addWidget(button)
    return row
