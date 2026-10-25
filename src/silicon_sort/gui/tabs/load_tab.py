"""Load a recording, confirm geometry, and mask channels."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from silicon_sort.errors import RecordingError
from silicon_sort.gui.widgets.parameters import combo, double_box, int_box
from silicon_sort.gui.widgets.probe_view import ProbeMap
from silicon_sort.gui.widgets.trace_view import TraceView
from silicon_sort.io.binary import hints_from_binary_json, load_sidecar, open_recording
from silicon_sort.io.geometry import parse_msk
from silicon_sort.preprocess.bad_channels import suggest_bad_channels


class LoadTab(QWidget):
    def __init__(self, window) -> None:
        super().__init__()
        self.window = window
        self.form = QWidget()
        layout = QVBoxLayout(self.form)
        layout.setContentsMargins(8, 8, 8, 8)
        self.preprocess_step = QPushButton("1  Preprocess")
        self.mask_step = QPushButton("2  Mask")
        for button in (self.preprocess_step, self.mask_step):
            button.setCheckable(True)
        self.preprocess_step.setChecked(True)
        steps = QButtonGroup(self.form)
        steps.setExclusive(True)
        steps.addButton(self.preprocess_step, 0)
        steps.addButton(self.mask_step, 1)
        step_row = QHBoxLayout()
        step_row.addWidget(self.preprocess_step, 1)
        step_row.addWidget(self.mask_step, 1)
        layout.addLayout(step_row)

        self.dat = QLineEdit()
        self.dat.setToolTip("Headerless interleaved binary, samples by channels.")
        self.cfg = QLineEdit()
        self.cfg.setToolTip("Probe geometry: a name line, then site channel x y in micrometers.")
        self.n_channels = int_box(32, 1, 1024, "Number of channels stored in the binary file.")
        self.sampling_rate = double_box(30000, 1000, 100000, 1000, "Sampling rate in Hz.", decimals=1)
        self.dtype = combo(
            ["int16", "int32", "uint16", "float32", "float64"],
            "int16",
            "Sample dtype. Intan amplifier files are int16.",
        )
        self.scale = double_box(0.195, 0.0, 1000, 0.001, "Microvolts per bit. Intan RHD2132 is 0.195.", decimals=4)
        self.offset = int_box(0, 0, 10**9, "Bytes to skip before the first sample.")
        recording = QFormLayout()
        recording.addRow("Recording", _browse_row(self.dat, self._browse_dat))
        recording.addRow("Geometry", _browse_row(self.cfg, self._browse_cfg))
        recording.addRow("Channels", self.n_channels)
        recording.addRow("Sampling rate", self.sampling_rate)
        recording.addRow("Dtype", self.dtype)
        recording.addRow("µV / bit", self.scale)
        recording.addRow("Offset", self.offset)

        self.preprocess = combo(
            ["off", "on"],
            "off",
            "Off keeps the raw recording for viewing, detection, and clustering. On bandpasses and applies the common median reference first. You can leave this off and still scroll the raw traces.",
        )
        self.highpass = double_box(300, 0.1, 10000, 50, "Highpass cutoff in Hz, used only when preprocessing is on.")
        self.lowpass = double_box(6000, 100, 20000, 100, "Lowpass cutoff in Hz.")
        self.reference = combo(
            ["global", "shank", "local"],
            "global",
            "Common median reference. Shanks split on x gaps. Local groups use the radius below.",
        )
        self.local_radius = double_box(100, 10, 500, 10, "Radius in µm for local-group referencing.")
        self.whiten = combo(["on", "off"], "on", "ZCA whitening is applied during detection and is not written to the cache.")
        filt = QFormLayout()
        filt.addRow("Preprocessing", self.preprocess)
        filt.addRow("Highpass Hz", self.highpass)
        filt.addRow("Lowpass Hz", self.lowpass)
        filt.addRow("Reference", self.reference)
        filt.addRow("Local radius", self.local_radius)
        filt.addRow("Whiten", self.whiten)
        self.run_button = QPushButton("Run preprocessing")
        self.run_button.setToolTip("Optional. Bandpass and common-median reference the recording into the cache. Detection uses it only when preprocessing is on.")

        preprocess_page = QWidget()
        preprocess_layout = QVBoxLayout(preprocess_page)
        preprocess_layout.setContentsMargins(0, 0, 0, 0)
        preprocess_layout.addWidget(_group("Recording", recording))
        filter_box = _group("Filter", filt)
        filter_box.layout().addWidget(self.run_button)
        preprocess_layout.addWidget(filter_box)
        preprocess_layout.addStretch(1)

        self.suggest_button = QPushButton("Suggest bad channels")
        self.suggest_button.setToolTip("Flag flat, railed, or very noisy channels from a few raw chunks. Nothing is masked until you accept.")
        self.accept_button = QPushButton("Accept suggestions")
        self.accept_button.setToolTip("Add the suggested channels to the mask. Channels you already masked stay masked.")
        self.mask_selected_button = QPushButton("Mask selected")
        self.mask_selected_button.setToolTip(
            "Mask the channels selected on the traces. Click a trace to select it, and Shift-click or Ctrl-click to add more. "
            "This adds to the current mask, including after you accept suggestions."
        )
        self.clear_button = QPushButton("Clear mask")
        self.mask_button = QPushButton("Import .msk")
        self.mask_button.setToolTip("Import a 1-based channel T/F mask. T excludes the channel.")
        note = QLabel("Click a trace to select it. Shift-click or Ctrl-click adds more. Masking adds to the channels you already chose.")
        note.setWordWrap(True)
        mask_layout = QVBoxLayout()
        for button in (
            self.suggest_button,
            self.accept_button,
            self.mask_selected_button,
            self.clear_button,
            self.mask_button,
        ):
            mask_layout.addWidget(button)
        mask_page = QWidget()
        mask_page_layout = QVBoxLayout(mask_page)
        mask_page_layout.setContentsMargins(0, 0, 0, 0)
        mask_page_layout.addWidget(note)
        mask_page_layout.addWidget(_group("Bad channels", mask_layout))
        mask_page_layout.addStretch(1)

        self.steps = QStackedWidget()
        self.steps.addWidget(preprocess_page)
        self.steps.addWidget(mask_page)
        steps.idClicked.connect(self.steps.setCurrentIndex)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(self.steps)
        layout.addWidget(scroll, 1)

        self.probe = ProbeMap()
        self.probe.setMinimumWidth(150)
        self.probe.setMaximumWidth(230)
        self.trace = TraceView()
        self.trace.multi_select = True
        self.view = QSplitter()
        self.view.setHandleWidth(2)
        self.view.setChildrenCollapsible(False)
        self.view.addWidget(self.trace)
        self.view.addWidget(self.probe)
        self.view.setStretchFactor(0, 1)
        self.view.setStretchFactor(1, 0)
        self.view.setSizes([1100, 180])

        self.suggest_button.clicked.connect(self.suggest)
        self.accept_button.clicked.connect(window.accept_suggestions)
        self.mask_selected_button.clicked.connect(window.mask_selected)
        self.clear_button.clicked.connect(window.clear_mask)
        self.mask_button.clicked.connect(self.import_mask)
        self.run_button.clicked.connect(self.run_preprocess)
        self.probe.channelClicked.connect(window.toggle_mask)
        self.trace.maskRequested.connect(window.toggle_mask)
        self.trace.selectionChanged.connect(window.pick_channel)

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
        self.show_recording()

    def _browse_cfg(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Geometry", "", "Config (*.cfg);;All files (*)")
        if not path:
            return
        self.cfg.setText(path)
        if self.dat.text().strip():
            self.show_recording()

    def show_recording(self) -> None:
        self.collect()
        settings = self.window.settings
        if not settings.dat_path:
            return
        try:
            if settings.cfg_path:
                recording, probe = self.window.pipeline.open()
                self.window.remember_geometry(probe)
            else:
                self.window.pipeline.ensure_meta()
                if self.window.pipeline.recording is not None:
                    self.window.pipeline.recording.close()
                recording = open_recording(settings)
                self.window.pipeline.recording = recording
                self.window.pipeline.probe = None
                self.window.remember_file_order(recording.n_channels)
        except Exception as exc:
            self.window.log(str(exc))
            return
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


def _group(title: str, body: QFormLayout | QVBoxLayout) -> QGroupBox:
    box = QGroupBox(title)
    box.setLayout(body)
    return box


def _browse_row(edit: QLineEdit, slot) -> QWidget:
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    button = QPushButton("Browse")
    button.clicked.connect(slot)
    layout.addWidget(edit, 1)
    layout.addWidget(button)
    return row
