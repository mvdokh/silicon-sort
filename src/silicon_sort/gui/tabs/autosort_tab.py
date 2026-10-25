"""Automatic merge, split, metrics, and Phy export."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from silicon_sort.gui.widgets.parameters import double_box


class AutosortTab(QWidget):
    def __init__(self, window) -> None:
        super().__init__()
        self.window = window
        self.form = QWidget()
        layout = QVBoxLayout(self.form)
        form = QFormLayout()
        layout.addLayout(form)
        self.merge_cosine = double_box(
            0.9,
            0.5,
            1,
            0.01,
            "Minimum template cosine before two neighboring sites can merge.",
            decimals=2,
        )
        self.merge_footprint = double_box(
            0.8,
            0.3,
            1,
            0.01,
            "Minimum correlation of the spatial footprints.",
            decimals=2,
        )
        self.refractory = double_box(
            1.5,
            0.5,
            5,
            0.1,
            "Refractory window in ms for ISI violations and the cross-correlogram.",
            decimals=2,
        )
        self.isi_split = double_box(
            0.1,
            0.01,
            0.5,
            0.01,
            "Split a cluster when the ISI violation rate exceeds this and amplitude is bimodal.",
            decimals=2,
        )
        form.addRow("Merge cosine", self.merge_cosine)
        form.addRow("Merge footprint", self.merge_footprint)
        form.addRow("Refractory ms", self.refractory)
        form.addRow("ISI split", self.isi_split)
        self.run_button = QPushButton("Run autosort")
        self.run_button.setToolTip(
            "Cluster every site with the counts from the Clusters tab, then merge and split."
        )
        self.export_button = QPushButton("Export Phy")
        self.export_button.setToolTip("Write spike_times, templates, channel map, params.py, and cluster_info.tsv.")
        layout.addWidget(self.run_button)
        layout.addWidget(self.export_button)
        layout.addStretch(1)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ["cluster", "rate Hz", "ISI viol.", "SNR", "peak ch", "amplitude", "width ms", "quality"]
        )
        self.table.verticalHeader().setVisible(False)
        self.view = self.table

        self.run_button.clicked.connect(self.run_sort)
        self.export_button.clicked.connect(self.export)

    def collect(self) -> None:
        settings = self.window.settings
        settings.merge_cosine = float(self.merge_cosine.value())
        settings.merge_footprint = float(self.merge_footprint.value())
        settings.refractory_ms = float(self.refractory.value())
        settings.isi_split_fraction = float(self.isi_split.value())

    def run_sort(self) -> None:
        self.window.collect_all()

        def work(token, progress):
            return self.window.pipeline.run_sort(progress=progress, cancel=token)

        self.window.run_task(work, lambda result: self.show_metrics(result.metrics))

    def show_metrics(self, metrics: list[dict]) -> None:
        self.table.setRowCount(len(metrics))
        for row, metric in enumerate(metrics):
            values = [
                metric["cluster_id"],
                f"{metric['firing_rate']:.2f}",
                f"{metric['isi_violation_rate']:.3f}",
                f"{metric['snr']:.2f}",
                metric["peak_channel"],
                f"{metric['amplitude']:.1f}",
                f"{metric['waveform_width_ms']:.2f}",
                metric["quality"],
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                self.table.setItem(row, column, item)
        self.table.resizeColumnsToContents()
        self.window.log(f"{len(metrics)} clusters")

    def export(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Phy output folder")
        self.window.collect_all()
        if directory:
            self.window.settings.output_dir = directory

        def work(token, progress):
            return str(self.window.pipeline.run_export(progress=progress, cancel=token))

        self.window.run_task(work, lambda path: self._exported(path))

    def _exported(self, path: str) -> None:
        result = self.window.pipeline.result
        if result is not None:
            self.show_metrics(result.metrics)
        self.window.log(f"Phy folder: {path}")
