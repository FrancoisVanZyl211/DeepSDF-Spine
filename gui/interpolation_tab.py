# interpolation_tab.py

import sys
import os
import numpy as np
import torch

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QLineEdit, QSlider, QMessageBox, QScrollArea,
    QGroupBox, QFrame, QGridLayout, QSplitter, QFileDialog, QDialog,
    QPlainTextEdit, QProgressBar
)
from PyQt6.QtCore import Qt, pyqtSlot, QThread, pyqtSignal
from PyQt6.QtGui import QGuiApplication

# Interpolation logic
from scripts.latent_interpolation import (
    interpolate_latents,
    predict_sdf_with_latent
)
from scripts.model_multishape import MultiShapeDecoder
from scripts.mesh_viewer_widget import MeshViewerWidget
from skimage import measure

# ----------------------------------
#  Worker for background interpolation
# ----------------------------------
class InterpolationWorker(QThread):
    finishedSignal = pyqtSignal(object)
    errorSignal = pyqtSignal(str)
    progressSignal = pyqtSignal(int)

    def __init__(self, model, shape0_id, shape1_id, steps, custom_z, device, parent=None):
        super().__init__(parent)
        self.model = model
        self.shape0_id = shape0_id
        self.shape1_id = shape1_id
        self.steps = steps
        self.custom_z = custom_z
        self.device = device

    def run(self):
        """Runs interpolation + SDF generation in a thread so the UI doesn't freeze."""
        try:
            results = []
            if self.custom_z is not None:
                sdf_3d = predict_sdf_with_latent(self.model, self.custom_z, device=self.device)
                results.append(("Custom Vector", sdf_3d))
            else:
                total = self.steps
                step_counter = 0
                for alpha, z in interpolate_latents(self.model, self.shape0_id, self.shape1_id, steps=self.steps):
                    if self.isInterruptionRequested():
                        return
                    z = z.to(self.device)
                    sdf_3d = predict_sdf_with_latent(self.model, z, device=self.device)
                    results.append((f"{alpha:.2f}", sdf_3d))

                    step_counter += 1
                    progress_percent = int(step_counter / total * 100)
                    self.progressSignal.emit(progress_percent)

            self.finishedSignal.emit(results)

        except Exception as e:
            self.errorSignal.emit(str(e))

# ----------------------------------
#  The main Interpolation Tab Widget
# ----------------------------------
class InterpolationTabWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None
        self.worker = None
        self.shape0_id = 0
        self.shape1_id = 1
        self.z0 = None
        self.z1 = None
        self.initUI()
    
    def initUI(self):
        """
        Creates a QSplitter-based layout: left for controls, right for results.
        """

        # Main splitter
        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)

        # ---- Left Panel ----
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # 1) Load model button
        self.load_model_btn = QPushButton("Load Model (.pth)")
        self.load_model_btn.clicked.connect(self.load_model)
        left_layout.addWidget(self.load_model_btn)

        # 2) Interpolation group
        interp_group = QGroupBox("Latent Code Interpolation")
        form_layout = QFormLayout(interp_group)
       
        self.shape0_combo = QComboBox()
        self.shape0_combo.setEnabled(False)
        form_layout.addRow("Shape 0 ID:", self.shape0_combo)

        self.shape1_combo = QComboBox()
        self.shape1_combo.setEnabled(False)
        form_layout.addRow("Shape 1 ID:", self.shape1_combo)

        # Steps spin
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(2, 50)
        self.steps_spin.setValue(5)
        form_layout.addRow("Steps:", self.steps_spin)

        # Latent vector text box
        self.latent_vector_edit = QLineEdit()
        self.latent_vector_edit.setPlaceholderText("Enter latent vector (comma-separated)")
        self.latent_vector_edit.setEnabled(False)
        form_layout.addRow("Latent Code Vector:", self.latent_vector_edit)

        left_layout.addWidget(interp_group)

        # 2a) Latent Code button
        self.get_code_btn = QPushButton("Get Latent Code for shape0")
        self.get_code_btn.setEnabled(False)
        self.get_code_btn.clicked.connect(self.on_get_latent_code)
        left_layout.addWidget(self.get_code_btn)

        # 3) Run Interpolation
        self.run_interp_btn = QPushButton("Run Interpolation")
        self.run_interp_btn.setEnabled(False)
        self.run_interp_btn.clicked.connect(self.run_interpolation)
        left_layout.addWidget(self.run_interp_btn)

        # 4) Optional slider
        self.interp_slider = QSlider(Qt.Orientation.Horizontal)
        self.interp_slider.setRange(0, 100)
        self.interp_slider.setValue(50)
        self.interp_slider.valueChanged.connect(self.update_slider)
        left_layout.addWidget(self.interp_slider)

        left_layout.addStretch()
        self.splitter.addWidget(left_panel)

        # ---- Right Panel ----
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.results_container = QWidget()
        self.results_grid = QGridLayout(self.results_container)
        self.results_grid.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.scroll_area.setWidget(self.results_container)
        right_layout.addWidget(self.scroll_area)
        self.splitter.addWidget(right_panel)
        self.splitter.setSizes([400, 800])

        main_layout = QHBoxLayout(self)
        main_layout.addWidget(self.splitter)
        self.setLayout(main_layout)

    # ---------------------
    #  Load model
    # ---------------------
    @pyqtSlot()
    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Model Checkpoint (.pth)",
            "",
            "PyTorch Models (*.pth)"
        )
        if not path:
            return

        try:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(path, map_location=device)
            num_shapes = checkpoint['latent_codes.weight'].size(0)
            latent_dim = checkpoint['latent_codes.weight'].size(1)
            model = MultiShapeDecoder(num_shapes=num_shapes, args=None, latent_dim=latent_dim)
            model.load_state_dict(checkpoint)
            model.to(device)
            model.eval()
            self.model = model

            QMessageBox.information(
                self,
                "Load Model",
                f"Model loaded:\n{os.path.basename(path)}\nNum shapes={num_shapes}, LatentDim={latent_dim}."
            )

            self.shape0_combo.clear()
            self.shape1_combo.clear()
            for sid in range(num_shapes):
                self.shape0_combo.addItem(str(sid))
                self.shape1_combo.addItem(str(sid))

            self.shape0_combo.setEnabled(True)
            self.shape1_combo.setEnabled(True)
            self.latent_vector_edit.setEnabled(True)
            self.run_interp_btn.setEnabled(True)
            self.get_code_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Model Load Error", str(e))


    @pyqtSlot()
    def on_get_latent_code(self):
        """Retrieves the latent code for whichever shape ID is in shape0_combo and sets it in the text field."""
        if self.model is None:
            QMessageBox.warning(self, "No Model", "Please load a model first.")
            return
        shape_id_str = self.shape0_combo.currentText()
        if not shape_id_str:
            return
        shape_id = int(shape_id_str)

        z_known = self.model.latent_codes.weight[shape_id].detach().cpu().numpy()
        code_str = ", ".join(str(v) for v in z_known)
        self.latent_vector_edit.setText(code_str)

    # ---------------------
    #  Run Interpolation
    # ---------------------
    @pyqtSlot()
    def run_interpolation(self):
        if self.model is None:
            QMessageBox.warning(self, "No Model", "Please load a model first.")
            return

        shape0_id = int(self.shape0_combo.currentText())
        shape1_id = int(self.shape1_combo.currentText())
        steps = self.steps_spin.value()

        custom_z = None
        lv_str = self.latent_vector_edit.text().strip()
        if lv_str:
            try:
                arr = [float(x) for x in lv_str.split(",")]
                custom_z = torch.tensor(arr, dtype=torch.float32)
            except Exception:
                QMessageBox.warning(self, "Latent Vector Error", "Invalid format of latent vector.")
                return

        # Clear old results
        for i in reversed(range(self.results_grid.count())):
            w = self.results_grid.itemAt(i).widget()
            if w:
                w.setParent(None)

        # Show a progress dialog
        self.interp_dialog = QDialog(self)
        self.interp_dialog.setWindowTitle("Interpolation in Progress")
        self.interp_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        dlg_layout = QVBoxLayout(self.interp_dialog)

        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        dlg_layout.addWidget(self.log_edit)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        dlg_layout.addWidget(self.progress_bar)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.cancel_interpolation)
        dlg_layout.addWidget(cancel_btn)
        self.interp_dialog.show()

        # Start the worker
        device = next(self.model.parameters()).device
        if custom_z is not None:
            custom_z = custom_z.to(device)

        self.worker = InterpolationWorker(
            self.model,
            shape0_id,
            shape1_id,
            steps,
            custom_z,
            device
        )
        self.worker.progressSignal.connect(self.on_interpolation_progress)
        self.worker.finishedSignal.connect(self.on_interpolation_finished)
        self.worker.errorSignal.connect(self.on_interpolation_error)
        self.worker.start()

    @pyqtSlot(int)
    def on_interpolation_progress(self, val):
        self.progress_bar.setValue(val)
        self.log_edit.appendPlainText(f"Progress: {val}%")

    @pyqtSlot(object)
    def on_interpolation_finished(self, results):
        
        self.interp_dialog.close()
        row, col = 0, 0
        for label_str, sdf_3d in results:
            widget = self.build_mesh_viewer(sdf_3d, label_str)
            self.results_grid.addWidget(widget, row, col)
            col += 1
            if col > 2:
                col = 0
                row += 1

    @pyqtSlot(str)
    def on_interpolation_error(self, err_msg):
        self.interp_dialog.close()
        QMessageBox.critical(self, "Interpolation Error", err_msg)

    def cancel_interpolation(self):
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()
        self.interp_dialog.close()

    # ---------------------
    #  Build small 3D viewer
    # ---------------------
    def build_mesh_viewer(self, sdf_3d, label_str="Alpha"):
        verts, faces, normals, values = measure.marching_cubes(sdf_3d, level=0.0)

        wrapper = QFrame()
        layout = QVBoxLayout(wrapper)
        row_top = QHBoxLayout()
        lbl = QLabel(label_str)
        row_top.addWidget(lbl)

        mag_btn = QPushButton("🔍")
        mag_btn.setFixedSize(30, 30)
        mag_btn.clicked.connect(lambda: self.show_fullscreen_viewer(verts, faces, label_str))
        row_top.addWidget(mag_btn)
        layout.addLayout(row_top)

        mv = MeshViewerWidget(verts, faces)
        mv.setMinimumSize(200, 200)
        layout.addWidget(mv)
        return wrapper

    def show_fullscreen_viewer(self, verts, faces, title_str):
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Full 3D View - {title_str}")
        dlg.resize(1000, 800)
        dlg.setModal(False)

        dlayout = QVBoxLayout(dlg)
        big_viewer = MeshViewerWidget(verts, faces)
        dlayout.addWidget(big_viewer)
        dlg.show()

    @pyqtSlot(int)
    def update_slider(self, value):
        pass