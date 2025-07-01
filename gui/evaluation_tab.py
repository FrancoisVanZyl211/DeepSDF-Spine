import sys
import os
import glob
import math
import subprocess
import numpy as np
import trimesh
import torch

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QPushButton, QFileDialog, QComboBox, QLineEdit,
    QGroupBox, QSpinBox, QDoubleSpinBox, QSlider, QMessageBox,
    QDialog, QPlainTextEdit, QFormLayout, QScrollArea, QProgressDialog,
    QProgressBar, QFrame, QGridLayout
)
from PyQt6.QtCore import Qt, pyqtSlot, pyqtSignal, QThread
from PyQt6.QtGui import QPalette, QColor, QPainter, QPen, QPixmap, QGuiApplication

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.model_multishape import MultiShapeDecoder
from scripts.mesh_viewer_widget import MeshViewerWidget
from scripts.View_multishape import evaluate_sdf_grid, showMeshReconstruction
from scripts.Combine_PTS_files import combine_pts_files
from core.train_settings import TrainSettings

###############################################################################
# Helper Classes
###############################################################################
class ValidationResultsDialog(QDialog):
    def __init__(self, title: str, message: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(800, 600)
        self.setSizeGripEnabled(True)
        layout = QVBoxLayout(self)
        title_label = QLabel("File Validation Complete")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title_label)
        self.text_edit = QPlainTextEdit(self)
        self.text_edit.setReadOnly(True)
        self.text_edit.setPlainText(message)
        layout.addWidget(self.text_edit)
        btn_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        btn_layout.addStretch()
        btn_layout.addWidget(ok_button)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

class TrainingWorker(QThread):
    lineReceived = pyqtSignal(str)
    finishedSignal = pyqtSignal(int, str)

    def __init__(self, cmd, parent=None):
        super().__init__(parent)
        self.cmd = cmd
        self.process = None

    def run(self):
        self.process = subprocess.Popen(
            self.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        while True:
            line = self.process.stdout.readline()
            if line:
                self.lineReceived.emit(line.rstrip('\n'))
            else:
                if self.process.poll() is not None:
                    break
        retcode = self.process.returncode
        stderr_output = self.process.stderr.read()
        self.finishedSignal.emit(retcode, stderr_output)

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()

###############################################################################
# MultiViewWidget: the grid of shape thumbnails and status
###############################################################################
class MultiViewWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None
        self.initUI()

    def initUI(self):
        self.main_layout = QVBoxLayout(self)
        self.status_label = QLabel("Data has not been loaded. Please load via the Visualize Data Button (Bottom Left)")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        self.main_layout.addWidget(self.status_label)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.main_layout.addWidget(self.scroll_area)
        self.setLayout(self.main_layout)

    def set_model(self, model):
        self.model = model
        if self.model is None:
            self.status_label.setText("Data has not been loaded. Please load via the Visualize Data Button (Bottom Left)")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            return
        else:
            self.status_label.setText("Data Loaded")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")

        grid_container = QWidget()
        grid_layout = QGridLayout(grid_container)
        grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        num_shapes = self.model.latent_codes.weight.shape[0]
        max_cols = 4

        for shape_id in range(num_shapes):
            row = shape_id // max_cols
            col = shape_id % max_cols

            frame = QFrame()
            frame.setFrameStyle(QFrame.Shape.Panel | QFrame.Shadow.Raised)
            frame.setMinimumSize(150, 140)

            cell_layout = QVBoxLayout(frame)
            cell_layout.setContentsMargins(5, 5, 5, 5)
            cell_layout.setSpacing(5)

            top_layout = QHBoxLayout()
            shape_label = QLabel(f"Shape_ID = {shape_id}")
            shape_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            btn = QPushButton("🔍")
            btn.setFixedSize(30, 30)
            btn.setToolTip("View in 3D")
            btn.clicked.connect(lambda checked, sid=shape_id: self.view_shape(sid))
            top_layout.addStretch()
            top_layout.addWidget(shape_label)
            top_layout.addStretch()
            top_layout.addWidget(btn)
            cell_layout.addLayout(top_layout)
            cell_layout.addSpacing(4)
           
            canvas = QLabel()
            canvas.setStyleSheet("background-color: white;")
            canvas.setFixedSize(120, 70)
            cell_layout.addWidget(canvas, alignment=Qt.AlignmentFlag.AlignHCenter)
            grid_layout.addWidget(frame, row, col)

        self.scroll_area.setWidget(grid_container)

    def view_shape(self, shape_id):
        if self.model is None:
            QMessageBox.warning(self, "No Model", "No model loaded.")
            return
        try:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            self.model.eval()
            grid_N = 128
            max_xyz = 1.0
            min_xyz = -max_xyz
            sdf_values = evaluate_sdf_grid(self.model, shape_id, grid_N, min_xyz, max_xyz, device)
            verts, triangles = showMeshReconstruction(sdf_values)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error reconstructing shape {shape_id}: {e}")

###############################################################################
# EvaluationTabWidget: Main Evaluation Tab with left control groups and right panel.
###############################################################################
class EvaluationTabWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.saved_model_path = None
        self.full_input_folder_path = None
        self.full_output_folder_path = None
        self.pts_folder_path = None
        self.npy_output_path = None
        self.initUI()
    
    def initUI(self):
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        
        # --- LEFT PANEL ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # (A) Latent Code Interpolation Group (hidden by default)
        self.interp_left_group = QGroupBox("Latent Code Interpolation")
        self.interp_left_group.setVisible(False)
        interp_layout = QVBoxLayout()
        row_latent = QHBoxLayout()
        row_latent.addWidget(QLabel("Shape 0 ID:"))
        self.shape0_combo = QComboBox()
        for i in range(5):
            self.shape0_combo.addItem(str(i))
        row_latent.addWidget(self.shape0_combo)
        row_latent.addWidget(QLabel("Shape 1 ID:"))
        self.shape1_combo = QComboBox()
        for i in range(5):
            self.shape1_combo.addItem(str(i))
        row_latent.addWidget(self.shape1_combo)
        interp_layout.addLayout(row_latent)
        steps_layout = QHBoxLayout()
        steps_layout.addWidget(QLabel("Steps:"))
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(2, 50)
        self.steps_spin.setValue(5)
        steps_layout.addWidget(self.steps_spin)
        interp_layout.addLayout(steps_layout)
        self.interp_btn = QPushButton("Interpolate")
        self.interp_btn.clicked.connect(self.run_interpolation)
        interp_layout.addWidget(self.interp_btn)
        self.interp_left_group.setLayout(interp_layout)
        left_layout.addWidget(self.interp_left_group)
        
        # (B) Data Prep Group
        self.grp_data_prep = QGroupBox("Data Prep:")
        dp_layout = QVBoxLayout()
        row_input = QHBoxLayout()
        self.input_folder_label = QLabel("Input OBJ Folder: ❌<not selected>")
        btn_browse_obj = QPushButton("Browse OBJ Folder")
        btn_browse_obj.clicked.connect(self.browse_input_folder)
        row_input.addWidget(self.input_folder_label, stretch=1)
        row_input.addWidget(btn_browse_obj)
        dp_layout.addLayout(row_input)
        row_output = QHBoxLayout()
        self.output_folder_label = QLabel("Output PTS Folder: ❌<not selected>")
        btn_browse_pts_output = QPushButton("Browse PTS Folder")
        btn_browse_pts_output.clicked.connect(self.browse_output_folder)
        row_output.addWidget(self.output_folder_label, stretch=1)
        row_output.addWidget(btn_browse_pts_output)
        dp_layout.addLayout(row_output)
        row_samples = QHBoxLayout()
        row_samples.addWidget(QLabel("N_samples:"))
        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(100, 99999)
        self.samples_spin.setValue(9500)
        row_samples.addWidget(self.samples_spin)
        dp_layout.addLayout(row_samples)
        self.convert_btn = QPushButton("Convert OBJ → PTS")
        self.convert_btn.clicked.connect(self.convert_obj_pts)
        dp_layout.addWidget(self.convert_btn)
        self.grp_data_prep.setLayout(dp_layout)
        left_layout.addWidget(self.grp_data_prep)
        
        # (C) Data Combination Group
        self.combination_group = QGroupBox("Data Combination")
        comb_layout = QVBoxLayout()
        row_pts = QHBoxLayout()
        self.pts_folder_label = QLabel("Input PTS Folder:❌<not selected>")
        self.pts_folder_btn = QPushButton("Browse .PTS Folder")
        self.pts_folder_btn.clicked.connect(self.browse_pts_folder)
        row_pts.addWidget(self.pts_folder_label, stretch=1)
        row_pts.addWidget(self.pts_folder_btn)
        comb_layout.addLayout(row_pts)
        row_max_shapes = QHBoxLayout()
        row_max_shapes.addWidget(QLabel("Max Shapes:"))
        self.max_shapes_spin = QSpinBox()
        self.max_shapes_spin.setRange(1, 9999)
        self.max_shapes_spin.setValue(25)
        row_max_shapes.addWidget(self.max_shapes_spin)
        comb_layout.addLayout(row_max_shapes)
        row_npy = QHBoxLayout()
        row_npy.addWidget(QLabel("Output .npy File:"))
        self.npy_output_edit = QLineEdit("")
        btn_npy_browse = QPushButton("Browse")
        btn_npy_browse.clicked.connect(self.browse_npy_output)
        row_npy.addWidget(self.npy_output_edit, stretch=1)
        row_npy.addWidget(btn_npy_browse)
        comb_layout.addLayout(row_npy)
        self.combine_btn = QPushButton("Combine PTS Files")
        self.combine_btn.clicked.connect(self.combine_pts_files_ui)
        comb_layout.addWidget(self.combine_btn)
        self.combination_group.setLayout(comb_layout)
        left_layout.addWidget(self.combination_group)
        
        # (D) Data Check Group
        self.check_group = QGroupBox("Data Check:")
        dcheck_layout = QVBoxLayout()
        self.validate_btn = QPushButton("Validate Files")
        self.validate_btn.clicked.connect(self.validate_files)
        dcheck_layout.addWidget(self.validate_btn)
        self.check_group.setLayout(dcheck_layout)
        left_layout.addWidget(self.check_group)
        
        # (E) Data Training Group
        self.training_group = QGroupBox("Data Training:")
        train_layout = QVBoxLayout()
        row_mode = QHBoxLayout()
        self.training_options = QComboBox()
        self.training_options.addItems(["Multi Shape", "Single Shape"])
        row_mode.addWidget(self.training_options)
        self.save_model_btn = QPushButton("Save Model To")
        self.save_model_btn.clicked.connect(self.save_model_path)
        row_mode.addWidget(self.save_model_btn)
        train_layout.addLayout(row_mode)
        
        self.multi_shape_settings_group = QGroupBox("Multi-Shape Training Settings")
        ms_layout = QFormLayout()
        self.multi_shape_file_edit = QLineEdit("multi_shape_data.npy")
        self.multi_shape_file_browse_btn = QPushButton("Browse")
        self.multi_shape_file_browse_btn.clicked.connect(self.browse_multi_shape_file)
        hbox_file = QHBoxLayout()
        hbox_file.addWidget(self.multi_shape_file_edit)
        hbox_file.addWidget(self.multi_shape_file_browse_btn)
        ms_layout.addRow("Multi-shape File:", hbox_file)
        
        self.train_split_ratio_spin = QDoubleSpinBox()
        self.train_split_ratio_spin.setRange(0.0, 1.0)
        self.train_split_ratio_spin.setDecimals(2)
        self.train_split_ratio_spin.setValue(0.8)
        ms_layout.addRow("Train Split Ratio:", self.train_split_ratio_spin)
        
        self.latent_dim_spin = QSpinBox()
        self.latent_dim_spin.setRange(1, 1024)
        self.latent_dim_spin.setValue(64)
        ms_layout.addRow("Latent Dim:", self.latent_dim_spin)

        self.hidden_dim_spin = QSpinBox()
        self.hidden_dim_spin.setRange(32, 2048)
        self.hidden_dim_spin.setValue(512)
        ms_layout.addRow("Hidden Dim:", self.hidden_dim_spin)
        
        self.lr_dspin = QDoubleSpinBox()
        self.lr_dspin.setRange(1e-6, 1.0)
        self.lr_dspin.setDecimals(6)
        self.lr_dspin.setValue(0.0001)
        ms_layout.addRow("Learning Rate:", self.lr_dspin)
        
        self.weight_decay_dspin = QDoubleSpinBox()
        self.weight_decay_dspin.setRange(0.0, 1.0)
        self.weight_decay_dspin.setDecimals(6)
        self.weight_decay_dspin.setValue(0.0001)
        ms_layout.addRow("Weight Decay:", self.weight_decay_dspin)
        
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(10)
        ms_layout.addRow("Epochs:", self.epochs_spin)
        
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 65536)
        self.batch_spin.setValue(512)
        ms_layout.addRow("Train Batch Size:", self.batch_spin)
        
        self.sample_std_dspin = QDoubleSpinBox()
        self.sample_std_dspin.setRange(0.0, 1.0)
        self.sample_std_dspin.setDecimals(3)
        self.sample_std_dspin.setValue(0.05)
        ms_layout.addRow("Sample Std:", self.sample_std_dspin)

        self.dropout_dspin = QDoubleSpinBox()
        self.dropout_dspin.setRange(0.0, 0.9)
        self.dropout_dspin.setDecimals(2)
        self.dropout_dspin.setSingleStep(0.05)
        self.dropout_dspin.setValue(0.10)
        ms_layout.addRow("Drop-out:", self.dropout_dspin)
        
        self.N_samples_spin = QSpinBox()
        self.N_samples_spin.setRange(1, 10000)
        self.N_samples_spin.setValue(10)
        ms_layout.addRow("N_samples (perturbation):", self.N_samples_spin)
        
        self.gridN_spin = QSpinBox()
        self.gridN_spin.setRange(8, 1024)
        self.gridN_spin.setValue(128)
        ms_layout.addRow("Grid N:", self.gridN_spin)
        
        self.maxxyz_dspin = QDoubleSpinBox()
        self.maxxyz_dspin.setRange(0.1, 100.0)
        self.maxxyz_dspin.setValue(1.0)
        self.maxxyz_dspin.setDecimals(2)
        ms_layout.addRow("Max XYZ:", self.maxxyz_dspin)
        
        self.multi_shape_settings_group.setLayout(ms_layout)
        train_layout.addWidget(self.multi_shape_settings_group)
        
        self.train_btn = QPushButton("Start Training")
        self.train_btn.clicked.connect(self.start_training)
        train_layout.addWidget(self.train_btn)
        
        self.training_group.setLayout(train_layout)
        
        self.training_scroll_area = QScrollArea()
        self.training_scroll_area.setWidgetResizable(True)
        self.training_scroll_area.setWidget(self.training_group)
        left_layout.addWidget(self.training_scroll_area)
        
        # (F) Visualize Data
        self.grp_visualize = QGroupBox("Visualize Data")
        viz_layout = QVBoxLayout()
        btn_viz = QPushButton("Visualize Data")
        btn_viz.clicked.connect(self.visualize_data)
        viz_layout.addWidget(btn_viz)
        self.grp_visualize.setLayout(viz_layout)
        left_layout.addWidget(self.grp_visualize)
        
        self.status_label = QLabel("Ready")
        left_layout.addWidget(self.status_label)
        
        left_layout.addStretch()
        splitter.addWidget(left_panel)
        
        # --------------------------- RIGHT PANEL ---------------------------
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.multi_view = MultiViewWidget()
        right_layout.addWidget(self.multi_view)
        splitter.addWidget(right_panel)
        splitter.setSizes([350, 850])
        
        main_layout = QHBoxLayout(self)
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    # ===================== Callback Methods =====================
    @pyqtSlot()
    def browse_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder Containing .OBJ Files")
        if folder:
            self.full_input_folder_path = folder
            self.input_folder_label.setText("✅ Loaded .obj files (hover for details)")
            self.input_folder_label.setToolTip(folder)

    @pyqtSlot()
    def browse_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder for PTS Output")
        if folder:
            self.full_output_folder_path = folder
            self.output_folder_label.setText("✅ Output folder selected (hover for details)")
            self.output_folder_label.setToolTip(folder)

    @pyqtSlot()
    def convert_obj_pts(self):
        if not self.full_input_folder_path or not self.full_output_folder_path:
            QMessageBox.warning(self, "Missing Folders", "Please select valid input and output folders.")
            return
        obj_files = glob.glob(os.path.join(self.full_input_folder_path, "*.obj"))
        if not obj_files:
            QMessageBox.information(self, "No OBJ Files", "No .obj files found.")
            return
        n_samples = self.samples_spin.value()
        progress = QProgressDialog("Converting .obj to .pts...", "Cancel", 0, len(obj_files), self)
        progress.setWindowTitle("OBJ → PTS Conversion")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        converted_count = 0
        for i, obj_path in enumerate(obj_files):
            if progress.wasCanceled():
                break
            progress.setLabelText(f"Converting: {os.path.basename(obj_path)}")
            progress.setValue(i)
            try:
                mesh = trimesh.load(obj_path)
                if mesh.vertex_normals is None or len(mesh.vertex_normals) == 0:
                    mesh.compute_vertex_normals()
                sampled_points, face_indices = trimesh.sample.sample_surface(mesh, n_samples)
                face_normals = mesh.face_normals[face_indices]
                data = np.hstack((sampled_points, face_normals))
                pts_name = os.path.basename(obj_path).replace(".obj", ".pts")
                pts_path = os.path.join(self.full_output_folder_path, pts_name)
                np.savetxt(pts_path, data, fmt="%.6f")
                converted_count += 1
            except Exception as e:
                print(f"Failed to convert {obj_path}: {e}")
            QApplication.processEvents()
        progress.setValue(len(obj_files))
        QMessageBox.information(self, "Conversion Finished", f"Converted {converted_count} .obj files in:\n{self.full_output_folder_path}")
        self.status_label.setText("Conversion complete!")

    @pyqtSlot()
    def browse_pts_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder Containing .PTS Files")
        if folder:
            self.pts_folder_path = folder
            self.pts_folder_label.setText("✅ Loaded .pts files (hover for details)")
            self.pts_folder_label.setToolTip(folder)

    @pyqtSlot()
    def browse_npy_output(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Select Output .npy File", "", "NumPy Files (*.npy)")
        if file_path:
            self.npy_output_path = file_path
            self.npy_output_edit.setText(file_path)
            self.npy_output_edit.setToolTip(file_path)

    @pyqtSlot()
    def combine_pts_files_ui(self):
        if not self.pts_folder_path or not os.path.isdir(self.pts_folder_path):
            QMessageBox.warning(self, "Invalid Folder", "Please select a valid folder containing .pts files.")
            return
        out_npy = self.npy_output_edit.text()
        if not out_npy.endswith(".npy"):
            QMessageBox.warning(self, "Invalid Output File", "Please specify a valid .npy output path.")
            return
        max_shapes = self.max_shapes_spin.value()
        combine_pts_files(self.pts_folder_path, out_npy, max_shapes=max_shapes)
        QMessageBox.information(self, "Combination Complete", f"Combined .pts files into:\n{out_npy}\n(max_shapes={max_shapes})")

    @pyqtSlot()
    def validate_files(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder with .pts/.npy Files")
        if not folder:
            return
        pts_files = glob.glob(os.path.join(folder, "*.pts"))
        npy_files = glob.glob(os.path.join(folder, "*.npy"))
        pts_message = ""
        npy_message = ""
        count_valid_pts = 0
        for f in pts_files:
            try:
                data = np.loadtxt(f)
                if data.ndim != 2 or data.shape[1] != 6:
                    pts_message += f"\n❌ {os.path.basename(f)} has shape {data.shape}"
                else:
                    pts_message += f"\n✅ {os.path.basename(f)} OK ({data.shape[0]} rows)"
                    count_valid_pts += 1
            except Exception as e:
                pts_message += f"\n❌ {os.path.basename(f)} failed to load: {str(e)}"
        count_valid_npy = 0
        for f in npy_files:
            try:
                data = np.load(f)
                if data.ndim != 2:
                    npy_message += f"\n❌ {os.path.basename(f)} has shape {data.shape}"
                else:
                    npy_message += f"\n✅ {os.path.basename(f)} OK ({data.shape[0]} rows)"
                    count_valid_npy += 1
            except Exception as e:
                npy_message += f"\n❌ {os.path.basename(f)} load error: {str(e)}"
        summary = f"PTS Files:\n{pts_message}\n\nNPY Files:\n{npy_message}"
        dlg = ValidationResultsDialog("Validation Results", summary, self)
        dlg.exec()

    @pyqtSlot()
    def browse_multi_shape_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select multi_shape_data.npy", "", "NumPy Files (*.npy)")
        if file_path:
            self.multi_shape_file_edit.setText(file_path)
            self.multi_shape_file_edit.setToolTip(file_path)

    # (1) (hover message)
    @pyqtSlot()
    def save_model_path(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Model As", "", "PyTorch Models (*.pth)")
        if path:
            self.saved_model_path = path
            self.status_label.setText("Model will be saved to chosen location (Hover for details)")
            self.status_label.setToolTip(path)
        else:
            self.saved_model_path = None

    # (1.5) - Helpers
    @pyqtSlot()
    def _settings_from_widgets(self) -> TrainSettings:
        """Collect current UI values into a dataclass."""
        return TrainSettings(
            # paths
            multi_shape_file=self.multi_shape_file_edit.text(),
            out_model     = self.saved_model_path or "",
            train_split = self.train_split_ratio_spin.value(),
            latent_dim  = self.latent_dim_spin.value(),
            hidden_dim  = self.hidden_dim_spin.value(),
            dropout     = self.dropout_dspin.value(),
            lr          = self.lr_dspin.value(),
            weight_decay= self.weight_decay_dspin.value(),
            epochs      = self.epochs_spin.value(),
            batch_size  = self.batch_spin.value(),
            sample_std  = self.sample_std_dspin.value(),
            n_samples   = self.N_samples_spin.value(),
            grid_N      = self.gridN_spin.value(),
            max_xyz     = self.maxxyz_dspin.value(),
        )

    # (2) Data Training: Start Training callback
    @pyqtSlot()
    def start_training(self):
        script_name = ("train_singleshape.py"
                    if self.training_options.currentText() == "Single Shape"
                    else "train_multishape.py")
        script_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "scripts", script_name))

        if not os.path.isfile(script_path):
            QMessageBox.critical(self, "Error",
                                f"Training script not found:\n{script_path}")
            return

        # ----------------------------------------------- #
        # grab all widget values first
        # ----------------------------------------------- #
        settings = self._settings_from_widgets()

        if script_name == "train_multishape.py" and not settings.multi_shape_file:
            QMessageBox.warning(self, "Missing file",
                                "Select a multi-shape .npy first.")
            return

        if self.saved_model_path is None or self.saved_model_path.strip() == "":
            # use GUI/data/multi_shape_deepsdf.pth  (or single_shape_deepsdf.pth)
            default_name = ("single_shape_deepsdf.pth"
                            if script_name == "train_singleshape.py"
                            else "multi_shape_deepsdf.pth")
            self.saved_model_path = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "..", "data", default_name
            )
        # pass the path into Settings so the script gets it
        settings.out_model = self.saved_model_path
        # ------------------------------------------------------ #

        cfg_json = settings.to_json_temp()
        cmd = [sys.executable, "-u", script_path, "--config", cfg_json]

        # ---------- progress dialog ----------
        self.training_dialog = QDialog(self)
        self.training_dialog.setWindowTitle("Training in Progress")
        self.training_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        dlg_layout = QVBoxLayout(self.training_dialog)

        self.log_edit = QPlainTextEdit(readOnly=True)
        dlg_layout.addWidget(self.log_edit)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)         
        dlg_layout.addWidget(self.progress_bar)

        cancel_btn = QPushButton("Cancel")
        dlg_layout.addWidget(cancel_btn, alignment=Qt.AlignmentFlag.AlignRight)

        self.worker = TrainingWorker(cmd, self)
        self.worker.lineReceived.connect(self.log_edit.appendPlainText)
        self.worker.finishedSignal.connect(self.on_training_finished)
        cancel_btn.clicked.connect(self.worker.stop)

        self.training_dialog.show()
        self.worker.start()
        self.status_label.setText("Training started… please wait.")

    @pyqtSlot(str)
    def on_line_received(self, line: str):
        self.log_edit.appendPlainText(line)

    @pyqtSlot(int, str)
    def on_training_finished(self, retcode: int, stderr: str):
        self.training_dialog.close()
        if retcode == 0:
            QMessageBox.information(self, "Training Complete", "Training finished successfully.")
            self.status_label.setText("Training complete!")
        elif retcode == -9:
            QMessageBox.information(self, "Cancelled", "Training was cancelled.")
        else:
            QMessageBox.warning(
            self, "Training Error",
            f"Process exited with code {retcode}\n\n{stderr or '(no stderr)'}"
        )

    # (6) Visualize Data
    @pyqtSlot()
    def visualize_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model Checkpoint", "", "PyTorch Models (*.pth)")
        if not file_path:
            return
        checkpoint = torch.load(file_path, map_location=torch.device("cpu"))
        num_shapes = checkpoint['latent_codes.weight'].size(0)
        print(f"Detected {num_shapes} shapes from checkpoint.")
        
        num_shapes = checkpoint['latent_codes.weight'].size(0)
        latent_dim = checkpoint['latent_codes.weight'].size(1)

        model = MultiShapeDecoder(num_shapes=num_shapes, args=None, latent_dim=latent_dim)
        model.load_state_dict(checkpoint)
        model.eval()
        self.multi_view.set_model(model)

    # (7) Right Panel slider callback
    @pyqtSlot()
    def update_evaluation_visual(self, value):
        self.eval_visual.setText(f"Evaluation Visualization Param: {value}")

    @pyqtSlot()
    def run_interpolation(self):
        shape0_id = int(self.shape0_combo.currentText())
        shape1_id = int(self.shape1_combo.currentText())
        steps = self.steps_spin.value()
        out_folder = QFileDialog.getExistingDirectory(self, "Choose folder to save interpolation .obj files")
        if not out_folder:
            self.status_label.setText("Interpolation cancelled: no folder selected.")
            return
        QMessageBox.information(self, "Interpolation Complete",
                                f"Placeholder: Interpolated from {shape0_id} to {shape1_id} in {steps} steps.")
        self.status_label.setText("Interpolation complete!")