import os
import numpy as np
import trimesh
import torch
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
    QTableWidget, QTableWidgetItem, QMessageBox, QLabel, QSpinBox, QSplitter, QProgressBar
)
from PyQt6.QtCore import QThread, pyqtSignal, QObject
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from types import SimpleNamespace
from skimage import measure


def load_npy_mesh(path):
    data = np.load(path, allow_pickle=True)
    if isinstance(data, dict):
        verts = np.array(data.get("verts") or data.get("vertices"))
        faces = np.array(data.get("faces"))
    elif isinstance(data, np.ndarray):
        if data.shape[1] == 3:
            verts = data
            faces = np.array([])
        elif data.shape[1] >= 6:
            verts = data[:, :3]
            faces = np.array([])
        else:
            raise ValueError(f"Unsupported .npy array shape: {data.shape}")
    else:
        raise TypeError("Unsupported data format in .npy file.")
    if verts is None or len(verts) == 0:
        raise ValueError("Vertices not found or empty in .npy file.")
    return trimesh.Trimesh(vertices=verts, faces=faces if faces.size else None, process=False)


def load_mesh_file(path):
    if path.endswith(".npy"):
        return load_npy_mesh(path)
    else:
        return trimesh.load(path, force='mesh')

def chamfer_distance(mesh1, mesh2):
    d1 = mesh1.nearest.signed_distance(mesh2.vertices)
    d2 = mesh2.nearest.signed_distance(mesh1.vertices)
    return (np.abs(d1).mean() + np.abs(d2).mean()) / 2.0

class Worker(QObject):
    finished = pyqtSignal()
    result = pyqtSignal(object)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        result = self.fn()
        self.result.emit(result)
        self.finished.emit()

class AnalyzerTabWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.pred_meshes = []
        self.gt_meshes = []
        self.model = None
        self.decoded_mesh = None
        self.initUI()

    def initUI(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left control panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Model loading
        self.load_model_btn = QPushButton("Load .pth Model")
        self.load_model_btn.clicked.connect(self.load_pth_model)
        left_layout.addWidget(self.load_model_btn)

        shape_id_row = QHBoxLayout()
        self.shape_id_label = QLabel("Shape ID:")
        self.shape_id_spin = QSpinBox()
        self.shape_id_spin.setRange(0, 99)

        shape_id_row.addWidget(self.shape_id_label)
        shape_id_row.addWidget(self.shape_id_spin)
        left_layout.addLayout(shape_id_row)

        self.decode_btn = QPushButton("Decode Shape to Mesh")
        self.decode_btn.clicked.connect(self.decode_shape_to_mesh)
        left_layout.addWidget(self.decode_btn)

        self.export_btn = QPushButton("Export Decoded Mesh to .obj")
        self.export_btn.clicked.connect(self.export_decoded_mesh)
        left_layout.addWidget(self.export_btn)

        # Chamfer mesh loading
        self.load_pred_btn = QPushButton("Load Predicted Mesh (.obj or .npy)")
        self.load_pred_btn.clicked.connect(self.load_pred_mesh)
        left_layout.addWidget(self.load_pred_btn)

        self.load_gt_btn = QPushButton("Load Ground Truth Mesh (.obj or .npy)")
        self.load_gt_btn.clicked.connect(self.load_gt_mesh)
        left_layout.addWidget(self.load_gt_btn)

        self.compute_btn = QPushButton("Compute Chamfer Distance")
        self.compute_btn.clicked.connect(self.compute_chamfer)
        left_layout.addWidget(self.compute_btn)

        left_layout.addStretch()

        # Status label
        self.status_label = QLabel("Status: Ready")
        self.status_label.setFont(QFont("Arial", 10))
        left_layout.addWidget(self.status_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        splitter.addWidget(left_panel)

        # Chamfer results table
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Shape ID", "Chamfer Distance"])
        splitter.addWidget(self.table)
        splitter.setSizes([300, 800])

        main_layout = QVBoxLayout()
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def threaded(self, task_func, on_result):
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)

        self.thread = QThread()
        self.worker = Worker(task_func)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.worker.result.connect(on_result)

        def cleanup():
            self.progress_bar.setVisible(False)
            self.status_label.setText("Status: Done")

        self.worker.finished.connect(cleanup)
        self.thread.start()

    def load_pred_mesh(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Predicted Mesh Files", "", "Mesh Files (*.obj *.npy)")
        if files:
            try:
                self.pred_meshes = [load_mesh_file(f) for f in files]
                QMessageBox.information(self, "Loaded", f"Loaded {len(self.pred_meshes)} predicted mesh(es).")
            except Exception as e:
                QMessageBox.critical(self, "Load Error", str(e))

    def load_gt_mesh(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Ground Truth Mesh Files", "", "Mesh Files (*.obj *.npy)")
        if files:
            try:
                self.gt_meshes = [load_mesh_file(f) for f in files]
                QMessageBox.information(self, "Loaded", f"Loaded {len(self.gt_meshes)} ground-truth mesh(es).")
            except Exception as e:
                QMessageBox.critical(self, "Load Error", str(e))

    def compute_chamfer(self):
        if len(self.pred_meshes) != len(self.gt_meshes):
            QMessageBox.warning(self, "Mismatch", "Number of predicted and GT meshes must match.")
            return

        self.status_label.setText("Status: Computing Chamfer Distance...")
        self.table.setRowCount(0)

        def task():
            results = []
            for i, (pred, gt) in enumerate(zip(self.pred_meshes, self.gt_meshes)):
                try:
                    chamfer = chamfer_distance(pred, gt)
                    results.append((i, chamfer))
                except Exception as e:
                    results.append((i, f"Error: {str(e)}"))
            return results

        def handle_result(results):
            for i, value in results:
                row = self.table.rowCount()
                self.table.insertRow(row)
                self.table.setItem(row, 0, QTableWidgetItem(str(i)))
                self.table.setItem(row, 1, QTableWidgetItem(str(value)))
            self.status_label.setText("Status: Chamfer distances computed.")

        self.threaded(task, handle_result)

    def load_pth_model(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select .pth Model File", "", "Model Files (*.pth)")
        if file:
            try:
                from scripts.model_multishape import MultiShapeDecoder
                checkpoint = torch.load(file, map_location='cpu')
                latent_weight = checkpoint.get("latent_codes.weight")
                if latent_weight is None:
                    raise ValueError("Missing latent_codes.weight in checkpoint.")

                num_shapes, latent_dim = latent_weight.shape
                args = SimpleNamespace(hidden_dim=512, dropout=0.1)
                model = MultiShapeDecoder(num_shapes=num_shapes, args=args, latent_dim=latent_dim)
                model.load_state_dict(checkpoint)
                model.eval()

                self.model = model
                self.latent_dim = latent_dim
                self.status_label.setText(f"Status: Loaded model with {num_shapes} shape codes")
                self.shape_id_spin.setMaximum(num_shapes - 1)

            except Exception as e:
                QMessageBox.critical(self, "Model Load Error", str(e))

    def decode_shape_to_mesh(self):
        if self.model is None:
            QMessageBox.warning(self, "No Model", "Load a model first.")
            return

        shape_id = self.shape_id_spin.value()
        grid_N = 128

        def task():
            lin = np.linspace(-1, 1, grid_N)
            grid = np.stack(np.meshgrid(lin, lin, lin, indexing='ij'), -1).reshape(-1, 3)
            xyz_tensor = torch.tensor(grid, dtype=torch.float32)
            shape_ids = torch.full((len(grid),), shape_id, dtype=torch.long)

            with torch.no_grad():
                sdf_pred = self.model(xyz_tensor, shape_ids).numpy().reshape(grid_N, grid_N, grid_N)
            verts, faces, _, _ = measure.marching_cubes(sdf_pred, level=0.0)
            return trimesh.Trimesh(vertices=verts / grid_N * 2 - 1, faces=faces, process=False)

        def handle_result(mesh):
            self.decoded_mesh = mesh
            self.status_label.setText(f"Status: Decoded shape {shape_id}")

        self.status_label.setText("Status: Decoding...")
        self.threaded(task, handle_result)

    def export_decoded_mesh(self):
        if not hasattr(self, 'decoded_mesh'):
            QMessageBox.warning(self, "No Mesh", "Decode a shape first.")
            return
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Decoded Mesh As", "", "OBJ Files (*.obj)")
        if save_path:
            self.decoded_mesh.export(save_path)
            self.status_label.setText(f"Status: Saved mesh to {os.path.basename(save_path)}")