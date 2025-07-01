# interpolationSlider_tab.py
from __future__ import annotations
import os
import numpy as np
import torch
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton,
    QComboBox, QSlider, QMessageBox, QGroupBox, QFrame, QSplitter,
    QFileDialog, QWidget, QProgressDialog
)
from skimage import measure

from gui.base_tab import BaseTabWidget
from scripts.latent_interpolation import predict_sdf_with_latent
from scripts.model_multishape import MultiShapeDecoder
from scripts.mesh_viewer_widget import MeshViewerWidget


class InterpolationSliderTabWidget(BaseTabWidget):
    """
    Latent interpolation tab with a persistent MeshViewer and
    a pre-computed cache of meshes for instant slider response.
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        self.model = None
        self.z0 = self.z1 = None
        self.shape0_id = self.shape1_id = 0
        self.mesh_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.cache_steps = 101
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self):
        splitter = QSplitter(Qt.Orientation.Horizontal, self)

        # ------------------ LEFT PANEL ------------------
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        self.load_model_btn = QPushButton("Load Model (.pth)")
        self.load_model_btn.clicked.connect(self.load_model)
        left_layout.addWidget(self.load_model_btn)
        combo_group = QGroupBox("Select Shapes")
        combo_layout = QFormLayout(combo_group)

        self.shape0_combo = QComboBox(enabled=False)
        self.shape1_combo = QComboBox(enabled=False)
        combo_layout.addRow("Shape 0:", self.shape0_combo)
        combo_layout.addRow("Shape 1:", self.shape1_combo)
        left_layout.addWidget(combo_group)

        self.set_shapes_btn = QPushButton("Set Shapes for Slider", enabled=False)
        self.set_shapes_btn.clicked.connect(self.on_set_shapes)
        left_layout.addWidget(self.set_shapes_btn)

        self.alpha_slider = QSlider(Qt.Orientation.Horizontal, enabled=False)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.valueChanged.connect(self.update_interpolation)
        left_layout.addWidget(QLabel("Interpolation Slider (0 → 1)"))
        left_layout.addWidget(self.alpha_slider)

        left_layout.addStretch()
        splitter.addWidget(left_panel)

        # ------------------ RIGHT PANEL -----------------
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.viewer_container = QFrame()
        vlayout = QVBoxLayout(self.viewer_container)
        self.mv = MeshViewerWidget(np.empty((0, 3)), np.empty((0, 3), dtype=np.int32))
        vlayout.addWidget(self.mv)

        right_layout.addWidget(self.viewer_container)
        splitter.addWidget(right_panel)
        splitter.setSizes([380, 820])

        main = QHBoxLayout(self)
        main.addWidget(splitter)
        self.setLayout(main)

    # ------------------------------------------------------------------
    @pyqtSlot()
    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model Checkpoint (.pth)", "", "PyTorch Models (*.pth)"
        )
        if not path:
            return
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ckpt = torch.load(path, map_location=device)
            num_shapes = ckpt["latent_codes.weight"].size(0)
            latent_dim = ckpt["latent_codes.weight"].size(1)

            model = MultiShapeDecoder(num_shapes=num_shapes, latent_dim=latent_dim)
            model.load_state_dict(ckpt)
            model.to(device).eval()
            self.model = model

            QMessageBox.information(
                self, "Model Loaded",
                f"{os.path.basename(path)}\nShapes: {num_shapes} • Latent: {latent_dim}"
            )

            self.shape0_combo.clear()
            self.shape1_combo.clear()
            [self.shape0_combo.addItem(str(i)) for i in range(num_shapes)]
            [self.shape1_combo.addItem(str(i)) for i in range(num_shapes)]
            for w in (self.shape0_combo, self.shape1_combo, self.set_shapes_btn):
                w.setEnabled(True)
        except Exception as e:
            self.alert(str(e), level="error")

    # ------------------------------------------------------------------
    @pyqtSlot()
    def on_set_shapes(self):
        if self.model is None:
            return
        s0 = int(self.shape0_combo.currentText())
        s1 = int(self.shape1_combo.currentText())
        if s0 == s1:
            self.alert("Pick two different shapes.", level="warning")
            return

        self.shape0_id, self.shape1_id = s0, s1
        with torch.no_grad():
            self.z0 = self.model.latent_codes.weight[s0].detach().clone()
            self.z1 = self.model.latent_codes.weight[s1].detach().clone()

        self.alpha_slider.setEnabled(False)
        self.mesh_cache.clear()

        # progress dialog while pre-computing
        self.prog = QProgressDialog("Pre-computing mesh cache…", None, 0,
                                    self.cache_steps, self)
        self.prog.setWindowTitle("Please wait")
        self.prog.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.prog.show()

        self.threaded(self._precompute_cache, self._cache_ready)

    # ------------------ cache worker ------------------
    def _precompute_cache(self):
        cache = {}
        device = next(self.model.parameters()).device
        alphas = np.linspace(0.0, 1.0, self.cache_steps)

        with torch.no_grad():
            for idx, a in enumerate(alphas):
                z = (1 - a) * self.z0 + a * self.z1
                z = z.to(device)
                sdf = predict_sdf_with_latent(self.model, z,
                                              grid_N=32, max_xyz=1.0, device=device)
                verts, faces, *_ = measure.marching_cubes(sdf, level=0.0)
                cache[int(round(a * 100))] = (verts, faces)
                self.prog.setValue(idx + 1)
        return cache

    def _cache_ready(self, cache):
        self.mesh_cache = cache
        self.prog.close()
        self.alpha_slider.setEnabled(True)
        self.mv.update_mesh(*self.mesh_cache[0])

    # ------------------------------------------------------------------
    def update_interpolation(self):
        if not self.mesh_cache:
            return
        key = self.alpha_slider.value()
        verts, faces = self.mesh_cache[key]
        self.mv.update_mesh(verts, faces)