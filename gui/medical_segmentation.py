# medical_segmentation.py

import os
import numpy as np
import sys
import subprocess
import tempfile 
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QLabel, QPushButton,
    QLineEdit, QDoubleSpinBox, QSpinBox, QFormLayout, QGroupBox, QFileDialog,
    QScrollArea, QFrame, QSlider, QMessageBox, QDialog, QPlainTextEdit
)
from PyQt6.QtGui import QPixmap, QImage, QColor, QAction
from PyQt6.QtCore import Qt, pyqtSlot, QPoint, QThread, pyqtSignal
from skimage import measure, morphology

import nibabel as nib
import scipy.ndimage as ndi
try:
    from skimage.morphology import binary_fill_holes
except ImportError:
    from scipy.ndimage import binary_fill_holes
import traceback
import torch

try:
    from scripts.mesh_viewer_widget import MeshViewerWidget
    MESH_VIEWER_ENABLED = True
except ImportError:
    MESH_VIEWER_ENABLED = False
    print("Warning: MeshViewerWidget not found. 3D mesh display will be disabled.")
    class MeshViewerWidget(QWidget): # Placeholder if not found
        def __init__(self, verts=None, faces=None, parent=None):
            super().__init__(parent)
            layout = QVBoxLayout(self)
            label = QLabel("3D Mesh Viewer (disabled - MeshViewerWidget not found)")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label)

MAX_UNDO_STATES = 20

class SliceLabel(QLabel):
    def __init__(self, parent):
        super().__init__()
        self.parent_widget = parent
        self.setMouseTracking(True)
        self.drawing = False

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.parent_widget.draw_mode != 'none':
            self.drawing = True
            if self.parent_widget.mask_data is not None:
                 self.parent_widget.save_mask_state_for_undo()
            self.modify_mask(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.drawing:
                self.drawing = False

    def mouseMoveEvent(self, event):
        if self.drawing and self.parent_widget.draw_mode != 'none':
            self.modify_mask(event.pos())

    def modify_mask(self, pos: QPoint):
            mode = self.parent_widget.draw_mode          # 'draw' | 'erase' | 'roi' | 'none'
            if mode == 'none':
                return

            # ------------------------------------------------------------------ #
            # 1. map mouse-pos (px,py) → voxel indices (i,j,z)                   #
            # ------------------------------------------------------------------ #
            pm = self.pixmap()
            if pm is None or self.parent_widget.volume_data is None:
                return

            # centre offset of pixmap in label
            off_x = (self.width()  - pm.width())  // 2
            off_y = (self.height() - pm.height()) // 2
            if not (off_x <= pos.x() < off_x + pm.width() and
                    off_y <= pos.y() < off_y + pm.height()):
                return

            px = pos.x() - off_x
            py = pos.y() - off_y

            # slice dims (H rows, W cols)
            if self.parent_widget.slice_axis == 0:          
                H, W = self.parent_widget.volume_data.shape[1:]
            elif self.parent_widget.slice_axis == 1:        
                H = self.parent_widget.volume_data.shape[0]
                W = self.parent_widget.volume_data.shape[2]
            else:                                          
                H, W = self.parent_widget.volume_data.shape[:2]

            scl_x = pm.width()  / H
            scl_y = pm.height() / W
            if scl_x == 0 or scl_y == 0:
                return

            i = int(px / scl_x)
            j = int(W - 1 - py / scl_y)    

            half = self.parent_widget.brush_size_spin.value() // 2
            i_min, i_max = max(0, i - half), min(H - 1, i + half)
            j_min, j_max = max(0, j - half), min(W - 1, j + half)
            z = self.parent_widget.current_slice

            # ------------------------------------------------------------------ #
            # 2. pick target mask(s) and value                                   #
            # ------------------------------------------------------------------ #
            if mode == 'roi':                       # GREEN
                masks = [self.parent_widget.roi_mask]
                value = True                        # boolean
            elif mode == 'draw':                    # RED
                masks = [self.parent_widget.mask_data]
                value = 1                           
            else:                                   # ERASE
                masks = [self.parent_widget.mask_data,
                        self.parent_widget.roi_mask]
                value = 0

            # ------------------------------------------------------------------ #
            # 3. paint into the selected 3-D arrays                              #
            # ------------------------------------------------------------------ #
            for arr3d in masks:
                if arr3d is None:
                    continue

                if self.parent_widget.slice_axis == 0:
                    arr3d[z, i_min:i_max+1, j_min:j_max+1] = value
                elif self.parent_widget.slice_axis == 1:
                    arr3d[i_min:i_max+1, z, j_min:j_max+1] = value
                else:
                    arr3d[i_min:i_max+1, j_min:j_max+1, z] = value

            # ------------------------------------------------------------------ #
            # 4. refresh view and enable export                                  #
            # ------------------------------------------------------------------ #
            if mode == 'roi':
                self.parent_widget._fill_roi_slice(z)

            self.parent_widget.update_slice_view()
            self.parent_widget.export_btn.setEnabled(True)

class Worker(QThread):
    """A worker to run a script in a subprocess and emit its output."""
    lineReceived = pyqtSignal(str)
    finished = pyqtSignal(int, str)

    def __init__(self, cmd: list[str], parent=None):
        super().__init__(parent)
        self.cmd = cmd
        self.process = None

    def run(self):
        try:
            self.process = subprocess.Popen(
                self.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )
            for line in iter(self.process.stdout.readline, ''):
                self.lineReceived.emit(line.strip())
            
            stderr_output = self.process.stderr.read()
            self.process.wait()
            self.finished.emit(self.process.returncode, stderr_output)
        except Exception as e:
            self.finished.emit(-1, str(e))

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()

class MedicalSegmentationTabWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.nifti_path = None
        self.volume_data = None
        self.mask_data = None
        self.slice_axis = 2
        self.current_slice = 0
        self.draw_mode = 'none' # ['none', 'draw', 'erase']

        # Create custom mask
        self.roi_mask   = None
        self.roi_mode   = False

        self.undo_stack = []
        self.redo_stack = []
        self.fitter_model_path = None
        self.last_fit_obj_path = None
        self.last_mask_path_for_fit = None
        self.fit_worker = None
        self.initUI()

    def _toggle_roi_mode(self):
        if self.roi_button.isChecked():
            self.draw_mode = 'roi'
            self.draw_button.setChecked(False)
            self.erase_button.setChecked(False)
        else:
            self.draw_mode = 'none'


    def initUI(self):
        # ------------------------------------------------------------------ #
        # 0.  basic splitter ------------------------------------------------#
        # ------------------------------------------------------------------ #
        main_layout = QHBoxLayout(self)
        splitter     = QSplitter(Qt.Orientation.Horizontal, self)
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

        # ================================================================== #
        # 1.  -------- LEFT-HAND CONTROL PANEL ----------------------------- #
        # ================================================================== #
        left_panel  = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # A) NIfTI loader --------------------------------------------------- #
        self.load_nifti_btn = QPushButton("Load NIfTI")
        self.load_nifti_btn.clicked.connect(self.on_load_nifti)
        left_layout.addWidget(self.load_nifti_btn)

        # B) automated segmentation group ---------------------------------- #
        segment_box  = QGroupBox("Automated Segmentation")
        segment_form = QFormLayout(segment_box)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(-2000, 2000)
        self.threshold_spin.setValue(15.0)
        segment_form.addRow("Bone Threshold (HU)", self.threshold_spin)

        self.min_size_spin = QSpinBox()
        self.min_size_spin.setRange(1, 999_999)
        self.min_size_spin.setValue(1000)
        segment_form.addRow("Min size (vox)", self.min_size_spin)

        self.segment_btn = QPushButton("Segment Largest Region")
        self.segment_btn.setEnabled(False)
        self.segment_btn.clicked.connect(self.on_segment)
        segment_form.addRow(self.segment_btn)
        left_layout.addWidget(segment_box)

        # C) slice index slider ------------------------------------------- #
        slice_box  = QGroupBox("Slice Viewer")
        slice_form = QFormLayout(slice_box)
        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setRange(0, 0)
        self.slice_slider.valueChanged.connect(self.on_slice_changed)
        slice_form.addRow("Slice Index", self.slice_slider)
        left_layout.addWidget(slice_box)

        # D) drawing / ROI tools ------------------------------------------ #
        tool_box  = QGroupBox("Drawing Tools")
        tool_form = QFormLayout(tool_box)

        # ---- create the buttons ----------------------------------------- #
        self.draw_button  = QPushButton("Draw")    # red mask (foreground)
        self.erase_button = QPushButton("Erase")   # erases red OR green
        self.roi_button   = QPushButton("ROI")     # green mask (limit seg)

        for b in (self.draw_button, self.erase_button, self.roi_button):
            b.setCheckable(True)

        self.draw_mode = 'none'                    # 'draw'|'erase'|'roi'|'none'
        def _hook(btn, mode_name:str):
            def _cb(checked: bool):
                if checked:
                    for other in (self.draw_button, self.erase_button, self.roi_button):
                        if other is not btn:
                            other.blockSignals(True)
                            other.setChecked(False)
                            other.blockSignals(False)
                    self.draw_mode = mode_name
                else:
                    if self.draw_mode == mode_name:
                        self.draw_mode = 'none'
            return _cb

        self.draw_button.toggled .connect(_hook(self.draw_button , 'draw' ))
        self.erase_button.toggled.connect(_hook(self.erase_button, 'erase'))
        self.roi_button.toggled  .connect(_hook(self.roi_button  , 'roi'  ))

        # ---- lay them out ------------------------------------------------ #
        row_btn = QHBoxLayout()
        row_btn.addWidget(self.draw_button)
        row_btn.addWidget(self.erase_button)
        row_btn.addWidget(self.roi_button)
        tool_form.addRow(row_btn)

        # ---- brush size -------------------------------------------------- #
        self.brush_size_spin = QSpinBox()
        self.brush_size_spin.setRange(1, 50)
        self.brush_size_spin.setValue(5)
        self.brush_size_spin.setSuffix(" px")
        tool_form.addRow("Brush size", self.brush_size_spin)

        # ---- undo / redo ------------------------------------------------- #
        self.undo_button = QPushButton("Undo")
        self.redo_button = QPushButton("Redo")
        self.undo_button.clicked.connect(self.on_undo)
        self.redo_button.clicked.connect(self.on_redo)
        self.undo_button.setEnabled(False)
        self.redo_button.setEnabled(False)

        row_ur = QHBoxLayout()
        row_ur.addWidget(self.undo_button)
        row_ur.addWidget(self.redo_button)
        tool_form.addRow(row_ur)
        left_layout.addWidget(tool_box)

        self.export_mask_btn = QPushButton("Export Mask(Red Mask) → .nii")
        self.export_mask_btn.clicked.connect(self.export_mask_to_nii)
        left_layout.addWidget(self.export_mask_btn)

        # E) export -------------------------------------------------------- #
        self.export_btn = QPushButton("Export Mask → .obj")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.on_export_obj)
        left_layout.addWidget(self.export_btn)

        # F) fit DeepSDF model to mask ------------------------------------ #
        fit_box = QGroupBox("Fit Model to Mask")
        fit_form = QFormLayout(fit_box)

        self.load_fitter_model_btn = QPushButton("Load Trained Model (.pth)")
        self.model_status_label = QLabel("No model loaded.")
        self.model_status_label.setStyleSheet("font-style: italic; color: gray;")
        
        row_fit_params = QHBoxLayout()
        self.iters_spin = QSpinBox()
        self.iters_spin.setRange(100, 10000)
        self.iters_spin.setValue(1200)
        self.iters_spin.setToolTip("Number of optimization iterations.")
        row_fit_params.addWidget(QLabel("Iters:"))
        row_fit_params.addWidget(self.iters_spin)
        
        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(1000, 50000)
        self.samples_spin.setValue(8000)
        self.samples_spin.setToolTip("Number of points to sample for the loss calculation.")
        row_fit_params.addWidget(QLabel("Samples:"))
        row_fit_params.addWidget(self.samples_spin)

        self.fit_btn = QPushButton("Fit DeepSDF to Current Mask")
        self.check_align_btn = QPushButton("Check Alignment of Last Fit")
        self.fit_btn.setEnabled(False)
        self.check_align_btn.setEnabled(False)

        fit_form.addRow(self.load_fitter_model_btn)
        fit_form.addRow(self.model_status_label)
        fit_form.addRow(row_fit_params)
        fit_form.addRow(self.fit_btn)
        fit_form.addRow(self.check_align_btn)
        left_layout.addWidget(fit_box)

        left_layout.addStretch()
        splitter.addWidget(left_panel)

        # ================================================================== #
        # 2.  -------- RIGHT-HAND VIEWER PANEL ----------------------------- #
        # ================================================================== #
        right_panel  = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # 2D slice --------------------------------------------------------- #
        self.slice_label = SliceLabel(self)
        self.slice_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.slice_label.setStyleSheet("background-color:black; color:white;")
        right_layout.addWidget(self.slice_label, stretch=1)

        # 3D mesh viewer--------------------------------------- #
        if MESH_VIEWER_ENABLED:
            self.mesh_container = QFrame()
            self.mesh_container.setLayout(QVBoxLayout())
            right_layout.addWidget(self.mesh_container, stretch=1)
        else:
            placeholder = QLabel("3D mesh viewer disabled")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            right_layout.addWidget(placeholder, stretch=1)

        splitter.addWidget(right_panel)
        splitter.setSizes([350, 850])

        self.load_fitter_model_btn.clicked.connect(self.on_load_fitter_model)
        self.fit_btn.clicked.connect(self.on_fit_model)
        self.check_align_btn.clicked.connect(self.on_check_alignment)

    def get_brush_size(self):
        return self.brush_size_spin.value()

    def save_mask_state_for_undo(self):
        if self.mask_data is None or self.roi_mask is None:
            return

        if len(self.undo_stack) >= MAX_UNDO_STATES:
            self.undo_stack.pop(0)

        self.undo_stack.append((
            self.mask_data.copy(),
            self.roi_mask.copy()
        ))
        self.redo_stack.clear()
        self.update_undo_redo_buttons()

    def on_undo(self):
        if not self.undo_stack:
            return
       
        self.redo_stack.append((
            self.mask_data.copy(),
            self.roi_mask.copy()
        ))
       
        self.mask_data, self.roi_mask = self.undo_stack.pop()
        self.update_slice_view()
        self.update_undo_redo_buttons()

    def on_redo(self):
        if not self.redo_stack:
            return
        self.undo_stack.append((
            self.mask_data.copy(),
            self.roi_mask.copy()
        ))
        self.mask_data, self.roi_mask = self.redo_stack.pop()
        self.update_slice_view()
        self.update_undo_redo_buttons()

    def update_undo_redo_buttons(self):
        self.undo_button.setEnabled(len(self.undo_stack) > 0)
        self.redo_button.setEnabled(len(self.redo_stack) > 0)

    @pyqtSlot()
    def on_load_nifti(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select NIfTI File", "", "NIfTI Files (*.nii *.nii.gz)")
        if not file_path:
            return
        self.nifti_path = file_path

        try:
            nii_img = nib.load(file_path)
            data = nii_img.get_fdata()
            self.volume_data = data.astype(np.float32)

            self.mask_data = np.zeros_like(self.volume_data, dtype=np.uint8) # red mask
            self.roi_mask  = np.zeros_like(self.volume_data, dtype=bool) # green mask

            print(f"Loaded volume shape={self.volume_data.shape}, range=({self.volume_data.min()}..{self.volume_data.max()})")

            self.segment_btn.setEnabled(True)
            self.export_btn.setEnabled(False)

            self.undo_stack.clear()
            self.redo_stack.clear()
            self.update_undo_redo_buttons()

            slices_along_axis = self.volume_data.shape[self.slice_axis]
            self.slice_slider.setRange(0, slices_along_axis - 1)
            self.slice_slider.setValue(slices_along_axis // 2)
            self.current_slice = self.slice_slider.value()
            self.update_slice_view()

        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load {file_path}:\n{str(e)}")
            self.volume_data = None
            self.mask_data = None
            self.segment_btn.setEnabled(False)
            self.export_btn.setEnabled(False)
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.update_undo_redo_buttons()

    def update_slice_view(self):
        # ------------------------------------------------------------------
        # 0. early-out if nothing loaded
        # ------------------------------------------------------------------
        if self.volume_data is None:
            blank = QPixmap(self.slice_label.size())
            blank.fill(Qt.GlobalColor.black)
            self.slice_label.setPixmap(blank)
            self.slice_label.setText("No volume loaded")
            return

        # ------------------------------------------------------------------
        # 1. pull the correct intensity slice + the two masks
        # ------------------------------------------------------------------
        z = self.current_slice
        if self.slice_axis == 0:
            slice_2d = self.volume_data[z, :, :]
            mask_2d  = self.mask_data[z,  :, :]  if self.mask_data is not None else None
            roi_2d   = self.roi_mask[z,   :, :]  if getattr(self, "roi_mask", None) is not None else None
        elif self.slice_axis == 1:
            slice_2d = self.volume_data[:, z, :]
            mask_2d  = self.mask_data[:,  z, :]  if self.mask_data is not None else None
            roi_2d   = self.roi_mask[:,   z, :]  if getattr(self, "roi_mask", None) is not None else None
        else: 
            slice_2d = self.volume_data[:, :, z]
            mask_2d  = self.mask_data[:,  :, z]  if self.mask_data is not None else None
            roi_2d   = self.roi_mask[:,   :, z]  if getattr(self, "roi_mask", None) is not None else None

        # ------------------------------------------------------------------
        # 2. Rotation
        # ------------------------------------------------------------------
        slice_rot = np.rot90(slice_2d)
        if slice_rot.max() > slice_rot.min():
            norm = (slice_rot - slice_rot.min()) / (slice_rot.max() - slice_rot.min())
        else:
            norm = np.zeros_like(slice_rot)
        gray_u8 = (norm * 255).astype(np.uint8)

        h, w = gray_u8.shape
        qimg = QImage(gray_u8.data, w, h, w, QImage.Format.Format_Grayscale8)
        qimg = qimg.convertToFormat(QImage.Format.Format_ARGB32)

        ptr = qimg.bits();  ptr.setsize(h * w * 4)
        rgba = np.frombuffer(ptr, np.uint8).reshape((h, w, 4))


        if roi_2d is not None:
            green = np.array([0, 255, 0, 120], dtype=np.uint8)    # B,G,R,A
            rgba[np.rot90(roi_2d) > 0] = green
        # ------------------------------------------------------------------
        
        if mask_2d is not None:
            red = np.array([0, 0, 255, 120], dtype=np.uint8)      # B,G,R,A
            rgba[np.rot90(mask_2d) > 0] = red

        pix = QPixmap.fromImage(qimg)
        self.slice_label.setPixmap(
            pix.scaled(self.slice_label.width(),
                    self.slice_label.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation)
        )

    @pyqtSlot(int)
    def on_slice_changed(self, value):
        self.current_slice = value
        self.update_slice_view()

    def _fill_roi_slice(self, z_idx: int):
        """Fill the area enclosed by the green border on slice z_idx."""
        if self.roi_mask is None:
            return
        sl = self.roi_mask[:, :, z_idx] if self.slice_axis == 2 else (
            self.roi_mask[:, z_idx, :]  if self.slice_axis == 1 else
            self.roi_mask[z_idx, :, :])
        sl_filled = binary_fill_holes(sl)
        if self.slice_axis == 2:
            self.roi_mask[:, :, z_idx] |= sl_filled
        elif self.slice_axis == 1:
            self.roi_mask[:, z_idx, :]  |= sl_filled
        else:
            self.roi_mask[z_idx, :, :]  |= sl_filled

    def _propagate_roi(self):
        """
        Copy the nearest drawn ROI slice into all empty slices between
        the first and last ROI slice.
        """
        if self.roi_mask is None:
            return
        # along Z (axial)
        roi_any = self.roi_mask.any(axis=(0, 1)) 
        idx     = np.where(roi_any)[0]
        if len(idx) < 2:                               
            return
        z_first, z_last = idx[0], idx[-1]
        prev = None
        for z in range(z_first, z_last + 1):
            if roi_any[z]:
                prev = z
            else:
                
                nxt = idx[idx > z][0]         
                use = prev if (z - prev) <= (nxt - z) else nxt
                if self.slice_axis == 2:
                    self.roi_mask[:, :, z] = self.roi_mask[:, :, use]
                elif self.slice_axis == 1:
                    self.roi_mask[:, z, :]  = self.roi_mask[:, use, :]
                else:
                    self.roi_mask[z, :, :]  = self.roi_mask[use, :, :]

    @pyqtSlot()
    def on_segment(self):
        if self.volume_data is None:
            QMessageBox.warning(self, "No volume", "Load a .nii first.")
            return

        # -------- 1.  prepare / densify the ROI ---------------------------
        if self.roi_mask is not None and self.roi_mask.any():
            for z in range(self.roi_mask.shape[2]):
                self._fill_roi_slice(z)
            self._propagate_roi()                       
            roi_bool = self.roi_mask                    
        else:
            roi_bool = None                             

        # -------- 2.  binary threshold  -----------------------------------
        thr    = self.threshold_spin.value()
        min_sz = self.min_size_spin.value()

        binary = self.volume_data > thr                  
        if roi_bool is not None:                         
            binary &= roi_bool

        # -------- 3.  clean-up & largest CC -------------------------------
        binary = morphology.remove_small_holes(binary, area_threshold=500)

        labeled = measure.label(binary, connectivity=1)
        props   = measure.regionprops(labeled)

        if not props:
            QMessageBox.information(self, "No region",
                                    "No connected component above threshold.")
            return

        biggest = max(props, key=lambda r: r.area)
        if biggest.area < min_sz:
            QMessageBox.information(self, "Small region",
                                    f"Largest CC = {biggest.area:.1f} vox < min {min_sz}.")
            return

        largest_mask = labeled == biggest.label   

        # optional tidy-up
        largest_mask = morphology.remove_small_holes(largest_mask, 1000)

        # -------- 4.  commit to class mask & refresh ----------------------
        self.mask_data[:] = 0                       
        self.mask_data[largest_mask] = 1               
        self.export_btn.setEnabled(True)
        self.update_slice_view()

    @pyqtSlot()
    def export_mask_to_nii(self):
        if self.mask_data is None or not self.mask_data.any():
            QMessageBox.information(self, "No mask",
                                    "Run segmentation first (red voxels).")
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save mask (.nii)", "", "NIfTI files (*.nii *.nii.gz)")
        if not out_path:
            return

        ref_img = nib.load(self.nifti_path)
        mask_img = nib.Nifti1Image(self.mask_data.astype(np.uint8),
                                ref_img.affine,
                                ref_img.header)
        nib.save(mask_img, out_path)
        QMessageBox.information(self, "Saved",
                                f"Segmentation mask written to:\n{out_path}")

    @pyqtSlot()
    def on_export_obj(self):
        if self.mask_data is None or np.sum(self.mask_data) == 0:
            QMessageBox.warning(self, "No Mask", "Mask is empty or not yet created/segmented. Draw or segment first.")
            return

        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save Mesh as .obj", "", "Wavefront OBJ (*.obj)"
        )
        if not out_path:
            return

        try:
            verts, faces, _, _ = measure.marching_cubes( 
                self.mask_data.astype(np.float32), level=0.5
            )
            print(f"marching_cubes → {len(verts)} verts, {len(faces)} faces")

            if len(verts) == 0:
                QMessageBox.information(self, "Empty Mesh", "Marching cubes resulted in an empty mesh. Nothing to export.")
                return

            if self.nifti_path:
                 nii = nib.load(self.nifti_path)
                 zooms = nii.header.get_zooms()[:3]
                 verts = verts * np.asarray(zooms)[None, :]
            else:
                 print("Warning: NIfTI path not available. Exporting mesh without voxel spacing.")


            with open(out_path, "w") as f:
                for vx, vy, vz in verts:
                    f.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")
                for a, b, c in faces: 
                    f.write(f"f {a+1} {b+1} {c+1}\n") 

            QMessageBox.information(
                self, "Exported",
                f"Saved mesh ({len(verts)} verts) in world-space mm units (if NIfTI was loaded):\n{out_path}"
            )

            if MESH_VIEWER_ENABLED:
                self.show_3d_mesh(verts, faces)
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export .obj:\n{str(e)}")
            print(f"Export error: {e}")
            traceback.print_exc()

    def show_3d_mesh(self, verts, faces):
        if not MESH_VIEWER_ENABLED or not isinstance(self.mesh_container, QFrame):
            return
            
        layout = self.mesh_container.layout()
        if layout is None:
            layout = QVBoxLayout(self.mesh_container)
            self.mesh_container.setLayout(layout)

        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        viewer = MeshViewerWidget(verts, faces)
        viewer.setMinimumSize(300, 300)
        layout.addWidget(viewer)
    
    @pyqtSlot()
    def on_load_fitter_model(self):
        """Opens a file dialog to select the trained .pth model for fitting."""
        path, _ = QFileDialog.getOpenFileName(self, "Select Trained Model", "", "PyTorch Models (*.pth)")
        if path:
            self.fitter_model_path = path
            self.model_status_label.setText(f"Loaded: {os.path.basename(path)}")
            self.model_status_label.setStyleSheet("font-style: normal; color: green;")
            self.fit_btn.setEnabled(True)

    @pyqtSlot()
    def on_fit_model(self):
        """Runs the fit_latent_mri.py script in a background thread."""
        # --- 1. More Robust Input Validation ---
        if self.volume_data is None:
            QMessageBox.warning(self, "Error", "Please load a NIfTI volume first.")
            return
        if self.fitter_model_path is None:
            QMessageBox.warning(self, "Error", "Please load a trained model first.")
            return
        if self.roi_mask is None or not self.roi_mask.any():
            QMessageBox.warning(self, "Error", "Please draw a green ROI mask first.")
            return

        # --- 2. Save the green ROI mask to a temporary NIfTI file ---
        try:
            ref_img = nib.load(self.nifti_path)
            mask_to_save = self.roi_mask.astype(np.uint8)
            mask_img = nib.Nifti1Image(mask_to_save, ref_img.affine, ref_img.header)
            
            # Use a context manager to be safer, but store path outside
            with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
                self.last_mask_path_for_fit = tmp.name
                nib.save(mask_img, self.last_mask_path_for_fit)

        except Exception as e:
            QMessageBox.critical(self, "Mask Save Error", f"Could not save temporary mask file: {e}")
            return
            
        # --- 3. Define output path and command ---
        base_name = os.path.basename(self.nifti_path).rsplit('.', 1)[0].replace('.nii', '')
        output_dir = os.path.dirname(self.nifti_path)
        self.last_fit_obj_path = os.path.join(output_dir, f"{base_name}_fit_roi.obj")

        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "fit_latent_mri.py"))
        
        cmd = [
            sys.executable, "-u", script_path,
            "--weights", self.fitter_model_path,
            "--mri", self.nifti_path,
            "--mask", self.last_mask_path_for_fit,
            "--out_obj", self.last_fit_obj_path,
            "--iters", str(self.iters_spin.value()),
            "--samples", str(self.samples_spin.value()),
            "--device", "cuda:0" if torch.cuda.is_available() else "cpu"
        ]

        # --- 4. Launch the worker and show a progress dialog ---
        self.fit_dialog = QDialog(self)
        self.fit_dialog.setWindowTitle("Fitting in Progress...")
        dlg_layout = QVBoxLayout(self.fit_dialog)
        self.fit_log = QPlainTextEdit()
        self.fit_log.setReadOnly(True)
        cancel_btn = QPushButton("Cancel")
        dlg_layout.addWidget(self.fit_log)
        dlg_layout.addWidget(cancel_btn)
        self.fit_dialog.resize(600, 400)

        self.fit_worker = Worker(cmd, self)
        self.fit_worker.lineReceived.connect(self.fit_log.appendPlainText)
        self.fit_worker.finished.connect(self.on_fit_finished)
        cancel_btn.clicked.connect(self.fit_worker.stop)
        
        self.fit_worker.start()
        self.fit_dialog.exec()

    def on_fit_finished(self, retcode, stderr):
        """Called when the fitting process is complete."""
        self.fit_dialog.close()
        if retcode == 0:
            QMessageBox.information(self, "Success", f"Fitting complete!\nOutput mesh saved to:\n{self.last_fit_obj_path}")
            self.check_align_btn.setEnabled(True)
            self.on_check_alignment()
        else:
            QMessageBox.critical(self, "Fit Failed", f"The fitting script failed with code {retcode}.\n\nError:\n{stderr}")

    @pyqtSlot()
    def on_check_alignment(self):
        """Runs the check_alignment.py script to visualize the last fit."""
        if not self.last_fit_obj_path or not os.path.exists(self.last_fit_obj_path):
            QMessageBox.warning(self, "Error", "No fitted mesh found. Please run the fitting process first.")
            return
        if not self.roi_mask.any():
            QMessageBox.warning(self, "Error", "There is no active ROI mask to check against.")
            return

        mask_path_for_check = None
        try:
            ref_img = nib.load(self.nifti_path)
            mask_img = nib.Nifti1Image(self.roi_mask.astype(np.uint8), ref_img.affine, ref_img.header)
            with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp:
                nib.save(mask_img, tmp.name)
                mask_path_for_check = tmp.name

            script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "check_alignment.py"))
            
            cmd = [
                sys.executable, script_path,
                "--mri", self.nifti_path,
                "--mask", mask_path_for_check,
                "--mesh", self.last_fit_obj_path,
                "--show"
            ]
            subprocess.run(cmd, check=True)

        except Exception as e:
            QMessageBox.critical(self, "Check Alignment Error", f"Failed to run alignment check:\n{e}")
        finally:
            if mask_path_for_check and os.path.exists(mask_path_for_check):
                os.remove(mask_path_for_check)