import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt6.QtWidgets import QWidget, QVBoxLayout


class MeshViewerWidget(QWidget):
    """
    Very light wrapper around a GLViewWidget that shows one mesh
    and can be updated in-place via update_mesh().
    """
    def __init__(self, verts, faces, parent=None):
        super().__init__(parent)

        # ----------- UI boilerplate -----------
        layout = QVBoxLayout(self)
        self.view = gl.GLViewWidget()
        layout.addWidget(self.view)
        self.setLayout(layout)
        self.view.setBackgroundColor("w")

        # ----------- initial mesh -------------
        self.meshdata = gl.MeshData(vertexes=verts, faces=faces)
        self.mesh_item = gl.GLMeshItem(
            meshdata=self.meshdata,
            smooth=True,
            shader=None,      
            drawEdges=True,
            edgeColor=(0, 0, 0, 1),
            color=(0.8, 0.8, 0.8, 1),
        )
        self.view.addItem(self.mesh_item)
        self.__reset_camera(verts)


    def update_mesh(self, verts: np.ndarray, faces: np.ndarray):
        """
        Swap the viewer’s geometry with new vertex/face arrays.
        """
        self.meshdata.setVertexes(verts)
        self.meshdata.setFaces(faces)
        self.mesh_item.setMeshData(meshdata=self.meshdata)
        self.__reset_camera(verts)
        self.view.update()

    # ------------------------------------------------------------
    # Helper to keep camera centred on current geometry
    # ------------------------------------------------------------
    def __reset_camera(self, verts: np.ndarray):
        if verts.size == 0:
            return
        bb_min = verts.min(axis=0)
        bb_max = verts.max(axis=0)
        center = (bb_min + bb_max) / 2
        size = (bb_max - bb_min).max()

        self.view.opts["center"] = pg.Vector(*center)
        self.view.setCameraPosition(
            distance=size * 2.0,
            azimuth=0,
            elevation=10,
        )