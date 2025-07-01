import sys
import os
import subprocess
import numpy as np
import math
import trimesh
import glob
import torch
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget
from PyQt6.QtCore import Qt, pyqtSlot, pyqtSignal, QThread
from PyQt6.QtGui import QPalette, QColor, QPainter, QPen, QPixmap, QGuiApplication

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from gui.evaluation_tab import EvaluationTabWidget
from gui.medical_segmentation import MedicalSegmentationTabWidget
from gui.interpolation_tab import InterpolationTabWidget
from gui.interpolationSlider_tab import InterpolationSliderTabWidget
from gui.analyzer_tab import AnalyzerTabWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeepSDF - GUI")
        self.setMinimumSize(1200, 700)
        self.initUI()
    
    def initUI(self):
        tabs = QTabWidget()
        self.setCentralWidget(tabs)
        
        evaluation_tab = EvaluationTabWidget()
        segmentation_tab = MedicalSegmentationTabWidget()
        interpolation_tab = InterpolationTabWidget()
        interpolationSlider_tab = InterpolationSliderTabWidget()
        analyzer_tab = AnalyzerTabWidget()
        
        tabs.addTab(evaluation_tab, "Evaluation")
        tabs.addTab(segmentation_tab, "Medical Segmentation")
        tabs.addTab(interpolation_tab, "Interpolation")
        tabs.addTab(interpolationSlider_tab, "Interpolation Slider")
        tabs.addTab(analyzer_tab, "Analyzer")

def main():
    app = QApplication(sys.argv)
    
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(240, 240, 240))
    palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Text, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorRole.Button, QColor(230, 230, 230))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(0, 0, 0))
    app.setPalette(palette)
    window = MainWindow()
    
    screen = QGuiApplication.primaryScreen()
    if screen is not None:
        screen_geometry = screen.availableGeometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()
        window.resize(int(screen_width * 0.9), int(screen_height * 0.9))
        window.move((screen_width - window.width()) // 2, (screen_height - window.height()) // 2)
    
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()