 _____   ______  ______ ______       _____  _____   ______ 
|  __ \ |  ____||  ____||  __ \     / ____||  __ \ |  ____|
| |  | || |__   | |__   | |__) |    | (___ | |  | || |__   
| |  | ||  __|  |  __|  |  ___/     \___  \| |  | ||  __|  
| |__| || |____ | |____ | |         ____) || |__| || |     
|_____/ |______||______||_|         |_____/|_____/ |_|     

# Brief
This project provides a graphical user interface (GUI) for training DeepSDF models and fitting them to 3D medical scans.It streamlines the entire workflow, from data preparation and model training to interactive segmentation and final mesh extraction. This application was developed as part of a Master's thesis.

# GitHub
https://github.com/FrancoisVanZyl211/DeepSDF-Spine/tree/main

## Features
    *Interactive NIfTI Viewer: Load and scroll through .nii medical scans.
    *Semi-Automatic Segmentation: Use thresholding and region-growing to create initial anatomical masks.
    *Manual ROI Tools: Manually draw, erase, and refine regions of interest (ROI) directly on scan slices.
    *One-Click Model Fitting: Apply a pre-trained DeepSDF model to a new mask to generate a smooth 3D mesh with a single button click.
    *3D Visualization: Instantly view the final fitted mesh and check its alignment.

## Requirements
torch==2.1.0
numpy==1.26.2
PyQt6==6.8.1
pyqtgraph==0.13.7
nibabel==5.3.2
scikit-image==0.20.0
scipy==1.15.2
trimesh==4.6.5
open3d==0.17.0
matplotlib==3.10.1

## How to Run the Application
All functionality is handled through the main graphical interface. To launch the application, run the main_window.py file.

## How to install and Setup
Please create a dedicated Python environment to avoid conflicts with other packages

Conda to create a new environment. Open your Anaconda Prompt / terminal and run:
```bash
conda create -n deepsdf-gui python=3.10
conda activate deepsdf-gui
```
Once finished, install all the necessary packages via
```bash
pip install -r requirements.txt
```
Once finish installing the required packages. Please run the main_window.py file in the GUI folder or
```bash
python gui/main_window.py
```