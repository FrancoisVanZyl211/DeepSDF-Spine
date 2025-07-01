# DeepSDF-Spine
# Interactive DeepSDF-based Vertebra Fitting from MRI
This repository contains the source code and documentation for the APPS597 project report, "3D Shape Representation using DeepSDF on MRI vertebrae," completed at the University of Otago. The project is a Python application with a comprehensive PyQt5 GUI for training a multi-shape DeepSDF model and interactively fitting it to medical imaging data, specifically MRI scans of human lumbar vertebrae.

## Video Demonstration
A full video demonstration of the software's features, including the interactive fitting process, is available here:
   * Video Example

* **Medical Image Viewer:** Load and visualize volumetric NIfTI scans (MRI/CT) with a 2D slice viewer.
* **Interactive Segmentation Tools:**
    * Automated bone-thresholding to generate an initial segmentation guess.
    * Manual brush tools (draw/erase) to define a precise Region of Interest (ROI).
* **Offline DeepSDF Training Pipeline:**
    * Pre-processing tools to convert `.obj` meshes into signed distance samples (`.pts` files).
    * A GUI-driven training module to train a multi-shape DeepSDF model on a collection of shapes.
* **Interactive Model Fitting:**
    * Fit a trained DeepSDF model to a target MRI scan using a user-defined ROI.
    * Implements a novel **gradient-weighted loss function** to robustly guide the fit to anatomical edges in low-contrast MRI data.
* **Latent Space Exploration:**
    * Tools to visualize and interactively interpolate between learned shapes in the latent space, demonstrating the quality of the generative model.
* **Quantitative Analysis:**
    * Built-in tools to compute **Chamfer Distance** for reconstruction accuracy and **AABB-IoU** for fitting alignment.
 
Technology Stack
* **Language:** Python 3
* **Core Libraries:** PyTorch, NumPy, SciPy
* **GUI:** PyQt5
* **3D Visualization:** Open3D, Matplotlib
* **Medical Imaging:** SimpleITK, NiBabel

## ⚙️ Setup and Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/FrancoisVanZyl211/DeepSDF-Spine/tree/main
    cd folder
    ```
2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download Datasets:**
    * The training data used in this project can be sourced from the [3D Lumbar Vertebrae Data Set](https://figshare.com/articles/dataset/3D_Lumbar_Vertebrae_Data_Set/3493643) and the [Spine Segmentation from CT Scans](https://www.kaggle.com/datasets/pycadmk/spine-segmentation-from-ct-scans) dataset.
    * Place the data in the `/data` folder

## Usage & Workflow

The application is divided into two main workflows, managed through the GUI.

### 1. Offline Model Training (`Evaluation` Tab)
1.  Use the **Data Prep** tools to convert a folder of `.obj` meshes into `.pts` files.
2.  Use the **Data Combination** tool to aggregate the `.pts` files into a single `.npy` training dataset.
3.  In the **Data Training** section, configure the model hyperparameters (e.g., learning rate, epochs) and select your `.npy` file.
4.  Click **Start Training** to generate a `.pth` model file.
5.  Use the **Visualize Data** or **Interpolation** tabs to validate the quality of the trained model.

### 2. Interactive Fitting (`Medical Segmentation` Tab)
1.  Click **Load NIfTI** to load a patient's MRI scan.
2.  Use the automated segmentation or manual brush tools to draw a green **ROI** around the target vertebra.
3.  Click **Load Trained Model (.pth)** and select the model file generated in the offline phase.
4.  Configure fitting parameters (iterations, samples).
5.  Click **Fit DeepSDF to Current Mask** to begin the optimisation.
6.  Once complete, the final mesh can be saved or inspected with the **Check Alignment of Last Fit** tool.
  
## Citing This Work
If you use this work, please cite the accompanying project report:
```bibtex
@misc{vanzyl2025deepsdf,
  author       = {Francois Van Zyl},
  title        = {3D Shape Representation using DeepSDF on MRI vertebrae},
  year         = {2025},
  institution  = {University of Otago},
  howpublished = {APPS597 Project Report}
}
