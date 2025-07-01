# DeepSDF-Spine
Master's project for 3D reconstruction of vertebrae from MRI using DeepSDF, PyTorch, and a novel gradient-weighted loss function.
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
