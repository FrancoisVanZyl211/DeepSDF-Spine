# deep_ct_pipeline.py

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage import measure, morphology

def view_nifti_slices(nifti_file, slice_axis=2, num_display=8):
    """
    Quickly display slices from a raw .nii volume along the specified axis.
    """
    img = nib.load(nifti_file)
    data = img.get_fdata()
    print(f"Loaded NIfTI file '{nifti_file}' shape={data.shape}, range=({data.min()}..{data.max()})")

    num_slices = data.shape[slice_axis]
    indices = np.linspace(0, num_slices-1, num_display, dtype=int)

    fig, axes = plt.subplots(1, num_display, figsize=(15, 4))
    for i, idx in enumerate(indices):
        if slice_axis == 0:
            slice_data = data[idx, :, :]
        elif slice_axis == 1:
            slice_data = data[:, idx, :]
        else:
            slice_data = data[:, :, idx]
        axes[i].imshow(np.rot90(slice_data), cmap='gray')
        axes[i].set_title(f"Slice {idx}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

def segment_vertebra_ct(nifti_file, bone_threshold=15, min_size=1000):
    """
    1) Load .nii
    2) Threshold for bone
    3) Largest connected component
    4) Return mask + raw data
    """
    img = nib.load(nifti_file)
    data = img.get_fdata()
    print(f"Segmenting '{nifti_file}' with threshold={bone_threshold}")

    binary_mask = data > bone_threshold
    binary_mask = morphology.remove_small_holes(binary_mask, area_threshold=500)

    labeled = measure.label(binary_mask, connectivity=1)
    props = measure.regionprops(labeled)
    if not props:
        print("No connected regions above threshold.")
        return None, data

    largest_region = max(props, key=lambda r: r.area)
    largest_mask = (labeled == largest_region.label)
    largest_mask = morphology.remove_small_holes(largest_mask, area_threshold=1000)

    if largest_region.area < min_size:
        print("Largest region is quite small. Adjust threshold or min_size.")
        return largest_mask, data
    return largest_mask, data

def preview_slices(volume, mask=None, slice_axis=2, num_slices=6):
    """
    Display the volume with optional mask overlay in red.
    """
    slices = volume.shape[slice_axis]
    indices = np.linspace(0, slices-1, num_slices, dtype=int)

    fig, axes = plt.subplots(1, num_slices, figsize=(15, 4))
    for i, idx in enumerate(indices):
        if slice_axis == 0:
            slice_img = volume[idx, :, :]
            slice_msk = mask[idx, :, :] if mask is not None else None
        elif slice_axis == 1:
            slice_img = volume[:, idx, :]
            slice_msk = mask[:, idx, :] if mask is not None else None
        else:
            slice_img = volume[:, :, idx]
            slice_msk = mask[:, :, idx] if mask is not None else None

        axes[i].imshow(np.rot90(slice_img), cmap='gray')
        if slice_msk is not None:
            axes[i].imshow(np.rot90(slice_msk), cmap='autumn', alpha=0.4)
        axes[i].set_title(f"Slice {idx}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

def export_mask_as_obj(nii_mask_file, out_obj_file, level=0.5):
    """
    Load a mask from .nii, apply marching cubes, and save .obj
    """
    img = nib.load(nii_mask_file)
    mask_data = img.get_fdata()
    binary_mask = mask_data > level

    verts, faces, normals, values = measure.marching_cubes(volume=binary_mask, level=0.5)
    print(f"Marching cubes => {len(verts)} verts, {len(faces)} faces.")
    
    with open(out_obj_file, 'w') as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            # .obj is 1-based
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print(f"Saved mesh to {out_obj_file}")

def main():
    """
    Example usage of the combined pipeline.
    1. View raw CT slices
    2. Segment largest vertebra
    3. Preview overlay
    4. Save mask as .nii
    5. Convert that mask to .obj
    """
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_folder = os.path.join(script_dir, "..", "data")
    ct_folder    = os.path.join(data_folder, "ct_files")
    mask_folder  = os.path.join(data_folder, "mask_output")
    mesh_folder  = os.path.join(data_folder, "mesh_output")
    os.makedirs(ct_folder, exist_ok=True)
    os.makedirs(mask_folder, exist_ok=True)
    os.makedirs(mesh_folder, exist_ok=True)

    # File paths for input CT and output files
    ct_file  = os.path.join(ct_folder, "case_0027.nii")
    mask_nii = os.path.join(mask_folder, "Mask.nii")
    mesh_obj = os.path.join(mesh_folder, "Object.obj")
    view_nifti_slices(ct_file, slice_axis=2, num_display=6)
    mask, data = segment_vertebra_ct(ct_file, bone_threshold=15)
    if mask is None:
        print("Mask was empty, aborting.")
        return

    preview_slices(data, mask=mask, slice_axis=2, num_slices=6)
    import nibabel as nib
    original_img = nib.load(ct_file)
    mask_img = nib.Nifti1Image(mask.astype(np.uint8), affine=original_img.affine)
    nib.save(mask_img, mask_nii)
    print(f"Saved mask to {mask_nii}")
    export_mask_as_obj(mask_nii, mesh_obj, level=0.5)

if __name__ == "__main__":
    main()