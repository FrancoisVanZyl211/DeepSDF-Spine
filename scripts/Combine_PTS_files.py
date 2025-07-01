import os
import glob
import numpy as np

def combine_pts_files(pts_folder, output_npy, max_shapes=25):
    """
    Combine at most 'max_shapes' .pts files from 'pts_folder' into a single .npy file.
    """
    pts_files = sorted(glob.glob(os.path.join(pts_folder, "*.pts")))
    if not pts_files:
        print(f"No .pts files found in {pts_folder}. Exiting")
        return
    pts_files = pts_files[:max_shapes]
    all_points = []
    all_normals = []
    all_shape_ids = []

    print("Found .pts files (limited to %d shapes):" % max_shapes)
    for i, f in enumerate(pts_files):
        print(f" [{i}] {f}")

    for shape_idx, pts_path in enumerate(pts_files):
        data = np.loadtxt(pts_path)
        xyz = data[:, :3]
        normals = data[:, 3:6]

        shape_ids = np.full((xyz.shape[0], 1), shape_idx, dtype=np.int32)

        all_points.append(xyz)
        all_normals.append(normals)
        all_shape_ids.append(shape_ids)
    
    all_points = np.vstack(all_points)    # shape [N, 3]
    all_normals = np.vstack(all_normals)  # shape [N, 3]
    all_shape_ids = np.vstack(all_shape_ids)   # shape [N, 1]

    combined_data = np.hstack((all_points, all_normals, all_shape_ids))
    np.save(output_npy, combined_data)
    
    print(f"\nCombined data shape: {combined_data.shape}")
    print(f"Saved combined data to {output_npy}")

def main():
    pts_folder = r"your_pts_folder_path"  # Replace with your actual folder path
    output_npy = r"your_pts_folder_path"
    combine_pts_files(pts_folder, output_npy, max_shapes=25)

if __name__ == "__main__":
    main()