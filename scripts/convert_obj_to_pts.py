import os
import numpy as np
import trimesh

def process_obj_file(obj_path, pts_path, n_samples=9500):
    print(f"Processing {obj_path} with n_samples={n_samples}...")

    # Load mesh
    mesh = trimesh.load(obj_path)
    sampled_points, face_indices = trimesh.sample.sample_surface(mesh, n_samples)
    if mesh.vertex_normals is None or len(mesh.vertex_normals) == 0:
        mesh.compute_vertex_normals()

    face_normals = mesh.face_normals[face_indices]
    data = np.hstack((sampled_points, face_normals))
    np.savetxt(pts_path, data, fmt="%.6f")

    print(f"Saved: {pts_path}")

def convert_obj_files(obj_dir, out_dir, n_samples=9500, progress_callback=None):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    obj_files = [f for f in os.listdir(obj_dir) if f.endswith(".obj")]
    total = len(obj_files)
    if total == 0:
        print(f"No .obj files found in {obj_dir}")
        return

    for i, obj_file in enumerate(obj_files):
        if progress_callback:
            progress_callback(i, total, obj_file)

        obj_path = os.path.join(obj_dir, obj_file)
        pts_file = obj_file.replace(".obj", ".pts")
        pts_path = os.path.join(out_dir, pts_file)

        process_obj_file(obj_path, pts_path, n_samples=n_samples)

    if progress_callback:
        progress_callback(total, total, None)

    print("Conversion Completed :)")

if __name__ == "__main__":
    OBJ_Input_DIR = r"C:\Users\Me\MyOBJFolder"
    PTS_OUT_DIR   = r"C:\Users\Me\MyPTSFolder"
    N_samples     = 9500

    convert_obj_files(OBJ_Input_DIR, PTS_OUT_DIR, n_samples=N_samples)