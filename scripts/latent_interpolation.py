import torch
import numpy as np
import sys
import os
from skimage import measure
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.model_multishape import MultiShapeDecoder

def interpolate_latents(model, shape0_id, shape1_id, steps=5):
    """
    Yields 'steps' number of latent vectors linearly interpolated
    between shape0_id's latent and shape1_id's latent.
    """
    z0 = model.latent_codes.weight[shape0_id].detach()
    z1 = model.latent_codes.weight[shape1_id].detach()
    for alpha in np.linspace(0, 1, steps):
        z = (1 - alpha) * z0 + alpha * z1
        yield alpha, z

def predict_sdf_with_latent(model, z, grid_N=128, max_xyz=1.0, device=None):
    """
    Evaluate the SDF for a single latent vector z across a 3D bounding box,
    returning a [grid_N, grid_N, grid_N] array for marching cubes.
    """
    model.eval()
    min_xyz = -max_xyz
    spacing = (max_xyz - min_xyz) / (grid_N - 1)

    sdf_values = np.zeros((grid_N, grid_N, grid_N), dtype=np.float32)

    with torch.no_grad():
        for ix in range(grid_N):
            x_val = min_xyz + ix * spacing
            for iy in range(grid_N):
                y_val = min_xyz + iy * spacing
                z_coords = [min_xyz + iz * spacing for iz in range(grid_N)]
                coords_np = np.array([[x_val, y_val, zc] for zc in z_coords], dtype=np.float32)
                coords_tensor = torch.from_numpy(coords_np).to(device)

                z_batched = z.unsqueeze(0).repeat(coords_tensor.shape[0], 1)
                sdf_pred = model.forward_with_latent(coords_tensor, z_batched)
                sdf_pred = sdf_pred.squeeze().cpu().numpy()

                for iz in range(grid_N):
                    sdf_values[ix, iy, iz] = sdf_pred[iz]

    return sdf_values

def export_sdf_to_obj(sdf_3d, out_obj, level=0.0):
    """
    Run marching cubes on the SDF volume and export a mesh to .obj.
    """
    verts, faces, normals, values = measure.marching_cubes(sdf_3d, level=level)
    with open(out_obj, 'w') as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

def main():
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MultiShapeDecoder(num_shapes=5, args=None, latent_dim=64).to(device)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    checkpoint_path = os.path.join(script_dir, "..", "data", "multi_shape_deepsdf.pth")
    
    # Debug prints to check path and working directory
    print("Current working directory:", os.getcwd())
    print("Looking for checkpoint at:", checkpoint_path)
    
    # Load the weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # Interpolate between shape IDs 0 and 1
    shape0_id = 0
    shape1_id = 1
    steps = 5

    out_dir = os.path.join(script_dir, "..", "data", "interpolation_meshes")
    os.makedirs(out_dir, exist_ok=True)

    for alpha, z in interpolate_latents(model, shape0_id, shape1_id, steps=steps):
        z = z.to(device)
        # Evaluate SDF for this interpolated latent
        sdf_3d = predict_sdf_with_latent(model, z, grid_N=128, max_xyz=1.0, device=device)
        # Export to .obj
        out_obj = os.path.join(out_dir, f"interp_{alpha:.2f}.obj")
        export_sdf_to_obj(sdf_3d, out_obj, level=0.0)
        print(f"Saved shape alpha={alpha:.2f} to {out_obj}")