import os
import sys
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.model_multishape import MultiShapeDecoder
from scripts.utils import showMeshReconstruction

def evaluate_sdf_grid(model, shape_id, grid_N, min_xyz, max_xyz, device, batch_size=4096):
    """
    Vectorized evaluation of the SDF on a full 3D grid.
    Args:
        model: The trained MultiShapeDecoder model.
        shape_id: The shape identifier for reconstruction.
        grid_N: Resolution of the 3D grid (grid_N x grid_N x grid_N).
        min_xyz: Minimum coordinate value (e.g., -max_xyz).
        max_xyz: Maximum coordinate value (e.g., max_xyz).
        device: Torch device (cpu or cuda).
        batch_size: Number of points to process per forward pass.
        
    Returns:
        sdf_values: A 3D numpy array of shape (grid_N, grid_N, grid_N) containing the evaluated SDF.
    """
    lin = np.linspace(min_xyz, max_xyz, grid_N)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing='ij')
    grid_points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1)
    grid_points_tensor = torch.from_numpy(grid_points.astype(np.float32)).to(device)
    shape_id_tensor = torch.tensor([shape_id], dtype=torch.long, device=device)
    shape_ids = shape_id_tensor.repeat(grid_points_tensor.shape[0])
    
    sdf_values_list = []
    model.eval()
    with torch.no_grad():
        for i in range(0, grid_points_tensor.shape[0], batch_size):
            batch_coords = grid_points_tensor[i:i+batch_size]
            batch_shape_ids = shape_ids[i:i+batch_size]
            sdf_pred = model(batch_coords, batch_shape_ids)
            sdf_values_list.append(sdf_pred.squeeze().cpu().numpy())
    
    sdf_values = np.concatenate(sdf_values_list, axis=0)
    sdf_values = sdf_values.reshape(grid_N, grid_N, grid_N)
    return sdf_values

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    model = MultiShapeDecoder(args.num_shapes, args, latent_dim=args.latent_dim).to(device)
    print(f"Loading weights from {args.weights}")
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    print(f"Will reconstruct shape_id={args.shape_id}")
    min_xyz = -args.max_xyz
    max_xyz = args.max_xyz
    grid_N = args.grid_N

    print("Evaluating SDF grid...")
    sdf_values = evaluate_sdf_grid(model, args.shape_id, grid_N, min_xyz, max_xyz, device)
    print("Running marching cubes...")
    verts, triangles = showMeshReconstruction(sdf_values)

    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_folder = os.path.join(script_dir, "..", "data")
    os.makedirs(data_folder, exist_ok=True)
    out_obj = os.path.join(data_folder, f"test_shape{args.shape_id}.obj")

    with open(out_obj, 'w') as outfile:
        for v in verts:
            outfile.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for f in triangles:
            outfile.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")
    print(f"Saved mesh to {out_obj}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="multi_shape_deepsdf.pth",
                        help="Path to the trained model weights")
    parser.add_argument("--num_shapes", type=int, default=86,
                        help="Number of distinct shapes in the dataset (needed to init model)")
    parser.add_argument("--latent_dim", type=int, default=64,
                        help="Latent code dimension used in training")
    parser.add_argument("--shape_id", type=int, default=0,
                        help="Which shape to reconstruct (0 <= shape_id < num_shapes)")
    parser.add_argument("--max_xyz", type=float, default=1.0,
                        help="Bounding box range: [-max_xyz, max_xyz]^3")
    parser.add_argument("--grid_N", type=int, default=128,
                        help="Resolution of the 3D grid for reconstruction")
    args = parser.parse_args()
    main(args)

    # example commad: python View_multishape.py --weights ../data/multi_shape_deepsdf.pth --num_shapes 86 --latent_dim 64  --shape_id 2 --max_xyz 1.0 --grid_N 128