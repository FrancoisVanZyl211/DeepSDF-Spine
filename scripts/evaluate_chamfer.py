import os
import csv
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from scipy.spatial import cKDTree
from model_multishape import MultiShapeDecoder
from skimage import measure
import trimesh

##############################################################################
# 1) HELPER FUNCTIONS
##############################################################################

def load_mapping_file(csv_path):
    mapping = {}
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        shape_id_str, obj_name = line.split(',')
        shape_id = int(shape_id_str)
        mapping[shape_id] = obj_name
    return mapping

def sample_points_from_implicit(model, shape_id, grid_N=128, max_xyz=1.0, device='cpu'):
    """
    1) Evaluate the SDF on a 3D grid for the given shape_id,
    2) Run marching cubes to get the surface mesh,
    3) Return its vertices as a point cloud.
    """
    model.eval()
    min_xyz = -max_xyz
    spacing = (max_xyz - min_xyz) / (grid_N - 1)

    sdf_values = np.zeros((grid_N, grid_N, grid_N), dtype=np.float32)
    shape_id_torch = torch.tensor([shape_id], dtype=torch.long, device=device)

    with torch.no_grad():
        for ix in range(grid_N):
            x_val = min_xyz + ix * spacing
            for iy in range(grid_N):
                y_val = min_xyz + iy * spacing
                z_coords = [min_xyz + iz * spacing for iz in range(grid_N)]
                coords_np = np.array([[x_val, y_val, zc] for zc in z_coords], dtype=np.float32)
                coords_tensor = torch.from_numpy(coords_np).to(device)

                shape_id_batch = shape_id_torch.repeat(coords_tensor.shape[0])
                sdf_pred = model(coords_tensor, shape_id_batch)
                sdf_pred = sdf_pred.squeeze().cpu().numpy()
                sdf_values[ix, iy, :] = sdf_pred

    verts, faces, normals, values = measure.marching_cubes(sdf_values, level=0.0)
    pc_verts = []
    for (vx, vy, vz) in verts:
        wx = min_xyz + vx * spacing
        wy = min_xyz + vy * spacing
        wz = min_xyz + vz * spacing
        pc_verts.append((wx, wy, wz))

    pc_verts = np.array(pc_verts, dtype=np.float32)
    return pc_verts

def sample_points_from_obj(obj_path, n_samples=20000):
    """
    Loads a mesh from 'obj_path' using trimesh and samples 'n_samples' points.
    Returns a [n_samples, 3] array of points.
    """
    mesh = trimesh.load(obj_path)
    if mesh.is_empty:
        print(f"[WARN] '{obj_path}' is an empty mesh? Returning 0 points.")
        return np.zeros((0, 3), dtype=np.float32)
    pts, _ = trimesh.sample.sample_surface(mesh, n_samples)
    return pts.astype(np.float32)

def chamfer_distance(pc1, pc2):
    """
    Chamfer Distance between pc1 and pc2, each shape [N,3].
    We do squared L2 distance:
       CD = mean_{x in pc1} [ min_{y in pc2} ||x-y||^2 ]
          + mean_{y in pc2} [ min_{x in pc1} ||x-y||^2 ].
    """
    tree2 = cKDTree(pc2)
    dist1, _ = tree2.query(pc1, k=1)
    tree1 = cKDTree(pc1)
    dist2, _ = tree1.query(pc2, k=1)
    # Use squared distances
    cd_val = (dist1**2).mean() + (dist2**2).mean()
    return cd_val

##############################################################################
# 2) MAIN EVALUATION FUNCTION
##############################################################################
def main():
    parser = argparse.ArgumentParser(description="Evaluate Chamfer Distance for a multi-shape model.")
    parser.add_argument("--weights", type=str, default="../data/multi_shape_deepsdf.pth",
                        help="Path to the trained model weights")
    parser.add_argument("--num_shapes", type=int, default=5,
                        help="Number of distinct shapes in the dataset")
    parser.add_argument("--latent_dim", type=int, default=64,
                        help="Latent embedding dimension used in training")
    parser.add_argument("--mapping_csv", type=str, default="shape_mapping.csv",
                        help="CSV that maps shape_id -> ground-truth .obj filename")
    parser.add_argument("--gt_obj_folder", type=str, default="C:/Users/You/DataSet",
                        help="Folder containing the ground-truth .obj files")
    parser.add_argument("--gt_samples", type=int, default=20000,
                        help="Number of points to sample from each ground-truth .obj")
    parser.add_argument("--grid_N", type=int, default=128,
                        help="Grid resolution for reconstruction")
    parser.add_argument("--max_xyz", type=float, default=1.0,
                        help="Bounding box coordinate limit for reconstruction [-max_xyz..max_xyz].")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    print(f"Loading model weights from: {args.weights}")
    model = MultiShapeDecoder(num_shapes=args.num_shapes, args=None, latent_dim=args.latent_dim)
    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    mapping = load_mapping_file(args.mapping_csv)

    all_cds = []
    used_count = 0
    for shape_id in range(args.num_shapes):
        
        if shape_id not in mapping:
            print(f"[WARN] shape_id={shape_id} not found in {args.mapping_csv}, skipping.")
            continue
        gt_obj_file = mapping[shape_id]
        gt_obj_path = os.path.join(args.gt_obj_folder, gt_obj_file)
        if not os.path.isfile(gt_obj_path):
            print(f"[WARN] Ground truth .obj not found: {gt_obj_path}")
            continue

        pred_pc = sample_points_from_implicit(model,
                                              shape_id=shape_id,
                                              grid_N=args.grid_N,
                                              max_xyz=args.max_xyz,
                                              device=device)
        
        gt_pc = sample_points_from_obj(gt_obj_path, n_samples=args.gt_samples)

        
        cd_val = chamfer_distance(pred_pc, gt_pc)
        all_cds.append(cd_val)
        used_count += 1
        print(f"shape_id={shape_id}, {gt_obj_file}: CD={cd_val:.6f}")

    
    if len(all_cds) == 0:
        print("No shapes evaluated. Check CSV or shape_id range.")
    else:
        avg_cd = np.mean(all_cds)
        print(f"\nAverage Chamfer Distance over {used_count} shapes = {avg_cd:.6f}")

##############################################################################
# 3) ENTRY POINT
##############################################################################
if __name__ == "__main__":
    main()

# python evaluate_chamfer.py --weights ../data/multi_shape_deepsdf.pth --num_shapes 5 --obj_name_pattern "L1_{}.obj" --gt_obj_folder "C:/Users/GGPC/Downloads/SpineData/LumbarVertebraeDataSet/dataphotoscanfinalpublish" --grid_N 128 --max_xyz 1.0 --gt_samples 20000