import os, argparse, numpy as np, nibabel as nib
from skimage import measure
import trimesh

def save_pts(verts, normals, out_path):
    """verts,normals -> Nx6 ascii file"""
    arr = np.hstack((verts, normals))
    np.savetxt(out_path, arr, fmt="%.6f")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mask", required=True, help=".nii.gz binary mask")
    p.add_argument("--out_pts", required=True)
    p.add_argument("--n_samples", type=int, default=10000)
    args = p.parse_args()

    mask = (nib.load(args.mask).get_fdata() > 0)
    verts, faces, norms, _ = measure.marching_cubes(mask, level=0.5)
    mesh = trimesh.Trimesh(verts, faces, vertex_normals=norms)
    pts, face_idx = trimesh.sample.sample_surface(mesh, args.n_samples)
    nrm = mesh.face_normals[face_idx]
    save_pts(pts, nrm, args.out_pts)
    print("saved", args.out_pts)

if __name__ == "__main__":
    main()
