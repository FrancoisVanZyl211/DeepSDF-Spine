# scripts/fit_latent_mri.py
import os, sys, argparse
import numpy as np
import nibabel as nib
import torch
from   scipy import ndimage as ndi
from   skimage import measure
from scipy.ndimage import gaussian_filter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.model_multishape import MultiShapeDecoder

# ───────────────────────── helpers ──────────────────────────────────
def euler_xyz_to_mat(r: torch.Tensor) -> torch.Tensor:
    """r: (3,) Euler angles (rad)  → 3×3 rotation matrix"""
    rx, ry, rz = r
    cx, cy, cz = torch.cos(r)
    sx, sy, sz = torch.sin(r)
    row0 = torch.stack(( cy*cz,      -cy*sz,     sy), 0)
    row1 = torch.stack(( sx*sy*cz+cx*sz,  -sx*sy*sz+cx*cz, -sx*cy), 0)
    row2 = torch.stack((-cx*sy*cz+sx*sz,   cx*sy*sz+sx*cz,  cx*cy), 0)
    return torch.stack((row0, row1, row2), 0)

# ───────────────────────── main ─────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Fit DeepSDF latent + pose to a vertebra mask")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--mri",     required=True)
    ap.add_argument("--mask",    required=True)
    ap.add_argument("--iters",   type=int, default=1200)
    ap.add_argument("--samples", type=int, default=10_000)
    ap.add_argument("--out_obj", default="fit_shape.obj")
    ap.add_argument("--latent_dim", type=int)
    ap.add_argument("--device",
                    default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    # ── load model ──────────────────────────────────────────────────
    ckpt = torch.load(args.weights, map_location=device)
    L    = args.latent_dim or ckpt["latent_codes.weight"].shape[1]
    S    = ckpt["latent_codes.weight"].shape[0]

    model = MultiShapeDecoder(num_shapes=S, args=None, latent_dim=L).to(device)
    model.load_state_dict(ckpt);  model.eval()

    # ── load volume & mask ─────────────────────────────────────────
    vol   = nib.load(args.mri ).get_fdata().astype(np.float32)
    mask  = nib.load(args.mask).get_fdata().astype(bool)
    if vol.shape != mask.shape:
        raise RuntimeError("volume and mask shapes differ")

    # gradient magnitude
    gx, gy, gz = np.gradient(vol)
    g_mag = np.sqrt(gx**2 + gy**2 + gz**2);  g_mag[~mask] = 0.0

    # ── build 2-voxel shell & sample points ────────────────────────
    shell = np.logical_xor(
        ndi.binary_dilation(mask, iterations=2),
        ndi.binary_erosion (mask, iterations=2)
    )
    kji = np.column_stack(np.where(shell))
    if len(kji) < args.samples:
        raise RuntimeError(f"shell only has {len(kji)} voxels")

    sel   = kji[np.random.choice(len(kji), args.samples, False)]
    zoom  = nib.load(args.mri).header.get_zooms()[:3]
    xyz_mm = sel[:, ::-1] * zoom
    g_np   = g_mag[tuple(sel.T)]
    inside = mask[tuple(sel.T)].astype(np.float32)

    # normalise cube -------------------------------------
    bb_min, bb_max = xyz_mm.min(0), xyz_mm.max(0)
    bb_c = 0.5*(bb_min+bb_max);  bb_r = 0.5*(bb_max-bb_min).max()
    xyz = ((xyz_mm - bb_c)/bb_r).astype(np.float32)

    xyz_t = torch.from_numpy(xyz).to(device, dtype=torch.float32)  # <-- 2
    g_t   = torch.from_numpy(g_np.astype(np.float32)).to(device).unsqueeze(1)
    in_t  = torch.from_numpy(inside.astype(np.float32)).to(device).unsqueeze(1)

    z      = torch.zeros(L,  device=device, requires_grad=True)
    r      = torch.zeros(3,  device=device, requires_grad=True)
    t      = torch.zeros(3,  device=device, requires_grad=True)
    log_s  = torch.zeros(1,  device=device, requires_grad=True)
    opt    = torch.optim.Adam([z, r, t, log_s], lr=1e-3)

    print(f"optimising z({L})  r,t,s  for {args.iters} iters …")
    N = xyz_t.shape[0]

    for it in range(args.iters):
        opt.zero_grad()
        s = torch.exp(log_s)
        xyz_tf   = s * (xyz_t @ euler_xyz_to_mat(r).T) + t
        sdf_pred = model.forward_with_latent(
                       xyz_tf, z.unsqueeze(0).expand(N, -1))

        L_surface = torch.mean(torch.abs(sdf_pred) / (g_t + 1e-2))
        sign      = in_t*2.0 - 1.0
        L_sign    = torch.mean(torch.relu(sign * sdf_pred))

        lambda_lat = 1e-4
        L_lat      = lambda_lat * torch.mean(z**2)

        loss      = L_surface + 0.5*L_sign + L_lat
        loss.backward()
        opt.step()

      
        if it % 100 == 0 or it == args.iters - 1:
            deg = r.detach()*180/np.pi
            print(f"iter {it:4d} | "
                  f"loss {loss.item():.4e} | "
                  f"t {t.detach().cpu().numpy()} | "
                  f"r(deg){deg.cpu().numpy()} | "
                  f"s {s.item():.3f}")

    # ── reconstruct mesh ──────────────────────────────────────────
    print("\n[info] reconstructing mesh from fitted latent code...")
    grid_N = 128
    lin = np.linspace(-1, 1, grid_N, dtype=np.float32)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")
    coords  = np.stack([X, Y, Z], -1).reshape(-1, 3)
    coords_t = torch.from_numpy(coords).to(device)

    with torch.no_grad():
        sdf_grid = model.forward_with_latent(
            coords_t,
            z.unsqueeze(0).expand(coords_t.shape[0], -1)
        ).cpu().view(grid_N, grid_N, grid_N).numpy()

    sdf_grid = gaussian_filter(sdf_grid, sigma=1.0)
    if sdf_grid.min() > 0 or sdf_grid.max() < 0:
        print("\n[warn] SDF grid does not cross zero "
              "(min %.3f  max %.3f) – skipping mesh export."
              % (sdf_grid.min(), sdf_grid.max()))
        return

    verts, faces, *_ = measure.marching_cubes(sdf_grid, level=0.0)
    verts_ndc = (verts / (grid_N - 1.0) * 2.0) - 1.0  # Normalized to [-1, 1]
    verts_t = torch.from_numpy(verts_ndc).to(device, dtype=torch.float32)


    with torch.no_grad():
        s = torch.exp(log_s)
        R = euler_xyz_to_mat(r)
        
        verts_posed_t = (verts_t - t) @ R / s
        verts_posed = verts_posed_t.cpu().numpy()


    verts_mm  = (verts_posed * bb_r) + bb_c
    verts_ijk = (verts_mm / zoom)[:, ::-1]


    with open(args.out_obj, "w") as f:
        for vx, vy, vz in verts_ijk:
            f.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")
        for a, b, c in faces:
            f.write(f"f {a+1} {b+1} {c+1}\n")

    print("mesh saved:", args.out_obj)
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()

# command CLI:
# python scripts/fit_latent_mri.py --weights data/10000Samples/5ShapesGradientTesting.pth --mri     data/10000Samples/Vol_case_0042.nii --mask    data/10000Samples/Seg_case_0042.nii --iters   1000 --samples 8000 --out_obj data/10000Samples/case_042_fit_roi.obj --device  cuda:0