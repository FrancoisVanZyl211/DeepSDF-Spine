# scripts/check_alignment.py
import argparse
import numpy as np
import nibabel as nib
import trimesh
import sys
from PyQt6.QtGui import QGuiApplication

# ----------------------------------------------------------------------
def tm_to_o3d(tm: trimesh.Trimesh, color=(0.8, 0.8, 0.8)):
    """Convert a trimesh.Trimesh → open3d TriangleMesh."""
    import open3d as o3d
    m = o3d.geometry.TriangleMesh()
    m.vertices  = o3d.utility.Vector3dVector(np.asarray(tm.vertices))
    m.triangles = o3d.utility.Vector3iVector(np.asarray(tm.faces))
    m.compute_vertex_normals()
    m.paint_uniform_color(color)
    return m

# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mri",   required=True, help="reference MRI/CT volume")
    ap.add_argument("--mask",  required=True, help="binary mask (.nii)")
    ap.add_argument("--mesh",  required=True, help="fitted DeepSDF mesh (.obj)")
    ap.add_argument("--show",  action="store_true", help="open viewer")
    ap.add_argument("--mesh-mm", action="store_true",
                    help="OBJ vertices are in millimetres; convert to voxels")
    args = ap.parse_args()

    # 1 ─ MRI affine ---------------------------------------------------
    nii = nib.load(args.mri)
    print("Affine (voxel→mm):\n", nii.affine)

    # 2 ─ load mask ------------------------------------
    mask = nib.load(args.mask).get_fdata() > 0

    # 3 ─ load fitted mesh --------------------------------------------
    sdf_mesh = trimesh.load(args.mesh, force='mesh')
    print(f"[SDF ] vertices: {len(sdf_mesh.vertices):,}")

    if args.mesh_mm:
        spacing = np.asarray(nii.header.get_zooms()[:3])
        sdf_mesh.vertices = (sdf_mesh.vertices / spacing)[:, ::-1]
        print("[info] converted mesh from mm to voxel indices")

    # 4 ─ numeric IoU --------------------------------------------------
    bb_mask      = np.vstack(mask.nonzero()).T
    bb_mask_min  = bb_mask.min(axis=0);  bb_mask_max = bb_mask.max(axis=0)
    bb_sdf_min   = sdf_mesh.vertices.min(axis=0)
    bb_sdf_max   = sdf_mesh.vertices.max(axis=0)

    print("\nAxis-aligned bounding boxes (voxel indices)")
    print(" mask :", np.column_stack((bb_mask_min, bb_mask_max)).T)
    print(" sdf  :", np.column_stack((bb_sdf_min,  bb_sdf_max )).T)

    inter_min = np.maximum(bb_mask_min, bb_sdf_min)
    inter_max = np.minimum(bb_mask_max, bb_sdf_max)
    inter_sz  = np.clip(inter_max - inter_min, 0, None)
    iou = inter_sz.prod() / (
        (bb_mask_max - bb_mask_min).prod()
        + (bb_sdf_max  - bb_sdf_min ).prod()
        - inter_sz.prod() + 1e-6
    )
    print(f"\nAABB Intersection over Union: {iou:.3f}")
    if not args.show:
        return

    try:
        import open3d as o3d
        app = QGuiApplication.instance() or QGuiApplication(sys.argv)
        screen = app.primaryScreen()
        if screen:
            screen_geometry = screen.availableGeometry()
            width = int(screen_geometry.width() * 0.8)
            height = int(screen_geometry.height() * 0.8)
            left = (screen_geometry.width() - width) // 2
            top = (screen_geometry.height() - height) // 2
        else:
            width, height, left, top = 1280, 720, 50, 50

        sdf_o3d = tm_to_o3d(sdf_mesh, color=[0.2, 0.9, 0.2])
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='Alignment Check', width=width, height=height, left=left, top=top)
        vis.add_geometry(sdf_o3d)
        vis.run()
        vis.destroy_window()

    except ImportError:
        print("[info] open3d not installed – falling back to trimesh viewer.")
        sdf_mesh.visual.face_colors = [50, 230, 50, 150]
        trimesh.Scene(sdf_mesh).show()


# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()

## some CLI commands
# python scripts/check_alignment.py --mri data\10000Samples\Vol_case_0042.nii --mask data\10000Samples\Seg_case_0042.nii --mesh data\10000Samples\case_042_fit_roi.obj --show