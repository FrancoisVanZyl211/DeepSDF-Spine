import torch.utils.data as data
import numpy as np
import math
import torch
import os
import errno
import open3d as o3d
from skimage import measure

def mkdir_p(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def isdir(dirname):
    return os.path.isdir(dirname)

def normalize_pts(input_pts):
    center_point = np.mean(input_pts, axis=0)
    center_point = center_point[np.newaxis, :]
    centered_pts = input_pts - center_point

    largest_radius = np.amax(np.sqrt(np.sum(centered_pts ** 2, axis=1)))
    normalized_pts = centered_pts / largest_radius
    return normalized_pts


def normalize_normals(input_normals):
    normals_magnitude = np.sqrt(np.sum(input_normals ** 2, axis=1))
    normals_magnitude = normals_magnitude[:, np.newaxis]
    normalized_normals = input_normals / normals_magnitude
    return normalized_normals

def showMeshReconstruction(IF):
    """
    calls marching cubes on the input implicit function sampled in the 3D grid
    and shows the reconstruction mesh
    Args:
        IF    : implicit function sampled at the grid points
    Returns:
        verts, triangles: vertices and triangles of the polygon mesh after iso-surfacing it at level 0
    """
    from PyQt6.QtGui import QGuiApplication
    verts, triangles, normals, values = measure.marching_cubes(IF, 0)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(triangles.astype(np.int32))
    mesh.compute_vertex_normals()

    # ==================== Dynamically size the Open3D window ====================
    screen = QGuiApplication.primaryScreen()
    if screen is not None:
        screen_geometry = screen.availableGeometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()
        desired_w = int(screen_width * 0.7)
        desired_h = int(screen_height * 0.7)
    else:
        desired_w = 800
        desired_h = 600

    # Create a Visualizer with specified size/position
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name='Open3D Mesh',
        width=desired_w,
        height=desired_h,
        left=(screen_width - desired_w)//2 if screen else 50,
        top=(screen_height - desired_h)//2 if screen else 50
    )
    vis.add_geometry(mesh)
    vis.run()
    vis.destroy_window()
    return verts, triangles

def signed_distance(p, surface_points, surface_normals):
    """
    Computes the signed distance between a point p and a surface defined by a set of points and normals.

    Args:
        p (ndarray): 3D point.
        points (ndarray): 3D points that define the surface.
        normals (ndarray): 3D normals of the surface at the corresponding points.

    Returns:
        The signed distance between the point p and the surface.
    """
    diffs = p - surface_points
    dists = np.sqrt(np.sum(diffs**2, axis=-1))
    signs = np.sign(np.sum(diffs * surface_normals, axis=-1))
    return dists * signs 


class SdfDataset(data.Dataset):
    def __init__(self, points=None, normals=None, shape_ids=None, phase='train', args=None):
        """
        points:    [M, 3] surface points from multi-shape data
        normals:   [M, 3] surface normals for those points
        shape_ids: [M, 1] integer shape ID for each point
        """
        self.phase = phase
        self.args = args
        self.points = points
        self.normals = normals
        self.shape_ids = shape_ids.squeeze()
        self.sample_std = args.sample_std

        # Batches
        if phase == 'test':
            self.bs = args.test_batch
            max_dimensions = np.ones((3,)) * args.max_xyz
            min_dimensions = -np.ones((3,)) * args.max_xyz
            bounding_box_dimensions = max_dimensions - min_dimensions
            grid_spacing = max(bounding_box_dimensions) / (args.grid_N - 9)

            X, Y, Z = np.meshgrid(
                np.arange(min_dimensions[0] - 4*grid_spacing, max_dimensions[0] + 4*grid_spacing, grid_spacing),
                np.arange(min_dimensions[1] - 4*grid_spacing, max_dimensions[1] + 4*grid_spacing, grid_spacing),
                np.arange(min_dimensions[2] - 4*grid_spacing, max_dimensions[2] + 4*grid_spacing, grid_spacing),
            )
            self.grid_shape = X.shape
            self.samples_xyz = np.array([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)]).T
            self.number_samples = self.samples_xyz.shape[0]
            self.number_batches = math.ceil(self.number_samples / self.bs)
        else:
            # Train or Validation
            self.bs = args.train_batch
            M = self.points.shape[0]
            self.number_points = M
            self.number_samples = int(M * args.N_samples)
            self.number_batches = math.ceil(self.number_samples / self.bs)

            self.samples_xyz = np.zeros((self.number_samples, 3), dtype=np.float32)
            self.samples_sdf = np.zeros((self.number_samples,), dtype=np.float32)
            self.samples_shape_id = np.zeros((self.number_samples,), dtype=np.int32)

            if phase == 'val':
                print("val init")

            else:
                print("train init")

            idx_start = 0
            for i in range(M):
                point = self.points[i, :]     # shape [3]
                normal = self.normals[i, :]   # shape [3]
                sid = self.shape_ids[i]       # shape ID

                sample_point = (
                    np.tile(point, (args.N_samples, 1))
                    + np.random.normal(0, self.sample_std, size=(args.N_samples, 3)) 
                    * np.tile(normal, (args.N_samples, 1))
                )

                sdf_vals = np.zeros((args.N_samples,), dtype=np.float32)
                diffs = sample_point - point[None, :]
                dist = np.linalg.norm(diffs, axis=1)
                sign = np.sign(np.sum(diffs * normal[None, :], axis=1))
                sdf_vals = dist * sign

                idx_end = idx_start + args.N_samples
                self.samples_xyz[idx_start:idx_end, :] = sample_point
                self.samples_sdf[idx_start:idx_end] = sdf_vals
                self.samples_shape_id[idx_start:idx_end] = sid
                idx_start = idx_end

            perm = np.random.permutation(self.number_samples)
            self.samples_xyz = self.samples_xyz[perm]
            self.samples_sdf = self.samples_sdf[perm]
            self.samples_shape_id = self.samples_shape_id[perm]

    def __len__(self):
        return self.number_batches

    def __getitem__(self, idx):
        start_idx = idx * self.bs
        end_idx = min(start_idx + self.bs, self.number_samples)

        if self.phase == 'test':
            xyz = self.samples_xyz[start_idx:end_idx, :]
            return {'xyz': torch.FloatTensor(xyz)}
        else:
            xyz = self.samples_xyz[start_idx:end_idx, :]
            sdf = self.samples_sdf[start_idx:end_idx]
            sid = self.samples_shape_id[start_idx:end_idx]

            return {
                'xyz': torch.FloatTensor(xyz),
                'gt_sdf': torch.FloatTensor(sdf),
                'shape_id': torch.LongTensor(sid)
            }