import numpy as np
from numba import cuda, float32  # type: ignore
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from enum import IntEnum
from dataclasses import dataclass
from typing import TypeAlias
import time
import argparse
from pathlib import Path

import pypcd4
import plotly.graph_objects as go  # type: ignore
from numpy.typing import NDArray

Position: TypeAlias = tuple[float, float, float]


@dataclass
class BeamConfig:
    use_dilation: bool = True
    divergence_deg: float = 0.15
    min_radius_m: float = 0.01  # near-field footprint (m), accounts for aperture/jitter
    extra_vox_inflate: int = 0


class ClusterVisMode(IntEnum):
    ANY_VISIBLE = 0   # cluster is visible if ANY sampled surface voxel is LoS-visible
    ALL_VISIBLE = 1   # cluster is visible only if ALL sampled surface voxels are LoS-visible


class ClusterVisibility(IntEnum):
    VISIBLE = 0
    OCCLUDED = 1
    OUT_OF_FOV = 2  # TODO: not implemented yet, use angle


def cudagrid(shape, threadsperblock=None):
    """
    Automatically configure (blockspergrid, threadsperblock) 
    for 1D, 2D, or 3D CUDA kernels.

    Parameters
    ----------
    shape : tuple
        Shape of the array (1D, 2D, or 3D).
    threadsperblock : tuple or None
        Threads per block. If None, a heuristic is used.

    Returns
    -------
    (blockspergrid, threadsperblock)
    """
    ndim = len(shape)

    if threadsperblock is None:
        if ndim == 1:
            threadsperblock = (256,)
        elif ndim == 2:
            threadsperblock = (16, 16)
        elif ndim == 3:
            threadsperblock = (8, 8, 8)
        else:
            raise ValueError("Only 1D, 2D, or 3D supported")

    blockspergrid = tuple(math.ceil(s / t) for s, t in zip(shape, threadsperblock))
    return blockspergrid, threadsperblock


@cuda.jit
def raycast_kernel(occ: NDArray[np.uint8], visibility_grid: NDArray[np.int32], sensor: Position, dirs: NDArray[np.float32], max_range: float, cell_size: float) -> None:
    """Marks all free voxels traversed by a ray as visible, stopping at the first occupied voxel.

    This uses a DDA voxel traversal algorithm from Amanatides & Woo (1987).
    The `visibility_grid` is modified in place.
    """
    W, D, H = occ.shape
    idx = cuda.grid(1)
    if idx >= dirs.shape[0]:
        return

    sx, sy, sz = sensor[0], sensor[1], sensor[2]
    dx, dy, dz = dirs[idx, 0], dirs[idx, 1], dirs[idx, 2]

    vx = int(sx)
    vy = int(sy)
    vz = int(sz)

    if vz < 0 or vz >= D or vy < 0 or vy >= H or vx < 0 or vx >= W:
        return

    step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
    step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)
    step_z = 1 if dz > 0 else (-1 if dz < 0 else 0)

    if dx == 0.0:
        tMaxX, tDeltaX = math.inf, math.inf
    else:
        if dx > 0:
            tMaxX = ((math.floor(sx) + 1.0 - sx) * cell_size) / dx
        else:  # dx < 0
            tMaxX = ((sx - math.floor(sx)) * cell_size) / -dx
        tDeltaX = cell_size / abs(dx)

    if dy == 0.0:
        tMaxY, tDeltaY = math.inf, math.inf
    else:
        if dy > 0:
            tMaxY = ((math.floor(sy) + 1.0 - sy) * cell_size) / dy
        else: # dy < 0
            tMaxY = ((sy - math.floor(sy)) * cell_size) / -dy
        tDeltaY = cell_size / abs(dy)

    if dz == 0.0:
        tMaxZ, tDeltaZ = math.inf, math.inf
    else:
        if dz > 0:
            tMaxZ = ((math.floor(sz) + 1.0 - sz) * cell_size) / dz
        else:  # dz < 0
            tMaxZ = ((sz - math.floor(sz)) * cell_size) / -dz
        tDeltaZ = cell_size / abs(dz)

    t = 0.0
    max_t = max_range

    if occ[vx, vy, vz] != 0:
        return
    visibility_grid[vx, vy, vz] = 1

    while t <= max_t:
        if tMaxX <= tMaxY and tMaxX <= tMaxZ:
            vx += step_x
            t = tMaxX
            tMaxX += tDeltaX
        elif tMaxY <= tMaxX and tMaxY <= tMaxZ:
            vy += step_y
            t = tMaxY
            tMaxY += tDeltaY
        else:
            vz += step_z
            t = tMaxZ
            tMaxZ += tDeltaZ

        if vz < 0 or vz >= D or vy < 0 or vy >= H or vx < 0 or vx >= W:
            break

        if occ[vx, vy, vz] != 0:
            break

        visibility_grid[vx, vy, vz] = 1


@cuda.jit(device=True)
def ceil_impl_device(val: float) -> int:
    """Device-side implementation of math.ceil for a float."""
    return int(val) if val == int(val) else int(val + 1.0)


@cuda.jit(device=True)
def tan_impl_device(rad: float) -> float:
    """Device-side implementation of math.tan for a float."""
    return math.sin(rad) / math.cos(rad)


@cuda.jit(device=True)
def is_path_clear_device(occ: NDArray[np.uint8], x0: int, y0: int, z0: int, x1: int, y1: int, z1: int) -> bool:
    """Checks for obstacles on the line segment BETWEEN two voxels using DDA."""
    dx = x1 - x0
    dy = y1 - y0
    dz = z1 - z0

    steps = max(abs(dx), abs(dy), abs(dz))

    if steps <= 1:
        return True

    x_inc = float32(dx) / float32(steps)
    y_inc = float32(dy) / float32(steps)
    z_inc = float32(dz) / float32(steps)

    for i in range(1, steps):
        x_f = float32(x0) + i * x_inc
        y_f = float32(y0) + i * y_inc
        z_f = float32(z0) + i * z_inc
        
        vx, vy, vz = int(round(x_f)), int(round(y_f)), int(round(z_f))

        if vx == x0 and vy == y0 and vz == z0:
            continue

        if vx == x1 and vy == y1 and vz == z1:
            break
            
        if occ[vx, vy, vz] != 0:
            return False

    return True


@cuda.jit
def dilate_visibility_kernel(
        in_vis: NDArray[np.int32],
        out_vis: NDArray[np.int32],
        occ: NDArray[np.uint8],
        sensor_pos: Position,
        cell_size: float,
        divergence_deg: float,
        min_radius_m: float,
        extra_vox_inflate: int
) -> None:
    W, D, H = in_vis.shape
    vx, vy, vz = cuda.grid(3)
    if vx >= W or vy >= D or vy >= H:
        return
    if in_vis[vx, vy, vz] == 0:
        return

    sx, sy, sz = sensor_pos[0], sensor_pos[1], sensor_pos[2]
    dist_vox_sq = float32((vx - sx)**2 + (vy - sy)**2 + (vz - sz)**2)
    dist_m = math.sqrt(dist_vox_sq) * cell_size
    divergence_rad = divergence_deg * (math.pi / 180.0)
    r_m = max(min_radius_m, dist_m * tan_impl_device(divergence_rad * 0.5))
    r = ceil_impl_device(r_m / cell_size) + extra_vox_inflate

    for dz in range(-r, r+1):
        z = vz + dz
        if z < 0 or z >= H: continue
        for dy in range(-r, r+1):
            y = vy + dy
            if y < 0 or y >= D: continue
            for dx in range(-r, r+1):
                x = vx + dx
                if x < 0 or x >= W: continue
                
                if dx*dx + dy*dy + dz*dz <= r*r:
                    # Check that the destination is free AND the path to it is clear.
                    if occ[x, y, z] == 0 and is_path_clear_device(occ, vx, vy, vz, x, y, z):
                        out_vis[x, y, z] = 1

def get_visibility_grid(
        occ: NDArray[np.uint8],
        sensor_voxel_xyz: NDArray[np.float32],
        dirs: NDArray[np.float32],
        max_range: float = 100.0,
        cell_size: float = 1.0,
        beam_config: BeamConfig = BeamConfig(),
) -> NDArray[np.int32]:
    visibility_grid = np.zeros_like(occ, dtype=np.int32)

    d_occ = cuda.to_device(occ)
    d_visibility_grid = cuda.to_device(visibility_grid)
    d_sensor = cuda.to_device(np.array(sensor_voxel_xyz, dtype=np.float32))
    d_dirs = cuda.to_device(dirs.astype(np.float32))

    raycast_kernel[*cudagrid((dirs.shape[0],))](d_occ, d_visibility_grid, d_sensor, d_dirs, float32(max_range), float32(cell_size))

    if not beam_config.use_dilation:
        return d_visibility_grid.copy_to_host()

    d_dilated_visibility_grid = cuda.device_array_like(d_visibility_grid)
    dilate_visibility_kernel[*cudagrid(occ.shape)](
        d_visibility_grid, d_dilated_visibility_grid, d_occ, d_sensor,
        float32(cell_size), float32(beam_config.divergence_deg),
        float32(beam_config.min_radius_m), beam_config.extra_vox_inflate
    )
    return d_dilated_visibility_grid.copy_to_host()


def get_visibility_grid_batch(occ_sensor_list, dirs, max_range=100.0, 
                                       cell_size=1.0, beam_config=BeamConfig()):
    """Process batch using CUDA streams for overlapping computation"""
    # set min based on available gpu memory
    streams = [cuda.stream() for _ in range(min(8, len(occ_sensor_list)))]
    results = []

    for i, (occ, sensor) in enumerate(occ_sensor_list):
        stream = streams[i % len(streams)]
        with stream.auto_synchronize():
            result = get_visibility_grid(occ, sensor, dirs, max_range, 
                                       cell_size, beam_config)
            results.append(result)
    
    cuda.synchronize()
    return results


def pos_to_cell_idx(d: float, h: float, w: float, cell_size: float) -> tuple[int, int, int]:
    """Converts world coordinates to integer voxel indices."""
    return (int(d / cell_size), int(h / cell_size), int(w / cell_size))


def make_scene_from_pcd(
        pcd_path: Path,
        grid_shape_vox: tuple[int, int, int],
        cell_size: float,
        grid_origin_m: NDArray[np.float32],
) -> NDArray[np.uint8]:
    pc = pypcd4.PointCloud.from_path(pcd_path)
    occ = np.zeros(grid_shape_vox, dtype=np.uint8)
    points_xyz = pc.numpy()[:, :3]
    points_xyz[:, 2] += 2  # input pcd was shifted down 2 meters. TODO: later no need
    print(f"Loaded {points_xyz.shape[0]} points.")

    points_relative_to_origin_m = points_xyz - grid_origin_m
    indices_xyz = np.floor(points_relative_to_origin_m / cell_size).astype(int)

    W, D, H = grid_shape_vox
    valid_mask = (
        (indices_xyz[:, 0] >= 0) & (indices_xyz[:, 0] < W) &
        (indices_xyz[:, 1] >= 0) & (indices_xyz[:, 1] < D) &
        (indices_xyz[:, 2] >= 0) & (indices_xyz[:, 2] < H)
    )
    valid_indices_xyz = indices_xyz[valid_mask]

    num_invalid_indices = len(indices_xyz) - len(valid_indices_xyz)
    if num_invalid_indices > 0:
        print('Number of points outside grid bounds:', num_invalid_indices)

    if valid_indices_xyz.shape[0] == 0:
        print("WARNING: No points from the PCD file fall within the specified grid volume.")
        return occ
    
    occ[tuple(valid_indices_xyz.T)] = 1
    return occ


def sample_fov_dirs(
    h_samples: int = 1000, v_samples: int = 1000, fov_h_deg: float = 360.0, fov_v_deg: float = 104.0
) -> NDArray[np.float32]:
    """Samples ray direction vectors on a uniform grid within a specified FOV.

    The FOV is centered around the positive X-axis.

    Args:
        h_samples: Number of horizontal samples (azimuth).
        v_samples: Number of vertical samples (elevation).
        fov_h_deg: Horizontal field of view in degrees.
        fov_v_deg: Vertical field of view in degrees.

    Returns:
        A (N, 3) array of normalized direction vectors (dx, dy, dz).
    """
    phi_range = np.deg2rad(np.linspace(-fov_h_deg / 2.0, fov_h_deg / 2.0, h_samples))    # azimuth
    theta_range = np.deg2rad(np.linspace(-fov_v_deg / 2.0, fov_v_deg / 2.0, v_samples))  # elevation
    
    phi, theta = np.meshgrid(phi_range, theta_range)
    
    dx = np.cos(theta) * np.cos(phi)
    dy = np.cos(theta) * np.sin(phi)
    dz = np.sin(theta)
    
    dirs = np.stack([dx.ravel(), dy.ravel(), dz.ravel()], axis=1).astype(np.float32)
    
    norms = np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8
    dirs /= norms
    return dirs


def plot_3d_scene_web(
    occ: NDArray[np.uint8],
    sensor_zyx: Position,
    dirs: NDArray[np.float32],
    max_range: float,
    num_rays_to_plot: int = 50,
    out_path: str = "scene_3d_view.html",
) -> None:
    """Creates and saves an interactive 3D web-based view of the scene using Plotly."""
    # The simulation uses (z, y, x) numpy indexing.
    # Plotly uses a standard Cartesian (x, y, z) coordinate system.
    # We must consistently swap the axes when creating plotable objects.

    plot_data = []

    # 1. Trace for the occupied voxels (PCD)
    occ_indices_zyx = np.argwhere(occ == 1)
    if occ_indices_zyx.size > 0:
        # Swap axes from (z, y, x) to (x, y, z) for plotting
        occ_indices_xyz = occ_indices_zyx[:, ::-1]
        pcd_trace = go.Scatter3d(
            x=occ_indices_xyz[:, 0], y=occ_indices_xyz[:, 1], z=occ_indices_xyz[:, 2],
            mode='markers',
            marker=dict(size=2, color='black', opacity=0.7),
            name='Occupied Voxels (PCD)'
        )
        plot_data.append(pcd_trace)

    # 2. Trace for the sensor position
    # Swap axes from (z, y, x) to (x, y, z) for plotting
    sensor_xyz = np.array([sensor_zyx[2], sensor_zyx[1], sensor_zyx[0]])
    sensor_trace = go.Scatter3d(
        x=[sensor_xyz[0]], y=[sensor_xyz[1]], z=[sensor_xyz[2]],
        mode='markers',
        marker=dict(size=8, color='yellow', symbol='diamond', line=dict(color='black', width=2)),
        name='Sensor Position'
    )
    plot_data.append(sensor_trace)

    # 3. Trace for the FOV rays
    # Subsample the rays to avoid creating a huge HTML file and cluttering the view
    rng = np.random.default_rng(0)
    num_rays_to_plot = min(num_rays_to_plot, dirs.shape[0])
    ray_indices = rng.choice(dirs.shape[0], size=num_rays_to_plot, replace=False)
    sampled_dirs_xyz = dirs[ray_indices] # Dirs are already in (x, y, z)

    # To draw many separate lines efficiently in Plotly, we build a single trace
    # and separate the line segments with `None` values.
    ray_x, ray_y, ray_z = [], [], []
    for i in range(num_rays_to_plot):
        start_point = sensor_xyz
        end_point = sensor_xyz + sampled_dirs_xyz[i] * max_range
        
        ray_x.extend([start_point[0], end_point[0], None])
        ray_y.extend([start_point[1], end_point[1], None])
        ray_z.extend([start_point[2], end_point[2], None])
    
    ray_trace = go.Scatter3d(
        x=ray_x, y=ray_y, z=ray_z,
        mode='lines',
        line=dict(color='cyan', width=1),
        name='FOV Rays'
    )
    plot_data.append(ray_trace)

    fig = go.Figure(data=plot_data)
    fig.update_layout(
        title='3D Scene Visualization',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectratio=dict(x=1, y=1, z=1),
            aspectmode='data', # Ensures proportions are correct
            camera_eye=dict(x=1.5, y=1.5, z=1.5)
        ),
        legend_orientation="h"
    )

    print(f"\n[ 3D VISUALIZATION ]")
    print(f"- Saving interactive 3D scene to: {out_path}")
    fig.write_html(out_path)
    print("-  -> Open this file in a web browser to view the scene.")

def plot_slice(
        occ: NDArray[np.uint8],
        vis: NDArray[np.int32],
        cluster: NDArray[np.uint8],
        z_idx: int | float,
        sensor_zyx: NDArray[np.float32],
        out_path: str = "example_result.png",
) -> None:
    """Plots a 2D slice of the combined occupancy, visibility, and cluster grids."""
    occ_slice = occ[:, :, int(z_idx)]
    vis_slice = vis[:, :, int(z_idx)]
    cluster_slice = cluster[:, :, int(z_idx)]

    # 0: Empty/Unseen (white)
    # 1: Occupied (black)
    # 3: Visible Free Space (light green)
    # 5: Cluster (Occluded) (light red)
    # 6: Cluster (Visible) (cyan)
    merged_grid = np.zeros_like(occ_slice, dtype=np.uint8)
    
    # Build up layers, with later layers overwriting earlier ones
    merged_grid[vis_slice != 0] = 3
    # Overlay cluster information
    is_cluster = cluster_slice == 1
    is_visible = vis_slice != 0
    merged_grid[is_cluster & ~is_visible] = 5
    merged_grid[is_cluster & is_visible] = 6
    # Occupied voxels have the highest priority
    merged_grid[occ_slice == 1] = 1

    cmap = mcolors.ListedColormap([
        'white',      # 0: Empty
        'black',      # 1: Occupied
        '#E0E0E0',    # 2: (unused) In FOV Free
        '#A0E0A0',    # 3: Visible Free
        '#696969',    # 4: (unused) Cluster (Out of FOV)
        '#F08080',    # 5: Cluster (Occluded)
        '#00FFFF'     # 6: Cluster (Visible)
    ])
    bounds = np.arange(-0.5, 7.5, 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(20, 20))
    img = ax.imshow(merged_grid, cmap=cmap, norm=norm, origin="lower")

    ax.plot(sensor_zyx[0], sensor_zyx[1], 'y*', markersize=20, markeredgecolor='black', label='Sensor Origin')
    ax.legend()
    
    ax.set_title(f"Occupancy, Visibility & Cluster Status (z={int(z_idx)})")
    
    cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04, ticks=np.arange(7))
    cbar.ax.set_yticklabels(['Empty', 'Occupied', 'In FOV Free', 'Visible Free', 'Cluster (Out of FOV)', 'Cluster (Occluded)', 'Cluster (Visible)'])
    cbar.set_label("Voxel State")

    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()


def get_cluster_visibility_status(
        visibility_grid: NDArray[np.int32],
        cluster_mask: NDArray[np.uint8],
        mode: ClusterVisMode = ClusterVisMode.ANY_VISIBLE,
        max_samples: int = 512,
        rng_seed: int = 0,
) -> ClusterVisibility:
    """Determines a cluster's visibility status based on sampled voxels.

    Args:
        visibility_grid: The grid indicating line-of-sight visible voxels.
        cluster_mask: A grid where non-zero values indicate the cluster's voxels.
        mode: The visibility mode (ANY_VISIBLE or ALL_VISIBLE).
        max_samples: The maximum number of voxels to sample from the cluster for the check.
        rng_seed: The random seed for sampling.

    Returns:
        The visibility status of the cluster.
    """
    x_coords, y_coords, z_coords = np.nonzero(cluster_mask > 0)
    num_voxels = z_coords.shape[0]
    if num_voxels == 0:
        return ClusterVisibility.OUT_OF_FOV # No points to check

    if num_voxels > max_samples:
        rng = np.random.default_rng(rng_seed)
        sample_indices = rng.choice(num_voxels, size=max_samples, replace=False)
        xv, yv, zv = x_coords[sample_indices], y_coords[sample_indices], z_coords[sample_indices]
    else:
        xv, yv, zv = x_coords, y_coords, z_coords

    W, D, H = visibility_grid.shape
    valid_indices = (xv < W) & (yv < D) & (zv < H)
    if not np.any(valid_indices):
        return ClusterVisibility.OUT_OF_FOV # All points outside grid bounds
    
    xv, yv, zv = xv[valid_indices], yv[valid_indices], zv[valid_indices]
    
    visibility_flags = visibility_grid[xv, yv, zv].astype(bool)

    if mode == ClusterVisMode.ANY_VISIBLE:
        return ClusterVisibility.VISIBLE if np.any(visibility_flags) else ClusterVisibility.OCCLUDED
    elif mode == ClusterVisMode.ALL_VISIBLE:
        return ClusterVisibility.VISIBLE if np.all(visibility_flags) else ClusterVisibility.OCCLUDED
    
    return ClusterVisibility.OCCLUDED


def get_cluster_visibility_status_batch(
        items: list[tuple[NDArray[np.int32], NDArray[np.uint8], ClusterVisMode]],
        max_samples: int = 512,
        rng_seed: int = 0,
) -> list[ClusterVisibility]:
    """Determines cluster visibility status for a batch of items.

    Args:
        items: A list of tuples, each containing (visibility_grid, cluster_mask, mode) 
            for a single cluster check.
        max_samples: The maximum number of voxels to sample from each cluster.
        rng_seed: The random seed for sampling.

    Returns:
        A list of ClusterVisibility statuses corresponding to the input items.
    """
    return [get_cluster_visibility_status(
                visibility_grid,
                cluster_mask,
                mode=mode,
                max_samples=max_samples,
                rng_seed=rng_seed,
            ) for visibility_grid, cluster_mask, mode in items]


if __name__ == "__main__":
    if not cuda.is_available():
        raise RuntimeError("Numba CUDA is not available. Example requires a CUDA-capable GPU and CUDA toolkit.")

    parser = argparse.ArgumentParser(description="Voxel-Based Visibility Analysis")
    parser.add_argument(
        "--pcd_file", type=Path, default=None,
        help="Path to a .pcd file to use for the occupancy grid. If not provided, a default scene is generated."
    )
    parser.add_argument(
        "--grid_dims_m", type=str, default="40,40,40",
        help="Grid dimensions in meters as 'Width,Depth,Height' (X,Y,Z)."
    )
    parser.add_argument(
        "--sensor_pos_m", type=str, default="20,20,20",
        help="Sensor position in world coordinates (X,Y,Z) in meters."
    )
    parser.add_argument(
        "--grid_origin_m", type=str, default="-20,-20,-20",
        help="World coordinate (x,y,z) in meters that maps to grid voxel (0,0,0). Used only with --pcd_file."
    )
    parser.add_argument(
        "--cell_size", type=float, default=0.2,
        help="Size of a single voxel in meters."
    )
    parser.add_argument(
        "--plot_3d_web", action="store_true",
        help="If set, generates an interactive 3D plot of the scene as an HTML file (scene_3d_view.html)."
    )
    args = parser.parse_args()

    start_setup_time = time.time()

    grid_dims_m = tuple(map(float, args.grid_dims_m.split(',')))
    grid_shape_vox = pos_to_cell_idx(grid_dims_m[0], grid_dims_m[1], grid_dims_m[2], args.cell_size)

    sensor = np.array(list(map(float, args.sensor_pos_m.split(',')))) / args.cell_size

    grid_origin_m = np.array(list(map(float, args.grid_origin_m.split(','))))
    # we expect that points are in global or vehicle coord system
    # probably world and we use grid_origin for centering
    occ = make_scene_from_pcd(
        pcd_path=args.pcd_file,
        grid_shape_vox=grid_shape_vox,
        cell_size=args.cell_size,
        grid_origin_m=grid_origin_m,
    )

    start_ray_time = time.time()
    h_samples, v_samples = 900, 128
    fov_h_deg, fov_v_deg = 360.0, 105.0
    dirs = sample_fov_dirs(h_samples=h_samples, v_samples=v_samples, fov_h_deg=fov_h_deg, fov_v_deg=fov_v_deg)
    ray_sampling_time = time.time() - start_ray_time
    print(f"Num Rays:            {dirs.shape[0]} ({h_samples}x{v_samples})")
    print(f"FOV (H x V):         {fov_h_deg}° x {fov_v_deg}°")

    h_angular_res_deg = fov_h_deg / h_samples
    v_angular_res_deg = fov_v_deg / v_samples
    required_divergence_deg = max(h_angular_res_deg, v_angular_res_deg)
    print(f"Auto-calculated beam divergence: {required_divergence_deg:.4f} degrees")

    beam_config = BeamConfig(
        use_dilation=True,
        divergence_deg=required_divergence_deg,
        min_radius_m=0,
        extra_vox_inflate=0,
    )
    
    cluster = np.zeros_like(occ, dtype=np.uint8)
    # if not args.pcd_file:
    #     # This cluster definition is tuned for the default procedural scene
    #     z0, z1 = int(10 / args.cell_size), int(36 / args.cell_size)
    #     y0, y1 = int(10 / args.cell_size), int(30 / args.cell_size)
    #     x0, x1 = int(10 / args.cell_size), int(56 / args.cell_size)
    #     cluster[z0:z1, y0:y1, x0:x1] = 1

    warmup_result = get_visibility_grid(occ, sensor, dirs[:1], beam_config=beam_config)

    print("\n--- DEMO 1: Single Run (Visibility + FOV Grids) ---")
    start_time = time.time()
    vis_grid = get_visibility_grid_batch(
        [(occ, sensor)], dirs,
        max_range=40.0, cell_size=args.cell_size,
        beam_config=beam_config
    )[0]
    full_time = time.time() - start_time
    print(f"- Grid Computation Time:   {full_time:.4f}s")

    start_time = time.time()
    any_vis = get_cluster_visibility_status(vis_grid, cluster, mode=ClusterVisMode.ANY_VISIBLE)
    all_vis = get_cluster_visibility_status(vis_grid, cluster, mode=ClusterVisMode.ALL_VISIBLE)
    single_cluster_time = time.time() - start_time
    print(f"- Cluster Check Time:      {single_cluster_time:.4f}s")
    print(f"  - Result (ANY_VISIBLE):  {any_vis.name}")
    print(f"  - Result (ALL_VISIBLE):  {all_vis.name}")

    start_plot_time = time.time()
    print("- Plotting slices (view slice*.png for results)...")
    for z_idx in range(100, 101):
    # for z_idx in range(96, 108):
        if 0 <= z_idx < occ.shape[0]:
            plot_slice(occ, vis_grid, cluster, z_idx=z_idx, sensor_zyx=sensor, out_path=f"slice_{z_idx}.png")
    plot_time = time.time() - start_plot_time
    print(f"- Plotting Time:           {plot_time:.4f}s")

 
    # --- DEMO 4: Interactive Web-Based 3D Visualization ---
    if args.plot_3d_web:
        plot_3d_scene_web(
            occ=occ,
            sensor_zyx=sensor,
            dirs=dirs,
            max_range=40.0, # Use the same max_range as Demo 1
        )
