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
import plotly.graph_objects as go
from numpy.typing import NDArray

Position: TypeAlias = tuple[float, float, float]

CELL_SIZE = 0.2


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
    OUT_OF_FOV = 2


@cuda.jit
def fov_kernel(fov_grid: NDArray[np.int32], sensor: Position, dirs: NDArray[np.float32], max_range: float, cell_size: float) -> None:
    """Marks all voxels traversed by a ray as being within the FOV.

    This uses a DDA voxel traversal algorithm (Amanatides & Woo, 1987) and ignores occupancy.
    The `fov_grid` is modified in place.
    """
    D, H, W = fov_grid.shape
    idx = cuda.grid(1)
    if idx >= dirs.shape[0]:
        return

    sz, sy, sx = sensor[0], sensor[1], sensor[2]
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
        tMaxX, tDeltaX = 1e30, 1e30
    else:
        if dx > 0:
            tMaxX = ((math.floor(sx) + 1.0 - sx) * cell_size) / dx
        else:
            tMaxX = ((sx - math.floor(sx)) * cell_size) / -dx
        tDeltaX = cell_size / abs(dx)

    if dy == 0.0:
        tMaxY, tDeltaY = 1e30, 1e30
    else:
        if dy > 0:
            tMaxY = ((math.floor(sy) + 1.0 - sy) * cell_size) / dy
        else:
            tMaxY = ((sy - math.floor(sy)) * cell_size) / -dy
        tDeltaY = cell_size / abs(dy)

    if dz == 0.0:
        tMaxZ, tDeltaZ = 1e30, 1e30
    else:
        if dz > 0:
            tMaxZ = ((math.floor(sz) + 1.0 - sz) * cell_size) / dz
        else:
            tMaxZ = ((sz - math.floor(sz)) * cell_size) / -dz
        tDeltaZ = cell_size / abs(dz)

    t = 0.0
    max_t = max_range
    fov_grid[vz, vy, vx] = 1

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

        fov_grid[vz, vy, vx] = 1


@cuda.jit
def raycast_kernel(occ: NDArray[np.uint8], visibility_grid: NDArray[np.int32], sensor: Position, dirs: NDArray[np.float32], max_range: float, cell_size: float) -> None:
    """Marks all free voxels traversed by a ray as visible, stopping at the first occupied voxel.

    This uses a DDA voxel traversal algorithm (Amanatides & Woo, 1987).
    The `visibility_grid` is modified in place.
    """
    D, H, W = occ.shape
    idx = cuda.grid(1)
    if idx >= dirs.shape[0]:
        return

    sz, sy, sx = sensor[0], sensor[1], sensor[2]
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
        tMaxX, tDeltaX = 1e30, 1e30
    else:
        if dx > 0:
            tMaxX = ((math.floor(sx) + 1.0 - sx) * cell_size) / dx
        else: # dx < 0
            tMaxX = ((sx - math.floor(sx)) * cell_size) / -dx
        tDeltaX = cell_size / abs(dx)

    if dy == 0.0:
        tMaxY, tDeltaY = 1e30, 1e30
    else:
        if dy > 0:
            tMaxY = ((math.floor(sy) + 1.0 - sy) * cell_size) / dy
        else: # dy < 0
            tMaxY = ((sy - math.floor(sy)) * cell_size) / -dy
        tDeltaY = cell_size / abs(dy)

    if dz == 0.0:
        tMaxZ, tDeltaZ = 1e30, 1e30
    else:
        if dz > 0:
            tMaxZ = ((math.floor(sz) + 1.0 - sz) * cell_size) / dz
        else: # dz < 0
            tMaxZ = ((sz - math.floor(sz)) * cell_size) / -dz
        tDeltaZ = cell_size / abs(dz)

    t = 0.0
    max_t = max_range

    if occ[vz, vy, vx] != 0:
        return
    visibility_grid[vz, vy, vx] = 1

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

        if occ[vz, vy, vx] != 0:
            break

        visibility_grid[vz, vy, vx] = 1


@cuda.jit(device=True)
def ceil_impl_device(val: float) -> int:
    """Device-side implementation of math.ceil for a float."""
    return int(val) if val == int(val) else int(val + 1.0)


@cuda.jit(device=True)
def tan_impl_device(rad: float) -> float:
    """Device-side implementation of math.tan for a float."""
    return math.sin(rad) / math.cos(rad)


@cuda.jit(device=True)
def is_path_clear_device(occ: NDArray[np.uint8], z0: int, y0: int, x0: int, z1: int, y1: int, x1: int) -> bool:
    """Checks for obstacles on the line segment BETWEEN two voxels using DDA."""
    dz = z1 - z0
    dy = y1 - y0
    dx = x1 - x0

    steps = max(abs(dx), abs(dy), abs(dz))

    if steps <= 1:
        return True

    z_inc = float32(dz) / float32(steps)
    y_inc = float32(dy) / float32(steps)
    x_inc = float32(dx) / float32(steps)

    for i in range(1, steps):
        z_f = float32(z0) + i * z_inc
        y_f = float32(y0) + i * y_inc
        x_f = float32(x0) + i * x_inc
        
        vz, vy, vx = int(round(z_f)), int(round(y_f)), int(round(x_f))

        if vz == z0 and vy == y0 and vx == x0:
            continue

        if vz == z1 and vy == y1 and vx == x1:
            break
            
        if occ[vz, vy, vx] != 0:
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
        extra_vox_inflate: int,
        check_occ: bool,
) -> None:
    D, H, W = in_vis.shape
    vz, vy, vx = cuda.grid(3)
    if vz >= D or vy >= H or vx >= W:
        return
    if in_vis[vz, vy, vx] == 0:
        return

    sz, sy, sx = sensor_pos[0], sensor_pos[1], sensor_pos[2]
    dist_vox_sq = float32((vz - sz)**2 + (vy - sy)**2 + (vx - sx)**2)
    dist_m = math.sqrt(dist_vox_sq) * cell_size
    divergence_rad = divergence_deg * (3.141592653589793 / 180.0)
    r_m = max(min_radius_m, dist_m * tan_impl_device(divergence_rad * 0.5))
    r = ceil_impl_device(r_m / cell_size) + extra_vox_inflate

    for dz in range(-r, r+1):
        z = vz + dz
        if z < 0 or z >= D: continue
        for dy in range(-r, r+1):
            y = vy + dy
            if y < 0 or y >= H: continue
            for dx in range(-r, r+1):
                x = vx + dx
                if x < 0 or x >= W: continue
                
                if dx*dx + dy*dy + dz*dz <= r*r:
                    # When dilating the FOV grid (check_occ=False), we ignore obstacles entirely.
                    # When dilating the visibility grid (check_occ=True), we must not dilate
                    # into an obstacle, nor should we dilate THROUGH an obstacle.
                    if check_occ:
                        # Check that the destination is free AND the path to it is clear.
                        if occ[z, y, x] == 0 and is_path_clear_device(occ, vz, vy, vx, z, y, x):
                            out_vis[z, y, x] = 1
                    else:
                        # This is the FOV grid case, ignore occupancy.
                        out_vis[z, y, x] = 1

def get_visibility_grid(
        occ: NDArray[np.uint8],
        sensor_voxel_xyz: Position,
        dirs: NDArray[np.float32],
        max_range: float = 100.0,
        cell_size: float = 1.0,
        beam_config: BeamConfig = BeamConfig(),
        compute_fov_grid: bool = True,
) -> tuple[NDArray[np.int32], NDArray[np.int32] | None]:
    """Computes the visibility grid and, optionally, the Field-of-View (FOV) grid."""
    visibility_grid = np.zeros_like(occ, dtype=np.int32)
    fov_grid = np.zeros_like(occ, dtype=np.int32) if compute_fov_grid else None

    d_occ = cuda.to_device(occ)
    d_visibility_grid = cuda.to_device(visibility_grid)
    d_sensor = cuda.to_device(np.array(sensor_voxel_xyz, dtype=np.float32))
    d_dirs = cuda.to_device(dirs.astype(np.float32))

    threads = 256
    blocks = (dirs.shape[0] + threads - 1) // threads

    raycast_kernel[blocks, threads](d_occ, d_visibility_grid, d_sensor, d_dirs, float32(max_range), float32(cell_size))

    if compute_fov_grid:
        d_fov_grid = cuda.to_device(fov_grid)
        fov_kernel[blocks, threads](d_fov_grid, d_sensor, d_dirs, float32(max_range), float32(cell_size))

    if beam_config.use_dilation:
        d_dilated_visibility_grid = cuda.device_array_like(d_visibility_grid)
        TPB = (8, 8, 8)
        grid_z = (occ.shape[0] + TPB[0] - 1) // TPB[0]
        grid_y = (occ.shape[1] + TPB[1] - 1) // TPB[1]
        grid_x = (occ.shape[2] + TPB[2] - 1) // TPB[2]
        bpg = (grid_z, grid_y, grid_x)

        dilate_visibility_kernel[bpg, TPB](
            d_visibility_grid, d_dilated_visibility_grid, d_occ, d_sensor,
            float32(cell_size), float32(beam_config.divergence_deg),
            float32(beam_config.min_radius_m), int(beam_config.extra_vox_inflate),
            True  # check_occ = True
        )
        d_dilated_visibility_grid.copy_to_host(visibility_grid)
        
        if compute_fov_grid:
            d_dilated_fov_grid = cuda.device_array_like(d_fov_grid)
            dilate_visibility_kernel[bpg, TPB](
                d_fov_grid, d_dilated_fov_grid, d_occ, d_sensor,
                float32(cell_size), float32(beam_config.divergence_deg),
                float32(beam_config.min_radius_m), int(beam_config.extra_vox_inflate),
                False  # check_occ = False
            )
            d_dilated_fov_grid.copy_to_host(fov_grid)

    else:
        d_visibility_grid.copy_to_host(visibility_grid)
        if compute_fov_grid:
            d_fov_grid.copy_to_host(fov_grid)

    return visibility_grid, fov_grid


def get_visibility_grid_batch(
        occ_sensor_list: list[tuple[NDArray[np.uint8], Position]],
        dirs: NDArray[np.float32],
        max_range: float = 100.0,
        cell_size: float = 1.0,
        beam_config: BeamConfig = BeamConfig(),
        compute_fov_grid: bool = True,
) -> list[tuple[NDArray[np.int32], NDArray[np.int32] | None]]:
    """Computes visibility grids for a batch of occupancy grids and sensor positions."""
    return [get_visibility_grid(
                occ,
                sensor,
                dirs,
                max_range=max_range,
                cell_size=cell_size,
                beam_config=beam_config,
                compute_fov_grid=compute_fov_grid,
             ) for occ, sensor in occ_sensor_list]


def pos_to_cell_idx(d: float, h: float, w: float, cell_size: float) -> tuple[int, int, int]:
    """Converts world coordinates to integer voxel indices."""
    return (int(d / cell_size), int(h / cell_size), int(w / cell_size))


def make_scene_from_pcd(
    pcd_path: Path,
    grid_shape_vox: tuple[int, int, int],
    cell_size: float,
    grid_origin_m: Position,
) -> NDArray[np.uint8]:
    """Creates an occupancy grid from a PCD file.

    The grid's voxel (0,0,0) corresponds to the real-world `grid_origin_m` coordinate.

    Args:
        pcd_path: Path to a .pcd file.
        grid_shape_vox: The desired (D, H, W) shape of the grid in voxels.
        cell_size: The size of a single voxel in meters.
        grid_origin_m: The (z, y, x) world coordinate that corresponds to voxel (0,0,0).

    Returns:
        An occupancy grid with points from the PCD file marked as occupied.
    """
    print(f"- Loading points from: {pcd_path}")
    pc = pypcd4.PointCloud.from_path(pcd_path)
    occ = np.zeros(grid_shape_vox, dtype=np.uint8)

    # pypcd gives points in [x, y, z] order, units are in meters.
    points_xyz = pc.numpy()[:, :3]
    points_xyz[:, 2] += 2  # input pcd was shifted down 2 meters
    print(f"  - Loaded {points_xyz.shape[0]} points.")

    # grid_origin_m is (z, y, x) in meters. Convert it to (x, y, z) for vector math.
    grid_origin_m_xyz = np.array([grid_origin_m[2], grid_origin_m[1], grid_origin_m[0]])

    # 1. Translate points to be relative to the grid's origin. Coordinates are still in meters.
    points_relative_to_origin_m = points_xyz - grid_origin_m_xyz

    # 2. Convert the relative meter coordinates into integer voxel indices.
    indices_xyz = np.floor(points_relative_to_origin_m / cell_size).astype(int)

    # 3. Create a boolean mask for points that fall within the grid boundaries.
    # Grid shape is (D, H, W), which corresponds to (z, y, x) indices.
    D, H, W = grid_shape_vox
    valid_mask = (
        (indices_xyz[:, 0] >= 0) & (indices_xyz[:, 0] < W) &  # Check X against W
        (indices_xyz[:, 1] >= 0) & (indices_xyz[:, 1] < H) &  # Check Y against H
        (indices_xyz[:, 2] >= 0) & (indices_xyz[:, 2] < D)    # Check Z against D
    )

    # 4. Filter to get only the valid indices (still in x,y,z order).
    valid_indices_xyz = indices_xyz[valid_mask]

    if valid_indices_xyz.shape[0] == 0:
        print("  - WARNING: No points from the PCD file fall within the specified grid volume.")
        return occ

    print(f"  - {valid_indices_xyz.shape[0]} points fall within the grid volume.")

    # 5. Use the valid indices to set the corresponding voxels to 1 (occupied).
    # The occ grid is indexed by (z, y, x), so we must supply the columns in that order.
    z_indices = valid_indices_xyz[:, 2]
    y_indices = valid_indices_xyz[:, 1]
    x_indices = valid_indices_xyz[:, 0]
    occ[z_indices, y_indices, x_indices] = 1

    return occ

def make_scene(D_m: float = 40.0, H_m: float = 40.0, W_m: float = 40.0, cell_size: float = 1.0) -> tuple[NDArray[np.uint8], Position]:
    """Creates an example scene with obstacles and a sensor position.

    Args:
        D_m: The depth of the grid in meters.
        H_m: The height of the grid in meters.
        W_m: The width of the grid in meters.
        cell_size: The size of a single voxel in meters.

    Returns:
        A tuple containing:
        - occ: A (D, H, W) occupancy grid with obstacles.
        - sensor: The (z, y, x) sensor position in continuous voxel coordinates.
    """
    # Scene Geometry Constants (in meters)
    WALL_X_POS = 30.0
    BOX_Z_RANGE = (15.0, 36.0)
    BOX_Y_RANGE = (15.0, 30.0)
    BOX_X_RANGE = (20.0, 56.0)
    SENSOR_POSITION = (min(32.0, D_m - cell_size), min(32.0, H_m - cell_size), min(8.0, W_m - cell_size))

    grid_shape = pos_to_cell_idx(D_m, H_m, W_m, cell_size)
    occ = np.zeros(grid_shape, dtype=np.uint8)

    wall_x_idx = int(WALL_X_POS / cell_size)
    if wall_x_idx < grid_shape[2]:
        occ[:, :, wall_x_idx] = 1

    # box obstacle
    z0_idx, z1_idx = int(BOX_Z_RANGE[0] / cell_size), int(BOX_Z_RANGE[1] / cell_size)
    y0_idx, y1_idx = int(BOX_Y_RANGE[0] / cell_size), int(BOX_Y_RANGE[1] / cell_size)
    x0_idx, x1_idx = int(BOX_X_RANGE[0] / cell_size), int(BOX_X_RANGE[1] / cell_size)
    if z0_idx < z1_idx and y0_idx < y1_idx and x0_idx < x1_idx:
        occ[z0_idx:z1_idx, y0_idx:y1_idx, x0_idx:x1_idx] = 1

    sensor_idx = (
        SENSOR_POSITION[0] / cell_size,
        SENSOR_POSITION[1] / cell_size,
        SENSOR_POSITION[2] / cell_size,
    )
    return occ, sensor_idx


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
        fov: NDArray[np.int32],
        cluster: NDArray[np.uint8],
        z_idx: int | float,
        sensor_zyx: Position | None = None,
        out_path: str = "example_result.png",
) -> None:
    """Plots a 2D slice of the combined occupancy, visibility, FOV, and cluster grids."""
    occ_slice = occ[int(z_idx)]
    vis_slice = vis[int(z_idx)]
    fov_slice = fov[int(z_idx)]
    cluster_slice = cluster[int(z_idx)]

    # 0: Empty/Unseen (white)
    # 1: Occupied (black)
    # 2: In FOV Free Space (light gray)
    # 3: Visible Free Space (light green)
    # 4: Cluster (Out of FOV) (dark gray)
    # 5: Cluster (Occluded) (light red)
    # 6: Cluster (Visible) (cyan)
    merged_grid = np.zeros_like(occ_slice, dtype=np.uint8)
    
    # Build up layers, with later layers overwriting earlier ones
    merged_grid[fov_slice != 0] = 2
    merged_grid[vis_slice != 0] = 3
    # Overlay cluster information
    is_cluster = cluster_slice == 1
    is_visible = vis_slice != 0
    is_in_fov = fov_slice != 0
    merged_grid[is_cluster & ~is_in_fov] = 4
    merged_grid[is_cluster & is_in_fov & ~is_visible] = 5
    merged_grid[is_cluster & is_visible] = 6
    # Occupied voxels have the highest priority
    merged_grid[occ_slice == 1] = 1

    cmap = mcolors.ListedColormap([
        'white',      # 0: Empty
        'black',      # 1: Occupied
        '#E0E0E0',    # 2: In FOV Free
        '#A0E0A0',    # 3: Visible Free
        '#696969',    # 4: Cluster (Out of FOV)
        '#F08080',    # 5: Cluster (Occluded)
        '#00FFFF'     # 6: Cluster (Visible)
    ])
    bounds = np.arange(-0.5, 7.5, 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(20, 20))
    img = ax.imshow(merged_grid, cmap=cmap, norm=norm, origin="lower")

    if sensor_zyx is not None and int(z_idx) == int(sensor_zyx[0]):
        ax.plot(sensor_zyx[2], sensor_zyx[1], 'y*', markersize=20, markeredgecolor='black', label='Sensor Origin')
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
        fov_grid: NDArray[np.int32] | None,
        cluster_mask: NDArray[np.uint8],
        mode: ClusterVisMode = ClusterVisMode.ANY_VISIBLE,
        max_samples: int = 512,
        rng_seed: int = 0,
) -> ClusterVisibility:
    """Determines a cluster's visibility status based on sampled voxels.

    Args:
        visibility_grid: The grid indicating line-of-sight visible voxels.
        fov_grid: An optional grid indicating all voxels within the sensor's FOV.
            If provided, allows distinguishing between OCCLUDED and OUT_OF_FOV.
        cluster_mask: A grid where non-zero values indicate the cluster's voxels.
        mode: The visibility mode (ANY_VISIBLE or ALL_VISIBLE).
        max_samples: The maximum number of voxels to sample from the cluster for the check.
        rng_seed: The random seed for sampling.

    Returns:
        The visibility status of the cluster.
    """
    z_coords, y_coords, x_coords = np.nonzero(cluster_mask > 0)
    num_voxels = z_coords.shape[0]
    if num_voxels == 0:
        return ClusterVisibility.OUT_OF_FOV # No points to check

    if num_voxels > max_samples:
        rng = np.random.default_rng(rng_seed)
        sample_indices = rng.choice(num_voxels, size=max_samples, replace=False)
        zv, yv, xv = z_coords[sample_indices], y_coords[sample_indices], x_coords[sample_indices]
    else:
        zv, yv, xv = z_coords, y_coords, x_coords

    D, H, W = visibility_grid.shape
    valid_indices = (zv < D) & (yv < H) & (xv < W)
    if not np.any(valid_indices):
        return ClusterVisibility.OUT_OF_FOV # All points outside grid bounds
    
    zv, yv, xv = zv[valid_indices], yv[valid_indices], xv[valid_indices]
    
    visibility_flags = visibility_grid[zv, yv, xv].astype(bool)
    
    # If FOV grid is not provided, we can't distinguish between occluded and out-of-fov.
    if fov_grid is None:
        if np.any(visibility_flags):
            return ClusterVisibility.VISIBLE
        else:
            return ClusterVisibility.OCCLUDED

    fov_flags = fov_grid[zv, yv, xv].astype(bool)

    if mode == ClusterVisMode.ANY_VISIBLE:
        if np.any(visibility_flags):
            return ClusterVisibility.VISIBLE
        elif np.any(fov_flags):
            return ClusterVisibility.OCCLUDED
        else:
            return ClusterVisibility.OUT_OF_FOV
    elif mode == ClusterVisMode.ALL_VISIBLE:
        if np.all(visibility_flags):
            return ClusterVisibility.VISIBLE
        elif np.all(fov_flags):
            return ClusterVisibility.OCCLUDED
        else:
            return ClusterVisibility.OUT_OF_FOV
    
    return ClusterVisibility.OUT_OF_FOV


def get_cluster_visibility_status_batch(
        items: list[tuple[NDArray[np.int32], NDArray[np.int32] | None, NDArray[np.uint8], ClusterVisMode]],
        max_samples: int = 512,
        rng_seed: int = 0,
) -> list[ClusterVisibility]:
    """Determines cluster visibility status for a batch of items.

    Args:
        items: A list of tuples, each containing (visibility_grid, fov_grid,
            cluster_mask, mode) for a single cluster check.
        max_samples: The maximum number of voxels to sample from each cluster.
        rng_seed: The random seed for sampling.

    Returns:
        A list of ClusterVisibility statuses corresponding to the input items.
    """
    return [get_cluster_visibility_status(
                visibility_grid,
                fov_grid,
                cluster_mask,
                mode=mode,
                max_samples=max_samples,
                rng_seed=rng_seed,
            ) for visibility_grid, fov_grid, cluster_mask, mode in items]


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
        help="Grid dimensions (depth,height,width) in meters as 'D,H,W' (z,y,x)."
    )
    parser.add_argument(
        "--grid_origin_m", type=str, default="-20,-20,-20",
        help="World coordinate (z,y,x) in meters that maps to grid voxel (0,0,0). Used only with --pcd_file."
    )
    parser.add_argument(
        "--sensor_pos_m", type=str, default="20,20,20",
        help="Sensor position (z,y,x) in meters."
    )
    parser.add_argument(
        "--cell_size", type=float, default=CELL_SIZE,
        help="Size of a single voxel in meters."
    )
    parser.add_argument(
        "--plot_3d_web", action="store_true",
        help="If set, generates an interactive 3D plot of the scene as an HTML file (scene_3d_view.html)."
    )
    args = parser.parse_args()

    print("===== Voxel-Based Visibility Analysis =====")
    print("\n[ SETUP ]")
    start_setup_time = time.time()

    # --- Convert meter-based arguments to voxel-based coordinates ---
    # User provides grid dimensions in meters, we calculate voxel shape
    grid_dims_m = tuple(map(float, args.grid_dims_m.split(',')))
    grid_shape_vox = pos_to_cell_idx(grid_dims_m[0], grid_dims_m[1], grid_dims_m[2], args.cell_size)

    # User provides sensor position in meters, we convert to continuous voxel coordinates
    sensor_pos_m = tuple(map(float, args.sensor_pos_m.split(',')))
    sensor = (
        sensor_pos_m[0] / args.cell_size,
        sensor_pos_m[1] / args.cell_size,
        sensor_pos_m[2] / args.cell_size,
    )

    if args.pcd_file:
        if not args.pcd_file.exists():
            raise FileNotFoundError(f"PCD file not found: {args.pcd_file}")

        print(f"- Generating scene from PCD file: {args.pcd_file}")
        grid_origin_m = tuple(map(float, args.grid_origin_m.split(',')))

        occ = make_scene_from_pcd(
            pcd_path=args.pcd_file,
            grid_shape_vox=grid_shape_vox,
            cell_size=args.cell_size,
            grid_origin_m=grid_origin_m,
        )
    else:
        print("- Generating default procedural scene.")
        # The sensor position from the CLI (`--sensor_pos_m`) is used.
        # We call make_scene to generate obstacles based on the grid dimensions.
        occ, _ = make_scene(
            D_m=grid_dims_m[0], H_m=grid_dims_m[1], W_m=grid_dims_m[2],
            cell_size=args.cell_size
        )

    scene_build_time = time.time() - start_setup_time
    print(f"- Scene Build Time:      {scene_build_time:.4f}s")
    print(f"  - Grid Shape:          {occ.shape}")
    print(f"  - Cell Size:           {args.cell_size}m")

    start_ray_time = time.time()
    h_samples, v_samples = 900, 128
    fov_h_deg, fov_v_deg = 360.0, 105.0
    dirs = sample_fov_dirs(h_samples=h_samples, v_samples=v_samples, fov_h_deg=fov_h_deg, fov_v_deg=fov_v_deg)
    ray_sampling_time = time.time() - start_ray_time
    print(f"- Ray Sampling Time:     {ray_sampling_time:.4f}s")
    print(f"  - Num Rays:            {dirs.shape[0]} ({h_samples}x{v_samples})")
    print(f"  - FOV (H x V):         {fov_h_deg}° x {fov_v_deg}°")

    h_angular_res_deg = fov_h_deg / h_samples
    v_angular_res_deg = fov_v_deg / v_samples
    required_divergence_deg = max(h_angular_res_deg, v_angular_res_deg)
    print(f"- Auto-calculated beam divergence: {required_divergence_deg:.4f} degrees")

    beam_config = BeamConfig(
        use_dilation=True,
        divergence_deg=required_divergence_deg,
        min_radius_m=0,
        extra_vox_inflate=0,
    )
    
    # Define a consistent cluster for all demos
    cluster = np.zeros_like(occ, dtype=np.uint8)
    if not args.pcd_file:
        # This cluster definition is tuned for the default procedural scene
        z0, z1 = int(10 / args.cell_size), int(36 / args.cell_size)
        y0, y1 = int(10 / args.cell_size), int(30 / args.cell_size)
        x0, x1 = int(10 / args.cell_size), int(56 / args.cell_size)
        cluster[z0:z1, y0:y1, x0:x1] = 1
    else:
        # For PCD files, let's make the cluster the entire occupied space.
        # This is useful for checking the visibility of the whole point cloud.
        cluster = occ.copy()

    # --- WARMUP ---
    # First run includes a one-time JIT compilation cost. We run it once
    # to ensure subsequent measurements are for execution time only.
    print("\n[ WARMUP ]")
    print("- Running once to trigger CUDA JIT compilation...")
    _ = get_visibility_grid(occ, sensor, dirs[:1], beam_config=beam_config, compute_fov_grid=True)
    print("- Warmup complete.")

    # --- DEMO 1: Single Run (Visibility + FOV Grids) ---
    print("\n--- DEMO 1: Single Run (Visibility + FOV Grids) ---")
    start_time = time.time()
    vis_grid, fov_grid = get_visibility_grid(
        occ, sensor, dirs,
        max_range=40.0, cell_size=args.cell_size,
        beam_config=beam_config, compute_fov_grid=True
    )
    full_time = time.time() - start_time
    print(f"- Grid Computation Time:   {full_time:.4f}s")

    start_time = time.time()
    any_vis = get_cluster_visibility_status(vis_grid, fov_grid, cluster, mode=ClusterVisMode.ANY_VISIBLE)
    all_vis = get_cluster_visibility_status(vis_grid, fov_grid, cluster, mode=ClusterVisMode.ALL_VISIBLE)
    single_cluster_time = time.time() - start_time
    print(f"- Cluster Check Time:      {single_cluster_time:.4f}s")
    print(f"  - Result (ANY_VISIBLE):  {any_vis.name}")
    print(f"  - Result (ALL_VISIBLE):  {all_vis.name}")

    start_plot_time = time.time()
    print("- Plotting slices (view slice*.png for results)...")
    if fov_grid is not None:
        for z_idx in range(96, 108):
            if 0 <= z_idx < occ.shape[0]:
                plot_slice(occ, vis_grid, fov_grid, cluster, z_idx=z_idx, sensor_zyx=sensor, out_path=f"slice_{z_idx}.png")
    plot_time = time.time() - start_plot_time
    print(f"- Plotting Time:           {plot_time:.4f}s")

    # --- DEMO 2: Single Run (Visibility-Only Grid) ---
    print("\n--- DEMO 2: Single Run (Visibility-Only Grid) ---")
    start_time = time.time()
    vis_only_grid, no_fov_grid = get_visibility_grid(
        occ, sensor, dirs,
        max_range=40.0, cell_size=args.cell_size,
        beam_config=beam_config, compute_fov_grid=False
    )
    vis_only_time = time.time() - start_time
    print(f"- Grid Computation Time:   {vis_only_time:.4f}s")

    start_time = time.time()
    any_vis_simple = get_cluster_visibility_status(vis_only_grid, no_fov_grid, cluster, mode=ClusterVisMode.ANY_VISIBLE)
    all_vis_simple = get_cluster_visibility_status(vis_only_grid, no_fov_grid, cluster, mode=ClusterVisMode.ALL_VISIBLE)
    simple_cluster_time = time.time() - start_time
    print(f"- Cluster Check Time:      {simple_cluster_time:.4f}s")
    print(f"  - Result (ANY_VISIBLE):  {any_vis_simple.name} (Note: OUT_OF_FOV not distinguishable from OCCLUDED)")
    print(f"  - Result (ALL_VISIBLE):  {all_vis_simple.name} (Note: OUT_OF_FOV not distinguishable from OCCLUDED)")

    # --- DEMO 3: Batch Processing ---
    print("\n--- DEMO 3: Batch Processing ---")
    batch_size = 20
    print(f"- Batch Size:              {batch_size}")
    
    # Batch with Vis + FOV
    start_time = time.time()
    batch_results_full = get_visibility_grid_batch([(occ, sensor) for _ in range(batch_size)], dirs, max_range=40.0, cell_size=args.cell_size, beam_config=beam_config, compute_fov_grid=True)
    batch_grid_time = time.time() - start_time
    avg_batch_grid_time = batch_grid_time / batch_size
    print(f"- Grid Comp Time (Vis+FOV):{batch_grid_time:>8.4f}s (Avg: {avg_batch_grid_time:.4f}s per item)")
    
    # Batch with Vis-Only
    start_time = time.time()
    batch_results_vis_only = get_visibility_grid_batch([(occ, sensor) for _ in range(batch_size)], dirs, max_range=40.0, cell_size=args.cell_size, beam_config=beam_config, compute_fov_grid=False)
    batch_vis_only_time = time.time() - start_time
    avg_batch_vis_only_time = batch_vis_only_time / batch_size
    print(f"- Grid Comp Time (Vis-Only):{batch_vis_only_time:>7.4f}s (Avg: {avg_batch_vis_only_time:.4f}s per item)")

    # Batch cluster check (using the full results)
    start_time = time.time()
    batch_items = [(vis_grid, fov_grid, cluster, ClusterVisMode.ANY_VISIBLE) for vis_grid, fov_grid in batch_results_full]
    batch_cluster_vis = get_cluster_visibility_status_batch(batch_items)
    batch_cluster_time = time.time() - start_time
    avg_batch_cluster_time = batch_cluster_time / batch_size
    print(f"- Cluster Check Time:      {batch_cluster_time:>8.4f}s (Avg: {avg_batch_cluster_time:.4f}s per item)")

    # --- Performance Summary ---
    print("\n======================= PERFORMANCE SUMMARY =======================")
    print(f"| {'Operation':<32} | {'Time (Single)':>15} | {'Time (Batch Avg)':>18} |")
    print(f"|{'-'*34}|{'-'*17}|{'-'*20}|")
    print(f"| Grid Computation (Vis + FOV)   | {full_time:>15.4f}s | {avg_batch_grid_time:>18.4f}s |")
    print(f"| Grid Computation (Vis-Only)    | {vis_only_time:>15.4f}s | {avg_batch_vis_only_time:>18.4f}s |")
    print(f"| Cluster Visibility Check       | {single_cluster_time:>15.4f}s | {avg_batch_cluster_time:>18.4f}s |")
    print("---------------------------------------------------------------------")
    print("=====================================================================")

    # --- DEMO 4: Interactive Web-Based 3D Visualization ---
    if args.plot_3d_web:
        plot_3d_scene_web(
            occ=occ,
            sensor_zyx=sensor,
            dirs=dirs,
            max_range=40.0, # Use the same max_range as Demo 1
        )
