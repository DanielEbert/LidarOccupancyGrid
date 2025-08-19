import numpy as np
from numba import cuda, float32
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from enum import IntEnum
from dataclasses import dataclass

DEFAULT_BEAM_DIVERGENCE_DEG = 0.15  # typical for some LiDARs; adjustable
DEFAULT_MIN_BEAM_RADIUS_M   = 0.01  # near-field footprint (m), accounts for aperture/jitter
DEFAULT_DILATION_SAFETY_VOX = 0     # extra conservative vox inflation (integer vox)
CELL_SIZE = 0.2

@dataclass
class BeamConfig:
    use_dilation: bool = True
    divergence_deg: float = DEFAULT_BEAM_DIVERGENCE_DEG
    min_radius_m: float = DEFAULT_MIN_BEAM_RADIUS_M
    extra_vox_inflate: int = DEFAULT_DILATION_SAFETY_VOX

class ClusterVisMode(IntEnum):
    ANY_VISIBLE = 0   # cluster is visible if ANY sampled surface voxel is LoS-visible
    ALL_VISIBLE = 1   # cluster is visible only if ALL sampled surface voxels are LoS-visible

@cuda.jit
def raycast_kernel(occ, visibility_grid, sensor, dirs, max_range, cell_size):
    """
    DDA voxel traversal (Amanatides & Woo, 1987).
    Marks all free voxels traversed by each ray as visible (1), and stops at first occupied voxel.
    """
    D, H, W = occ.shape
    idx = cuda.grid(1)
    if idx >= dirs.shape[0]:
        return

    # Sensor in voxel coordinates (float)
    sz, sy, sx = sensor[0], sensor[1], sensor[2]
    dx, dy, dz = dirs[idx, 0], dirs[idx, 1], dirs[idx, 2]

    # Start voxel
    vx = int(sx)
    vy = int(sy)
    vz = int(sz)

    # Bounds check
    if vz < 0 or vz >= D or vy < 0 or vy >= H or vx < 0 or vx >= W:
        return

    step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
    step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)
    step_z = 1 if dz > 0 else (-1 if dz < 0 else 0)

    # Voxel boundary positions (in voxel coords)
    # t is measured in world meters. Distance to next voxel boundary along each axis:
    # boundary coordinate in voxel space => convert to world via cell_size inside t computations.
    def axis_init(s_coord, dir_comp, step):
        if dir_comp == 0.0:
            return 1e30, 1e30  # never crosses
        # next boundary in voxel coords
        next_boundary = (float(int(s_coord) + (1 if step > 0 else 0)))
        # distance (in voxels) to boundary
        dist_vox = (next_boundary - s_coord) if step > 0 else (s_coord - next_boundary)
        # convert to world meters along this axis via dividing by |dir_comp| in voxel units/meter:
        # Along the ray, advancing param t (meters) moves dir_comp/cell_size voxels per meter.
        # So voxel distance -> meters: dist_vox / (|dir_comp|/cell_size) = dist_vox * cell_size / |dir_comp|
        # Use abs() for clarity
        t_max = (dist_vox * cell_size) / abs(dir_comp)
        t_delta = cell_size / abs(dir_comp)
        return t_max, t_delta

    tMaxX, tDeltaX = axis_init(sx, dx, step_x)
    tMaxY, tDeltaY = axis_init(sy, dy, step_y)
    tMaxZ, tDeltaZ = axis_init(sz, dz, step_z)

    # Traverse until max_range or out of grid
    t = 0.0
    max_t = max_range

    # Visit starting voxel
    if occ[vz, vy, vx] != 0:
        # sensor spawned inside occupied; we still stop
        return
    visibility_grid[vz, vy, vx] = 1

    while t <= max_t:
        # Choose next axis to step
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
            # Out of grid
            break

        if occ[vz, vy, vx] != 0:
            # Occupied = stop
            break

        visibility_grid[vz, vy, vx] = 1

# Dilation approximates real LiDAR divergence and reduces "invisible" gaps from sparse sampling.
@cuda.jit
def dilate_visibility_kernel(in_vis, out_vis, occ, radius_vox):
    D, H, W = in_vis.shape
    vz, vy, vx = cuda.grid(3)
    if vz >= D or vy >= H or vx >= W:
        return
    if in_vis[vz, vy, vx] == 0:
        return
    r = radius_vox
    for dz in range(-r, r+1):
        z = vz + dz
        if z < 0 or z >= D: 
            continue
        for dy in range(-r, r+1):
            y = vy + dy
            if y < 0 or y >= H:
                continue
            for dx in range(-r, r+1):
                x = vx + dx
                if x < 0 or x >= W:
                    continue
                # spherical neighborhood
                if dx*dx + dy*dy + dz*dz <= r*r:
                    # check if the voxel is not an obstacle
                    if occ[z, y, x] == 0:
                        out_vis[z, y, x] = 1

def get_visibility_grid(
        occ,
        sensor_voxel_xyz,
        dirs,
        max_range=100.0,
        cell_size=1.0,
        beam_config: BeamConfig = BeamConfig(),
):
    """
    Compute visible free voxels by DDA ray traversal, then conservatively inflate
    by a voxel radius derived from LiDAR beam divergence (without adding new rays).
    """
    visibility_grid = np.zeros_like(occ, dtype=np.int32)

    d_occ = cuda.to_device(occ)
    d_visibility_grid = cuda.to_device(visibility_grid)
    d_sensor = cuda.to_device(np.array(sensor_voxel_xyz, dtype=np.float32))
    d_dirs = cuda.to_device(dirs.astype(np.float32))

    threads = 256
    blocks = (dirs.shape[0] + threads - 1) // threads

    # DDA ray traversal
    raycast_kernel[blocks, threads](d_occ, d_visibility_grid, d_sensor, d_dirs, float32(max_range), float32(cell_size))

    # Optional beam dilation
    if beam_config.use_dilation:
        r_m = max(beam_config.min_radius_m, 0.5 * float(max_range) * np.tan(np.deg2rad(beam_config.divergence_deg) * 0.5))
        r_vox = int(np.ceil(r_m / float(cell_size))) + int(beam_config.extra_vox_inflate)
        if r_vox > 0:
            # Create a new device array for the dilated output, using the original raycast result as input
            d_dilated_visibility_grid = cuda.device_array_like(d_visibility_grid)
            TPB = (8, 8, 8)
            grid_z = (occ.shape[0] + TPB[0] - 1) // TPB[0]
            grid_y = (occ.shape[1] + TPB[1] - 1) // TPB[1]
            grid_x = (occ.shape[2] + TPB[2] - 1) // TPB[2]
            dilate_visibility_kernel[(grid_z, grid_y, grid_x), TPB](d_visibility_grid, d_dilated_visibility_grid, d_occ, r_vox)
            # Copy the final dilated grid to the host
            d_dilated_visibility_grid.copy_to_host(visibility_grid)
        else:
            # If no dilation occurs, copy the original raycast grid to the host
            d_visibility_grid.copy_to_host(visibility_grid)
    else:
        # If dilation is disabled, copy the original raycast grid to the host
        d_visibility_grid.copy_to_host(visibility_grid)

    return visibility_grid

def get_visibility_grid_batch(
    occ_sensor_list,
    dirs,
    max_range=100.0,
    cell_size=1.0,
    beam_config: BeamConfig = BeamConfig(),
):
    """
    Batch version of get_visibility_grid.
    occ_sensor_list: list of (occ, sensor_voxel_xyz) tuples
    Returns: list of visibility grids
    """
    return [get_visibility_grid(
                occ,
                sensor,
                dirs,
                max_range=max_range,
                cell_size=cell_size,
                beam_config=beam_config,
             ) for occ, sensor in occ_sensor_list]

def pos_to_cell_idx(d: int, h: int, w: int, cell_size: float):
    return (
        int(d / cell_size),
        int(h / cell_size),
        int(w / cell_size),
    )

# Example scene
def make_scene(D=40, H=40, W=40, cell_size: float = 1):
    """
    Create a simple synthetic scene:
      - A wall perpendicular to +X at x=40 (world units)
      - A solid box obstacle near x~52 (world units)
      - Sensor at (z=32, y=32, x=8) (world units)
    Returns:
      occ: uint8[D,H,W]
      sensor: (z,y,x) float32 (voxel indices)
    """
    # --- Scene Geometry Constants (in world units) ---
    WALL_X_POS = 30.0
    BOX_Z_RANGE = (15.0, 36.0)
    BOX_Y_RANGE = (15.0, 30.0)
    BOX_X_RANGE = (20.0, 56.0)
    SENSOR_POSITION = (min(32.0, D - 1.0), min(32.0, H - 1.0), min(8.0, W - 1.0))
    # ------------------------------------

    grid_shape = pos_to_cell_idx(D, H, W, cell_size)
    occ = np.zeros(grid_shape, dtype=np.uint8)

    # Add a wall
    wall_x_idx = int(WALL_X_POS / cell_size)
    if wall_x_idx < grid_shape[2]:
        occ[:, :, wall_x_idx] = 1

    # Add a box obstacle
    z0_idx, z1_idx = int(BOX_Z_RANGE[0] / cell_size), int(BOX_Z_RANGE[1] / cell_size)
    y0_idx, y1_idx = int(BOX_Y_RANGE[0] / cell_size), int(BOX_Y_RANGE[1] / cell_size)
    x0_idx, x1_idx = int(BOX_X_RANGE[0] / cell_size), int(BOX_X_RANGE[1] / cell_size)
    if z0_idx < z1_idx and y0_idx < y1_idx and x0_idx < x1_idx:
        occ[z0_idx:z1_idx, y0_idx:y1_idx, x0_idx:x1_idx] = 1

    # Define sensor position
    sensor_idx = tuple([s / cell_size for s in SENSOR_POSITION])
    return occ, sensor_idx

def sample_fov_dirs(N=50000, fov_h_deg=60.0, fov_v_deg=30.0, seed=0):
    """
    Sample N ray directions within a forward-facing FOV around +X axis.
    fov_h_deg: horizontal FOV (azimuth)
    fov_v_deg: vertical FOV (elevation)
    Returns float32[N,3] (dx,dy,dz), unit vectors.
    """
    rng = np.random.default_rng(seed)
    phi = np.deg2rad(rng.uniform(-fov_h_deg / 2.0, fov_h_deg / 2.0, size=N))    # azimuth
    theta = np.deg2rad(rng.uniform(-fov_v_deg / 2.0, fov_v_deg / 2.0, size=N))  # elevation
    dx = np.cos(theta) * np.cos(phi)
    dy = np.cos(theta) * np.sin(phi)
    dz = np.sin(theta)
    dirs = np.stack([dx, dy, dz], axis=1).astype(np.float32)
    # normalize for numerical safety
    norms = np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8
    dirs /= norms
    return dirs

def plot_slice(occ, vis, cluster, z_idx, out_path="example_result.png"):
    """
    Plot merged occupancy, visibility, and cluster status for a given z-slice.
    """
    occ_slice = occ[int(z_idx)]
    vis_slice = vis[int(z_idx)]
    cluster_slice = cluster[int(z_idx)]

    # Create a new merged grid with values representing different states.
    # 0: Empty/Unseen (white)
    # 1: Occupied (black)
    # 2: Visible Free Space (light green)
    # 3: Cluster (Occluded) (light red)
    # 4: Cluster (Visible) (cyan)
    merged_grid = np.zeros_like(occ_slice, dtype=np.uint8)
    
    # Start with base visibility
    merged_grid[vis_slice != 0] = 2
    # Overlay cluster information
    merged_grid[cluster_slice == 1] = 3
    merged_grid[(cluster_slice == 1) & (vis_slice != 0)] = 4
    # Occupied voxels have the highest priority
    merged_grid[occ_slice == 1] = 1

    # Create a custom colormap for the discrete values.
    cmap = mcolors.ListedColormap([
        'white',      # 0: Empty
        'black',      # 1: Occupied
        '#A0E0A0',    # 2: Visible Free
        '#F08080',    # 3: Cluster (Occluded)
        '#00FFFF'     # 4: Cluster (Visible)
    ])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(20, 20))
    
    img = ax.imshow(merged_grid, cmap=cmap, norm=norm, origin="lower")
    
    ax.set_title(f"Occupancy, Visibility & Cluster Status (z={int(z_idx)})")
    
    cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1, 2, 3, 4])
    cbar.ax.set_yticklabels(['Empty', 'Occupied', 'Visible Free', 'Cluster (Occluded)', 'Cluster (Visible)'])
    cbar.set_label("Voxel State")

    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()

def is_cluster_visible(
    visibility_grid,
    cluster_mask,
    mode=ClusterVisMode.ANY_VISIBLE,
    max_samples=512,
    rng_seed=0,
):
    """
    Determine if a cluster is visible by checking for overlap between its voxels
    and a pre-computed visibility grid. Assumes cluster_mask contains surface voxels.

    - ANY_VISIBLE: True if any sampled voxel is in a visible grid cell.
    - ALL_VISIBLE: True only if all sampled voxels are in visible grid cells.
    """
    # Use np.nonzero for better performance; it returns a tuple of arrays (z_coords, y_coords, x_coords)
    z_coords, y_coords, x_coords = np.nonzero(cluster_mask > 0)

    num_voxels = z_coords.shape[0]
    if num_voxels == 0:
        return False

    # Sample voxels if there are too many, to keep the check fast.
    if num_voxels > max_samples:
        rng = np.random.default_rng(rng_seed)
        sample_indices = rng.choice(num_voxels, size=max_samples, replace=False)
        zv, yv, xv = z_coords[sample_indices], y_coords[sample_indices], x_coords[sample_indices]
    else:
        zv, yv, xv = z_coords, y_coords, x_coords
    
    # Ensure coordinates are within the grid bounds before indexing
    D, H, W = visibility_grid.shape
    valid_indices = (zv < D) & (yv < H) & (xv < W)
    if not np.any(valid_indices):
        return False # No points are even inside the grid
    
    zv, yv, xv = zv[valid_indices], yv[valid_indices], xv[valid_indices]
    
    visibility_flags = visibility_grid[zv, yv, xv]

    if mode == ClusterVisMode.ANY_VISIBLE:
        return np.any(visibility_flags).item()
    elif mode == ClusterVisMode.ALL_VISIBLE:
        return np.all(visibility_flags).item()
    
    return False

def is_cluster_visible_batch(
    items,
    max_samples=512,
    rng_seed=0,
):
    """
    Batch version of is_cluster_visible.
    items: list of (visibility_grid, cluster_mask, mode)
    Returns: list of booleans
    """
    return [is_cluster_visible(
                visibility_grid,
                cluster_mask,
                mode=mode,
                max_samples=max_samples,
                rng_seed=rng_seed,
            ) for visibility_grid, cluster_mask, mode in items]

# TODO: coverage_grid (float32) accumulating per-voxel solid angle coverage (e.g., 1/range^2 or solid angle per beam).
# Threshold for stricter visibility decisions. See main task notes for details.
if __name__ == "__main__":
    import time
    
    if not cuda.is_available():
        raise RuntimeError("Numba CUDA is not available. Example requires a CUDA-capable GPU and CUDA toolkit.")
    
    # Build scene
    print("Building scene...")
    start_time = time.time()
    occ, sensor = make_scene(cell_size=CELL_SIZE)
    print(f"Scene build time: {time.time() - start_time:.4f}s")

    # Sample rays within FOV
    print("Sampling ray directions...")
    start_time = time.time()
    dirs = sample_fov_dirs(N=50000, fov_h_deg=60.0, fov_v_deg=30.0, seed=0)
    print(f"Ray sampling time: {time.time() - start_time:.4f}s")
    
    # Single visibility grid computation
    print("Computing visibility grid (single)...")
    start_time = time.time()
    beam_config = BeamConfig(
        use_dilation=True,
        divergence_deg=0.15,
        min_radius_m=0.01,
        extra_vox_inflate=0,
    )
    counts = get_visibility_grid(
        occ,
        sensor,
        dirs,
        max_range=40.0,
        cell_size=CELL_SIZE,
        beam_config=beam_config,
    )
    single_time = time.time() - start_time
    print(f"Single visibility grid time: {single_time:.4f}s")
    print("visibility grid shape:", counts.shape)

    # Example cluster visibility check: reuse the box region from make_scene as a "cluster"
    cluster = np.zeros_like(occ, dtype=np.uint8)
    z0_idx, z1_idx = int(10 / CELL_SIZE), int(36 / CELL_SIZE)
    y0_idx, y1_idx = int(10 / CELL_SIZE), int(30 / CELL_SIZE)
    x0_idx, x1_idx = int(10 / CELL_SIZE), int(56 / CELL_SIZE)
    cluster[z0_idx:z1_idx, y0_idx:y1_idx, x0_idx:x1_idx] = 1
    
    print("Plotting slice...")
    start_time = time.time()
    plot_slice(occ, counts, cluster, z_idx=int(sensor[0]), out_path="example_result.png")
    print(f"Plot time: {time.time() - start_time:.4f}s")

    print("Computing cluster visibility (single calls)...")
    start_time = time.time()
    any_vis = is_cluster_visible(counts, cluster, mode=ClusterVisMode.ANY_VISIBLE)
    all_vis = is_cluster_visible(counts, cluster, mode=ClusterVisMode.ALL_VISIBLE)
    single_cluster_time = time.time() - start_time
    print(f"Single cluster visibility time: {single_cluster_time:.4f}s")
    print(f"Cluster visibility -> ANY_VISIBLE={any_vis}, ALL_VISIBLE={all_vis}")

    # Example batch usage
    print("Computing visibility grid (batch)...")
    batch_size = 20
    start_time = time.time()
    batch_results = get_visibility_grid_batch([(occ, sensor) for _ in range(batch_size)], dirs, max_range=40.0, cell_size=CELL_SIZE, beam_config=beam_config)
    batch_grid_time = time.time() - start_time
    print(f"Batch visibility grid time: {batch_grid_time:.4f}s")
    print(f"Batch visibility grid time per sample: {batch_grid_time/batch_size:.4f}s")
    print("Batch get_visibility_grid returned:", len(batch_results))

    print("Computing cluster visibility (batch)...")
    start_time = time.time()
    batch_cluster_vis = is_cluster_visible_batch([(counts, cluster, ClusterVisMode.ANY_VISIBLE) for _ in range(batch_size)])
    batch_cluster_time = time.time() - start_time
    print(f"Batch cluster visibility time: {batch_cluster_time:.4f}s")
    print(f"Batch cluster visibility time per sample: {batch_cluster_time/batch_size:.4f}s")
    print("Batch cluster visibility:", batch_cluster_vis)

    # Performance comparison
    print("\n--- Performance Comparison ---")
    print(f"Single visibility grid: {single_time:.4f}s")
    print(f"Batch visibility grid:  {batch_grid_time:.4f}s")
    print(f"Single cluster checks:  {single_cluster_time:.4f}s")
    print(f"Batch cluster checks:   {batch_cluster_time:.4f}s")
