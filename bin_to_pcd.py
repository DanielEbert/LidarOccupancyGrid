import pypcd4
import os
import numpy as np

bins_path = '/root/P/lidar/semantic_kitti/dataset/SemanticKitti/sequences/00/velodyne/'
out_path = 'pcds'

os.makedirs(out_path, exist_ok=True)

for filename in os.listdir(bins_path):
    filepath = os.path.join(bins_path, filename)

    with open(filepath, 'rb') as f:
        arr = np.frombuffer(f.read(), dtype=np.float32)

    # reshape into N x 4 array
    points = arr.reshape(-1, 4)

    out_filepath = os.path.join(out_path, filename.replace('.bin', '.pcd'))
    pypcd4.PointCloud.from_xyzi_points(points).save(out_filepath)
