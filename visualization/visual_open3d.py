import open3d as o3d
import numpy as np

# pcd = open3d.io.read_point_cloud("/home/yzy/Downloads/2011_09_26/2011_09_26_drive_0023_sync/velodyne_points/data/0000000000.bin", format='xyz')
pcd_np = np.fromfile("/home/yzy/Downloads/2011_09_26/2011_09_26_drive_0023_sync/velodyne_points/data/0000000000.bin", dtype=np.float32)
pcd_np = pcd_np.reshape(-1,4)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pcd_np[:, :3])
print(pcd)
print(np.asarray(pcd))
o3d.visualization.draw_geometries([pcd])