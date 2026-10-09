import numpy as np
import open3d as o3d

path = "runs/visibility_obj/views/view_00015.npz"

z = np.load(path)

points = z["points"]
states = z["state"]

# 0 FREE
# 1 SURFACE
# 2 UNKNOWN
# 3 INVALID

colors = np.zeros((len(points), 3))

colors[states == 0] = [0.2, 0.7, 1.0]   # free
colors[states == 1] = [1.0, 0.2, 0.2]   # surface
colors[states == 2] = [0.7, 0.7, 0.7]   # unknown
colors[states == 3] = [1.0, 0.8, 0.1]   # invalid

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([pcd])