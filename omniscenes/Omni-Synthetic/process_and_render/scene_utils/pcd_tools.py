import os

import numpy as np
import trimesh
from scipy.spatial import KDTree

def downsample_point_cloud(point_cloud, target_points):

    if target_points >= point_cloud.shape[0]:
        return point_cloud
    indices = np.random.choice(point_cloud.shape[0], target_points, replace=False)
    downsampled_point_cloud = point_cloud[indices]
    return downsampled_point_cloud

def norm_coords(coords):

    min_coords = np.min(coords, axis=0)
    max_coords = np.max(coords, axis=0)
    norm_coords = (coords - min_coords) / np.max((max_coords - min_coords))
    return norm_coords

def downsample_mesh_naive(mesh, target_num_faces=15000):
    num_faces = len(mesh.triangles)
    if num_faces > target_num_faces:
        # print(f"Number of vertices: {num_vertices}, downsampling to {target_num_vertices}")
        # downsamlped_mesh = mesh.simplify_quadric_decimation(target_num_faces, aggression=5)
        downsamlped_mesh = mesh.simplify_quadric_decimation(percent=0.8, aggression=5)
        return downsamlped_mesh
    else:
        return mesh
    

def downsample_mesh_scene(mesh_path, max_iterations=10, target_size_mb=55):
    mesh_temp = trimesh.load(mesh_path)
    mesh_temp.export(mesh_path)
    mesh = trimesh.load(mesh_path).process()
    original_size_mb = os.path.getsize(mesh_path) / (1024 * 1024)
    print(f"[INFO] The original mesh size is {original_size_mb:.2f}MB")
    if original_size_mb > target_size_mb:
        percent = target_size_mb / original_size_mb
        for _ in range(max_iterations):
            print(f'[INFO] The mesh size is larger than 55MB')
            print(f"Original mesh vertices: {len(mesh.vertices)}, Original mesh triangles: {len(mesh.triangles)}")
            print(f"[INFO] Current percent is {percent:.2f}")
            mesh_downsampled = mesh.simplify_quadric_decimation(percent=percent, aggression=5)
            print(f"Current mesh vertices: {len(mesh_downsampled.vertices)}, Current mesh triangles: {len(mesh_downsampled.triangles)}")
        # o3d.io.write_triangle_mesh(mesh_path, mesh_downsampled, write_ascii=False)
            mesh_downsampled.export(mesh_path)
            current_size_mb = os.path.getsize(mesh_path) / (1024 * 1024)
            print(f"[INFO] The current size is {os.path.getsize(mesh_path) / (1024 * 1024):.2f}MB")
            if current_size_mb <= target_size_mb:
                break
            if current_size_mb > target_size_mb:
                percent *= 0.9
            else:
                percent *= 1.1
            print("="*80)
        print(f"[INFO] The downsampled mesh has been saved in {mesh_path}, the size is {os.path.getsize(mesh_path) / (1024 * 1024):.2f}MB")



def paint_color(pc_wo_color, pc_with_color_norm):
    # pc_wo_color cant be normalized, because it is the original point cloud without color

    # pc_with_color_norm normalized by (coords - min_coords) / (max_coords - min_coords)

    norm_coords_with_color = pc_with_color_norm[:, :3] 
    colors = pc_with_color_norm[:, 3:]  
    coords_wo_color = pc_wo_color[:, :3] 
    norm_coords_wo_color = norm_coords(coords_wo_color)
    

    kdtree = KDTree(norm_coords_with_color)
    _, indices = kdtree.query(norm_coords_wo_color)

    colored_pc_wo_color = np.hstack((coords_wo_color, colors[indices]))
    return colored_pc_wo_color

def paint_pure_color(pc_wo_color, pure_color, downsample_points=10000):

    pc_wo_color = downsample_point_cloud(pc_wo_color, downsample_points)
    color_points_number = pc_wo_color.shape[0]
    paint_color = np.array([pure_color] * color_points_number).astype(np.float32)
    return np.hstack([pc_wo_color, paint_color])

def paint_plane_color(pc_wo_color, pure_color=(0.73, 0.73, 0.73)):
    paint_color = np.array([pure_color] * pc_wo_color.shape[0]).astype(np.float32)
    return np.hstack([pc_wo_color, paint_color])

def transform_pc(pc, transform):
    points = pc[:, :3]
    other_feature = pc[:, 3:]
    ones = np.ones((pc.shape[0], 1))
    points_h = np.hstack([points, ones])
    points_transformed_h = np.dot(points_h, transform)
    coords_transformed = points_transformed_h[:, :3] / points_transformed_h[:, 3][:, np.newaxis]
    return coords_transformed