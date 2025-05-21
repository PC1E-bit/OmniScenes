import numpy as np
import os
import sys
import open3d as o3d
import json

def vis_pc(pc, color = True):
    if color:
        points = pc[:, :3]  
        colors = pc[:, 3:]  
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        points = pc[:, :3]  
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])

def downsample_point_cloud(point_cloud, target_ratio=0.1):

    target_points = int(point_cloud.shape[0] * target_ratio)
    # if target_points >= point_cloud.shape[0]:
    #     return point_cloud
    indices = np.random.choice(point_cloud.shape[0], target_points, replace=False)
    downsampled_point_cloud = point_cloud[indices]
    return downsampled_point_cloud


def downsample_scene(point_cloud):
    points_number = point_cloud.shape[0]
    target_points = 10000000    # The default sampling point is 10000000.
    if points_number <= target_points:
        return point_cloud
    indices = np.random.choice(point_cloud.shape[0], target_points, replace=False)
    downsampled_point_cloud = point_cloud[indices]
    return downsampled_point_cloud

        
def merge_pcs_from_folder(folder_path):
    all_pcs = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.npy'):
            file_path = os.path.join(folder_path, file_name)
            try:
                pc = np.load(file_path)
            except:
                pass 
            if file_name == "xingzhe10290.npy":
                pc = downsample_point_cloud(pc, 1000)
            print("point cloud shape: ", pc.shape)
            print(f"name: {file_name}")
            print("="*80)
            all_pcs.append(pc)
    merged_pc = np.concatenate(all_pcs, axis=0)
    # save merged pc
    merged_pc_path = os.path.join(folder_path, 'all_scene/merged_pc.npy')
    if not os.path.exists(os.path.dirname(merged_pc_path)):
        os.makedirs(os.path.dirname(merged_pc_path))
        np.save(merged_pc_path, merged_pc)
        print(f"Merged point cloud saved to {merged_pc_path}")
    vis_pc(merged_pc, True)

def pc_norm(pc):
    """ pc: NxC, return NxC """
    xyz = pc[:, :3]
    other_feature = pc[:, 3:]

    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid
    m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
    xyz = xyz / m

    pc = np.concatenate((xyz, other_feature), axis=1)
    return pc, centroid, m

def pc_norm_use_info(pc, centroid, scale):

    xyz = pc[:, :3]
    other_feature = pc[:, 3:]

    xyz = xyz - centroid
    xyz = xyz / scale

    pc = np.concatenate((xyz, other_feature), axis=1)
    return pc

def mesh_norm(mesh):
    """ Normalize the mesh's vertices such that it's centered at origin and fits within a unit sphere """
    vertices = np.asarray(mesh.vertices)
    centroid = np.mean(vertices, axis=0)
    vertices = vertices - centroid
    m = np.max(np.sqrt(np.sum(vertices ** 2, axis=1)))
    vertices = vertices / m
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return mesh, centroid, m

def mesh_norm_use_info(mesh, centroid, scale):
    vertices = np.asarray(mesh.vertices)
    vertices = vertices - centroid
    vertices = vertices / scale
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return mesh



def draw_bbox(server, bbox, color=(1, 0, 0), thickness=0.001):
    min_coords_ori, max_coords_ori = bbox
    bbox_scale = (max_coords_ori - min_coords_ori)
    expand_scale = bbox_scale * 0.01
    min_coords = min_coords_ori - expand_scale
    max_coords = max_coords_ori + expand_scale
    x_min, y_min, z_min = min_coords
    x_max, y_max, z_max = max_coords
    line_points =[]
    vertices = [
        [x_min, y_min, z_min], [x_max, y_min, z_min],
        [x_min, y_max, z_min], [x_max, y_max, z_min],
        [x_min, y_min, z_max], [x_max, y_min, z_max],
        [x_min, y_max, z_max], [x_max, y_max, z_max]
    ]
    edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    for edge in edges:
        start, end = vertices[edge[0]], vertices[edge[1]]
        points_on_edge = np.linspace(start, end, 1000)
        line_points.append(points_on_edge)

    line_points = np.vstack(line_points)
    colors = np.array([color] * line_points.shape[0]).astype(np.float32)
    server.scene.add_point_cloud("bbox", line_points, colors, point_size=thickness)



def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    # print("load json success")
    return data

def save_json(file_path, data):
    new_path = file_path.replace(".json", "_rechck1.json")
    with open(new_path, "w") as f:
        json.dump(data, f, indent=4)

def get_info(file_name, json_info):
    
    for key, values in json_info.items():
        if key == "Lights" or key == "Cameras":
            continue
        for value in values:
            if value["name"] == file_name.split(".")[0]:
                value.update({"category": key})
                # print("get info success")
                return value
    # print("get info failed")
    return None

def modify_json(file_name, json_info, feedback):
    for key, values in json_info.items():
        if key == "Lights" or key == "Cameras":
            continue
        for value in values:
            if value["name"] == file_name.split(".")[0]:
                value.update({"feedback": feedback})
                return json_info


if __name__ == '__main__':

    dir_name = os.path.dirname(os.path.abspath(sys.argv[0]))
    folder_path = os.path.join(dir_name, 'flagged/0091/pcs')
    # test_folder_path = os.path.join(dir_name, 'flagged/ds_pcs')
    npys = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    pts = [f for f in os.listdir(folder_path) if f.endswith('.pt')]
    # test_npys = [f for f in os.listdir(test_folder_path) if f.endswith('.npy')]
    pcs = []
    count = 0
    # for npy in npys:
    for npy in npys:
        # if npy in test_npys:

        pc = np.load(os.path.join(folder_path, npy))
        # pc = pc.numpy()
        pcs.append(pc)
            # vis_pc(pc)
        # scene_path = os.path.join(folder_path, 'all_scene')
        # pc = np.load(os.path.join(scene_path, '一层.npy'))
        print(pc.shape)
        count += 1
        # print()
    pcs = np.concatenate(pcs, axis=0)
    # downsampled_pc = downsample_scene(pc)
    print(f"count: {count}")
    vis_pc(pcs, True)
    # merge_pcs_from_folder(folder_path)