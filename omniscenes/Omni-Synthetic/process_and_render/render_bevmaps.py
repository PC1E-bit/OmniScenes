import argparse
import omni
import json
import os
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})
from pxr import Usd, UsdGeom, UsdLux
import numpy as np
import open3d as o3d
import cv2

import sys
from urllib.parse import quote
from scene_utils.usd_utils import remove_empty_prims, recursive_parse_new, get_mesh_from_points_and_faces, sample_points_from_mesh, sample_points_from_prim
from scene_utils.usd_utils import fix_mdls
from scene_utils.usd_utils import IsEmpty, IsObjXform

from omni.isaac.sensor import Camera
from omni.isaac.core import World

from omni.isaac.core.utils.stage import open_stage
from omni.isaac.core.utils.stage import add_reference_to_stage
import omni.isaac.core.utils.numpy.rotations as rot_utils
from omni.isaac.core.utils.prims import delete_prim, create_prim
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
from math import ceil
from scene_utils.usd_utils import filter_free_noise, strip_world_prim, enumerate_lights, turnoff_original_lights
from scene_utils.geometry_tools import extract_floor_heights, fix_floorheight, generate_intrinsic, build_transformation_mat
# turn on the camera light
# import omni.kit.actions.core
# action_registry = omni.kit.actions.core.get_action_registry()
# action = action_registry.get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_camera")
# action.execute()



parser = argparse.ArgumentParser()
parser.add_argument("--scene_start_index", type=int, default=1)
parser.add_argument("--scene_end_index", type=int, default=2)
parser.add_argument("--scene_id", type=int, default=1)
parser.add_argument("--data_path", help="Path to the grutopia file", default='/ailab/user/caopeizhou/data/GRScenes')
parser.add_argument("--output_dir", help="Path to where the data should be saved", default="/ailab/user/caopeizhou/projects/GRRegion/output")
parser.add_argument("--part", type=int, default=1)
parser.add_argument("--usd_folder", type=int, default=1)
parser.add_argument("--mdl_path", help="Path to the mdl file", default='/ailab/user/caopeizhou/projects/GRRegion/mdls/default.mdl')
parser.add_argument("--use_scene_str", type=bool, required=True)
parser.add_argument("--scene_str", type=str)

args = parser.parse_known_args()[0]






# omni.kit.commands.execute('ChangeSetting', path='/rtx/post/histogram/enabled', value=True)
# omni.kit.commands.execute('ChangeSetting', path='/rtx/post/histogram/whiteScale', value=8.5)
# omni.kit.commands.execute('ChangeSetting', path='/rtx/post/histogram/minEV', value=500.0)




def find_path(dir, endswith):
    paths = sorted(os.listdir(dir))
    for p in paths:
        if endswith in p and 'copy' not in p:
            target_path = os.path.join(dir,p)
            if endswith == '.usd':
                target_path = quote(target_path, safe=':/')
            return target_path


def convert_usd_to_points(stage, meters_per_unit, json_data, use_json_data=True):
    remove_empty_prims(stage)
    world_prim = stage.GetPrimAtPath("/World/scene")
    scene_root = strip_world_prim(world_prim)
    noise_list = []
    if use_json_data:
        for prim_info in json_data:
            prim_name = prim_info['name'].replace("_MightBeGlass", "")
            prim_type = prim_info['feedback']
            if prim_type == 'confirm':
                noise_list.append(prim_name)
        print(noise_list)
    prims_all = [p for p in scene_root.GetAllChildren() if p.IsA(UsdGeom.Mesh) or p.IsA(UsdGeom.Xform) and not IsEmpty(p) and IsObjXform(p)]
    pcs_all = []
    sample_points_number = 100000
    for prim in prims_all:
        if prim.GetName() in noise_list:
            continue
        try:
            pcs, mesh = sample_points_from_prim(prim, sample_points_number)
            pcs_all.append(pcs)
            print(prim.GetName())
        except:
            prims_all.remove(prim)
            continue
    pcs_all = np.concatenate(pcs_all, axis=0) * meters_per_unit
    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(pcs_all)
    scene_pcd.colors = o3d.utility.Vector3dVector(np.ones_like(pcs_all)*0.4)
    scene_pcd = scene_pcd.voxel_down_sample(0.05)
    return scene_pcd, prims_all

def get_best_resoluiton(floor_width, floor_height, meters_per_unit):
    if meters_per_unit != 0.001:
        meters_per_unit = 0.001
    floor_width_mm = floor_width / meters_per_unit
    floor_height_mm = floor_height / meters_per_unit
    downsample_scale = 0.1
    width = np.floor(np.floor(floor_width_mm) * downsample_scale)
    height = np.floor(np.floor(floor_height_mm) * downsample_scale)
    return (int(width), int(height))


# data path and loop
# data_path = '/ailab/user/caopeizhou/data/GRScenes/part2/110_usd'
# output_path = '/ailab/user/caopeizhou/projects/GRRegion/output'
# mdl_path = '/ailab/user/caopeizhou/projects/GRRegion/mdls/default.mdl'

# all_list = sorted(os.listdir(data_path))
# loop_list = all_list
usd_folder = f"{args.usd_folder}_usd"
part_folder = f"part{args.part}"
data_path = os.path.join(args.data_path, part_folder, usd_folder)
output_path = os.path.join(args.output_dir, part_folder, usd_folder)
# if args.scene_end_index > len(os.listdir(data_path)):
#     args.scene_end_index = len(os.listdir(data_path))
if args.use_scene_str:
    house_id = args.scene_str
else:
    house_id = sorted(os.listdir(data_path))[args.scene_id]

if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)

mdl_path = args.mdl_path
# for house_id in house_ids:
print(house_id)
if not os.path.exists("%s/%s"%(output_path, house_id)):
    usd_path = find_path(os.path.join(data_path, house_id), endswith='.usd')
    usd_file_name = os.path.split(usd_path)[-1].replace(".usd", "")
    json_path = find_path(os.path.join(data_path, house_id), endswith='.json')
    json_data = None
    if json_path is not None:
        print(json_path)
        with open(json_path, 'r') as f:
            json_data = json.load(f)
    print(usd_path)
    world = World(stage_units_in_meters=1.0)
    fix_mdls(usd_path, mdl_path)
    add_reference_to_stage(usd_path, "/World/scene")
    stage = omni.usd.get_context().get_stage()
    meters_per_unit = Usd.Stage.Open(usd_path).GetMetadata('metersPerUnit')
    print(meters_per_unit)
    enumerate_lights(stage)
    # turnoff_original_lights(stage)
    scene_pcd_without_labeled_noise, prims_all = convert_usd_to_points(stage, meters_per_unit, json_data, False)
    scene_pcd_without_xy_free_noise = filter_free_noise(scene_pcd_without_labeled_noise)    
    scene_pcd_without_free_noise = filter_free_noise(scene_pcd_without_xy_free_noise, plane='xz')     
    scene_pcd = scene_pcd_without_free_noise
    points = np.array(scene_pcd.points)
    floor_heights = extract_floor_heights(points)
    # initialize the camera function
    camera_bev = Camera(
        prim_path="/World/camera",
        position=np.array([0,0,0]),
        # dt=0.05,
        resolution=(1280, 720),
        orientation=rot_utils.euler_angles_to_quats(np.array([0.0, 90.0, 90.0]), degrees=True))  # YXZ
    
    camera_sample = Camera(
        prim_path = "/World/camera_sample",
        position=np.array([0,0,0]),
        dt=0.05,
        resolution=(800, 600),
        orientation=rot_utils.euler_angles_to_quats(np.array([0.0, 0.0, 90.0]), degrees=True)        
    )

    # camera_bev.set_focal_length(1.4)
    # camera_bev.set_focus_distance(0.205)
    camera_bev.set_clipping_range(0.0001,10000000)
    world.reset()
    camera_bev.initialize()
    camera_bev.add_motion_vectors_to_frame()
    camera_bev.add_distance_to_image_plane_to_frame()
    # for i in range(30):
    #     world.step(render=True)
    camera_sample.set_focal_length(1)
    camera_sample.set_clipping_range(0.001, 10000000)
    camera_sample.initialize()
    camera_sample.add_distance_to_image_plane_to_frame()

    camera_height = 1.8
    os.makedirs("%s/%s/"%(output_path, house_id), exist_ok=False)
    for floor_index, floor in enumerate(floor_heights):

        current_floor = floor[0]
        current_floor = fix_floorheight(current_floor, prims_all, meters_per_unit)
        floor_upper_bound = current_floor + 0.1
        floor_lower_bound = current_floor - 0.1
        floor_points = points[np.where((points[:,2]<floor_upper_bound) & (points[:,2]>floor_lower_bound))]
        floor_xy = floor_points[:,:2]
        floor_pcd = o3d.geometry.PointCloud()
        floor_pcd.points = o3d.utility.Vector3dVector(floor_points)

        floor_x_max, floor_x_min = np.max(floor_xy[:,0]), np.min(floor_xy[:,0])
        floor_y_max, floor_y_min = np.max(floor_xy[:,1]), np.min(floor_xy[:,1])
        floor_x_center = (floor_x_max + floor_x_min) / 2
        floor_y_center = (floor_y_max + floor_y_min) / 2
        floor_width = floor_x_max - floor_x_min
        floor_height = floor_y_max - floor_y_min
        light_scale = max(floor_width, floor_height) / meters_per_unit * 2
        


        # BEV camera setting
        camera_bev.set_projection_mode("orthographic")
        scale = 1.0
        camera_bev.set_horizontal_aperture(floor_width / meters_per_unit * scale)
        width, height = get_best_resoluiton(floor_width, floor_height, meters_per_unit)
        camera_bev.set_resolution((width, height))
        # BEV camera position
        bev_camera_translation = np.array([floor_x_center, floor_y_center, current_floor + camera_height])
        bev_camera_rotation = [0, 0, 0]
        bev_camera_extrinsic = build_transformation_mat(bev_camera_translation, bev_camera_rotation)
        bev_camera_pos = bev_camera_extrinsic[0:3,3]/meters_per_unit
        if bev_camera_pos[2] != camera_height * 1000:
            bev_camera_pos[2] = ceil(bev_camera_pos[2])
        print(bev_camera_pos)
        light_pos = np.array([bev_camera_extrinsic[0,3], bev_camera_extrinsic[1,3], bev_camera_extrinsic[2,3] - 0.3]) / meters_per_unit
        print(light_pos)
        print(light_scale)
        bev_camera_rot = bev_camera_extrinsic[0:3,0:3]
        camera_bev.set_world_pose(bev_camera_pos)
        # BEV light setting
        light_path = "/World/light"
        create_prim(
            prim_path=light_path,
            prim_type="RectLight",
            position=light_pos,
            orientation=rot_utils.euler_angles_to_quats(np.array([0.0, 0.0, 0.0]), degrees=True),
            attributes={"inputs:width": light_scale, "inputs:height": light_scale, "inputs:intensity": 2000}
        )

        for i in range(20):
            world.step(render=True)

        rgb_bev = cv2.cvtColor(camera_bev.get_rgba()[:,:,:3],cv2.COLOR_BGR2RGB)
        
        
        os.makedirs("%s/%s/bevmap_%d/"%(output_path, house_id, floor_index), exist_ok=False)
        cv2.imwrite(os.path.join(output_path, house_id, f'bevmap_{floor_index}/', f"bev_map.jpg"), rgb_bev)
        # o3d.io.write_point_cloud(os.path.join(output_path, house_id, "bevmap", f"bevmap_{floor_index}.ply"), floor_pcd)

        # delete_prim(light_path)

        # sample camera setting
        # turn on the camera light
        # action_registry = omni.kit.actions.core.get_action_registry()
        # action = action_registry.get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_stage")
        # action.execute()

        # omni.kit.commands.execute('ChangeSetting', path='/rtx/post/histogram/enabled', value=True)
        # omni.kit.commands.execute('ChangeSetting', path='/rtx/post/histogram/whiteScale', value=4.5)

        # sample camera position
        downsampled_floor_pcd = floor_pcd.voxel_down_sample(2)
        render_sample_points_xy = np.array(downsampled_floor_pcd.points)[:, :2]
        render_sample_points = np.concatenate((render_sample_points_xy, np.ones((render_sample_points_xy.shape[0], 1)) * bev_camera_translation[2]), axis=-1)
        camera_rotations = []
        delta = 45.0 
        iter_number = 360.0 / delta
        base_rotation = np.array([25.0, 0.0, 0.0])
        for i in range(int(iter_number)):
            rotation = base_rotation.copy()
            rotation[2] = base_rotation[2] + i * delta
            # camera_rotations.append(rotation)
            print(rotation)
            camera_ext = build_transformation_mat(np.array([0, 0, 0]), rotation)
            camera_rot = camera_ext[0:3,0:3]
            camera_euler_angles = rot_utils.rot_matrices_to_quats(camera_rot)
            camera_euler_angles = rot_utils.quats_to_euler_angles(camera_euler_angles)
            camera_euler_angles[0],camera_euler_angles[1],camera_euler_angles[2] = camera_euler_angles[1],np.clip((np.pi/2 - camera_euler_angles[0])/2.0,0,np.pi/8),camera_euler_angles[2]+np.pi/2
            camera_rotations.append(camera_euler_angles)

        os.makedirs("%s/%s/regions_%d/"%(output_path, house_id, floor_index), exist_ok=True)
        sample_points = []
        for idx, render_point in enumerate(render_sample_points):
            os.makedirs("%s/%s/regions_%d/sample_%d"%(output_path, house_id, floor_index, idx), exist_ok=False)
            print(f"sample idx {idx}")
            sample_points.append(render_point.tolist())
            camera_pos = render_point / meters_per_unit
            for rot_idx, camera_rotation in enumerate(camera_rotations):
                print(rot_idx, camera_rotation)
            # orient = compute_camera_quaternion(target_points[idx], render_point)
            # print(target_points[idx], render_point)
                camera_sample.set_world_pose(camera_pos, rot_utils.euler_angles_to_quats(camera_rotation))
                for i in range(30):
                    world.step(render=True)
                render_rgb = cv2.cvtColor(camera_sample.get_rgba()[:,:,:3], cv2.COLOR_BGR2RGB)
                cv2.imwrite(os.path.join(output_path, house_id, f"regions_{floor_index}/sample_{idx}/render_{rot_idx}.jpg"), render_rgb)
        save_dict = {"floor_height": floor, "sample_points": sample_points, "bev_camera_translation": bev_camera_translation.tolist()}
        json_object = json.dumps(save_dict, indent=4)
        with open("%s/%s/camera_info_%d.json"%(output_path, house_id, floor_index), "w") as outfile:
            outfile.write(json_object)
        delete_prim("/World/light")


