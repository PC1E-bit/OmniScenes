import os
import time
import yaml
import sys
import argparse
import shutil
from tqdm import tqdm
from urllib.parse import quote
from collections import defaultdict

import imageio
import isaacsim
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})
import omni
import omni.isaac.core.utils.numpy.rotations as rot_utils
from omni.isaac.core.utils.prims import delete_prim, create_prim
from omni.isaac.sensor import Camera
from omni.isaac.core import World
import pxr
import open3d as o3d
import trimesh
from pxr import Usd, UsdGeom, Gf, Sdf
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

from scene_utils.usd_tools import compute_bbox, is_bad_prim
from scene_utils.usd_tools import get_prims_wo_structure, sample_points_from_prim, get_transform_from_prim, is_all_light_xform
from scene_utils.pcd_tools import norm_coords, paint_color, paint_pure_color, paint_plane_color, transform_pc, downsample_mesh_naive

# omni.kit.commands.execute('ChangeSetting', path='/rtx/post/histogram/enabled', value=True)
# omni.kit.commands.execute('ChangeSetting', path='/rtx/post/histogram/whiteScale', value=5.5)
# # omni.kit.commands.execute('ChangeSetting', path='/rtx/post/histogram/minEV', value=500.0)


def compute_camera_quaternion(target_position, camera_position, up_vector=np.array([0, 0, 1])):
    forward_vector = np.array(camera_position) - np.array(target_position)
    forward_vector = forward_vector / np.linalg.norm(forward_vector)  
    # default_forward = np.array([0, 0, 1])
    right_vector = np.cross(up_vector, forward_vector)
    right_vector = right_vector / np.linalg.norm(right_vector)
    corrected_up_vector = np.cross(forward_vector, right_vector)
    corrected_up_vector = corrected_up_vector / np.linalg.norm(corrected_up_vector)
    rotation_matrix = np.stack([right_vector, corrected_up_vector, forward_vector], axis=1)
    quaternion = R.from_matrix(rotation_matrix).as_quat()
    return np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])


def compute_camera_quaternion_pcd(camera_position, target_position):
    forward_vector = np.array(target_position) - np.array(camera_position)
    forward_vector = forward_vector / np.linalg.norm(forward_vector)  # Normalize the vector
    default_forward = np.array([0, 0, 1])
    rotation_axis = np.cross(default_forward, forward_vector)
    rotation_axis_norm = np.linalg.norm(rotation_axis)
    if rotation_axis_norm < 1e-6:
        if np.dot(default_forward, forward_vector) < 0:
            quaternion = R.from_rotvec(np.pi * np.array([1, 0, 0])).as_quat()
        else:
            quaternion = np.array([0, 0, 0, 1])  # Identity quaternion
    else:
        rotation_angle = np.arccos(np.dot(default_forward, forward_vector))
        rotation_axis = rotation_axis / rotation_axis_norm
        quaternion = R.from_rotvec(rotation_angle * rotation_axis).as_quat()
    return np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])



def render_thumbnails(model_path, model_name, output_dir, render_method, world, cameras):
    '''
    Render the thumbnails of given model, save the images or video.
    Input: The usd model path
    Output: The rendered images or video.

    Args:
        model_path: the path of the model usd file
        model_name: the name of the model, debugging purpose only
        output_dir: the output directory to save the rendered images or video
        render_method: the method to render the images or video, 'images' or 'video'
        world: the initialized world
        cameras: the initialized cameras
    '''
    render_and_save = False    

    # Parse the model path to get the index
    relative_path = model_path.split('rawUsd/')[1]
    scene_index = relative_path.split('/models/')[0]

    # Load the model into the world
    show_prim_path = "/World/Show"
    try:
        delete_prim(show_prim_path)
    except:
        pass
    prim = create_prim(show_prim_path, position=(0,0,0), scale=(1, 1, 1), usd_path=model_path)
    stage = omni.usd.get_context().get_stage()
    prim_check_path = "/World/Show/Instance"
    prim_check = stage.GetPrimAtPath(prim_check_path)
    print(prim.GetPath())
    if is_bad_prim(prim_check) or is_all_light_xform(prim_check):
        print(f"{model_name} is bad, delete and skip.")
        print(f"Delete {os.path.dirname(model_path)}")
        return render_and_save
    else:
        print("prim created")
    
    # Get the model's bbox to place the cameras
    bbox = compute_bbox(prim)
    min_point = np.array(bbox.min)
    max_point = np.array(bbox.max)
    mx,my,mz = max_point - min_point
    center = (min_point + max_point) * 0.5
    max_l = np.sqrt(mx*mx + my*my + mz*mz) / 2
    distance = 1.5 * max_l * 1.732

    # Render and save
    if render_method == 'images':
        # Place the cameras
        for idx, camera in enumerate(cameras):
            base_dir = 0
            delta = np.pi/4
            if idx < 8:
                alpha = base_dir + idx * delta
                base_vec = np.array([np.cos(alpha), np.sin(alpha), -np.sin(np.pi/4)])
                camera_position = center - base_vec * (max_l + distance)
            elif idx < 12:
                alpha = np.pi/4 + (idx - 8) * delta * 2 
                base_vec = np.array([np.cos(alpha), np.sin(alpha), -np.sin(np.pi/4)])
                camera_position = center + base_vec * (max_l + distance)
            elif idx == 12:
                camera_position = center - np.array([0, 0, 1.5*max_l + distance])
                orient = rot_utils.euler_angles_to_quats(np.array([0.0, -90.0, 90.0]), degrees=True)
                camera.set_world_pose(camera_position, orient)
                continue
            elif idx == 13:
                camera_position = center + np.array([0, 0, 1.5*max_l + distance])
                orient = rot_utils.euler_angles_to_quats(np.array([0.0, 90.0, 90.0]), degrees=True)
                camera.set_world_pose(camera_position, orient)
                continue
            orient = compute_camera_quaternion(center, camera_position)
            camera.set_world_pose(camera_position, orient, camera_axes = "usd")

        # Render
        start_render = time.time()
        for _ in range(10):
            world.step()    
        end_render = time.time()
        print(f"[INFO] The render time for {model_name} in {scene_index} is {end_render - start_render:.3f} seconds.")

        # Save the images
        for idx, camera in enumerate(cameras):
            rgb = 1.0 * camera.get_rgb() / 255.
            rgb = (rgb * 255).astype(np.uint8)
            image_name = f"{model_name}_{idx}.png"
            output_path = os.path.join(output_dir, image_name)
            Image.fromarray(rgb).save(output_path)

    elif render_method == 'video':
        # Place the cameras
        for idx, camera in enumerate(cameras):
            iter_num = len(cameras)
            base_dir = np.pi/(iter_num/2)
            delta = 4*np.pi/iter_num
            if idx < iter_num/2:
                alpha = base_dir + idx * delta
                base_vec = np.array([np.cos(alpha), np.sin(alpha), -np.sin(np.pi/4)])
                camera_position = center - base_vec * (max_l + distance)
            else:
                alpha = base_dir + (idx - iter_num/2) * delta
                base_vec = np.array([np.cos(alpha), np.sin(alpha), -np.sin(np.pi/4)])
                camera_position = center + base_vec * (max_l + distance)
            orient = compute_camera_quaternion(center, camera_position)
            camera.set_world_pose(camera_position, orient, camera_axes = "usd")

        # Render
        start_render = time.time()
        for i in range(15):
            world.step()    
        end_render = time.time()
        print(f"[INFO] The render time for {model_name} in {scene_index} is {end_render - start_render:.3f} seconds.")

        # Save the video
        image_paths = []
        frame_list = []
        for idx, camera_video in enumerate(cameras):
            rgb = 1.0 * camera_video.get_rgb() / 255.
            rgb = (rgb * 255).astype(np.uint8)
            image_name = f"{model_name}_{idx}.png"
            output_folder = os.path.join(output_dir, f"rgb/")
            output_path = os.path.join(output_dir, f"rgb/{image_name}")
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            Image.fromarray(rgb).save(output_path)
            image_paths.append(output_path)
            frame_list.append(rgb)
        gif_path = os.path.join(output_dir, f"{model_name}.gif")
        frames = [Image.open(image) for image in image_paths]
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0)

        # mp4 format
        # mp4_path = os.path.join(output_dir, f"{model_name}.mp4")
        # fps = 10  
        # with imageio.get_writer(mp4_path, fps=fps) as writer:
        #     for frame in frame_list:
        #         writer.append_data(frame)

        # for image_path in image_paths:
        #     os.remove(image_path)

    delete_prim(show_prim_path)
    render_and_save = True
    return render_and_save

def get_thumbnails_pipeline(usd_models_path, dest_dir, render_method):

    '''
    The pipeline of getting the thumbnails. Loop processing all the models in the given path.
    Args:
        usd_models: The path contains all instance models in one scene
        dest_dir: The output directory to save the rendered images or video
    '''    

    # Cameras configs
    camera_configs = {
        "images":{
            'count': 14,  # 8(Downward-angles) + 4(Upward-angles) + 2(Top-Down)
            'resolution': (350, 350),
            'clipping_range': (0.001, 10000000),
            'frequency': 20,
            'extra_features': []  
        },
        "video":{
            'count': 32,
            'resolution': (400, 400),
            'clipping_range': (0.001, 10000000),
            'frequency': 20
        }
    }

    # Environment configs
    env_configs = {
        "images":{
            'backgorund_usda_path': "./background/background1.usda",
        },
        "video":{
            'backgorund_usda_path': "./background/background1.usda",
        }
    }

    # 1. Set up the world and cameras
    world = World()
    world.reset()
    create_prim(
        prim_path="/World/background",
        usd_path=env_configs[render_method]['backgorund_usda_path']
    )
    cameras = []
    camera_config = camera_configs[render_method]
    for i in range(camera_config['count']):
        camera = Camera(
            prim_path=f"/World/cam{i}",
            frequency=camera_config['frequency'],
            resolution=camera_config['resolution'],
        )
        camera.set_clipping_range(*camera_config['clipping_range'])
        camera.initialize()
        camera.add_distance_to_image_plane_to_frame()
        cameras.append(camera)

    # 2. Load the models
    for model_name in tqdm(os.listdir(usd_models_path)):
        # check whether the model has been rendered
        output_folder = "multi_views" if render_method != "video" else "videos"
        output_files_number = camera_config["count"] if render_method != "video" else 2
        output_path = os.path.join(dest_dir, f'{output_folder}/{model_name}')
        if os.path.exists(output_path) and len(os.listdir(output_path)) == output_files_number:
            print(f"[INFO] The {model_name} has been rendered, skip it!")
            continue
        else:
            os.makedirs(output_path, exist_ok=True)
        
        # render the thumbnails
        model_usd_path = os.path.join(usd_models_path, f'{model_name}/instance.usd')
        if not os.path.exists(model_usd_path):
            continue

        success = render_thumbnails(model_usd_path, model_name, output_path, render_method, world, cameras)
        if not success:
            print(f"[INFO] The {model_name} cannot be rendered, delete and skip.")
            shutil.rmtree(output_path)
            origin_model_path = os.path.dirname(model_usd_path)
            try:
                shutil.rmtree(origin_model_path)
            except OSError as e:
                print(f"[ERROR] {e}")

    delete_prim("/World/background")

        


        
def render_pcd_with_color(model_path, model_name, world, cameras):
    '''
    Args:
        model_path: the path of the model usd file
        model_name: the name of the model, debugging purpose only
        world: the simulation world
        cameras: the list of cameras to capture the point cloud data
    Returns:
        The painted point cloud data, centered at the origin.
    '''
    # Parse the model path to get the index
    relative_path = model_path.split('rawUsd/')[1]
    scene_index = relative_path.split('/models/')[0]


    # Load the model into the world
    show_prim_path = "/World/Show"
    prim = create_prim(show_prim_path, position=(0,0,0), scale=(1, 1, 1), usd_path=model_path)
    stage = omni.usd.get_context().get_stage()
    prim_check_path = "/World/Show/Instance"
    prim_check = stage.GetPrimAtPath(prim_check_path)
    print(os.path.dirname(model_path))
    if is_bad_prim(prim_check) or is_all_light_xform(prim_check):
        print(f"{model_name} is bad, delete and skip.")
        # shutil.rmtree(os.path.dirname(model_path))
        print(f"Delete {os.path.dirname(model_path)}")
        return None
    else:
        print("prim created")

    # Compute the bounding box of the model
    bbox = compute_bbox(prim)
    min_point = np.array(bbox.min)
    max_point = np.array(bbox.max)
    mx,my,mz = max_point - min_point
    center = (min_point + max_point) * 0.5
    max_l = np.sqrt(mx*mx + my*my + mz*mz) / 2
    distance = 1.3 * max_l * 1.732
    cameras_number = len(cameras)
    base_dir = np.pi/(cameras_number/2)
    delta = 4 * np.pi/cameras_number

    # Place the cameras
    for idx, camera in enumerate(cameras):
        if idx < cameras_number/2:
            alpha = base_dir + idx * np.pi * (2/(cameras_number/2))
            base_vec = np.array([np.cos(alpha), np.sin(alpha), -np.sin(np.pi/4)])
            center_idx = center - base_vec * (max_l + distance)
        else:
            alpha = base_dir + (idx - cameras_number/2) * np.pi * (2/(cameras_number/2))
            base_vec = np.array([np.cos(alpha), np.sin(alpha), -np.sin(np.pi/4)])
            center_idx = center + base_vec * (max_l + distance)
        orient = compute_camera_quaternion_pcd(center, center_idx)
        camera.set_world_pose(center_idx, orient, camera_axes = "usd")

    # Render
    start_render = time.time()
    for _ in range(10):
        world.step()    
    end_render = time.time()
    print(f"[INFO] The render time of pcd for {model_name} in {scene_index} is {end_render - start_render:.3f} seconds.")

    # Merge the captured pcd
    pc_all = []
    color_all = []
    for camera in cameras:
        rgb = 1.0 * camera.get_rgb() / 255.
        pc = camera.get_pointcloud()
        depth = camera.get_depth()

        depth = np.array(depth)
        depth_flatten = depth.ravel()
        colors = rgb.reshape((-1, 3))
        valid_mask = depth_flatten != np.inf
        pc = [_ for _ in pc[valid_mask]]
        colors = [(r, g, b) for r,g,b in colors[valid_mask]]
        # print(colors)
        pc_all += pc
        color_all += colors
    pc_all = np.array(pc_all)
    color_all = np.array(color_all)
    pcd_with_color = np.concatenate([pc_all, color_all], -1)
    
    # Delete the prim
    delete_prim(show_prim_path)

    return pcd_with_color


def get_painted_pcd_pipeline(usd_models_path, scene_usd_file_path, dest_path, painting=True):

    # 0. (Optional) Group the duplicate prims

    # 1. Set up the world and cameras
    world = World()
    world.reset()
    create_prim(
        prim_path="/World/background",
        usd_path="/cpfs/user/caopeizhou/projects/OmniScenes/omniscenes/Omni-Synthetic/process_and_render/background/background1.usda"
    )
    cameras = []
    cameras_number = 6
    for i in range(cameras_number):
        camera = Camera(
            prim_path = f"/World/c_{i}",
            frequency = 20,
            resolution = (300, 300),  # If the resolution is too small, the camera will not pick up the color of slightly transparent objects, such as curtains
            # orientation = None
        )
        camera.set_clipping_range(0.000001, 100000000)
        camera.initialize()
        camera.add_distance_to_image_plane_to_frame()
        cameras.append(camera)

    # 2. Get the original prim from the scene usd file
    scene_stage = Usd.Stage.Open(scene_usd_file_path)
    instance_prims = get_prims_wo_structure(scene_stage)

    # 3. Get colored pcd from render_pcd_with_color()
    pcs_output_dir = os.path.join(dest_path, "pcs")
    meshes_output_dir = os.path.join(dest_path, "meshes")

    for model_name in tqdm(os.listdir(usd_models_path)):
        pcs_output_path = os.path.join(pcs_output_dir, f"{model_name}.npy")
        if os.path.exists(pcs_output_path):
            print(f"[INFO] The {model_name} has been rendered, skip it!")
            continue
        
        meshes_output_path = os.path.join(meshes_output_dir, f"{model_name}.ply")
        if os.path.exists(meshes_output_path):
            print(f"[INFO] The mesh of {model_name} has been generated, skip it!")
            continue

        model_usd_path = os.path.join(usd_models_path, f'{model_name}/instance.usd')
        if not os.path.exists(model_usd_path):  
            continue
        pcd_with_color = render_pcd_with_color(model_usd_path, model_name, world, cameras)
        print(f"[INFO] Rendering data shape: {pcd_with_color.shape}")
        # debug 
        # raw_render_npy_path = os.path.join(dest_path, f"{model_name}_raw.npy")
        # np.save(raw_render_npy_path, pcd_with_color)

        is_glass = False
        if pcd_with_color.shape[0] == 0:
            is_glass = True

        # find the corresponding prim in the scene usd file
        prim_from_scene = None
        for prim in instance_prims:
            if prim.GetName() == model_name:
                prim_from_scene = prim
                break
        if prim_from_scene:
            points_from_usd, mesh_from_usd = sample_points_from_prim(prim_from_scene, num_points=15000) 
            mesh_from_usd = trimesh.Trimesh(vertices=np.asarray(mesh_from_usd.vertices), faces=np.asarray(mesh_from_usd.triangles))
            # downsampled_mesh = downsample_mesh_naive(mesh_from_usd)
            downsampled_mesh = mesh_from_usd
            print(f"Downsample mesh from {len(mesh_from_usd.triangles)} to {len(downsampled_mesh.triangles)} triangles")
            if is_glass:
                glass_color = (0.863, 0.957, 0.961)
                painted_pcd = paint_pure_color(points_from_usd, glass_color, downsample_points = 15000)
                painted_pcd_float32 = painted_pcd.astype(np.float32)
                np.save(pcs_output_path, painted_pcd_float32)

                colored_vertices = paint_pure_color(np.asarray(downsampled_mesh.vertices), glass_color, downsample_points = 15000)
                downsampled_mesh.visual.vertex_colors = colored_vertices[:, 3:]
                downsampled_mesh.export(meshes_output_path)

            else:
                transform_matrix = get_transform_from_prim(prim_from_scene)
                pcd_wo_color = points_from_usd.copy()
                # norm_pcd_wo_color = norm_coords(pcd_wo_color)
                transformed_pcd_with_color = transform_pc(pcd_with_color, transform_matrix)
                norm_transformed_coords = norm_coords(transformed_pcd_with_color)
                norm_pcd_with_color = np.concatenate((norm_transformed_coords, pcd_with_color[:, 3:]), axis = 1)
                painted_pcd_wo_color = paint_color(pcd_wo_color, norm_pcd_with_color)
                painted_data_float32 = painted_pcd_wo_color.astype(np.float32)

                colored_vertices = paint_color(np.asarray(downsampled_mesh.vertices), norm_pcd_with_color)
                downsampled_mesh.visual.vertex_colors = colored_vertices[:, 3:]

                # save
                np.save(pcs_output_path, painted_data_float32)
                downsampled_mesh.export(meshes_output_path)

    delete_prim("/World/background")




if __name__ == '__main__':
    
    # dir_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Path to the grutopia file", default='/cpfs/user/caopeizhou/data/3dassets/rawUsd')
    parser.add_argument("--output_dir", help="Path to where the data should be saved", default="/cpfs/user/caopeizhou/data/3dassets/rendered")
    parser.add_argument('--part', type=int, default=1, required=True)
    parser.add_argument('--usd', type=int, required=True)
    parser.add_argument("--scene", type=str, default=None)

    args = parser.parse_args()
    
    part_str = f"part{args.part}"
    usd_str = f"{args.usd}_usd"
    data_path = args.data_path
    usd_path = os.path.join(data_path, part_str, usd_str)


    if args.scene:
        scene_ids = [args.scene]
        print(f"[INFO] The scene idnex is specified, rendering scene {args.scene}.")
    else:
        scene_ids = sorted(os.listdir(usd_path))
        print(f"[INFO] The scene idnex is not specified, rendering all scenes in {usd_path}.")


    for scene_id in scene_ids:
        scene_path = os.path.join(usd_path, scene_id)
        print(scene_path)
        models_path = os.path.join(scene_path, "models")
        clean_usd_path = os.path.join(scene_path, ([f for f in os.listdir(scene_path) if f.endswith("_copy.usd")][0]))
        
        if os.path.exists(models_path):
            thumbnails_output_path = os.path.join(args.output_dir, f"{part_str}/{usd_str}/{scene_id}/thumbnails")
            multiviews_output_path = os.path.join(args.output_dir, f"{part_str}/{usd_str}/{scene_id}/thumbnails/multi_views")
            if os.path.exists(thumbnails_output_path) and len(os.listdir(multiviews_output_path)) == len(os.listdir(models_path)):
                print(f"[INFO] The thumbnails of {scene_id} has been rendered, skip it!")
                # continue
            else:
                os.makedirs(thumbnails_output_path, exist_ok=True)
                get_thumbnails_pipeline(models_path, thumbnails_output_path, render_method="images")

            output_path = os.path.join(args.output_dir, f"{part_str}/{usd_str}/{scene_id}")
            pcs_output_path = os.path.join(output_path, "pcs")
            meshes_output_path = os.path.join(output_path, "meshes")
            if os.path.exists(pcs_output_path) and len(os.listdir(pcs_output_path)) == len(os.listdir(models_path)):
                print(f"[INFO] The pcs of {scene_id} has been rendered, skip it!")
            else:
                os.makedirs(pcs_output_path, exist_ok=True)
                os.makedirs(meshes_output_path, exist_ok=True)
                get_painted_pcd_pipeline(models_path, clean_usd_path, output_path)

    
    simulation_app.close()
    exit()




