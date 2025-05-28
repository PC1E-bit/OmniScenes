import os
import time
import yaml
import sys
from tqdm import tqdm
import argparse
import shutil
import imageio

import isaacsim
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import omni
from omni.isaac.core.utils.prims import delete_prim, create_prim
from omni.isaac.sensor import Camera
from omni.isaac.core import World
from cleaning_utils.usd_utils import compute_bbox, is_bad_prim

import pxr
from pxr import Usd, UsdGeom, Gf, Sdf
import numpy as np
from PIL import Image
from collections import defaultdict
from scipy.spatial.transform import Rotation as R
import omni.isaac.core.utils.numpy.rotations as rot_utils

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



def get_thumbnails(usd_models_path, dest_path, camera_number=6, get_multi_view=False, render_method="3+3", get_video=False):
    '''
    Args:
        render_method: "3+3" or "4+2"
    
    '''
    def concat_with_borders(images, border_width, border_color):
        vertical_border = np.full(
            (image_height, border_width, 3), border_color, dtype=np.uint8
        )
        row_with_borders = images[0]
        for img in images[1:]:
            row_with_borders = np.hstack((row_with_borders, vertical_border, img))
        return row_with_borders
    
    world = World(stage_units_in_meters=1.0)
    world.reset()
    create_prim(
        prim_path="/World/background",
        usd_path="/ailab/user/caopeizhou/projects/SceneCleaning/background/background1.usda"
    )

    if get_video:
        frame_number = 32
        cameras_video = []
        for i in range(frame_number):
            try:
                delete_prim(f"/World/camera_video_{i}")
            except:
                pass
            camera_video = Camera(
                prim_path = f"/World/camera_video_{i}",
                frequency = 20,
                resolution = (400, 400),
            )
            camera_video.set_clipping_range(0.001, 10000000)
            camera_video.initialize()
            camera_video.add_distance_to_image_plane_to_frame()
            cameras_video.append(camera_video)

        # light_pos_dome = (1, 1, 1)
        # light_pos_distant = (1, 1, 9)
        # create_prim(
        #     prim_path="/World/light_dome",
        #     prim_type="DomeLight",
        #     position=light_pos_dome,
        #     orientation=rot_utils.euler_angles_to_quats(np.array([0.0, 0.0, 0.0]), degrees=True),
        #     attributes={"inputs:intensity": 1000, "inputs:colorTemperature":6250}
        # )

        # create_prim(
        #     prim_path="/World/light_distant",
        #     prim_type="DistantLight",
        #     position=light_pos_distant,
        #     orientation=rot_utils.euler_angles_to_quats(np.array([0.0, 0.0, 0.0]), degrees=True),
        #     attributes={"inputs:intensity": 500, "inputs:colorTemperature":6250}
        # )

        create_prim(
            prim_path="/World/background",
            usd_path="/ailab/user/caopeizhou/projects/SceneCleaning/background/background_2.usda"
        )
    else:
        # setup the cameras
        cameras_33 = []
        cameras_82 = []

        for i in range(camera_number):
            try:
                delete_prim(f"/World/c_{i}_33")
            except:
                pass
            camera_33 = Camera(
                prim_path = f"/World/c_{i}_33",
                frequency = 20,
                resolution = (350, 350),  # If the resolution is too small, the camera will not pick up the color of slightly transparent objects, such as curtains
                # orientation = None
            )
            camera_33.set_clipping_range(0.001, 10000000)
            camera_33.initialize()
            camera_33.add_distance_to_image_plane_to_frame()
            cameras_33.append(camera_33)
            
        for i in range(10):
            try:
                delete_prim(f"/World/c_{i}_82")
            except:
                pass
            camera_82 = Camera(
                prim_path = f"/World/c_{i}_82",
                frequency = 20,
                resolution = (350, 350),  
            )
            camera_82.set_clipping_range(0.001, 10000000)
            camera_82.initialize()
            camera_82.add_distance_to_image_plane_to_frame()
            cameras_82.append(camera_82)




    for model_name in tqdm(os.listdir(usd_models_path)):
        print(model_name)
        model_usd_path = os.path.join(usd_models_path, f'{model_name}/instance.usd')
        show_prim_path = "/World/Show"
        try:
            delete_prim(show_prim_path)
        except:
            pass
        prim = create_prim(show_prim_path, position=(0,0,0), scale=(0.01, 0.01, 0.01), usd_path=model_usd_path)
        stage = omni.usd.get_context().get_stage()
        print(prim.GetPath())
        print([f for f in stage.Traverse()])
        print("="*50)
        prim_check_path = "/World/Show/Instance"
        prim_check = stage.GetPrimAtPath(prim_check_path)
        if is_bad_prim(prim_check):
            print(f"{model_name} is bad, delete and skip.")
            shutil.rmtree(os.path.join(usd_models_path, f'{model_name}'))
            continue
        else:
            print("prim created")


        model_merge_views_path = os.path.join(dest_path, f'merge_views/')
        model_multi_views_33_path = os.path.join(dest_path, f'multi_views_33/{model_name}')
        model_multi_views_82_path = os.path.join(dest_path, f'multi_views_82/{model_name}')
        model_video_path = os.path.join(dest_path, f'videos/{model_name}')
        # check one model is done?
        merge_view_exist = False
        _33_views_exist = False
        _82_views_exist = False
        if not os.path.exists(model_merge_views_path):
            os.makedirs(model_merge_views_path)
        else:
            merge_views = os.listdir(model_merge_views_path)
            merge_image_name = f"{model_name}_{render_method}.png"
            if merge_image_name in merge_views:
                merge_view_exist = True
        if not os.path.exists(model_multi_views_33_path) and get_multi_view:
            os.makedirs(model_multi_views_33_path)
        elif os.path.exists(model_multi_views_33_path):
            images_number = os.listdir(model_multi_views_33_path)
            if len(images_number) == 6:
                _33_views_exist = True
        if not os.path.exists(model_multi_views_82_path) and get_multi_view:
            os.makedirs(model_multi_views_82_path)
        elif os.path.exists(model_multi_views_82_path):
            images_number = os.listdir(model_multi_views_82_path)
            if len(images_number) == 10:
                _82_views_exist = True

        if merge_view_exist and _33_views_exist and _82_views_exist:
            print(f"{model_name} is done, skip...........")
            continue

        if not os.path.exists(model_video_path) and get_video:
            os.makedirs(model_video_path)


        bbox = compute_bbox(prim)
        min_point = np.array(bbox.min)
        max_point = np.array(bbox.max)
        mx,my,mz = max_point - min_point
        center = (min_point + max_point) * 0.5
        max_l = np.sqrt(mx*mx + my*my + mz*mz) / 2
        distance = 1.5 * max_l * 1.732
        


        if get_video:
            for idx, camera in enumerate(cameras_video):
                iter_num = frame_number
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

            start_render = time.time()
            for i in range(15):
                world.step()    
            end_render = time.time()
            print(f"[INFO] The render time for {model_name} is {end_render - start_render} seconds.")

            image_paths = []
            frame_list = []

            for idx, camera_video in enumerate(cameras_video):
                rgb = 1.0 * camera_video.get_rgb() / 255.
                rgb = (rgb * 255).astype(np.uint8)
                image_name = f"{model_name}_{idx}.png"
                output_folder = os.path.join(model_video_path, f"rgb/")
                output_path = os.path.join(model_video_path, f"rgb/{image_name}")
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                Image.fromarray(rgb).save(output_path)
                image_paths.append(output_path)
                frame_list.append(rgb)

            frames = [Image.open(image) for image in image_paths]
            gif_path = os.path.join(model_video_path, f"{model_name}.gif")
            frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0)

            mp4_path = os.path.join(model_video_path, f"{model_name}.mp4")
            fps = 10  
            with imageio.get_writer(mp4_path, fps=fps) as writer:
                for frame in frame_list:
                    writer.append_data(frame)

            for image_path in image_paths:
                os.remove(image_path)
        else:
            # Place the cameras
            for idx, camera in enumerate(cameras_33):
                iter_num = camera_number
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

            for idx, camera in enumerate(cameras_82):
                base_dir = 0
                delta = np.pi/4
                if idx < 8:
                    alpha = base_dir + idx * delta
                    base_vec = np.array([np.cos(alpha), np.sin(alpha), -np.sin(np.pi/4)])
                    camera_position = center - base_vec * (max_l + distance)
                elif idx == 8:
                    camera_position = center - np.array([0, 0, 1.5*max_l + distance])
                    orient = rot_utils.euler_angles_to_quats(np.array([0.0, -90.0, 90.0]), degrees=True)
                    camera.set_world_pose(camera_position, orient)
                    continue
                elif idx ==9:
                    camera_position = center + np.array([0, 0, 1.5*max_l + distance])
                    orient = rot_utils.euler_angles_to_quats(np.array([0.0, 90.0, 90.0]), degrees=True)
                    camera.set_world_pose(camera_position, orient)
                    continue
                else:
                    continue
                orient = compute_camera_quaternion(center, camera_position)
                camera.set_world_pose(camera_position, orient, camera_axes = "usd")
            start_render = time.time()
            for _ in range(10):
                world.step()    
            end_render = time.time()
            print(f"[INFO] The render time for {model_name} is {end_render - start_render} seconds.")

        border_width = 10
        border_color = [255, 255, 255]  

        # Transform purpose
        for idx, camera in enumerate(cameras_82):
            rgb = 1.0 * camera.get_rgb() / 255.
            rgb = (rgb * 255).astype(np.uint8)
            image_name = f"{model_name}_{idx}.png"
            output_path = os.path.join(model_multi_views_82_path, image_name)
            Image.fromarray(rgb).save(output_path)

        # Sementic purpose
        all_images = []
        for idx, camera in enumerate(cameras_33):
            rgb = 1.0 * camera.get_rgb() / 255.
            rgb = (rgb * 255).astype(np.uint8)
            if get_multi_view:
                image_name = f"{model_name}_{idx}.png"
                output_path = os.path.join(model_multi_views_33_path, image_name)
                Image.fromarray(rgb).save(output_path)
            all_images.append(rgb)

        image_height, image_width, _ = all_images[0].shape
        half = len(all_images) // 2
        first_row = all_images[:half]
        second_row = all_images[half:]
        first_row_concat = concat_with_borders(first_row, border_width, border_color)
        second_row_concat = concat_with_borders(second_row, border_width, border_color)
        horizontal_border = np.full(
            (border_width, first_row_concat.shape[1], 3), border_color, dtype=np.uint8
        )
        final_image = np.vstack([first_row_concat, horizontal_border, second_row_concat])
        final_image_pil = Image.fromarray(final_image, 'RGB')
        merge_image_name = f"{model_name}_{render_method}.png"
        output_path = os.path.join(model_merge_views_path, merge_image_name)
        final_image_pil.save(output_path)

        try:
            delete_prim(show_prim_path)
        except:
            pass
    


def move_renders(model_path, dest_dir, image_type="single", move_video=False):
    '''
    Args:
        model_path: the model path
        dest_dir: the scene path
        image_type: "single" or "merge"
    '''

    model_name = os.path.basename(model_path)

    if image_type == 'single':
        dest_path = os.path.join(dest_dir, f"../multi_views/{model_name}")
    elif image_type == 'merge':
        dest_path = os.path.join(dest_dir, "../merge_views")
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    
    merge_views_path = os.path.join(model_path, 'merge_views')
    multi_views_path = os.path.join(model_path, 'multi_views')
    video_path = os.path.join(model_path, 'video')
    # print(image_contents)
    if image_type == "single":
        multi_views = sorted(os.listdir(multi_views_path))
        # print(single_images)    
        for i in range(6):
            dest_image_path = os.path.join(dest_path)
            source_path = os.path.join(multi_views_path, multi_views[i])
            # print(source_path, dest_image_path)
            shutil.copy(source_path, dest_image_path)
    elif image_type == "merge":
        target_image = os.listdir(merge_views_path)[0]
        dest_image_path = os.path.join(dest_path)
        source_path = os.path.join(merge_views_path, target_image)
        shutil.copy(source_path, dest_image_path)
    if move_video:
        video = os.listdir(video_path)[0]
        dest_video_path = os.path.join(dest_dir, '../videos')
        if not os.path.exists(dest_video_path):
            os.makedirs(dest_video_path)
        source_path = os.path.join(video_path, video)
        shutil.copy(source_path, dest_video_path)



if __name__ == '__main__':
    
    # dir_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Path to the grutopia file", default='/ailab/user/caopeizhou/data/GRScenes')
    parser.add_argument("--output_dir", help="Path to where the data should be saved", default="/ailab/user/caopeizhou/projects/SceneCleaning/output/instances")
    parser.add_argument('--scene_id', type=int)
    parser.add_argument('--part', type=int, default=1)
    parser.add_argument('--usd_folder', type=int, required=True)
    parser.add_argument("--use_scene_str", type=bool, required=True)
    parser.add_argument("--scene_str", type=str, required=True)

    args = parser.parse_args()
    
    part_str = f"part{args.part}"
    usd_str = f"{args.usd_folder}_usd"
    data_path = args.data_path
    usd_path = os.path.join(data_path, part_str, usd_str)
    # debug_usd_path = "/home/caopeizhou/projects/SceneCleaning/data/part2/110_usd_cleaned"
    # usd_path = debug_usd_path
    scene_list = sorted(os.listdir(usd_path))
    # scene_id = scene_list[args.scene_id]
    if args.use_scene_str:
        scene_id = args.scene_str
    else:
        scene_id = scene_list[args.scene_id]

    scene_path = os.path.join(usd_path, scene_id)
    print(scene_path)

    # usd_files = [f for f in os.listdir(scene_path) if f.endswith(".usd") or f.endswith(".usda")]
    # usd_file_path = os.path.join(scene_path, usd_files[0])
    models_path = os.path.join(scene_path, "models")
    

    if os.path.exists(models_path):
        output_path = os.path.join(args.output_dir, f"{part_str}/{usd_str}/{scene_id}/models_rendered")
        get_thumbnails(models_path, output_path, get_multi_view=True, render_method="3+3", get_video=False)
        model_rendered_path = output_path
        # for model_name in os.listdir(model_rendered_path):
        #     model_path = os.path.join(model_rendered_path, model_name)
            
        #     move_renders(model_path, output_path, image_type='single', move_video=False)
        #     move_renders(model_path, output_path, image_type='merge', move_video=False)

    exit()
    simulation_app.close()





