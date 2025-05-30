import sys
sys.path.extend(["./", "./Sapien_Scene_Sim"])

import os
import pdb
import copy
import json
import glob
import scipy

import trimesh
# import coacd  # coacd 必须在trimesh之后导入!!!!!!!!!!!

from tqdm import tqdm
import sapien.core as sapien
from sapien.core import pysapien
from sapien.utils import Viewer
import numpy as np

from concurrent.futures import ThreadPoolExecutor
import threading

from scipy.spatial.transform import Rotation as R

from Sapien_Scene_Sim.optimiz_large_bbox.optimize import optimize_one_scene
from Sapien_Scene_Sim.optimiz_large_bbox.bind_small_to_large import bind_one_scene
from Sapien_Scene_Sim.main import get_pose_data, save_bbox_info, lock_motion_rules



def load_info_for_simulation(
        not_visualize = True,
        render = False,
        make_large_bbox_ground_aligned = True,
        need_texture = False,
        use_scaled_collision = True
):
    from config import Configs
    config = Configs()

    base_dir = config.base_dir #"./scene_glbs/embodiedscene/scan"
    origin_glb_dir = config.origin_glb_dir # "./origin_glbs"
    # cate_to_cut_json_path = config.cate_to_cut_json_path 
    cate_with_large_size_path = config.cate_with_large_size_path
    cate_need_touch_ground_path = config.cate_need_touch_ground_path 
    collisoin_glb_dir = "/cpfs01/shared/landmark_3dgen/xuxudong_group/zhongweipeng/projects/layout/collisions" # "/oss/zhongweipeng/data/collisions" TODO
    scaled_collisoin_glb_dir = "/cpfs01/shared/landmark_3dgen/xuxudong_group/zhongweipeng/projects/layout/collisions_scaled"
 
    with open(cate_with_large_size_path, "r") as f: # "./optimiz_large_bbox/cate_with_large_size.json"
        cate_with_large_size = json.load(f) 
        cate_with_large_size = list(cate_with_large_size.keys())
    
    with open(cate_need_touch_ground_path, "r") as f: # "./optimiz_large_bbox/cate_with_large_size.json"
        cate_need_touch_ground = json.load(f) 
        cate_need_touch_ground = list(cate_need_touch_ground.keys())

    simulation_info_dict = {
        "base_dir":base_dir,
        "origin_glb_dir": origin_glb_dir,
        "cate_with_large_size":cate_with_large_size,
        "cate_need_touch_ground":cate_need_touch_ground,
        "render": render,
        "not_visualize": not_visualize,
        "make_large_bbox_ground_aligned": make_large_bbox_ground_aligned,
        "collisoin_glb_dir": collisoin_glb_dir,
        "need_texture": need_texture,
        "scaled_collisoin_glb_dir" : scaled_collisoin_glb_dir,
        "use_scaled_collision": use_scaled_collision
    }
    return simulation_info_dict

def decompose_transform(transform: np.ndarray):
    """
    将 4x4 变换矩阵分解为平移、旋转（四元数）和缩放。
    假设 transform 的形式为：
      [ R_scaled | t ]
      [  0  0  0 | 1 ]
    """
    # 提取平移（最后一列的前三个元素）
    translation = transform[:3, 3]

    # 提取 3x3 部分（旋转和缩放混合）
    M = transform[:3, :3]

    # 对每一列求范数作为缩放因子
    scale = np.array([
        np.linalg.norm(M[:, 0]),
        np.linalg.norm(M[:, 1]),
        np.linalg.norm(M[:, 2])
    ], dtype=np.float32)

    # 如果某个轴的 scale 为 0，注意处理以避免除零（这里假设不会为 0）
    # 计算纯旋转矩阵
    R_mat = np.zeros((3, 3), dtype=np.float32)
    R_mat[:, 0] = M[:, 0] / scale[0] if scale[0] != 0 else M[:, 0]
    R_mat[:, 1] = M[:, 1] / scale[1] if scale[1] != 0 else M[:, 1]
    R_mat[:, 2] = M[:, 2] / scale[2] if scale[2] != 0 else M[:, 2]

    # 将旋转矩阵转换为四元数（SciPy 返回 [x, y, z, w]）
    quat_xyzw = R.from_matrix(R_mat).as_quat()
    # 转换为 [w, x, y, z]，SAPIEN 的 Pose 构造函数通常要求这种顺序
    quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)

    return translation, quat, scale

def get_end_transform(instances_info):
    # 数据集位置 -》原点位置（0，0，0） -》场景位置
    #将物体从 原始的数据集中的位置 移动到 场景中的位置
    x_plus_90 = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])

    transform_to_origin = np.array(instances_info["transform_to_origin"])
    scale_factors = np.array(instances_info["scale_factors"])
    transform_from_origin = x_plus_90 @ np.array(instances_info["transform_from_origin"])

    # 构造 4x4 缩放矩阵 S
    scale_matrix = np.eye(4, dtype=np.float32)
    scale_matrix[0, 0] = scale_factors[0]
    scale_matrix[1, 1] = scale_factors[1]
    scale_matrix[2, 2] = scale_factors[2]

    final_transform = transform_from_origin @ scale_matrix @ transform_to_origin

    translation, quat, scale = decompose_transform(final_transform)

    return translation, quat, scale



def output_scene_as_glb(instances_infos, output_glb_path, filter_faraway_objects = False, need_texture = False):
    """
    多线程版本的save_scene函数，从origin—glb出发
    1. 使用 transform_to_origin 、scale_factors将mesh变换到原点位置
    2. 再使用保存的物理仿真之后的 bbox data 来 compose 整个scene
    """
    scene = trimesh.scene.Scene()
    lock = threading.Lock()

    def process_single_instance(instance, lock):
        """
        处理单个实例的函数，用于多线程处理
        """
        if instance["category"] in ["wall", "ceiling", "floor", "pipe"] or "geom_name" not in instance:
            return None

        if instance["3d_model"] == '':
            print("3d_model is empty. No retrieval reslut.")
            return None

        uid = instance["3d_model"][0]
        geometry_name = instance["geom_name"]
        origin_mesh_path = instance["mesh_glb_path"]

        # transform to space origin
        transform_to_origin = np.array(instance["transform_to_origin"])
        scale_factors = np.array(instance["scale_factors"])
        transform_scale = np.eye(4)
        transform_scale[0, 0] = scale_factors[0]
        transform_scale[1, 1] = scale_factors[1]
        transform_scale[2, 2] = scale_factors[2]
        # mesh.apply_transform(transform_to_origin)
        # mesh.apply_scale(scale_factors)

        # transform
        transform_final = np.eye(4)
        box_data = instance["bbox_after_simulation"]
        euler_angles = np.array(box_data[6:9])
        rotation_matrix = trimesh.transformations.euler_matrix(euler_angles[0], euler_angles[1], euler_angles[2], axes='rzxy')
        transform_final = rotation_matrix @ transform_final
        # mesh.apply_transform(rotation_matrix)

        #进行平移
        center = np.array(box_data[0:3])
        transform_final[:3, 3] = center
        # mesh.apply_translation(center)

        #最后再绕 X 轴旋转 -90 度，将整个场景保存为 Y-up，这样就使得场景有正确向上的朝向
        rotation_matrix = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        transform_final = rotation_matrix @ transform_final
        # mesh.apply_transform(rotation_matrix)

        total_transform = transform_final @ transform_scale @ transform_to_origin
        
        # load mesh
        if need_texture:
            # mesh_scene = trimesh.load(origin_mesh_path, process=False)
            # meshes = []
            # for geom_name in mesh_scene.geometry.keys():
            #     # 获取原始场景中的变换矩阵
            #     try:
            #         transform, _ = mesh_scene.graph.get(geom_name)
            #         mesh = mesh_scene.geometry[geom_name].copy()
            #         # 应用原始变换到顶点
            #         mesh.apply_transform(transform)
            #         meshes.append(mesh)
            #     except:
            #         mesh = mesh_scene.geometry[geom_name].copy()
            #         meshes.append(mesh)
            mesh_scene = trimesh.load(origin_mesh_path)
            meshes = [mesh_scene]
        else:
            mesh = trimesh.load(origin_mesh_path, force='mesh')
            meshes = [mesh]
        
        for mesh in meshes:
            mesh.apply_transform(total_transform)

            # scene.add_geometry(mesh, geom_name=geometry_name)

        
        # 使用锁来保护场景添加操作
        with lock:
            return meshes, geometry_name

    def process_instance(index):
        instance = instances_infos[index]
        result = process_single_instance(instance, lock)
        if result is not None:
            meshes, geometry_name = result
            for idx, mesh in enumerate(meshes):
                scene.add_geometry(mesh, geom_name=f"{geometry_name}_{idx}")

    # 使用线程池并行处理所有实例
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = []
        for index in range(len(instances_infos)):
            futures.append(executor.submit(process_instance, index))
        
        # 等待所有任务完成并处理异常
        for future in tqdm(futures, desc="exporting scene..."):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing instance: {str(e)}")

    # 去掉 scene glb 的材质
    print(f"add texture to glb :{need_texture}")
    if not need_texture:
        from trimesh.visual.texture import TextureVisuals
        empty_visual = TextureVisuals()
        for geometry in scene.geometry.values():
            geometry.visual = empty_visual
        
    if output_glb_path != None:
        trimesh.exchange.export.export_mesh(scene, output_glb_path)
    return scene

def load_glb_scene_to_sapien_multithread(
    scene_name,
    scene_dir, 
    collisoin_glb_dir,
    scene,
    instances_infos,
    cate_with_large_size,
    density,
    physical_material,
    scaled_collisoin_glb_dir,
    use_scaled_collision
):
    lock = threading.Lock()  # 用于同步Sapien场景修改操作
    def process_instance(index, instance):
        if "geom_name" not in instance:
            return

        geometry_name = instance["geom_name"]
        # print(f"{geometry_name} loading...")
        uid = instance["3d_model"][0]
        mesh_glb_path = instance["mesh_glb_path"]

        # 处理位姿信息
        translation_add = instance["delta_translation_after_large_obj_optimization"]
        translation, quat, scale = get_end_transform(instance)
        pose = sapien.Pose(p=translation + translation_add, q=quat)

        # 创建Actor构建器（无需加锁）
        builder = scene.create_actor_builder()

        # 添加可视化组件（无共享状态操作）
        builder.add_visual_from_file(
            filename=mesh_glb_path,
            pose=pose,
            scale=scale,
            material=None
        )

        if use_scaled_collision:
            # 如果有 scaled_collisoin_glb_dir
            if os.path.exists(os.path.join(scaled_collisoin_glb_dir, uid)):
                print(f"{uid} using scaled collisoin")
                if "/" in uid: # not objaverse
                    collisions_path = os.path.join(scaled_collisoin_glb_dir, uid, "*.glb")  
                else: # objaverse
                    collisions_path = os.path.join(scaled_collisoin_glb_dir, "objaverse", uid, "*.glb")
            else: # 没有 scaled_collisoin_glb_dir
                if "/" in uid: # not objaverse
                    collisions_path = os.path.join(collisoin_glb_dir, uid, "*.glb")  
                else: # objaverse
                    collisions_path = os.path.join(collisoin_glb_dir, "objaverse", uid, "*.glb")
        else:
            # 处理碰撞体路径
            if "/" in uid:
                collisions_path = os.path.join(collisoin_glb_dir, uid, "*.glb")
            else:
                collisions_path = os.path.join(collisoin_glb_dir, "objaverse", uid, "*.glb")
        collision_path_list = glob.glob(collisions_path)
        if not collision_path_list:
            raise ValueError(f"Collision files not found for: {collisions_path}")

        # 添加碰撞组件
        for path in collision_path_list:
            builder.add_collision_from_file(
                filename=path,
                pose=pose,
                scale=scale,
                material=physical_material,
                density=density
            )

        # 确定运动约束类型
        lock_motion_type = lock_motion_rules(
            instances_infos=instances_infos,
            index=index,
            cate_with_large_size=cate_with_large_size,
            scene_name = scene_name
        )

        # 使用锁同步场景修改操作
        with lock:
            if lock_motion_type == "LOCK XYZ AND ROTATION":
                builder.set_collision_groups(2, 2, 0, 0)
                mesh = builder.build_kinematic(name=geometry_name)
            elif lock_motion_type == "LOCK XY AND ROTATION":
                builder.set_collision_groups(2, 2, 0, 0)
                mesh = builder.build(name=geometry_name)
                mesh.lock_motion(x=True, y=True, z=False, rx=True, ry=True, rz=True)
            elif lock_motion_type == "LOCK ROTATION":
                builder.set_collision_groups(2, 2, 0, 0)
                mesh = builder.build(name=geometry_name)
                mesh.lock_motion(x=False, y=False, z=False, rx=True, ry=True, rz=True)
            elif lock_motion_type == "SPECIAL COLIISION GROUP":
                builder.set_collision_groups(1, 0, 0, 0)
                mesh = builder.build(name=geometry_name)
            else:
                builder.set_collision_groups(2, 2, 0, 0)
                mesh = builder.build(name=geometry_name)

    # 创建线程池并提交任务
    with ThreadPoolExecutor(max_workers = 16) as executor:
        futures = []
        for index, instance in enumerate(instances_infos):
            if "geom_name" not in instance:
                continue
            futures.append(executor.submit(process_instance, index, instance))
        
        # 等待所有任务完成并处理异常
        for future in tqdm(futures, desc = "Loading scene to simulator..."):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing instance: {str(e)}")

def load_glb_scene_to_sapien(
    scene_name,
    scene_dir, 
    collisoin_glb_dir,
    scene,
    instances_infos,
    cate_with_large_size,
    density,
    physical_material
):
    '''
    一个新的load函数，不需要原有场景glb，可以直接读取每个物体的collison
    '''
    for index, instance in enumerate(instances_infos):
        if "geom_name" not in instance.keys(): # 没有 retrieval 物体的话就跳过
            continue

        geometry_name = instance["geom_name"]
        print(f"{geometry_name} loading...")
        uid = instance["3d_model"][0]
        mesh_glb_path = instance["mesh_glb_path"]

        # pose infomation
        translation_add = instance["delta_translation_after_large_obj_optimization"]
        translation, quat, scale = get_end_transform(instance)
        pose = sapien.Pose(p = translation + translation_add, q = quat)

        # 创建actor
        builder = scene.create_actor_builder()
       
        # add visual
        builder.add_visual_from_file(
            filename = mesh_glb_path, 
            pose = pose, 
            scale = scale,
            material = None
        )

        # add collosion
        if "/" in uid:
            collisions_path = os.path.join(collisoin_glb_dir, uid, "*.glb")
        else:
            collisions_path = os.path.join(collisoin_glb_dir, "objaverse", uid, "*.glb")
        # pdb.set_trace()
        collision_path_list = glob.glob(collisions_path)
        if len(collision_path_list) == 0:
            raise ValueError(f"collision_path_list: {collision_path_list} not exist")
        for path in collision_path_list:
            builder.add_collision_from_file(filename = path, pose = pose, scale = scale,
                                            material = physical_material, density = density)


        # add lock motion type
        lock_motion_type = lock_motion_rules(instances_infos = instances_infos, index = index, cate_with_large_size = cate_with_large_size, scene_name = scene_name)

        if lock_motion_type == "LOCK XYZ AND ROTATION":
            builder.set_collision_groups(2,2,0,0)
            mesh = builder.build_kinematic(name = geometry_name)
            # mesh.lock_motion(x=True, y=True, z=True, rx=True, ry=True, rz=True)

        elif lock_motion_type == "LOCK ROTATION":
            builder.set_collision_groups(2,2,0,0)
            mesh = builder.build(name = geometry_name)
            mesh.lock_motion(x=False, y=False, z=False, rx=True, ry=True, rz=True)

        elif lock_motion_type == "SPECIAL COLIISION GROUP":
            builder.set_collision_groups(1,0,0,0)
            mesh = builder.build(name = geometry_name)

        else:
            builder.set_collision_groups(2,2,0,0)
            mesh = builder.build(name = geometry_name)
            # pass
            # mesh.set_damping(linear = 0.5, angular=0.8)
    pass

def simulation(
        render,
        scene_info_path,
        scene_name,
        scene_dir,
        origin_glb_dir,
        cate_with_large_size,
        output_glb_path,
        collisoin_glb_dir,
        need_texture,
        scaled_collisoin_glb_dir,
        use_scaled_collision
):
    # 正式进行物理仿真
    engine = sapien.Engine()
    if render:
        renderer = sapien.SapienRenderer()
        engine.set_renderer(renderer)

    scene = engine.create_scene()
    scene.set_timestep(1 / 100.0)

    # physics:
    scene_config = sapien.SceneConfig()

    physical_material: sapien.PhysicalMaterial = scene.create_physical_material(
        static_friction = 0.5,
        dynamic_friction = 0.5,
        restitution = 0.0,
    )
    ground_material = scene.create_physical_material(
        static_friction = 5,
        dynamic_friction = 5,
        restitution = 0.0,
    )
    density = 3000

    # Add ground
    ground_rb = scene.add_ground(altitude=0, material = ground_material) 
    ground_collision_shape = ground_rb.get_collision_shapes()[0]
    ground_collision_shape.set_collision_groups(group0=1, group1=3, group2=0, group3=0) 

    # Add Actors, load_scene

    with open(scene_info_path, "r") as f:
        instances_infos = json.load(f)

    # glb_scene = load_glb_scene_to_sapien(
    glb_scene = load_glb_scene_to_sapien_multithread(
        scene_name = scene_name,
        scene_dir = scene_dir,
        # glb_path = glb_path, 
        # origin_glb_dir = origin_glb_dir,
        collisoin_glb_dir = collisoin_glb_dir,
        scene = scene,
        instances_infos = instances_infos,
        # cate_to_cut = cate_to_cut,
        cate_with_large_size = cate_with_large_size,
        density = density,
        physical_material = physical_material,
        scaled_collisoin_glb_dir = scaled_collisoin_glb_dir,
        use_scaled_collision = use_scaled_collision
    )
    
    
    # set light and camera
    if render:
        scene.set_ambient_light([0.5, 0.5, 0.5])
        scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

        viewer = Viewer(renderer, resolutions=(1920, 1080))
        viewer.set_scene(scene)

        viewer.set_camera_xyz(x=-2, y=0, z=2.5)
        viewer.set_camera_rpy(r=0, p=-np.arctan2(2, 2), y=0)
        viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)


    # 导出仿真之前物体的位置
    begin_pose_data = get_pose_data(actors = scene.get_all_actors())



    for i in tqdm(range(6000), desc = "Simulating..."):
        if render and viewer.closed:
            break

        if i < 500:#1000
            for actor in scene.get_all_actors():
                # 给能够运动的物体进行减速，防止被弹飞
                if type(actor) != sapien.Actor:
                    continue
                velocity = actor.get_velocity()
                velocity_norm = np.linalg.norm(velocity)
                if velocity_norm > 1e-10:
                    actor.set_velocity(velocity / velocity_norm * np.array([0.0000001, 0.0000001, 0.01]))
                # actor.set_angular_velocity([0, 0, 0])
                # if i % 10 == 0:
                #     actor.set_velocity([0,0,0])

        scene.step()  
        if render:
            scene.update_render()
            viewer.render()


    
    # 导出仿真之后物体的位置
    end_pose_data = get_pose_data(actors = scene.get_all_actors())

    # 保存位姿
    for key in end_pose_data.keys():
        end_pose_data[key]["begin_pose"] = begin_pose_data[key]
    with open(os.path.join(scene_dir, "pose_data.json"), "w") as f:
        json.dump(end_pose_data, f, indent=4)
    
    # 保存仿真之后的info
    output_path = ".".join(scene_info_path.split(".")[:-1]) + "_with_simu.json"
    instances_infos_new = save_bbox_info(end_pose_data, copy.deepcopy(instances_infos), output_path)

    # 导出仿真之后的场景为 glb 文件
    output_scene_as_glb(
        instances_infos = instances_infos_new, 
        output_glb_path = output_glb_path,
        filter_faraway_objects = True,
        need_texture = need_texture)


def simulate_one_scene(
    scene_name,
    simulation_info_dict
):
    # load info dict
    base_dir = simulation_info_dict["base_dir"]
    origin_glb_dir = simulation_info_dict["origin_glb_dir"]
    cate_with_large_size = simulation_info_dict["cate_with_large_size"]
    cate_need_touch_ground = simulation_info_dict["cate_need_touch_ground"]
    render = simulation_info_dict["render"]
    not_visualize = simulation_info_dict["not_visualize"]
    make_large_bbox_ground_aligned = simulation_info_dict["make_large_bbox_ground_aligned"]
    collisoin_glb_dir = simulation_info_dict["collisoin_glb_dir"]
    need_texture = simulation_info_dict["need_texture"]
    scaled_collisoin_glb_dir = simulation_info_dict["scaled_collisoin_glb_dir"]
    use_scaled_collision = simulation_info_dict["use_scaled_collision"]

    scene_dir = os.path.join(base_dir, scene_name)
    

    # check if scene_dir exists
    if not os.path.exists(scene_dir):
        os.makedirs(scene_dir, exist_ok=True)

    # optimize one scene
    input_instance_infos_dir = os.path.join(scene_dir, "instances_info_with_retri_with_compose.json")
    output_instance_infos_dir = os.path.join(scene_dir, "instances_info_with_opti.json")
    optimize_one_scene(
        input_instance_infos_dir,
        output_instance_infos_dir,
        cate_with_large_size,
        cate_need_touch_ground,
        not_visualize,
        make_large_bbox_ground_aligned# 保持bbox的底面与地面平行
    )

    # bind small to large
    input_instance_infos_dir = os.path.join(scene_dir, "instances_info_with_opti.json")
    output_instance_infos_dir = os.path.join(scene_dir, "instances_info_with_opti_with_shift.json")
    bind_one_scene(
        input_instance_infos_dir,
        output_instance_infos_dir,
        cate_with_large_size
    )

    # use sapien to simulation
    scene_info_path = os.path.join(scene_dir,"instances_info_with_opti_with_shift.json")
    output_glb_path = os.path.join(scene_dir, "retrieval_scene_16_withpartnetmobility_simu.glb")
    simulation(
        render = render,
        scene_info_path = scene_info_path,
        scene_name = scene_name,
        scene_dir = scene_dir,
        origin_glb_dir = origin_glb_dir,
        cate_with_large_size = cate_with_large_size,
        output_glb_path = output_glb_path,
        collisoin_glb_dir = collisoin_glb_dir,
        need_texture = need_texture,
        scaled_collisoin_glb_dir = scaled_collisoin_glb_dir,
        use_scaled_collision = use_scaled_collision
    )


def test():
    matrix = np.array([[1.0, 0,  0.0, 10],
                        [0,  1.0,  0.0, 2.0],
                        [0.0,  0.0,  1.0, 3.0],
                        [0.0,  0.0,  0.0, 1.0]])
    translation, quat, scale = decompose_transform(matrix)
    print("translation:", translation)
    print("quat:", quat)
    print("scale:", scale)

if __name__ == "__main__":
    # test()
    
    simulation_info_dict = load_info_for_simulation(
        not_visualize = True,
        render = False,
        make_large_bbox_ground_aligned = True,
        need_texture =True
    )

    with open("/cpfs01/user/zhongweipeng/Projects/layout/embodiedscene/20_samples_250416/20_samples_250416.json", "r")as f:
        name_list = json.load(f)
    for scene_name in name_list[11:]:
        # scene_name = "arkitscenes/Training/48018233"
        print(f"processing scene_name: {scene_name}")
        simulate_one_scene(scene_name = scene_name, 
                        simulation_info_dict = simulation_info_dict)

        scene_files_sample_dir = "/cpfs01/user/zhongweipeng/Projects/layout/embodiedscene/20_samples_250416/"
        output_files_path = os.path.join(scene_files_sample_dir, scene_name)
        output_files_path = "/".join(output_files_path.split("/")[:-1])
        os.makedirs(output_files_path, exist_ok=True)
        os.system(f"cp -r /cpfs01/user/zhongweipeng/Projects/layout/embodiedscene/scan/{scene_name} {output_files_path}")

    # ./rclone copy --progress --transfers 200 --checkers 200  My-aliyun-oss:pjlab-lingjun-landmarks/zhongweipeng/data/collisions/ /cpfs01/shared/landmark_3dgen/xuxudong_group/zhongweipeng/projects/layout/collisions/
