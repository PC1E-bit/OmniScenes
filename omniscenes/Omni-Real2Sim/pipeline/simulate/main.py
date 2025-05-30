"""Create actors (rigid bodies).

The actor (or rigid body) in Sapien is created through a sapien.ActorBuilder.
An actor consists of both collision shapes (for physical simulation) and visual shapes (for rendering).
Note that an actor can have multiple collision and visual shapes,
and they do not need to correspond.

Concepts:
    - Create sapien.Actor by primitives (box, sphere, capsule)
    - Create sapien.Actor by mesh files
    - sapien.Pose
"""
import sys
sys.path.extend(["./", "./Sapien_Scene_Sim"])

import os
import pdb
import copy
import json
import glob
import scipy

import trimesh
import coacd  # coacd 必须在trimesh之后导入!!!!!!!!!!!

from tqdm import tqdm
import sapien.core as sapien
from sapien.core import pysapien
from sapien.utils import Viewer
import numpy as np

from Sapien_Scene_Sim.optimiz_large_bbox.bbox_utils import euler_to_matrix, get_obb_vertices, is_bbox1_inside_bbox2, is_point_inside_bbox, get_totated_bbox_iou
from Sapien_Scene_Sim.optimiz_large_bbox.bbox_utils  import get_point_face_distance, get_bbox_Z_faces

def save_mesh(vertices, faces, filename):
    if filename.split(".")[-1] == "obj":
        with open(filename, 'w') as f:
            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    elif filename.split(".")[-1] == "glb":
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(filename)
    else:
        type_f = filename.split(".")[-1]
        raise Exception(f"File format {type_f} not supported")


def get_convex_parts_file_name_list(mesh, mesh_path, parts_mesh_type, cut_to_decomposition):

    # input_file = "/home/zhongweipeng/Desktop/Projects/LayOut/Sapien_sim/scene_instance_glbs/1b8d76cdbfba43829563ba3fc99886a0.glb"
    # mesh = trimesh.load(input_file, force="mesh")
    

    if cut_to_decomposition:
        plane_origin = np.array([0, 0, 0])  # 平面上的一点
        plane_normal_1 = np.array([0, 0, 1])  # 平面的法向量
        plane_normal_2 = np.array([0, 0, -1]) 

        cut_mesh_1 = mesh.slice_plane(plane_origin, plane_normal_1)
        cut_mesh_2 = mesh.slice_plane(plane_origin, plane_normal_2) 

        mesh_1 = coacd.Mesh(cut_mesh_1.vertices, cut_mesh_1.faces)
        parts_1 = coacd.run_coacd(mesh_1) # a list of convex hulls.
        mesh_2 = coacd.Mesh(cut_mesh_2.vertices, cut_mesh_2.faces)
        parts_2 = coacd.run_coacd(mesh_2) # a list of convex hulls.

        parts = parts_1 + parts_2

    else:
        mesh = coacd.Mesh(mesh.vertices, mesh.faces)
        parts = coacd.run_coacd(mesh) # a list of convex hulls.

    file_name_list = []
    for i, part in enumerate(parts):

        vertices, faces = part
        file_name = ".".join(mesh_path.split(".")[:-1]) + f"_collision_part_{i}.{parts_mesh_type}"
        file_name_list.append(file_name)
        save_mesh(vertices, faces, file_name)

    return file_name_list

def get_origin_geometry(geometry, instances_info):
    # 场景位置 -》原点位置（0，0，0） -》数据集位置 
    #下面是一个逆转过程，将物体从场景中的位置移动到原始的数据集中的位置

    # 下面的transformation 是
    transform_to_origin = np.array(instances_info["transform_to_origin"])
    scale_factors = np.array(instances_info["scale_factors"])
    transform_from_origin = np.array(instances_info["transform_from_origin"])

    # transform from end location to origin
    translation = -transform_from_origin[:3,3]
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = np.linalg.inv(transform_from_origin[:3, :3]) 
    geometry.apply_translation(translation)
    geometry.apply_transform(rotation_matrix)

    # scale
    geometry.apply_scale(1 / scale_factors)

    # transform_from_origin to origin location
    translation = -transform_to_origin[:3,3]
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = np.linalg.inv(transform_to_origin[:3, :3]) 
    geometry.apply_translation(translation)
    geometry.apply_transform(rotation_matrix)

    return geometry

def get_end_geometry(geometry, instances_infos, i):
    # 数据集位置 -》原点位置（0，0，0） -》场景位置
    #将物体从 原始的数据集中的位置 移动到 场景中的位置
    instances_info = instances_infos[i]
    x_plus_90 = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])

    transform_to_origin = np.array(instances_info["transform_to_origin"])
    scale_factors = np.array(instances_info["scale_factors"])
    transform_from_origin = x_plus_90 @ np.array(instances_info["transform_from_origin"])

    # 从原始位置 到 原点location
    geometry.apply_transform(transform_to_origin)
    # scale
    geometry.apply_scale(scale_factors)
    # 从原点location 到 场景中的end位置
    geometry.apply_transform(transform_from_origin)

    return geometry


def get_end_transform_pose(instances_info):
    #返回一个 sapien.Pose，用于设置物体的位置
    # 数据集位置 -》原点位置（0，0，0） -》场景位置
    #将物体从 原始的数据集中的位置 移动到 场景中的位置
    x_plus_90 = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    x_minus_90 = trimesh.transformations.rotation_matrix(- np.pi / 2, [1, 0, 0])

    # 注意：由于 transform_to_origin、transform_from_origin是以 Y-up 的trimesh坐标系考量的，所以要加入两个来抵消
    transform_to_origin =  np.array(instances_info["transform_to_origin"])
    scale_factors = np.array(instances_info["scale_factors"])
    transform_from_origin = x_plus_90 @ np.array(instances_info["transform_from_origin"])

    # 创建缩放矩阵
    scale_matrix = np.eye(4)
    scale_matrix[:3, :3] = np.diag(scale_factors)

    # 合并变换：transform_from_origin @ scale_matrix @ transform_to_origin

    # final_transform = x_plus_90 @np.eye(4)#  x_plus_90
    # final_transform = transform_to_origin
    final_transform = transform_from_origin @ transform_to_origin
    # final_transform = transform_from_origin @ scale_matrix @ transform_to_origin

    return sapien.Pose.from_transformation_matrix(final_transform), scale_factors  # transform_to_pose(final_transform)


def lock_motion_rules(instances_infos , index , cate_with_large_size, scene_name):
    try:
        cate = instances_infos[index]["category"]
        cate_after_replace = instances_infos[index]["embodiedscan_uid_cate"]
        if cate not in cate_with_large_size:
            shift_type = instances_infos[index]["shift_type"]
        else:
            shift_type = "no type"

        possible_types = [
        "carpet type",
        "chair_table type",
        "doorframe door type",
        "inside_large type",
        "support type",
        "over type",
        "inside_large_enlarge type",
        "no type"
        ]

        if shift_type not in possible_types:
            print("shift_type not in possible_types")
            raise Exception("shift_type not in possible_types")

    except Exception as e:
        print(f"Error in instance_info: {index} lack of key _category_ or _embodiedscan_uid_cate_")
        return "NO LOCK"


    # 1.如果是大物体类别，之前已经经过了优化，直接判定为不能动
    if cate in cate_with_large_size:
        return "LOCK XYZ AND ROTATION"

    # 2.100%挂壁物体判定不能动
    elif cate in ["clock", "curtain", "switch", "shower", "mirror", "chandelier", "light", "blackboard", "vent", "socket", "plug", "decoration"] \
        or cate_after_replace in ["clock", "curtain", "switch", "shower", "mirror", "chandelier", "light", "blackboard", "vent", "socket", "plug", "decoration"]:
         return "LOCK XYZ AND ROTATION" 

    # 3.墙体结构物体判定为不能动
    elif cate in ["window", "door", "doorframe"]:
        return "LOCK XYZ AND ROTATION"
    
    # 4.如果有条件性地判断为挂壁物体，不能动
    elif cate in ["picture", "frame"] \
        or cate_after_replace in ["picture", "frame"]:
        if shift_type in ["support type"]:
            return "LOCK XY AND ROTATION"
        else:
            return "LOCK XYZ AND ROTATION"
    
    # 4+. 如果有条件性地判断为挂壁物体，不能动
    elif cate in ["tv", "screen", "monitor", "towel"]:
        
        # "support type" 需要自由下落
        if shift_type in ["support type"]:
            return "LOCK XY AND ROTATION"
        
        # 由于 arkitscenes 标注中没有 wall
        if "arkitscenes" in scene_name:
            return "LOCK XYZ AND ROTATION"
        
        object_loc_xyz = np.array(instances_infos[index]["bbox"][0:3])
        # 筛选出 wall 的所有bbox
        wall_bbox_list = []
        for instance_info_dict in instances_infos:
            if  instance_info_dict["category"] == "wall":
                wall_bbox_list.append(instance_info_dict["bbox"])
        # 如果贴近墙面
        in_wall_flag = False
        enlarged_wall_bbox_list = [item + np.array([0, 0, 0, 0.1, 0.1, 0.1, 0, 0, 0]) for item in wall_bbox_list]
        for enlarged_bbox in enlarged_wall_bbox_list:
            if is_point_inside_bbox(object_loc_xyz, enlarged_bbox):
                in_wall_flag = True
                break
        if in_wall_flag:
            return "LOCK XYZ AND ROTATION"
    
    # 5.如果物体是长条形的立起来的类别，不能转动防止倒下
    elif cate in ["lamp", "chair", "stool"] or cate_after_replace in ["lamp", "chair", "stool"]:
        return "LOCK ROTATION"
    
    # 6.如果是地毯，设置特殊碰撞组，地毯只与地面碰撞，不和其他物体碰撞。
    elif cate in ["carpet"]:
        return "SPECIAL COLIISION GROUP"
    
    else:
        return "NO LOCK"


def load_glb_scene_to_sapien(
        scene_name,
        scene_dir,
        glb_path, 
        origin_glb_dir,
        scene, 
        instances_infos,
        cate_to_cut_json_path, 
        cate_with_large_size_path,
        density, 
        physical_material
    ):

    glb_file  = trimesh.load(glb_path)

    with open(cate_to_cut_json_path, "r") as f:
        cate_to_cut = json.load(f)
        cate_to_cut = list(cate_to_cut.keys())
    with open(cate_with_large_size_path, "r") as f:
        cate_with_large_size = json.load(f) 
        cate_with_large_size = list(cate_with_large_size.keys())

    for geometry_name, geometry in tqdm(glb_file.geometry.items(), desc="Loading glb file for simulation..."):
        
        # 只仿真少部分物体：
        # if int(geometry_name.split("_")[0]) not in [9, 128]:
        #     continue

        # 得到 序号 和 uid 、cate
        prefix = f'{scene_name}_'
        rest = geometry_name[len(prefix):]
        index = int(rest.split("_")[0])
        uid = "_".join(rest.split("_")[1:])  # uid
        cate = instances_infos[index]["embodiedscan_uid_cate"]
        translation_add = instances_infos[index]["delta_translation_after_large_obj_optimization"]

        # 得到 origin_mesh 保存位置
        if "partnet_mobility" in uid:
            uid_split = uid.split("/")[-1] # 对于partnet_mobility 只取后面的小数字 
            origin_mesh_path = os.path.join(origin_glb_dir, uid, f"{uid_split}.glb")
        elif "gen_assets" in uid:
            uid_split = uid.split("/")[-1] # 对于 gen_assets 只取后面的小数字 
            origin_mesh_path = os.path.join(origin_glb_dir, uid, f"{uid_split}.glb")
        
        else:# objaverse
            origin_mesh_path = os.path.join(origin_glb_dir, "objaverse", uid, f"{uid}.glb")

        # 得到 end_mesh 保存位置
        end_mesh_dir = os.path.join(scene_dir, "instance_glbs", geometry_name.replace("/", "_"))
        end_mesh_path = os.path.join(end_mesh_dir, f"end.glb")

        # 如果已经有了origin_glb, 就不再导出
        if not os.path.exists(origin_mesh_path):
            if not os.path.exists("/".join(origin_mesh_path.split("/")[:-1])): #如果不存在这个文件夹就创建
                print("making directory: ", "/".join(origin_mesh_path.split("/")[:-1]))
                os.makedirs("/".join(origin_mesh_path.split("/")[:-1]))
            origin_mesh = get_origin_geometry(copy.deepcopy(geometry), instances_infos[index])
            origin_mesh.export(origin_mesh_path) # 保存为glb文件

        # 如果不存在 end_mesh_dir 就创建文件
        if not os.path.exists(end_mesh_dir):
            os.makedirs(end_mesh_dir)


        # 创建actor
        builder = scene.create_actor_builder()

        # add visual
        origin_mesh = trimesh.load(origin_mesh_path)
        origin_mesh = list(origin_mesh.geometry.values())[0]
        end_mesh = get_end_geometry(copy.deepcopy(origin_mesh), instances_infos, index)
        end_mesh.export(end_mesh_path) # 保存为glb文件
        # 加入一个平移
        builder.add_visual_from_file(filename = end_mesh_path, pose = sapien.Pose(translation_add), material = None)
        

        # add collision
        # if "1b8d76cdbfba4382" in uid or "partnet_mobility/46490" in uid:
        collision_path_list = ".".join(origin_mesh_path.split(".")[:-1]) + "_collision_part_" + "*" + ".glb"
        collision_path_list = glob.glob(collision_path_list)
        if len(collision_path_list) == 0:   # 如果没有找到对应的碰撞体文件，就根据 origin_mesh 重新生成
            if cate in cate_to_cut:
                cut_to_decomposition = True
            else: 
                cut_to_decomposition = False
            collision_path_list = get_convex_parts_file_name_list(origin_mesh, origin_mesh_path, 
                                                                    parts_mesh_type="glb", cut_to_decomposition = cut_to_decomposition)

        for path in collision_path_list:
            collision_part_num = path.split("_")[-1].split(".")[0]
            origin_collision = trimesh.load(path)
            origin_collision = list(origin_collision.geometry.values())[0]
            end_collision = get_end_geometry(copy.deepcopy(origin_collision), instances_infos, index)
            end_collision_path = os.path.join(end_mesh_dir ,f"end_collision_part_{collision_part_num}.glb")
            end_collision.export(end_collision_path) # 保存为glb文件

            builder.add_collision_from_file(filename = end_collision_path, pose = sapien.Pose(translation_add),
                                            material = physical_material, density = density)
                
        # else: 
        #     builder.add_collision_from_file(filename = end_mesh_path, material = physical_material, density = density)


        
        # pdb.set_trace()

        lock_motion_type = lock_motion_rules(instances_infos = instances_infos, index = index, cate_with_large_size = cate_with_large_size)

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
    return 





def get_pose_data(actors):
    pose_data = {}
    for actor in actors:
        pose = actor.get_pose()
        pose_data[actor.name] = {}
                
        pose_data[actor.name][f"position"] = pose.p.tolist()  # 位置 (x, y, z)
        pose_data[actor.name][f"rotation"] = pose.q.tolist()   # 旋转四元数 (w, x, y, z)
            
    return pose_data


def save_bbox_info(end_pose_data, instances_infos, output_path):
        # 处理位置数据，保存物体仿真之后的bbox信息

    for index, instance_info in enumerate(instances_infos):

        # 不存在 geom_name 则跳过
        if "geom_name" not in instance_info.keys():
            continue

        key = instance_info["geom_name"]
        if key not in end_pose_data.keys():
            continue
        

        # simulation data
        end_position = end_pose_data[key]["position"]
        end_rotation = end_pose_data[key]["rotation"]    

        # before simulation
        bbox = np.array(instance_info["bbox"])
        shift_before = np.array(instance_info["delta_translation_after_large_obj_optimization"])
        center = bbox[:3] + shift_before
        size = bbox[3:6]
        euler = bbox[6:9]

       
        euler_matrix = trimesh.transformations.euler_matrix(euler[0], euler[1], euler[2], axes='rzxy')
        euler_matrix[:3,3] = center
        matrix = euler_matrix # 仿真之前的矩阵

        matrix = trimesh.transformations.quaternion_matrix(end_rotation) @ matrix
        matrix[:3,3] += end_position #仿真之后的矩阵
        
        # 计算最后的bbox
        center_after = matrix[:3,3]
        euler_after = trimesh.transformations.euler_from_matrix(matrix, axes='rzxy')     
        bbox_after = np.concatenate([center_after, size, euler_after])

        instance_info["bbox_after_simulation"] = bbox_after.tolist()

    
    with open(output_path, "w") as f:
        json.dump(instances_infos, f, indent=4)

    return instances_infos


def output_scene_as_glb(instances_infos, output_glb_path, instance_glb_dir, scene_name, filter_faraway_objects):
    """
    这个版本的save_scene函数从origin—glb出发
    1. 使用 transform_to_origin 、scale_factors将mesh变换到原点位置
    2. 再使用保存的物理仿真之后的 bbox data 来 compose 整个scene
    
    """

    scene = trimesh.scene.Scene()

    for index in tqdm(range(len(instances_infos)), desc="exporting scene..."):
        instance = instances_infos[index]
        if instance["label"] not in [276 ,49, 104] : #如果不属于 wall/ceiling/floor 这一类
            uid = instance["3d_model"][0]

            if instance["3d_model"]=='':
                print("3d_model is empty. No retrieval reslut.")

            # elif uid not in exist_uid_path_dict.keys() and "/" not in uid: 
            #     print(f"3d_model uid {uid} not exist in glb folder.")

            else:
                uid = instance["3d_model"][0]
                geometry_name = instance["geom_name"]
                origin_mesh_path = instance["mesh_glb_path"]
                # if "/" in uid:
                #     uid_split = uid.split("/")[-1] # 对于partnet_mobility 只取后面的小数字 
                #     origin_mesh_path = os.path.join(instance_glb_dir, uid, f"{uid_split}.glb")
                # else: # objaverse
                #     origin_mesh_path = os.path.join(instance_glb_dir, "objaverse", uid, f"{uid}.glb")

                # transform to space origin
                transform_to_origin = np.array(instance["transform_to_origin"])
                scale_factors = np.array(instance["scale_factors"])
                
                # load mesh
                mesh = trimesh.load(origin_mesh_path)
                mesh = list(mesh.geometry.values())[0]
                mesh.apply_transform(transform_to_origin)
                mesh.apply_scale(scale_factors)

                # transform
                box_data = instance["bbox_after_simulation"]
                euler_angles = np.array(box_data[6:9])
                rotation_matrix = trimesh.transformations.euler_matrix(euler_angles[0], euler_angles[1], euler_angles[2], axes='rzxy')
                mesh.apply_transform(rotation_matrix)

                #进行平移
                center = np.array(box_data[0:3])
                mesh.apply_translation(center)

                #最后再绕 X 轴旋转 -90 度，将整个场景保存为 Y-up，这样就使得场景有正确向上的朝向
                rotation_matrix = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
                mesh.apply_transform(rotation_matrix)

                # print(f"import {geometry_name} successfully.")
                scene.add_geometry(mesh, geom_name  = geometry_name)

    # 去掉 scene glb 的材质
    from trimesh.visual.texture import TextureVisuals
    empty_visual = TextureVisuals()
    for geometry in scene.geometry.values():
        geometry.visual = empty_visual
        
    trimesh.exchange.export.export_mesh(scene, output_glb_path)


def one_scene_simulation(
        render = True, 
        optimize = True,
        bind_small_to_large = True,
        scene_name = "scannet/scene0000_00",
        config = None
):
    # scene paths
    base_dir = config.base_dir #"./scene_glbs/embodiedscene/scan"
    scene_dir = os.path.join(base_dir, scene_name)
    glb_path = os.path.join(scene_dir, "retrieval_scene_16_withpartnetmobility.glb")
    scene_info_path = os.path.join(scene_dir,"instances_info_with_opti_with_shift.json")
    output_glb_path = os.path.join(scene_dir, "retrieval_scene_16_withpartnetmobility_simu.glb")

    # check if scene_dir exists
    if not os.path.exists(scene_dir):
        os.makedirs(scene_dir, exist_ok=True)

    # common paths
    origin_glb_dir = config.origin_glb_dir # "./origin_glbs"
    cate_to_cut_json_path = config.cate_to_cut_json_path #"./cate_need_to_cut_decomposition.json"
    cate_with_large_size_path = config.cate_with_large_size_path# "./optimiz_large_bbox/cate_with_large_size.json"
    cate_need_touch_ground_path = config.cate_need_touch_ground_path # "/cpfs01/user/zhongweipeng/Projects/layout/Sapien_Scene_Sim/optimiz_large_bbox/cate_need_touch_ground.json"


    # 优化大型家具位置，调用 optimize.py 来执行
    if optimize:
        python_command = config.python_command # "/home/zhongweipeng/anaconda3/envs/partnet/bin/python"
        optimize_python_file_path = config.optimize_python_file_path #"./optimiz_large_bbox/optimize.py"
        input_instance_infos_dir = os.path.join(scene_dir, "instances_info_with_retri_uid_withpartnetmobility.json")
        output_instance_infos_dir = os.path.join(scene_dir, "instances_info_with_opti.json") 
        command = (
                f"{python_command} {optimize_python_file_path}" +\
                f" --input_instance_infos_dir {input_instance_infos_dir}" +\
                f" --output_instance_infos_dir {output_instance_infos_dir}" +\
                f" --cate_with_large_size_path {cate_with_large_size_path}" +\
                f" --cate_need_touch_ground_path {cate_need_touch_ground_path}"
            )
        if config.make_large_bbox_ground_aligned:
            command = command + f" --make_large_bbox_ground_aligned"
        if not render:
            command = command + f" --not_visualize"
        os.system(command)
        # pdb.set_trace()

    # 将场景中的小型物体与大物体进行绑定，调用 bind_small_to_large.py 来执行
    if bind_small_to_large:
        python_command = config.python_command # "/home/zhongweipeng/anaconda3/envs/partnet/bin/python"
        bind_small_python_file_path = config.bind_small_python_file_path #"./optimiz_large_bbox/bind_small_to_large.py"
        input_instance_infos_dir = os.path.join(scene_dir, "instances_info_with_opti.json")
        output_instance_infos_dir = os.path.join(scene_dir, "instances_info_with_opti_with_shift.json")
        command = (
            f"{python_command} {bind_small_python_file_path}" +\
            f" --input_instance_infos_dir {input_instance_infos_dir}" +\
            f" --output_instance_infos_dir {output_instance_infos_dir}" +\
            f" --cate_with_large_size_path {cate_with_large_size_path}"
        )
        os.system(command)
        # pdb.set_trace()




    # 正式进行物理仿真
    engine = sapien.Engine()
    if render:
        renderer = sapien.SapienRenderer()
        engine.set_renderer(renderer)

    scene = engine.create_scene()
    scene.set_timestep(1 / 100.0)

    # physics:
    scene_config = sapien.SceneConfig()
    print(scene_config.gravity)
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
    glb_scene = load_glb_scene_to_sapien(
        scene_name = scene_name,
        scene_dir = scene_dir,
        glb_path = glb_path, 
        origin_glb_dir = origin_glb_dir,
        scene = scene,
        instances_infos = instances_infos,
        cate_to_cut_json_path = cate_to_cut_json_path,
        cate_with_large_size_path = cate_with_large_size_path,
        density = density,
        physical_material = physical_material
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

        if i < 10:#1000
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
        instance_glb_dir = origin_glb_dir,
        scene_name = scene_name,
        filter_faraway_objects = True)



if __name__ == '__main__':
    from config import Configs
    config = Configs()
    one_scene_simulation(
        render = False, 
        optimize = True,
        bind_small_to_large = True,
        scene_name="scannet/scene0001_00",
        config = config
    )


    