
import os
import cv2
import pdb
import math
import json
import time
import trimesh
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
from trimesh.transformations import rotation_matrix
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
sys.path.extend(["./"])
from codes_ybq.utils.uid_utils import get_mesh_path_from_uid


with open("/cpfs01/user/zhongweipeng/Projects/layout/cate_with_apparent_direction.json", "r") as f:
    cate_with_apparent_direction = json.load(f)
    cate_with_apparent_direction = list(cate_with_apparent_direction.keys())

def normalize_rotation_to_90_degrees(rotation_degrees):
    """
    将角度归一化到最接近的0、90、180或270度
    
    Args:
        rotation_degrees: 旋转角度(度)
    
    Returns:
        归一化后的角度(度)
    """
    # 确保角度在0-360度之间
    rotation_degrees = rotation_degrees % 360
    
    # 找到最接近的90度倍数
    nearest_90_multiple = round(rotation_degrees / 90) * 90
    
    # 确保结果在0-360度之间
    return nearest_90_multiple % 360

def rot_mat():
    rotation_angle1 = -np.pi / 2  # 90度转换为弧度，并且因为是顺时针，所以是负值
    rotation_angle2 = -np.pi / 2  # 90度转换为弧度，并且因为是顺时针，所以是负值
    rotation_angle3 = -np.pi   # 90度转换为弧度，并且因为是顺时针，所以是负值
    rotation_matrix_forward = np.array([
        [np.cos(rotation_angle1), 0, np.sin(rotation_angle1), 0],
        [0, 1, 0, 0],
        [-np.sin(rotation_angle1), 0, np.cos(rotation_angle1), 0],
        [0, 0, 0, 1]
    ])
    rotation_matrix_backward = np.array([
        [np.cos(rotation_angle2), 0, np.sin(rotation_angle2), 0],
        [0, 1, 0, 0],
        [-np.sin(rotation_angle2), 0, np.cos(rotation_angle2), 0],
        [0, 0, 0, 1]
    ])
    rotation_matrix_left = np.array([
        [1, 0, 0, 0],            # 沿x轴旋转不影响x坐标
        [0, np.cos(rotation_angle1), -np.sin(rotation_angle1), 0],  # y坐标根据旋转角度变化
        [0, np.sin(rotation_angle1), np.cos(rotation_angle1), 0],  # z坐标根据旋转角度变化
        [0, 0, 0, 1]              # 最后一行用于齐次坐标
    ])
    rotation_matrix_right = np.array([
        [1, 0, 0, 0],            # 沿x轴旋转不影响x坐标
        [0, np.cos(rotation_angle2), -np.sin(rotation_angle2), 0],  # y坐标根据旋转角度变化
        [0, np.sin(rotation_angle2), np.cos(rotation_angle2), 0],  # z坐标根据旋转角度变化
        [0, 0, 0, 1]              # 最后一行用于齐次坐标
    ])
    rotation_matrix_upside_down = np.array([
        [np.cos(rotation_angle3), 0, np.sin(rotation_angle3), 0],
        [0, 1, 0, 0],
        [-np.sin(rotation_angle3), 0, np.cos(rotation_angle3), 0],
        [0, 0, 0, 1]
    ])

    rotation_matrix_towards = np.array([
        [np.cos(rotation_angle3), -np.sin(rotation_angle3), 0, 0],
        [np.sin(rotation_angle3), np.cos(rotation_angle3), 0, 0],
        [0, 0, 1, 0],  # 沿z轴旋转不影响z坐标
        [0, 0, 0, 1]   # 最后一行用于齐次坐标
    ])

def quaternion_to_rotation_matrix(quaternion):
    w, x, y, z = quaternion
    theta = 2 * np.arccos(w)

    if abs(theta) < 1e-6:
        return np.eye(4)

    axis_x, axis_y, axis_z = x, y, z
    axis_norm = np.linalg.norm([(axis_x), (axis_y), (axis_z)])
    axis_x /= axis_norm
    axis_y /= axis_norm
    axis_z /= axis_norm

    rotation_matrix = np.array([
        [np.cos(theta) + axis_x**2 * (1 - np.cos(theta)), 
         axis_x * axis_y * (1 - np.cos(theta)) - axis_z * np.sin(theta), 
         axis_x * axis_z * (1 - np.cos(theta)) + axis_y * np.sin(theta), 0],
        [
            axis_y * axis_x * (1 - np.cos(theta)) + axis_z * np.sin(theta), 
            np.cos(theta) + axis_y**2 * (1 - np.cos(theta)), 
            axis_y * axis_z * (1 - np.cos(theta)) - axis_x * np.sin(theta), 0
        ], 
        [
            axis_z * axis_x * (1 - np.cos(theta)) - axis_y * np.sin(theta), 
            axis_z * axis_y * (1 - np.cos(theta)) + axis_x * np.sin(theta), 
            np.cos(theta) + axis_z**2 * (1 - np.cos(theta)), 0
        ],
        [0, 0, 0, 1]
    ])

    return rotation_matrix

def find_file_path(file_list, file_name):
    for file_path in file_list:
        if file_name in file_path:
            return file_path
        
def get_bbox_rotation_transform_z(target_bbox, candidate_bbox):
    '''
    根据target bbox、 candidate bbox的长宽高确定出target bbox 需要的旋转transform矩阵
    只绕Z轴旋转
    '''
    target_bbox_indecies = np.argsort(target_bbox[0:2]) 
    candidate_bbox_indecies = np.argsort(candidate_bbox[0:2]) 
    if np.sum(target_bbox_indecies == candidate_bbox_indecies) == 2:
        return np.eye(4)
    else:
        rotate_axis = [0, 0, 1]
        rotation_matrix = trimesh.transformations.rotation_matrix(np.pi / 2, rotate_axis)
        return rotation_matrix

def get_bbox_rotation_transform(target_bbox, candidate_bbox):
    '''
    根据target bbox、 candidate bbox的长宽高确定出target bbox 需要的旋转transform矩阵
    '''
    # pdb.set_trace()
    target_bbox_indecies = np.argsort(target_bbox) # 0 2 1
    candidate_bbox_indecies = np.argsort(candidate_bbox) # 1 0 2

    if np.sum(target_bbox_indecies == candidate_bbox_indecies) == 3:
        rotation_matrix = np.eye(4)
    elif np.sum(target_bbox_indecies == candidate_bbox_indecies) == 1:
        rotate_axis = (target_bbox_indecies == candidate_bbox_indecies)
        rotation_matrix = trimesh.transformations.rotation_matrix(np.pi / 2, rotate_axis)
    elif np.sum(target_bbox_indecies == candidate_bbox_indecies) == 0:

        # candidate_x_axis_match_target_which_axis = np.where(target_bbox_indecies==candidate_bbox_indecies[0])[0]
        # if candidate_x_axis_match_target_which_axis == 1:
        #     rotate_axis_1 = [0, 0, 1]
        #     rotate_axis_2 = [0, 1, 0]
        # elif candidate_x_axis_match_target_which_axis == 2:
        #     rotate_axis_1 = [0, 1, 0]
        #     rotate_axis_2 = [0, 0, 1]
        rotate_axis_1 = [0, 1, 0]
        rotate_axis_2 = [0, 0, 1]
        m1 = trimesh.transformations.rotation_matrix(np.pi / 2, rotate_axis_1)
        m2 = trimesh.transformations.rotation_matrix(np.pi / 2, rotate_axis_2)


        # rotation_matrix = m2 @ m1
        rotation_matrix = np.eye(4)
    return rotation_matrix

    pass

def get_cate_from_uid(uid, cate_dict, partnetmobility_cate_dict):
    if "/" not in uid:
        for cate in cate_dict.keys():
            for uid_ in cate_dict[cate]:
                if uid == uid_:
                    return cate
    else:
        for cate in partnetmobility_cate_dict.keys():
            for uid_ in partnetmobility_cate_dict[cate]:
                if uid == uid_:
                    return cate
    return "Unkonwn category,wrong uid"

def get_bbox_change_info(instances_infos, index):
    instance = instances_infos[index]
    bbox = instances_infos[index]["bbox"]
    # euler_angles = bbox[6:9]
    # 保持 stool、chandelier 直立
    if instance["embodiedscan_uid_cate"] in ["stool"]:
        bbox[7] = 0 if abs(bbox[7]) < 0.01 else bbox[7]
        bbox[8] = 0 if abs(bbox[8]) < 0.01 else bbox[8]
    if instance["embodiedscan_uid_cate"] in ["chandelier"]:
        bbox[7] = 0
        bbox[8] = 0

    return bbox

def process_one_instance(
        index, 
        uid, 
        instance, 
        glb_folder, 
        gapartnet_glb_folder, 
        partnet_mobility_glb_folder, 
        gen_assets_glb_folder, 
        exist_uid_path_dict, 
        useful_uid_rotate_dict,
        XY_same_scale, 
        not_compose_retrieval_glb_scene,
        cate_dict_new_bbox_info,
        use_xysort_in_retrieval
):
    glb_path = get_mesh_path_from_uid(uid, exist_uid_path_dict)
    # if "/" not in uid: #obja资产
    #     glb_path = os.path.join(glb_folder, exist_uid_path_dict[uid])#'/mnt/petrelfs/yinbaiqiao/EmbodiedScene/bed_chair/' + instance["3d_model"][0]+'.glb'
    # elif "partnet_mobility_part/" in uid: #gapartnet 资产
    #     glb_path = os.path.join(gapartnet_glb_folder, "PartNet", uid, "total_with_origin_rotation.glb")
    # elif "partnet_mobility/" in uid: #partnet_mobility 资产
    #     glb_path = os.path.join(partnet_mobility_glb_folder, uid,  "whole.obj")
    # elif "gen_assets/" in uid:
    #     glb_path = os.path.join(gen_assets_glb_folder, uid + ".glb")

    

    # 初始化总变换矩阵
    transform_to_origin = np.eye(4)  # 移动到原点之前的 transform
    transform_from_origin = np.eye(4)  # 从原点移动到最后位置的 transform
    scale_factors = np.ones(3)  # 在原点的拉伸 scale

    # # 尝试减少面数 ,有很多bug，放弃
    # n_faces = len(mesh.faces)
    # target_face_count = 1000
    # if n_faces > target_face_count:
    #     mesh = mesh.simplify_quadric_decimation(face_count = target_face_count)

    # mesh = trimesh.load(glb_path, force='mesh')
    # if isinstance(mesh, trimesh.path.path.Path3D): # Path3D：表示 3D 路径（如曲线、线段等），而不是三角形网格。
    #     print("Path3D")
    #     return None

    box_data = instance["bbox"]
    # 把bbox的中心先移动到原点来
    try:
        mesh_origen_centroid = cate_dict_new_bbox_info[uid]["center"] # mesh.bounding_box.centroid
        mesh_origen_extents = cate_dict_new_bbox_info[uid]["origin_extents"] # mesh.bounding_box_oriented.extents
    except:
        # 没有找到这个 uid 的 bbox info
        print(f"uid:{uid} not found in bbox info")
        return None
    
    # delete huge objects
    target_size = box_data[3:6]
    if target_size[0] > 15 or target_size[1] > 15 or target_size[2] > 15 or target_size[0]*target_size[1]*target_size[2] > 100:
        return None

    # mesh.apply_translation(-centroid) 
    transform_to_origin[:3, 3] -= mesh_origen_centroid

    # 由于glb、obj文件的坐标关系，导入的时候需要绕X轴旋转90度（embodiedscan 标注和 blender 采用的是 Z-up 坐标，而trimesh采用的是 Y-up 坐标）
    rotation_matrix = trimesh.transformations.euler_matrix(0, 0.5 * np.pi, 0, axes='rzxy')
    # mesh.apply_transform(rotation_matrix)
    transform_to_origin = rotation_matrix @ transform_to_origin

    # 如果是 partnet_mobility 的话, 由于 obj 没有对正, 需要绕Z轴旋转90度
    if "partnet_mobility/" in uid:
        rotation_matrix = trimesh.transformations.rotation_matrix(0.5 * np.pi, [0, 0, 1])
        # mesh.apply_transform(rotation_matrix)
        transform_to_origin = rotation_matrix @ transform_to_origin

    # 经过上面的步骤，所有物体的正面都朝向 Y 轴负方向，但是由于在Embodied所有物体的正面都是朝向 X 正方向，所以要把物体绕Z轴旋转90度
    rotation_matrix = trimesh.transformations.rotation_matrix(0.5 * np.pi, [0, 0, 1])
    # mesh.apply_transform(rotation_matrix)
    transform_to_origin = rotation_matrix @ transform_to_origin

    # 如果是 partnet_mobility 或者 gapartnet 的["pen", "remote", "phone"]类别的话,需要绕Z轴旋转180度，再绕Y轴旋转90度
    # TODO:有待确定 phone 和 pen的朝向是不是这样的
    if "partnet" in uid and instance["dataset_uid_cate"] in ["pen", "remote", "phone"]:
        rotation_matrix_1 = trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1])
        rotation_matrix_2 = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
        # mesh.apply_transform(rotation_matrix_1)
        # mesh.apply_transform(rotation_matrix_2)
        transform_to_origin = rotation_matrix_2 @ rotation_matrix_1 @ transform_to_origin

    # 根据useful_uid_rotate_dict 确定出物体转正的角度
    if uid in useful_uid_rotate_dict and useful_uid_rotate_dict[uid] != "None":
        rot_angle = useful_uid_rotate_dict[uid]
        rot_angle = normalize_rotation_to_90_degrees(rot_angle)
        rot_angle = float(rot_angle) * np.pi / 180.0
        rotation_matrix = trimesh.transformations.rotation_matrix(rot_angle, [0, 0, 1])
        # mesh.apply_transform(rotation_matrix)
        transform_to_origin = rotation_matrix @ transform_to_origin

    # # 把bbox的中心先移动到原点来
    # centroid = mesh.bounding_box.centroid
    # # mesh.apply_translation(-centroid) 
    # transform_to_origin[:3, 3] -= centroid

    # 根据target bbox和candidate bbox的长宽高确定出target bbox 需要的旋转transform矩阵
    # TODO:对于一些有明显正面的物体，我们将不允许它按照 bbox 的尺寸来旋转90度，因为这样会导致朝向错误
    # TODO:使用 jinkun 提供的已经转正的物体来替代 objaverse 的物体 。似乎不太可行。
    global cate_with_apparent_direction
    if use_xysort_in_retrieval:
        if instance["embodiedscan_uid_cate"] not in cate_with_apparent_direction:
            size_vec = np.array([mesh_origen_extents[0], mesh_origen_extents[1], mesh_origen_extents[2], 0])
            size_vec = transform_to_origin @ size_vec
            size_vec = np.abs(size_vec[0:3])
            current_size = size_vec #mesh.bounding_box_oriented.extents
            assert np.linalg.norm(current_size) -  np.linalg.norm(mesh_origen_extents) < 1e-6, f"mesh {index} size error"
            target_size = np.array(box_data[3:6])
            rotation_matrix = get_bbox_rotation_transform_z(target_size, current_size)
            transform_to_origin = rotation_matrix @ transform_to_origin
            # mesh.apply_transform(rotation_matrix)

    # 缩放mesh
    size_vec = np.array([mesh_origen_extents[0], mesh_origen_extents[1], mesh_origen_extents[2], 0])
    size_vec = transform_to_origin @ size_vec
    size_vec = np.abs(size_vec[0:3])
    current_size = size_vec # mesh.bounding_box_oriented.extents
    assert np.linalg.norm(current_size) -  np.linalg.norm(mesh_origen_extents) < 1e-6, f"mesh {index} size error"
    target_size = np.array(box_data[3:6])
    scale_factors = target_size / current_size
    if XY_same_scale:
        # pdb.set_trace()
        scale_factors[0:2] = scale_factors[0:2].mean()
    if instance["label"] in [46]: #如果是地毯，那么只在XY上面拉伸
        ## TODO 加入地毯错误朝向导致过度拉伸的判断：：：
        if scale_factors[2]/scale_factors[0] > 150 or scale_factors[2]/scale_factors[1] > 150:
            # 地毯错误朝向导致过度拉伸
            if scale_factors[2]/scale_factors[0] > scale_factors[2]/scale_factors[1]:
                trimesh.transformations.rotation_matrix(0.5 * np.pi, [0, 1, 0]) # 绕 Y 轴转90度
                transform_to_origin = rotation_matrix @ transform_to_origin
            else:
                trimesh.transformations.rotation_matrix(0.5 * np.pi, [1, 0, 0]) # 绕 X 轴转90度
                transform_to_origin = rotation_matrix @ transform_to_origin
            # 再计算一次scale
            size_vec = np.array([mesh_origen_extents[0], mesh_origen_extents[1], mesh_origen_extents[2], 0])
            size_vec = transform_to_origin @ size_vec
            size_vec = np.abs(size_vec[0:3])
            current_size = size_vec # mesh.bounding_box_oriented.extents
            assert np.linalg.norm(current_size) -  np.linalg.norm(mesh_origen_extents) < 1e-6, f"mesh {index} size error"
            target_size = np.array(box_data[3:6])
            scale_factors = target_size / current_size
        scale_factors[2] = 1

    if instance["category"] in ["clothes"]: #如果是衣服，那么选择最短拉伸
        min_scale = min(scale_factors)
        scale_factors = np.array([min_scale, min_scale, min_scale])
        
    # mesh.apply_scale(scale_factors)
    

    # 将旋转角度从弧度转换为欧拉角,进行欧拉角旋转
    euler_angles = np.array(box_data[6:9])

    # # 根据类别修改bbox 
    # 保持 stool、chandelier 直立
    if instance["embodiedscan_uid_cate"] in ["stool", "chandelier"]:
        euler_angles[1] = 0 if abs(euler_angles[1]) < 0.01 else euler_angles[1]
        euler_angles[2] = 0 if abs(euler_angles[2]) < 0.01 else euler_angles[2]
    if instance["embodiedscan_uid_cate"] in ["chandelier"]:
        # print("instance[\"embodiedscan_uid_cate\"] in [\"chandelier\"]")
        euler_angles[1] = 0 
        euler_angles[2] = 0 

    rotation_matrix = trimesh.transformations.euler_matrix(euler_angles[0], euler_angles[1], euler_angles[2], axes='rzxy') #关于物体坐标系的ZXY欧拉角
                        # trimesh.transformations.euler_matrix(euler_angles[2], euler_angles[1], euler_angles[0])
    # mesh.apply_transform(rotation_matrix)
    transform_from_origin = rotation_matrix @ transform_from_origin

    #进行平移
    center = np.array(box_data[0:3])
    # mesh.apply_translation(center)
    transform_from_origin[:3, 3] += center
    
    #最后再绕 X 轴旋转 -90 度，将整个场景保存为 Y-up，这样就使得场景有正确向上的朝向
    rotation_matrix = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    # mesh.apply_transform(rotation_matrix)
    transform_from_origin = rotation_matrix @ transform_from_origin

    # print(f"mesh {index} has been processed.")
    if not_compose_retrieval_glb_scene:
        return None, index, uid, transform_to_origin, scale_factors, transform_from_origin, glb_path
    else:
        mesh = trimesh.load(glb_path, force='mesh')
        if isinstance(mesh, trimesh.path.path.Path3D): # Path3D：表示 3D 路径（如曲线、线段等），而不是三角形网格。
            print(f"uid:{uid} is Path3D type")
            return None
        mesh.apply_transform(transform_to_origin)
        mesh.apply_scale(scale_factors)
        mesh.apply_transform(transform_from_origin)
        return mesh, index, uid, transform_to_origin, scale_factors, transform_from_origin, glb_path

def load_info_for_compose_scene(
        not_compose_retrieval_glb_scene = True,
        XY_same_scale = False,
        use_xysort_in_retrieval=False,
        scene_files_dir = "/cpfs01/user/zhongweipeng/Projects/layout/embodiedscene/scan",
        glb_folder = "/oss/lvzhaoyang/chenkaixu/objaverse_mesh/glbs/",
        gapartnet_glb_folder = "/cpfs01/shared/landmark_3dgen/xuxudong_group/zhongweipeng/datasets/GAPartNet/dataset",
        gen_assets_glb_folder = "/cpfs01/shared/landmark_3dgen/xuxudong_group/zhongweipeng/projects/hunyuan3d/assets_for_retrieval",
        partnet_mobility_glb_folder = "/oss/zhongweipeng/data/3d_asset" ,#"/oss/lianxinyu/urdfs",#  # 
        gobja_to_obja_json_path = "/cpfs01/user/zhongweipeng/Projects/layout/gobjaverse_index_to_objaverse.json",
        useful_uid_rotate_dict_path = "/cpfs01/user/zhongweipeng/Projects/layout/useful_uid_rotate_dict.json",
        cate_dict_new_bbox_info_path = "/cpfs01/user/zhongweipeng/Projects/layout/json_files/AssetSizeReader/uid_bbox_info_dict_total.json", #'/cpfs01/user/zhongweipeng/Projects/layout/cate_dict_new_bbox_info.json',
        cate_to_delete_path = "/cpfs01/user/zhongweipeng/Projects/layout/cate_to_delete.json"
):
    # # paths
    # input_instance_infos_path = os.path.join(scene_files_dir, scene_name, "instances_info_with_retri_uid_withpartnetmobility.json")
    # output_instance_infos_path = input_instance_infos_path
    # # 读取 JSON 数据
    # with open(input_instance_infos_path, 'r') as file:
    #     instances_infos = json.load(file)

    with open(gobja_to_obja_json_path, 'r') as file:
        gobja_to_obja = json.load(file)
    with open(useful_uid_rotate_dict_path, 'r') as file:
        useful_uid_rotate_dict = json.load(file)
    with open(cate_dict_new_bbox_info_path, 'r') as file:
        cate_dict_new_bbox_info = json.load(file)
    with open(cate_to_delete_path, "r")as file:
        cate_to_delete = json.load(file)
        cate_to_delete = list(cate_to_delete.keys())

    exist_uid_path_dict = {path.split("/")[-1].split(".")[0]:path for path in gobja_to_obja.values()}

    compose_scene_info_dict = {
        "not_compose_retrieval_glb_scene":not_compose_retrieval_glb_scene,
        "XY_same_scale":XY_same_scale,
        "scene_files_dir":scene_files_dir,
        "glb_folder": glb_folder,
        "gapartnet_glb_folder": gapartnet_glb_folder,
        "gen_assets_glb_folder": gen_assets_glb_folder,
        "partnet_mobility_glb_folder": partnet_mobility_glb_folder,
        "useful_uid_rotate_dict": useful_uid_rotate_dict,
        "cate_dict_new_bbox_info": cate_dict_new_bbox_info,
        "exist_uid_path_dict": exist_uid_path_dict,
        "cate_to_delete": cate_to_delete,
        "use_xysort_in_retrieval" : use_xysort_in_retrieval
    }
    return compose_scene_info_dict


def main_mt(
        scene_name,
        not_compose_retrieval_glb_scene,
        XY_same_scale = False,
        scene_files_dir = "/cpfs01/user/zhongweipeng/Projects/layout/embodiedscene/scan",
        glb_folder = "/oss/lvzhaoyang/chenkaixu/objaverse_mesh/glbs/",
        gapartnet_glb_folder = "/cpfs01/shared/landmark_3dgen/xuxudong_group/zhongweipeng/datasets/GAPartNet/dataset",
        gen_assets_glb_folder = "/cpfs01/shared/landmark_3dgen/xuxudong_group/zhongweipeng/projects/hunyuan3d/assets_for_retrieval",
        partnet_mobility_glb_folder = "/oss/zhongweipeng/data/3d_asset" ,#"/oss/lianxinyu/urdfs",#  # 
        gobja_to_obja_json_path = "/cpfs01/user/zhongweipeng/Projects/layout/gobjaverse_index_to_objaverse.json",
        useful_uid_rotate_dict_path = "/cpfs01/user/zhongweipeng/Projects/layout/useful_uid_rotate_dict.json",
        cate_dict_new_bbox_info_path = '/cpfs01/user/zhongweipeng/Projects/layout/cate_dict_new_bbox_info.json'
):
    
    # JSON 文件路径
    # json_file_path = f"/cpfs01/user/zhongweipeng/Projects/layout/embodiedscene/scan/{scene_name}/instances_info_with_retri_uid.json"
    # json_file_path = f"/cpfs01/user/zhongweipeng/Projects/layout/embodiedscene/scan/{scene_name}/instances_info_with_retri_uid_withgapartnet.json"
    # json_file_path = f"/cpfs01/user/zhongweipeng/Projects/layout/embodiedscene/scan/{scene_name}/instances_info_with_retri_uid_withpartnetmobility.json"
                        #'/mnt/petrelfs/yinbaiqiao/EmbodiedScene/json_folder/more_info.json'

    # gapartnet_glb_folder = "/cpfs01/shared/landmark_3dgen/xuxudong_group/zhongweipeng/datasets/GAPartNet/dataset"
    # partnet_mobility_glb_folder = "/oss/lianxinyu/urdfs"# "/oss/zhongweipeng/data/3d_asset" # 
    # gobja_to_obja_json_path = "/cpfs01/user/zhongweipeng/Projects/layout/gobjaverse_index_to_objaverse.json"

    # paths
    input_instance_infos_path = os.path.join(scene_files_dir, scene_name, "instances_info_with_retri_uid_withpartnetmobility.json")
    output_instance_infos_path = input_instance_infos_path

    # 读取 JSON 数据
    with open(input_instance_infos_path, 'r') as file:
        instances_infos = json.load(file)
    with open(gobja_to_obja_json_path, 'r') as file:
        gobja_to_obja = json.load(file)
    with open(useful_uid_rotate_dict_path, 'r') as file:
        useful_uid_rotate_dict = json.load(file)
    with open(cate_dict_new_bbox_info_path, 'r') as file:
        cate_dict_new_bbox_info = json.load(file)

    exist_uid_path_dict = {path.split("/")[-1].split(".")[0]:path for path in gobja_to_obja.values()}
    # with open("/cpfs01/user/zhongweipeng/Projects/layout/exist_uid_path_dict.json", "w") as f:
    #     json.dump(exist_uid_path_dict, f, indent=4)
    #     pdb.set_trace() 
    # # 初始化一个空列表来存储文件名
    # file_names = []

    # # 检查路径是否存在
    # # 列出文件夹下的所有文件和文件夹
    # for item in os.listdir(glb_folder):
    #     # 构造完整的文件或文件夹路径
    #     name = (item.split('.'))[0]
    #     # print(name)
    #     file_names.append(name)  # 将文件名添加到列表中

    # scene_data = []
    # trimesh_meshes = []
    if not not_compose_retrieval_glb_scene:
        scene = trimesh.scene.Scene()


    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = []

        for index, instance in enumerate(instances_infos):

            # 遍历每个实例的键值对
            if instance["category"] not in ["wall" ,"ceiling", "floor", "pipe"] : #如果不属于 wall/ceiling/floor 这一类
                

                if instance["3d_model"]=='':
                    print("3d_model is empty. No retrieval reslut.")

                else:
                    uid = instance["3d_model"][0]

                    if uid not in exist_uid_path_dict.keys() and "/" not in uid:  
                        print(f"3d_model uid {uid} not exist in glb folder.")

                    else:
                        # 提交任务
                        futures.append(executor.submit(process_one_instance, index, uid, instance, 
                                                    glb_folder, gapartnet_glb_folder, partnet_mobility_glb_folder, gen_assets_glb_folder, exist_uid_path_dict,
                                                    useful_uid_rotate_dict, XY_same_scale, not_compose_retrieval_glb_scene, cate_dict_new_bbox_info))

            elif instance["label"] == 276:
                    # print("A Box!")
                    box_data = instance["bbox"]
                    center = np.array(box_data[0:3])

                    # extents = np.array([box_data[3]-box_data[0], box_data[4] - box_data[1], box_data[5] - box_data[2]])
                    extents = np.array(box_data[3:6])
                    # 将旋转角度从弧度转换为欧拉角
                    euler_angles = np.array(box_data[6:9])
                    
                    rotation_matrix = trimesh.transformations.euler_matrix(euler_angles[0], euler_angles[1], euler_angles[2], axes='rzxy')
                    transform = np.eye(4)
                    transform[:3, 3] = center

                    transform[:3, :3] = rotation_matrix[:3, :3]
                    bbox = trimesh.primitives.Box(extents=extents, transform=transform)
                    transparent_material = trimesh.visual.material.PBRMaterial(
                    baseColorFactor=[1.0, 0.0, 0.0, 0.1]  # RGB颜色设置为红色，alpha设置为0
                    #     metallicFactor=0.0,  # 金属度
                    #     roughnessFactor=0.5  # 粗糙度
                    )
                    bbox.visual.material = transparent_material
                    # scene.add_geometry(bbox)    

        for future in as_completed(futures): 
            ans = future.result()
            if ans == None:
                continue
            mesh, index, uid, transform_to_origin, scale_factors, transform_from_origin, mesh_glb_path = ans
            if not_compose_retrieval_glb_scene:
                geom_name = f'{scene_name}_{index}_{uid}'
                instances_infos[index]["geom_name"] = geom_name
                instances_infos[index]["mesh_glb_path"] = mesh_glb_path
                instances_infos[index]["transform_to_origin"] = transform_to_origin.tolist()
                instances_infos[index]["scale_factors"] = scale_factors.tolist()
                instances_infos[index]["transform_from_origin"] = transform_from_origin.tolist()
            else:
                if mesh != None:
                    geom_name = f'{scene_name}_{index}_{uid}'
                    scene.add_geometry(mesh, geom_name = geom_name)
                    instances_infos[index]["geom_name"] = geom_name
                    instances_infos[index]["mesh_glb_path"] = mesh_glb_path
                    instances_infos[index]["transform_to_origin"] = transform_to_origin.tolist()
                    instances_infos[index]["scale_factors"] = scale_factors.tolist()
                    instances_infos[index]["transform_from_origin"] = transform_from_origin.tolist()


    # 保存json_file_path
    with open(output_instance_infos_path, 'w') as file:
        json.dump(instances_infos, file, indent=4)

    # 去掉 scene glb 的材质
    if not not_compose_retrieval_glb_scene:
        from trimesh.visual.texture import TextureVisuals
        empty_visual = TextureVisuals()
        for geometry in scene.geometry.values():
            geometry.visual = empty_visual

        # 保存 scene glb
        t = time.time()
        scene_path = os.path.join(scene_files_dir, scene_name) # f"/cpfs01/user/zhongweipeng/Projects/layout/embodiedscene/scan/{scene_name}"
        os.makedirs(scene_path, exist_ok=True)
        trimesh.exchange.export.export_mesh(scene, os.path.join(scene_path, "retrieval_scene_16_withpartnetmobility.glb"))
        print(f"save glb file time: {time.time() - t}")

def compose_one_scene(
    scene_name,
    compose_scene_info_dict
):
    for key in compose_scene_info_dict.keys():
        if key == "not_compose_retrieval_glb_scene":
            not_compose_retrieval_glb_scene = compose_scene_info_dict[key]
        elif key == "XY_same_scale":
            XY_same_scale = compose_scene_info_dict[key]
        elif key == "scene_files_dir":
            scene_files_dir = compose_scene_info_dict[key]
        elif key == "glb_folder":                   
            glb_folder = compose_scene_info_dict[key]
        elif key == "gapartnet_glb_folder":
            gapartnet_glb_folder = compose_scene_info_dict[key]
        elif key == "gen_assets_glb_folder":  
            gen_assets_glb_folder = compose_scene_info_dict[key]
        elif key == "partnet_mobility_glb_folder":
            partnet_mobility_glb_folder = compose_scene_info_dict[key]
        elif key == "useful_uid_rotate_dict":
            useful_uid_rotate_dict = compose_scene_info_dict[key]
        elif key == "cate_dict_new_bbox_info":
            cate_dict_new_bbox_info = compose_scene_info_dict[key]
        elif key == "exist_uid_path_dict":
            exist_uid_path_dict = compose_scene_info_dict[key]
        elif key == "cate_to_delete":
            cate_to_delete = compose_scene_info_dict[key]
        elif key == "use_xysort_in_retrieval":
            use_xysort_in_retrieval = compose_scene_info_dict[key]
            

    # paths
    input_instance_infos_path = os.path.join(scene_files_dir, scene_name, "instances_info_with_retri.json")
    output_instance_infos_path = os.path.join(scene_files_dir, scene_name, "instances_info_with_retri_with_compose.json")
    # 读取 JSON 数据
    with open(input_instance_infos_path, 'r') as file:
        instances_infos = json.load(file)

    if not not_compose_retrieval_glb_scene:
        scene = trimesh.scene.Scene()
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for index, instance in enumerate(instances_infos):
            # 遍历每个实例的键值对
            if instance["category"] not in cate_to_delete: # ["wall" ,"ceiling", "floor", "pipe"] : #如果不属于 wall/ceiling/floor 这一类

                if instance["3d_model"]=='':
                    print("3d_model is empty. No retrieval reslut.")
                else:
                    uid = instance["3d_model"][0]
                    if uid not in exist_uid_path_dict.keys() and "/" not in uid:  
                        print(f"3d_model uid {uid} not exist in glb folder.")

                    else:
                        # 提交任务
                        futures.append(executor.submit(process_one_instance, index, uid, instance, 
                                                    glb_folder, gapartnet_glb_folder, partnet_mobility_glb_folder, gen_assets_glb_folder, exist_uid_path_dict,
                                                    useful_uid_rotate_dict, XY_same_scale, not_compose_retrieval_glb_scene, cate_dict_new_bbox_info, use_xysort_in_retrieval))
        with tqdm(total=len(futures), desc=f"Composing {scene_name}...") as pbar:
            for future in as_completed(futures): 
                pbar.update(1)
                ans = future.result()
                if ans == None:
                    continue
                mesh, index, uid, transform_to_origin, scale_factors, transform_from_origin, mesh_glb_path = ans

                if mesh != None or not_compose_retrieval_glb_scene:
                    geom_name = f'{scene_name}_{index}_{uid}'
                    instances_infos[index]["geom_name"] = geom_name
                    instances_infos[index]["mesh_glb_path"] = mesh_glb_path
                    instances_infos[index]["transform_to_origin"] = transform_to_origin.tolist()
                    instances_infos[index]["scale_factors"] = scale_factors.tolist()
                    instances_infos[index]["transform_from_origin"] = transform_from_origin.tolist()
                    if instances_infos[index]["category"] in ["stool", "chandelier"]:
                        new_bbox = get_bbox_change_info(instances_infos, index)
                        instances_infos[index]["bbox"] = new_bbox
                
                if mesh != None and not not_compose_retrieval_glb_scene:
                    geom_name = f'{scene_name}_{index}_{uid}'
                    scene.add_geometry(mesh, geom_name = geom_name)

                # # change bbox
                # if instances_infos[index]["category"] in ["stool", "chandelier"]:
                #     new_bbox = get_bbox_change_info(instances_infos, index)
                # else: 
                #     new_bbox = instances_infos[index]["bbox"]

                # if not_compose_retrieval_glb_scene:
                #     geom_name = f'{scene_name}_{index}_{uid}'
                #     instances_infos[index]["geom_name"] = geom_name
                #     instances_infos[index]["mesh_glb_path"] = mesh_glb_path
                #     instances_infos[index]["transform_to_origin"] = transform_to_origin.tolist()
                #     instances_infos[index]["scale_factors"] = scale_factors.tolist()
                #     instances_infos[index]["transform_from_origin"] = transform_from_origin.tolist()
                #     instances_infos[index]["bbox"] = new_bbox
                # else:
                #     if mesh != None:
                #         geom_name = f'{scene_name}_{index}_{uid}'
                #         scene.add_geometry(mesh, geom_name = geom_name)
                #         instances_infos[index]["geom_name"] = geom_name
                #         instances_infos[index]["mesh_glb_path"] = mesh_glb_path
                #         instances_infos[index]["transform_to_origin"] = transform_to_origin.tolist()
                #         instances_infos[index]["scale_factors"] = scale_factors.tolist()
                #         instances_infos[index]["transform_from_origin"] = transform_from_origin.tolist()
                #         instances_infos[index]["bbox"] = new_bbox

    # 保存json_file_path
    with open(output_instance_infos_path, 'w') as file:
        json.dump(instances_infos, file, indent=4)

    # 去掉 scene glb 的材质
    if not not_compose_retrieval_glb_scene:
        from trimesh.visual.texture import TextureVisuals
        empty_visual = TextureVisuals()
        for geometry in scene.geometry.values():
            geometry.visual = empty_visual

        # 保存 scene glb
        t = time.time()
        scene_path = os.path.join(scene_files_dir, scene_name) # f"/cpfs01/user/zhongweipeng/Projects/layout/embodiedscene/scan/{scene_name}"
        os.makedirs(scene_path, exist_ok=True)
        trimesh.exchange.export.export_mesh(scene, os.path.join(scene_path, "retrieval_scene_16_withpartnetmobility.glb"))
        print(f"save glb file time: {time.time() - t}")


def main(scene_name = "scannet/scene0000_00"):
    
    # JSON 文件路径
    json_file_path = f"/cpfs01/user/zhongweipeng/Projects/layout/embodiedscene/scan/{scene_name}/instances_info_with_retri_uid_withpartnetmobility.json"

    glb_folder = "/oss/lvzhaoyang/chenkaixu/objaverse_mesh/glbs/"
    gapartnet_glb_folder = "/cpfs01/shared/landmark_3dgen/xuxudong_group/zhongweipeng/datasets/GAPartNet/dataset"
    partnet_mobility_glb_folder = "/oss/lianxinyu/urdfs"

    gobja_to_obja_json_path = "/cpfs01/user/zhongweipeng/Projects/layout/gobjaverse_index_to_objaverse.json"
    # 读取 JSON 数据
    with open(json_file_path, 'r') as file:
        instances_infos = json.load(file)
    with open(gobja_to_obja_json_path, 'r') as file:
        gobja_to_obja = json.load(file)

    exist_uid_path_dict = {path.split("/")[-1].split(".")[0]:path for path in gobja_to_obja.values()}

    scene = trimesh.scene.Scene()

    i = 0


    for instance in instances_infos:
        i += 1
        # 遍历每个实例的键值对
        if instance["label"] not in [276 ,49, 104] : #如果不属于 wall/ceiling/floor 这一类
            uid = instance["3d_model"][0]

            if instance["3d_model"]=='':
                print("3d_model is empty. No retrieval reslut.")

            elif uid not in exist_uid_path_dict.keys() and "/" not in uid: 
                print(f"3d_model uid {uid} not exist in glb folder.")

            else:
                # 提交任务
                mesh, i, uid, transform_to_origin, scale_factors, transform_from_origin = process_one_instance(i, uid, instance, 
                                                glb_folder, gapartnet_glb_folder, partnet_mobility_glb_folder, exist_uid_path_dict)

                if mesh != None:
                    scene.add_geometry(mesh, geom_name='{}_{}'.format(i, uid))
                    instance["transform_to_origin"] = transform_to_origin.tolist()
                    instance["scale_factors"] = scale_factors.tolist()
                    instance["transform_from_origin"] = transform_from_origin.tolist()

        else:
                # print("A Box!")
                box_data = instance["bbox"]
                center = np.array(box_data[0:3])

                # extents = np.array([box_data[3]-box_data[0], box_data[4] - box_data[1], box_data[5] - box_data[2]])
                extents = np.array(box_data[3:6])
                # 将旋转角度从弧度转换为欧拉角
                euler_angles = np.array(box_data[6:9])
                
                rotation_matrix = trimesh.transformations.euler_matrix(euler_angles[2], euler_angles[1], euler_angles[0])
                transform = np.eye(4)
                transform[:3, 3] = center

                transform[:3, :3] = rotation_matrix[:3, :3]
                bbox = trimesh.primitives.Box(extents=extents, transform=transform)
                transparent_material = trimesh.visual.material.PBRMaterial(
                baseColorFactor=[1.0, 0.0, 0.0, 0.1]  # RGB颜色设置为红色，alpha设置为0
                #     metallicFactor=0.0,  # 金属度
                #     roughnessFactor=0.5  # 粗糙度
                )
                bbox.visual.material = transparent_material
                # scene.add_geometry(bbox)    


    
    # 保存json_file_path
    with open(json_file_path, 'w') as file:
        json.dump(instances_infos, file, indent=4)

    # 去掉 scene glb 的材质
    from trimesh.visual.texture import TextureVisuals
    empty_visual = TextureVisuals()
    for geometry in scene.geometry.values():
        geometry.visual = empty_visual

    # 保存 scene glb
    t = time.time()
    scene_path = f"/cpfs01/user/zhongweipeng/Projects/layout/embodiedscene/scan/{scene_name}"
    os.makedirs(scene_path, exist_ok=True)
    trimesh.exchange.export.export_mesh(scene, os.path.join(scene_path, "retrieval_scene_16_withpartnetmobility.glb"))
    print(f"save glb file time: {time.time() - t}")

def test():

    rotation_matrix = trimesh.transformations.euler_matrix(3.1415926535, 0, 0, axes='rzxy') #关于物体坐标系的ZXY欧拉角
    print(rotation_matrix)

def parse_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_name", required=True, type=str, help = "scene name, e.g. scannet/scene0000_00")
    parser.add_argument("--not_compose_retrieval_glb_scene", action="store_true", help="不把retrieval的物体组合为glb场景")
    # parser.add_argument("--scene_files_dir", required=True, type=str, help = "")
    # parser.add_argument("--semantic_label_file_path", required=True, type=str, help = "")
    # parser.add_argument("--cate_dict_new_path", required=True, type=str, help = "")
    # parser.add_argument("--partnetmobility_cate_addto_embodiedscan_path", required=True, type=str, help = "")
    # parser.add_argument("--partnetmobility_cate_dict_path", required=True, type=str, help = "")
    # parser.add_argument("--Cap3D_csv_file_path", required=True, type=str, help = "")
    # parser.add_argument("--uid_size_dict_path", required=True, type=str, help = "")
    # parser.add_argument("--partnetmobility_size_dict_path", required=True, type=str, help = "")

    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    t0 = time.time()


    args = parse_arguments()
    main_mt(
        args.scene_name,
        args.not_compose_retrieval_glb_scene
    )  


    # main_mt("scannet/scene0000_00")     

    print(f"Total time: {time.time() - t0}")