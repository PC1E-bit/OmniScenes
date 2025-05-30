import os
import json
import trimesh
import numpy as np
from tqdm import tqdm

import threading
from concurrent.futures import ThreadPoolExecutor

def load_info_for_uid():
    '''
    用于load info 来根据 uid 获取 mesh 的路径
    '''
    # load 必要信息
    gobja_to_obja_path = "/cpfs01/shared/landmark_3dgen/xuxudong_group/zhongweipeng/projects/layout/asset_process/gobjaverse_index_to_objaverse.json"
    useful_uid_rotate_dict_path = "/cpfs01/shared/landmark_3dgen/xuxudong_group/zhongweipeng/projects/layout/asset_process/useful_uid_rotate_dict.json"
    partnetmobility_cate_dict_path = "/cpfs01/shared/landmark_3dgen/xuxudong_group/zhongweipeng/projects/layout/asset_process/partnetmobility_cate_dict.json"

    with open(gobja_to_obja_path, 'r') as file:
        gobja_to_obja = json.load(file)
    exist_uid_path_dict = {path.split("/")[-1].split(".")[0]:path for path in gobja_to_obja.values()}
    # 旋转信息
    with open(useful_uid_rotate_dict_path, "r") as f:
        useful_uid_rotate_dict = json.load(f)
    # uid-类别信息
    with open(partnetmobility_cate_dict_path, "r") as f:
        cate_dict = json.load(f)
        partnetmobility_uid_cate_dict = {}
        for cate, uids in cate_dict.items():
            for p_uid in uids:
                partnetmobility_uid_cate_dict[p_uid] = cate

    info_for_uid = {
        "exist_uid_path_dict": exist_uid_path_dict,
        "useful_uid_rotate_dict": useful_uid_rotate_dict,
        "partnetmobility_uid_cate_dict": partnetmobility_uid_cate_dict
    }

    return info_for_uid


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

def get_mesh_path_from_uid(
        uid, 
        exist_uid_path_dict = None,
        glb_folder = "/oss/lvzhaoyang/chenkaixu/objaverse_mesh/glbs/",
        gapartnet_glb_folder = "/cpfs01/shared/landmark_3dgen/xuxudong_group/zhongweipeng/datasets/GAPartNet/dataset",
        gen_assets_glb_folder = "/cpfs01/shared/landmark_3dgen/xuxudong_group/zhongweipeng/projects/hunyuan3d/assets_for_retrieval",
        partnet_mobility_glb_folder = "/oss/zhongweipeng/data/3d_asset" ,
        gobja_to_obja_json_path = "/cpfs01/user/zhongweipeng/Projects/layout/gobjaverse_index_to_objaverse.json",
        
):
    """
    根据uid获取对应的mesh路径
    """
    if exist_uid_path_dict == None :
        with open(gobja_to_obja_json_path, 'r') as file:
            gobja_to_obja = json.load(file)
        exist_uid_path_dict = {path.split("/")[-1].split(".")[0]:path for path in gobja_to_obja.values()}

    if "/" not in uid: # objaverse 资产
        mesh_path = os.path.join(glb_folder, exist_uid_path_dict[uid])
        mesh_source = "objaverse"

    elif "gen_assets/" in uid: # 生成资产
        mesh_path = os.path.join(gen_assets_glb_folder, uid + ".glb")
        mesh_source = "gen_assets"
        
    else:  # partnet-mobility 资产
        if "partnet_mobility_part/" in uid: # gapartnet 资产
            mesh_path = os.path.join(gapartnet_glb_folder, "PartNet", uid, "total_with_origin_rotation.glb")
        elif "partnet_mobility/" in uid: # partnet_mobility 资产
            mesh_path = os.path.join(partnet_mobility_glb_folder, uid,  "whole.obj")

        mesh_source = "partnet_mobility"

    if os.path.exists(mesh_path):
        return mesh_path, mesh_source
    else:
        raise FileNotFoundError(f"Mesh path {mesh_path} does not exist.")

def get_mesh_from_uid(
        uid, 
        info_for_uid = None,
        force_mesh = True,
        return_transform_to_origin = False
    ):
    """
    根据uid获取对应的 trimesh.Trimesh 对象
    得到的所有 mesh 在 Z-up 坐标系中 正面朝向 X 轴正方向
    """
    # load info
    if info_for_uid is None:
        info_for_uid = load_info_for_uid()
    exist_uid_path_dict = info_for_uid["exist_uid_path_dict"]
    useful_uid_rotate_dict = info_for_uid["useful_uid_rotate_dict"]
    partnetmobility_uid_cate_dict = info_for_uid["partnetmobility_uid_cate_dict"]
    

    # 获取 mesh 路径
    mesh_path, mesh_source = get_mesh_path_from_uid(uid, exist_uid_path_dict)

    # 初始化总变换矩阵
    transform_to_origin = np.eye(4)  # 移动到原点的 transform

    # load mesh
    if force_mesh:
        mesh = trimesh.load(mesh_path, force='mesh')
    else:
        mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.path.path.Path3D): # Path3D：表示 3D 路径（如曲线、线段等），而不是三角形网格。
        print("Path3D")
        raise NotImplementedError("Path3D is not supported")

    # 把bbox的中心先移动到原点来
    mesh_origen_centroid =  mesh.bounding_box.centroid
    transform_to_origin[:3, 3] -= mesh_origen_centroid

    # 由于glb、obj文件的坐标关系，导入的时候需要绕X轴旋转90度（embodiedscan 标注和 blender 采用的是 Z-up 坐标，而trimesh采用的是 Y-up 坐标）
    rotation_matrix = trimesh.transformations.euler_matrix(0, 0.5 * np.pi, 0, axes='rzxy')
    transform_to_origin = rotation_matrix @ transform_to_origin

    # 如果是 partnet_mobility 的话, 需要绕Z轴旋转90度
    if "partnet_mobility/" in uid:
        rotation_matrix = trimesh.transformations.rotation_matrix(0.5 * np.pi, [0, 0, 1])
        transform_to_origin = rotation_matrix @ transform_to_origin

    # 经过上面的步骤，所有物体的正面都朝向 Y 轴负方向，但是由于在Embodied所有物体的正面都是朝向 X 正方向，所以要把物体绕Z轴旋转90度
    rotation_matrix = trimesh.transformations.rotation_matrix(0.5 * np.pi, [0, 0, 1])
    transform_to_origin = rotation_matrix @ transform_to_origin

    # 如果是 partnet_mobility 的["pen", "remote", "phone"]类别的话,需要绕Z轴旋转180度，再绕Y轴旋转90度
    if mesh_source == "partnet_mobility" and partnetmobility_uid_cate_dict[uid] in ["pen", "remote", "phone"]:
        rotation_matrix_1 = trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1])
        rotation_matrix_2 = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
        transform_to_origin = rotation_matrix_2 @ rotation_matrix_1 @ transform_to_origin

    # 根据 useful_uid_rotate_dict 确定出物体转正的角度
    if uid in useful_uid_rotate_dict and useful_uid_rotate_dict[uid] != "None":
        rot_angle = useful_uid_rotate_dict[uid]
        rot_angle = normalize_rotation_to_90_degrees(rot_angle)
        rot_angle = float(rot_angle) * np.pi / 180.0
        rotation_matrix = trimesh.transformations.rotation_matrix(rot_angle, [0, 0, 1])
        transform_to_origin = rotation_matrix @ transform_to_origin


    mesh.apply_transform(transform_to_origin)

    # 导出
    # name = uid.replace("/", "_")
    # mesh.export(f"/cpfs01/user/zhongweipeng/Projects/layout/codes_ybq/assets_process/{name}.glb")

    if return_transform_to_origin:
        # 返回 mesh 和 transform_to_origin
        return mesh, transform_to_origin
    else:
        return mesh

def output_scene_glb_from_instances_infos(instances_infos, output_glb_path, need_texture = False, 
                                          info_for_uid = None):
    """
    多线程版本的save_scene函数，从origin—glb出发
    instances_info格式：
        {
        "id": 0,
        "category": "table",
        "model_uid": "partnet_mobility/19836",
        "bbox": [
            -0.13590669212526996,
            0.9468732844139846,
            0.37732592929995423,
            1.20027183494089,
            1.2072017818500944,
            0.7449189902064007,
            -2.4517364761819476,
            0.0,
            -0.0
        ]
    }
    """
    scene = trimesh.scene.Scene()
    lock = threading.Lock()

    if info_for_uid is None:
        info_for_uid = load_info_for_uid()

    def process_single_instance(instance):
        """
        处理单个实例的函数，用于多线程处理
        """

        if instance["model_uid"] == '':
            print("3d_model is empty. No retrieval reslut.")
            return None

        uid = instance["model_uid"]
        geometry_name = instance["category"] + "-" + instance["model_uid"]

        mesh = get_mesh_from_uid(
                    uid, 
                    info_for_uid,
                    force_mesh = False if need_texture else True
                )
        


        # transform
        transform_final = np.eye(4)
        box_data = instance["bbox"]

        # scale
        mesh_size = mesh.bounding_box.extents
        target_size = np.array(box_data[3:6])
        scale = target_size / mesh_size
        scale_matrix = np.array([
                                    [scale[0], 0, 0, 0],
                                    [0, scale[1], 0, 0],
                                    [0, 0, scale[2], 0],
                                    [0, 0, 0, 1]
                                ])
        transform_final = scale_matrix @ transform_final

        # rotation
        euler_angles = np.array(box_data[6:9])
        rotation_matrix = trimesh.transformations.euler_matrix(euler_angles[0], euler_angles[1], euler_angles[2], axes='rzxy')
        transform_final = rotation_matrix @ transform_final

        # translation
        center = np.array(box_data[0:3])
        transform_final[:3, 3] = center

        #最后再绕 X 轴旋转 -90 度，将整个场景保存为 Y-up，这样就使得场景有正确向上的朝向
        rotation_matrix = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        transform_final = rotation_matrix @ transform_final

        total_transform = transform_final
        
        # load mesh
        mesh.apply_transform(total_transform)
        
        return [mesh], geometry_name

    def process_instance(index):
        instance = instances_infos[index]
        result = process_single_instance(instance)

        if result is not None:
            meshes, geometry_name = result
            with lock:
                for idx, mesh_item in enumerate(meshes):
                    scene.add_geometry(mesh_item, geom_name=f"{geometry_name}_{idx}")
            print(f"process_instance {index} done")

    # 使用线程池并行处理所有实例
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for index in range(len(instances_infos)):
            futures.append(executor.submit(process_instance, index))
        
        # 等待所有任务完成并处理异常
        for future in futures: #tqdm(futures, desc="exporting scene..."):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing instance: {str(e)}")

    # 去掉 scene glb 的材质
    # print(f"add texture to glb :{need_texture}")
    if not need_texture:
        from trimesh.visual.texture import TextureVisuals
        empty_visual = TextureVisuals()
        for geometry in scene.geometry.values():
            geometry.visual = empty_visual
        
    if output_glb_path != None:
        trimesh.exchange.export.export_mesh(scene, output_glb_path)
    return scene


if __name__ == "__main__":
    
    # mesh = get_mesh_from_uid(uid = "partnet_mobility/9281")
    # print(mesh.bounding_box.centroid)
    # import pdb
    # pdb.set_trace()
    # # mesh.export("/cpfs01/user/zhongweipeng/Projects/layout/codes_ybq/utils/sample.glb")

    json_path = ""
    output_glb_path = "/cpfs01/user/zhongweipeng/Projects/layout/codes_ybq/utils/sample.glb"
    with open(json_path, "r")as f:
        instances_infos = json.load(f)

    output_scene_glb_from_instances_infos(instances_infos, 
                                            output_glb_path, 
                                            need_texture = True, 
                                            info_for_uid = None)


    