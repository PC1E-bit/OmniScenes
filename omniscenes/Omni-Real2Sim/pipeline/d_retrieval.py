import os
import csv
import pdb
import json
import copy
import time
import math

import torch
import pickle
import trimesh
import argparse
import open_clip
import numpy as np
import compress_json
import compress_pickle

from tqdm import tqdm
from typing import Dict, Any
import torch.nn.functional as F
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
import sys
sys.path.extend(["./"])
# from codes_ybq.uid_utils.uid_utils import get_mesh_path_from_uid
# sys.path.extend(["./optimiz_large_bbox", "./Sapien_Scene_Sim/optimiz_large_bbox"])
from Sapien_Scene_Sim.optimiz_large_bbox.bbox_utils import euler_to_matrix, get_obb_vertices, is_bbox1_inside_bbox2, is_point_inside_bbox, get_totated_bbox_iou
from Sapien_Scene_Sim.optimiz_large_bbox.bbox_utils  import get_point_face_distance, get_bbox_Z_faces





ASSETS_VERSION = os.environ.get("ASSETS_VERSION", "2023_09_23")
# OBJATHOR_ASSETS_BASE_DIR = os.environ.get(
#     "OBJATHOR_ASSETS_BASE_DIR", os.path.expanduser(f"/cpfs01/user/yinbaiqiao/understand_3d/indoor/Objathor")
# )
OBJATHOR_ASSETS_BASE_DIR = os.environ.get(
    "OBJATHOR_ASSETS_BASE_DIR", os.path.expanduser(f"/cpfs01/user/zhongweipeng/Projects/layout/Holodeck/objathor-assets")
)
OBJATHOR_VERSIONED_DIR = os.path.join(OBJATHOR_ASSETS_BASE_DIR, ASSETS_VERSION)
OBJATHOR_ASSETS_DIR = os.path.join(OBJATHOR_VERSIONED_DIR, "assets")
OBJATHOR_FEATURES_DIR = os.path.join(OBJATHOR_VERSIONED_DIR, "features")
OBJATHOR_ANNOTATIONS_PATH = os.path.join(OBJATHOR_VERSIONED_DIR, "annotations.json.gz")

'''
def is_bbox1_inside_bbox2(bbox1, bbox2):
    """
    判断 bbox1 是否被 bbox2 包含
    bbox1 和 bbox2 的格式为 (center, size, euler_angles)
    """
    center1, size1, euler1 = bbox1[0:3], bbox1[3:6], bbox1[6:9]
    center2, size2, euler2 = bbox2[0:3], bbox2[3:6], bbox2[6:9]

    # 创建 trimesh 的 Box 对象
    # trimesh.transformations.euler_matrix(euler_angles[0], euler_angles[1], euler_angles[2], axes='rzxy')
    bbox1_mesh = trimesh.primitives.Box(extents=size1, transform=trimesh.transformations.euler_matrix(*euler1, axes='rzxy'))
    bbox1_mesh.apply_translation(center1)

    bbox2_mesh = trimesh.primitives.Box(extents=size2, transform=trimesh.transformations.euler_matrix(*euler2, axes='rzxy'))
    bbox2_mesh.apply_translation(center2)

    # 获取 bbox1 的所有顶点
    vertices1 = bbox1_mesh.vertices

    # 检查 bbox1 的所有顶点是否都在 bbox2 内部
    for vertex in vertices1:
        if not bbox2_mesh.contains([vertex]):
            return False
    return True
'''

def get_caption(uid, csv_file_rows):
    uids = [csv_file_row[0] for csv_file_row in csv_file_rows]
    if uid in uids:
        return csv_file_rows[uids.index(uid)][1]
    else:
        return None
    
def get_dataset_cate_from_uid(uid, cate_dict, partnetmobility_cate_dict):
    """
    返回每个数据集所标注的类别，返回的类别 **不一定** 在 embodiedscan 的 288 个类别中
    """
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

def get_embodiedscan_cate_from_uid(uid, cate_dict):
    """
    返回的类别在 embodiedscan 的 288 个类别中
    """

    for cate in cate_dict.keys():
        for uid_ in cate_dict[cate]:
            if uid == uid_:
                return cate

    return "Unkonwn category,wrong uid"

def get_asset_metadata(obj_data: Dict[str, Any]):
    if "assetMetadata" in obj_data:
        return obj_data["assetMetadata"]
    elif "thor_metadata" in obj_data:
        return obj_data["thor_metadata"]["assetMetadata"]
    else:
        raise ValueError("Can not find assetMetadata in obj_data")
    
def get_bbox_dims(obj_data: Dict[str, Any]):
    am = get_asset_metadata(obj_data)

    bbox_info = am["boundingBox"]

    if "x" in bbox_info:
        return bbox_info

    if "size" in bbox_info:
        return bbox_info["size"]

    mins = bbox_info["min"]
    maxs = bbox_info["max"]

    return {k: maxs[k] - mins[k] for k in ["x", "y", "z"]}

def normalize_rotation_to_90_degrees(input):
    """
    将角度归一化到最接近的0、90、180或270度
    
    Args:
        rotation_degrees: 旋转角度(度)
    
    Returns:
        归一化后的角度(度)
    """
    def angle_diff(angle1, angle2):
        """
        计算两个角度之间的最小差值 (考虑 0-360 循环).

        Args:
            angle1: 第一个角度 (0-360).
            angle2: 第二个角度 (0-360).

        Returns:
            float: 两个角度之间的最小差值 (0-180).
        """
        diff = abs(angle1 - angle2)
        return min(diff, 360 - diff)
    # 确保角度在0-360度之间
    rotation_degrees = input % 360
    
    # 找到最接近的90度倍数
    nearest_90_multiple = round(rotation_degrees / 90) * 90
    
    # 确保结果在0-360度之间
    output = nearest_90_multiple % 360

    if angle_diff(input, output) > 15:
        return None
    else:
        return output

def rotate_size_based_on_rotation(size, rotation_degrees):
    """
    根据旋转角度调整物体的尺寸
    
    Args:
        size: 原始尺寸 [x, y, z]
        rotation_degrees: 旋转角度(度)
    
    Returns:
        旋转后的尺寸 [x', y', z']
    """
    # 归一化到最接近的90度倍数
    normalized_rotation = normalize_rotation_to_90_degrees(rotation_degrees)
    if normalized_rotation is None:
        return None 
    
    # 如果旋转是0度或180度，x和y保持不变
    if normalized_rotation == 0 or normalized_rotation == 180:
        return size
    
    # 如果旋转是90度或270度，交换x和y
    elif normalized_rotation == 90 or normalized_rotation == 270:
        return [size[1], size[0], size[2]]
    
    else:
        # 这种情况理论上不应该出现，因为我们已经归一化了角度
        print(f"警告: 遇到非90度倍数的角度: {rotation_degrees}")
        return size

def get_bboxes_similarity(target_bbox,
                           candidate_bboxes,
                           without_xy_sort=False):
    target_bbox = target_bbox.reshape(1,-1) # [1,3]
    target_bbox_norm = target_bbox / np.linalg.norm(target_bbox, ord=2, axis=1, keepdims=True) # [1,3]
    candidate_bboxes_norm = candidate_bboxes / np.linalg.norm(candidate_bboxes, ord=2, axis=1, keepdims=True) # [n,3]

    if without_xy_sort:
        # 对于有明显朝向的物体，直接计算相似度，不进行XY轴排序
        cosine_similarity = candidate_bboxes_norm @ target_bbox_norm.T
        return cosine_similarity
    else:
        print("ERROR!!!THE CODE IS USING XY_SORT!!!!!!!!!!!!ERROR!!!THE CODE IS USING XY_SORT!!!!!!!!!!!!")
        # 提取 XY 轴和 Z 轴
        target_xy = target_bbox_norm[:, :2]  # [1, 2]
        target_z = target_bbox_norm[:, 2:]  # [1, 1]

        candidate_xy = candidate_bboxes_norm[:, :2]  # [n, 2]
        candidate_z = candidate_bboxes_norm[:, 2:]  # [n, 1]

        # 对 XY 轴进行从大到小排序
        target_xy_sorted = np.sort(target_xy, axis=1)[:, ::-1]  # [1, 2]
        candidate_xy_sorted = np.sort(candidate_xy, axis=1)[:, ::-1]  # [n, 2]

        # 重新组合排序后的 XY 轴和 Z 轴
        target_bbox_norm_sorted = np.concatenate([target_xy_sorted, target_z], axis=1)  # [1, 3]
        candidate_bboxes_norm_sorted = np.concatenate([candidate_xy_sorted, candidate_z], axis=1)  # [n, 3]
        
        #求余弦相似性
        cosine_similarity = candidate_bboxes_norm_sorted @ target_bbox_norm_sorted.T

        return cosine_similarity
    # target_bbox = target_bbox.reshape(1,-1) # [1,3]
    # target_bbox_norm = target_bbox / torch.norm(target_bbox, p=2, dim=1) # [1,3]
    # candidate_bboxes_norm = candidate_bboxes / torch.norm(candidate_bboxes, p=2, dim=1,keepdim=True) # [n,3]

    # if without_xy_sort:
    #     # 对于有明显朝向的物体，直接计算相似度，不进行XY轴排序
    #     cosine_similarity = torch.mm(candidate_bboxes_norm, target_bbox_norm.T)
    #     return cosine_similarity
    # else:
    #     # 提取 XY 轴和 Z 轴
    #     target_xy = target_bbox_norm[:, :2]  # [1, 2]
    #     target_z = target_bbox_norm[:, 2:]  # [1, 1]

    #     candidate_xy = candidate_bboxes_norm[:, :2]  # [n, 2]
    #     candidate_z = candidate_bboxes_norm[:, 2:]  # [n, 1]

    #     # 对 XY 轴进行从大到小排序
    #     target_xy_sorted = torch.sort(target_xy, descending=True)[0]  # [1, 2]
    #     candidate_xy_sorted = torch.sort(candidate_xy, descending=True, dim=1)[0]  # [n, 2]

    #     # 重新组合排序后的 XY 轴和 Z 轴
    #     target_bbox_norm_sorted = torch.cat([target_xy_sorted, target_z], dim=1)  # [1, 3]
    #     candidate_bboxes_norm_sorted = torch.cat([candidate_xy_sorted, candidate_z], dim=1)  # [n, 3]
        
    #     #求余弦相似性
    #     cosine_similarity = torch.mm(candidate_bboxes_norm_sorted, target_bbox_norm_sorted.T)

    #     return cosine_similarity

def get_object_replace_cate(index, instance_info_dict_list, scene_name,
                             floor_distance_threshold = 0.15, distance_threshold=4.0):
    
    '''
    返回object类别替换为的类
    return
    replace_cates : 替换为的类别列表
    replace_type:替换操作的类型
    '''
    
    object_replace_dict = { "floor_":["bin", "bag", "backpack", "basket", "shoe", "ball"],
                           "bed_couch" : ["toy", "pillow", "bag", "book", "backpack", "hat"],
                           "table_desk":["book", "plant", "lamp", "bottle", "socket", "cup", "vase", "bowl", "plate", "fruit", "teapot"],
                           "washroom_sink":["cup", "box", "bottle", "towel", "case", "soap","soap dish", "soap dispenser"],
                           "kitchen_stove":["bowl", "cup", "knife", "plate", "can", "fruit", "food"],
                            "cabinet_": ["box", "toy", "book", "hat", "bag", "cup", "shoe"],
                            "wall_high_type": ["picture"],
                            "wall_low_type": ["socket"]}
    
    instance_loc_xyz_list = [np.array(instance_info_dict["bbox"][0:3]) for instance_info_dict in instance_info_dict_list]
    object_loc_xyz = instance_loc_xyz_list[index]
    object_size_xyz = np.array(instance_info_dict_list[index]["bbox"][3:6])

    ## object 替换为 picture
    # 筛选出 wall 的所有bbox
    wall_bbox_list = []
    for instance_info_dict in instance_info_dict_list:
        if  instance_info_dict["category"] == "wall":
            wall_bbox_list.append(instance_info_dict["bbox"])
    # 如果 object 形状扁平并且贴近墙面
    flat_flag = False
    in_wall_flag = False

    if (object_size_xyz[1] + object_size_xyz[2]) / object_size_xyz[0] > 8:
        flat_flag = True

    if flat_flag:
        enlarged_wall_bbox_list = [item + np.array([0, 0, 0, 0.1, 0.1, 0.1, 0, 0, 0]) for item in wall_bbox_list]
        for enlarged_bbox in enlarged_wall_bbox_list:
            if is_point_inside_bbox(object_loc_xyz, enlarged_bbox):
                in_wall_flag = True
                break
    
    if in_wall_flag and flat_flag:
        # object 替换为 wall_high_type
        replace_cates = object_replace_dict["wall_high_type"]
        return replace_cates, "wall_high_type"

    # 如果 object 接近地面，则替换为地面类别
    distance_z_to_floor = object_loc_xyz[2] - 0.5*object_size_xyz[2] 
    if np.abs(distance_z_to_floor) < floor_distance_threshold:
        replace_cates = object_replace_dict["floor_"]
        return replace_cates, "floor_"

    # 计算 object到每个物体的距离,只计算三米方框范围内的距离
    index_distance_list = []
    for i, instance_loc_xyz in enumerate(instance_loc_xyz_list):
        horizen_distance = np.abs(instance_loc_xyz[0] - object_loc_xyz[0]) + np.abs(instance_loc_xyz[1] - object_loc_xyz[1])
        if horizen_distance <= distance_threshold: 
            index_distance_list.append((i, np.linalg.norm(object_loc_xyz - instance_loc_xyz)))
        
    # 判断是否在 cabinet 中
    for i, _ in index_distance_list[1:]:
        instance_cate = instance_info_dict_list[i]["category"]
        # cabinet_ type
        if instance_cate in ["cabinet"]:
            cabinet_bbox = instance_info_dict_list[i]["bbox"]
            object_bbox = instance_info_dict_list[index]["bbox"]
            if is_bbox1_inside_bbox2(object_bbox, cabinet_bbox):
                return object_replace_dict["cabinet_"], "cabinet_"
            
    # 按照距离从小到大排序
    index_distance_list = sorted(index_distance_list, key=lambda x: x[1])

    # 从近处到远处判断归类类型
    for i, _ in index_distance_list[1:]:
        instance_cate = instance_info_dict_list[i]["category"]
        intance_z = instance_loc_xyz_list[i][2]
        object_z = object_loc_xyz[2]
        
        if intance_z < object_z and instance_cate in ["bed", "couch"]:
            return object_replace_dict["bed_couch"], f"bed_couch, i:{intance_z}, o:{object_z}"
        
        if intance_z < object_z and instance_cate in ["table", "desk"]:
            return object_replace_dict["table_desk"], f"table_desk, i:{intance_z}, o:{object_z}"
        
        if instance_cate in ["sink", "toilet", "shower"]:
            return object_replace_dict["washroom_sink"], f"washroom_sink"
        
        if instance_cate in ["pot", "microwave", "stove", "dish rack", "dishwasher"]:
            return object_replace_dict["kitchen_stove"], f"kitchen_stove"

    # pdb.set_trace()
    return None, "Unkonwn which category to replace obj" 

def get_couch_type(index, input_instance_infos):
    '''
    根据 bbox 信息判断 couch 类型：
    N-couch for Normal
    L-couch for Left L shape, 从沙发中心向前看去，左边是长的
    R-couch for Right L shape, 从沙发中心向前看去，右边是长的

    index : couch 的 index
    input_instance_infos : 包含场景中所有信息的列表，每个元素是一个字典类似于：
    {
        "id": 1,
        "label": 282,
        "category": "couch",
        "bbox": [cx, cy, cz, dx, dy, dz, angle_z, angle_x, angle_y],有9个元素,分别是中心坐标,尺寸,ZXY内旋欧拉角
        "3d_model": ""
    },
    '''
    # 获取目标沙发的信息
    couch_info = input_instance_infos[index]
    bbox = couch_info["bbox"]
    
    # 提取沙发中心、尺寸和绕 Z 轴的旋转角度（假设单位为弧度）
    cx, cy, cz = bbox[0], bbox[1], bbox[2]
    dx, dy, dz = bbox[3], bbox[4], bbox[5]
    angle_z = bbox[6]
    
    # 计算沙发的局部坐标系
    # 假设沙发“前方”对应于局部 y 轴方向，则
    forward = np.array([math.cos(angle_z), math.sin(angle_z)])
    # 左侧向量：前方顺时针旋转90度得到左侧（或逆时针旋转-90度，根据坐标系定义，此处采用[-sin, cos]）
    left_vector = np.array([-math.sin(angle_z), math.cos(angle_z)])
    
    # 初始化左右侧是否被遮挡的info_list
    obstructed_info_list = []
    

    # 遍历其他物体（如茶几等），判断其是否位于沙发的 bbox 内
    for i, instance in enumerate(input_instance_infos):
        # 跳过自己
        if i == index:
            continue

        other_bbox = instance["bbox"]
        other_size = other_bbox[3:6]
        other_cate = instance["category"]
        # 如果是枕头等，则跳过
        if other_cate in ["pillow", "cushion", "blanket", "ceiling", "floor"]:
            continue
        # 如果其他物体的中心点在沙发的上面，则跳过
        if abs(other_bbox[2] - cz) > np.array(other_size).max() + dz/2:
            continue
        # 获取其他物体的中心点（这里只考虑平面坐标）
        ox, oy = other_bbox[0], other_bbox[1]
        # 相对位置（世界坐标下相对于沙发中心的偏移）
        relative = np.array([ox - cx, oy - cy])
        
        # 将相对位置投影到沙发局部坐标系
        proj_left = np.dot(relative, left_vector)
        proj_forward = np.dot(relative, forward)
        
        # 判断该点是否落在沙发的 bbox 内（假设沙发 bbox 在局部坐标系下长 dx（左右）和 dy（前后））
        if abs(proj_left) < dx/2 and abs(proj_forward) < dy/2:
            # 初始化左右侧是否被遮挡的标志
            left_obstructed = False
            right_obstructed = False
            # 根据投影到左侧向量上的值判断在沙发的左侧还是右侧
            if proj_left > 0:
                left_obstructed = True
            elif proj_left < 0:
                right_obstructed = True
            # 记录被遮挡的信息与other_bbox的体积
            obstructed_info_list.append((left_obstructed, right_obstructed, np.prod(other_size), copy.deepcopy(instance)))
        
    # 根据左右侧遮挡情况判断沙发类型
    # 例如：如果沙发左侧被遮挡（有茶几等在内），则说明从正面看，右侧较长，即为 R-couch
    # pdb.set_trace()
    if len(obstructed_info_list) == 0:
        return "N-couch"
    else:
        left_v = np.array([info[0]*info[2] for info in obstructed_info_list]).sum()
        right_v = np.array([info[1]*info[2] for info in obstructed_info_list]).sum()
        if left_v > right_v:
            return "R-couch"
        else:
            return "L-couch"

def get_candidate_uids(index, cate_dict, input_instance_infos, scene_name, object_cate_replace=True, add_gapartnet_asset_candidate = True):
    '''
    index : 第index个instance
    cate_dict :cate_dict
    input_instance_infos : 一个由instances的信息字典组成的列表
    object_cate_replace: 是否要对 object 这一个类别进行替换

    return :
    uids : candidate的uid列表
    replace_type: 进行替换操作的类型, None表示不进行替换
    '''
    cate = input_instance_infos[index]["category"]
    if  cate == 'object':
        if not object_cate_replace:
            return cate_dict[cate], None
        else:
            
            object_replace_cate_list, replace_type = get_object_replace_cate(index, input_instance_infos, scene_name) # 用于替换 object的类别列表

            if object_replace_cate_list == None:
                print(f"{index}, 没有找到替换object的类别")
                object_replace_cate_list = ["object"]

            uids = []
            for category in object_replace_cate_list:
                uids.extend(cate_dict[category])

            return uids, replace_type
        
    elif cate == 'couch':
        couch_type = get_couch_type(index, input_instance_infos)
        if couch_type in ["L-couch", "R-couch"]:
            candidate_uids = []
            for uid in cate_dict[cate]:
                if couch_type in uid:
                    candidate_uids.append(uid)
            return candidate_uids, couch_type
        else:
            return cate_dict[cate], None

    elif cate == 'stool':
        candidate_uids = []
        for uid in cate_dict[cate]:
            if "stable_stool" in uid:
                candidate_uids.append(uid)
        return candidate_uids, "stable_stool"

    elif cate == 'heater':
        # heater里面加入暖气片
        candidate_uids = cate_dict['heater'] + cate_dict["radiator"]
        return candidate_uids, "add radiator to heater"

    elif cate == "light":
        # light 里面加入 chandelier
        candidate_uids = cate_dict['light'] + cate_dict["chandelier"]
        return candidate_uids, "add chandelier to light"
    
    elif cate == "clothes dryer":
        # light 里面加入 chandelier
        candidate_uids = cate_dict['clothes dryer'] + cate_dict["washing machine"]
        return candidate_uids, "add washing machine to clothes dryer"

    else:
        return cate_dict[cate], None


class ObjathorRetriever:
    def __init__(
        self,
        clip_model,
        clip_preprocess,
        clip_tokenizer,
        sbert_model,
        retrieval_threshold,
    ):
        objathor_annotations = compress_json.load(OBJATHOR_ANNOTATIONS_PATH)
        self.database = {**objathor_annotations}

        objathor_clip_features_dict = compress_pickle.load(
            os.path.join(OBJATHOR_FEATURES_DIR, f"clip_features.pkl")
        )  # clip features
        objathor_sbert_features_dict = compress_pickle.load(
            os.path.join(OBJATHOR_FEATURES_DIR, f"sbert_features.pkl")
        )  # sbert features
        assert (
            objathor_clip_features_dict["uids"] == objathor_sbert_features_dict["uids"]
        )

        objathor_uids = objathor_clip_features_dict["uids"]
        objathor_clip_features = objathor_clip_features_dict["img_features"].astype(
            np.float32
        )
        objathor_sbert_features = objathor_sbert_features_dict["text_features"].astype(
            np.float32
        )
        # self.clip_features = torch.from_numpy(
        #     np.concatenate([objathor_clip_features, thor_clip_features], axis=0)
        # )
        self.clip_features = torch.from_numpy(objathor_clip_features)
        self.clip_features = F.normalize(self.clip_features, p=2, dim=-1)

        # self.sbert_features = torch.from_numpy(
        #     np.concatenate([objathor_sbert_features, thor_sbert_features], axis=0)
        # )
        self.sbert_features = objathor_sbert_features

        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.clip_tokenizer = clip_tokenizer
        self.sbert_model = sbert_model
        self.asset_ids = objathor_uids


        self.retrieval_threshold = retrieval_threshold

        self.use_text = True

    def retrieve_tt_ti_ii(self, queries, scene_features=None, threshold=8):
        with torch.no_grad():
            query_feature_clip = self.clip_model.encode_text(
                self.clip_tokenizer(queries)
            )

            query_feature_clip = F.normalize(query_feature_clip, p=2, dim=-1)

        clip_similarities = 100 * torch.einsum(
            "ij, lkj -> ilk", query_feature_clip, self.clip_features
        )
        clip_similarities = torch.max(clip_similarities, dim=-1).values
    

        query_feature_sbert = self.sbert_model.encode(
            queries, convert_to_tensor=True, show_progress_bar=False
        )
 

     
        sbert_similarities = query_feature_sbert @ self.sbert_features.T

        if self.use_text:
            similarities = clip_similarities + sbert_similarities
        else:
            similarities = clip_similarities

        scene_feature = np.concatenate(scene_features,axis=0)

        
        top_indices = torch.argsort(-similarities, dim=0)[:, :10] #[20,10]

        unsorted_results = []
        most_similar_ids = []
        most_similar_scores = []
        for i in range(query_feature_clip.size(0)):
        # for query_index, asset_index in zip(*top_indices):

                selected_indices = top_indices[i]
                print(self.clip_features[selected_indices].size())
                print(torch.from_numpy(scene_feature)[i].size())

                selected_images = self.clip_features[selected_indices].reshape(-1, 768)
                target_image = torch.from_numpy(scene_feature)[i].reshape(1, 768)
                print(selected_images.size())
                print(target_image.size())
                norm_selected_images = F.normalize(selected_images, p=2, dim=1)
                norm_target_image = F.normalize(target_image, p=2, dim=1)

                # 计算余弦相似性
                image_similarity = torch.mm(norm_selected_images, norm_target_image.T)
                most_similar_id = selected_indices[np.argmax(image_similarity)]
                most_similar_scores.append(np.max(image_similarity))
                most_similar_ids.append(self.asset_ids[most_similar_id])

            
        return [most_similar_ids, most_similar_scores]

    def retrieve_size_new(
            self, 
            candidates, 
            target_size, 
            uid_size_dict, 
            replace_type,
            partnetmobility_sim_add = 0.1, 
            partnetmobility_sim_multi = 1.0,
            useful_uid_rotate_dict = None, 
            cate_with_apparent_direction = None,
            cate_dict = None, 
            partnetmobility_cate_dict = None,
            target_category = None,
            use_xysort_in_retrieval = False
    ):  # 添加目标类别参数

        candidate_sizes = []
        candidates_new = []
        candidate_rotations = []  # 记录每个候选物体的旋转状态

        for uid in candidates:
            if "/" in uid: #如果不是 objaverse 的uid
                size = uid_size_dict[uid]
                
            else:
                try:
                    size = get_bbox_dims(self.database[uid]) #candidates的size
                    # 用objaverse中读到的size覆盖
                    size = uid_size_dict[uid]
                except KeyError as e:
                    # 如果键不存在，捕获 KeyError 异常，继续下一次循环
                    continue

            # 处理三维资产的转正，使得这些物体在Z-up坐标中正面朝 +Y 方向
            rotation = None
            if useful_uid_rotate_dict and uid in useful_uid_rotate_dict:
                rotation = useful_uid_rotate_dict[uid]
                if rotation != "None":
                    size = rotate_size_based_on_rotation(size, rotation)
            if size is None: # 当rotation和 0，90，180. 270差距大于15°时，size为None
                continue

            # 由于在Embodied所有物体的正面都是朝向 X 正方向，所以要把物体绕Z轴旋转90度，因此这里要把size的x和y交换，使得retrieval的形状更合理
            size = [size[1], size[0], size[2]]

            candidates_new.append(uid)   # 如果可以得到size，就把uid加入candidates_new
            candidate_sizes.append(size)
            candidate_rotations.append(rotation)

        ##################################################################################
        # 如果在holodeck和 objaverse交集中没有找到，就把 objaverse 所有cate物体加入 candidate
        if candidate_sizes==[]:   
            for uid in candidates: 
                try:
                    size = uid_size_dict[uid]
                    # 由于在Embodied所有物体的正面都是朝向 X 正方向，所以要把物体绕Z轴旋转90度，因此这里要把size的x和y交换，使得retrieval的形状更合理
                    size = [size[1], size[0], size[2]]
                except KeyError as e:
                    continue
                # 处理三维资产的转正，使得这些物体在Z-up坐标中正面朝 +Y 方向
                rotation = None
                if useful_uid_rotate_dict and uid in useful_uid_rotate_dict:
                    rotation = useful_uid_rotate_dict[uid]
                    if rotation != "None":
                        size = rotate_size_based_on_rotation(size, rotation)
                candidates_new.append(uid)
                candidate_sizes.append(size)
                candidate_rotations.append(rotation)

        if candidate_sizes==[]:  
            return None
        
        candidate_sizes = np.array(candidate_sizes)# , dtype = torch.float32) # [n,3]
        target_size_list = target_size[3:6]
        target_size =  np.array(target_size_list)
        
        # 根据目标类别判断是否需要XY排序
        if not use_xysort_in_retrieval:
            without_xy_sort = True
        else:
            without_xy_sort = (cate_with_apparent_direction and target_category in cate_with_apparent_direction) and  use_xysort_in_retrieval
        
        # 计算相似度
        similarities = get_bboxes_similarity(target_size, candidate_sizes, without_xy_sort=without_xy_sort)
        similarities = similarities.reshape(-1, 1)


        # 提高已转正物体的优先级
        rotation_bonus = 0.05  # 旋转奖励值
        for i, rotation in enumerate(candidate_rotations):
            if rotation is not None and rotation != "None":
                similarities[i] += rotation_bonus  # max 987: 0.9897


        candidates_with_size_difference = []

        for i, uid in enumerate(candidates_new):
            # 如果是gapartnet的uid,给他们的sim来一些增益,以保证它们的优先性; 同时如果是object类别的话就取消增益，避免object替换策略失效
            if "partnet" in uid and replace_type == None: 
                candidates_with_size_difference.append(
                    (uid, similarities[i].item() * partnetmobility_sim_multi + partnetmobility_sim_add)
                )
            else:
                candidates_with_size_difference.append(
                    (uid, similarities[i].item())
                )

        # sort the candidates by score
        candidates_with_size_difference = sorted(
            candidates_with_size_difference, key=lambda x: x[1], reverse=True
        )

        # if target_category == "table":
        #     # 使用matplotlib将相似度可视化,统计每个0.01区间相似度的；散点图，保存为图片
        #     import matplotlib.pyplot as plt
        #     import numpy as np
        #     import matplotlib
        #     matplotlib.use('Agg')
        #     # 1. 统计每个0.01区间相似度的数量
        #     sim_list = [x[1] for x in candidates_with_size_difference]
        #     sim_list = np.array(sim_list)
        #     sim_list = np.round(sim_list, 2)
        #     sim_list = sim_list.tolist()
        #     sim_dict = {}
        #     for sim in sim_list:
        #         if sim in sim_dict:
        #             sim_dict[sim] += 1
        #         else:
        #             sim_dict[sim] = 1
        #     # 2. 绘制散点图
        #     x = list(sim_dict.keys())
        #     y = list(sim_dict.values())
        #     plt.scatter(x, y)
        #     plt.xlabel('similarity')
        #     plt.ylabel('count')
        #     plt.title('similarity distribution')
        #     plt.savefig("similarity_distribution.png")
        #     pdb.set_trace()

        return candidates_with_size_difference
        
    def retrieve_size(self, image_feature, candidates, target_size,  threshold=8):
        all_features = []
        if image_feature !=[]:

            image_feature = np.concatenate(image_feature,axis=0)
            print(image_feature.size)

            image_similarities = 100 * torch.einsum(
            "ij, lkj -> ilk",  torch.from_numpy(image_feature),all_features)
            image_similarities = torch.max(image_similarities, dim=-1).values
            print(image_similarities.shape)
            # size_similarities = (size, all_size)
            # print(size_similarities.shape)
            similarities = image_similarities #+ size_similarities
            threshold_indices = torch.where(similarities > threshold)

            unsorted_results = []
            for query_index, asset_index in zip(*threshold_indices):
                score = similarities[query_index, asset_index].item()
                unsorted_results.append((self.asset_ids[asset_index], score))

            # Sorting the results in descending order by score
            result = sorted(unsorted_results, key=lambda x: x[1], reverse=True)
            # print(result)

            return result
        else: 
            candidate_sizes = []
            candidates_new = []

            #在 holodeck 的标注中去找这些candidate uid 有没有size信息
            for uid in candidates:
                try:
                    size = get_bbox_dims(self.database[uid]) #candidates的size
                except KeyError as e:
                    # 如果键不存在，捕获 KeyError 异常，继续下一次循环
                    continue

                size_list = [size["x"] , size["y"] , size["z"] ]
                # print(size_list)
                size_list.sort()  
                candidates_new.append(uid)   # 如果可以得到size，就把uid加入candidates_new
                candidate_sizes.append(size_list)

            # 没有candidate,返回 None
            if candidate_sizes==[]:    
                #  print("空的")
                 return None
            
            candidate_sizes = torch.tensor(candidate_sizes)
            target_size_list = [target_size[3]-target_size[0], target_size[4] - target_size[1], target_size[5] - target_size[2]]
            target_size_list.sort()
            target_size = torch.tensor(target_size_list)
            # print(target_size)
            size_difference = abs(candidate_sizes - target_size).mean(axis=1) / 100 #????这里的直接相减有待考虑
            size_difference = size_difference.tolist()
            # print(len(size_difference))

            candidates_with_size_difference = []
            # for i, (uid, score) in enumerate(candidates):
            for i, uid in enumerate(candidates_new):
                # print(i)
                candidates_with_size_difference.append(
                    (uid,  size_difference[i])
                )

            # sort the candidates by score
            candidates_with_size_difference = sorted(
                candidates_with_size_difference, key=lambda x: x[1], reverse=False
            )

            return candidates_with_size_difference

    def retrieve_tt_ti(self, queries, threshold=1):
        with torch.no_grad():
            query_feature_clip = self.clip_model.encode_text(
                self.clip_tokenizer(queries)
            )

            query_feature_clip = F.normalize(query_feature_clip, p=2, dim=-1)

        clip_similarities = 100 * torch.einsum(
            "ij, lkj -> ilk", query_feature_clip, self.clip_features
        )
        clip_similarities = torch.max(clip_similarities, dim=-1).values
      

        query_feature_sbert = self.sbert_model.encode(
            queries, convert_to_tensor=True, show_progress_bar=False
        )
 
        sbert_similarities = query_feature_sbert @ self.sbert_features.T

        if self.use_text:
            similarities = clip_similarities + sbert_similarities
        else:
            similarities = clip_similarities
        threshold_indices = torch.where(similarities > threshold)

        unsorted_results = []
        for query_index, asset_index in zip(*threshold_indices):
            score = similarities[query_index, asset_index].item()
            unsorted_results.append((self.asset_ids[asset_index], score))
        result = sorted(unsorted_results, key=lambda x: x[1], reverse=True)
        # print(result)

        return result[:200]

    def compute_size_difference(self, target_size, candidates):
        candidate_sizes = []
        for uid, _ in candidates:
            size = get_bbox_dims(self.database[uid])
            size_list = [size["x"] * 100, size["y"] * 100, size["z"] * 100]
            size_list.sort()
            candidate_sizes.append(size_list)

        candidate_sizes = torch.tensor(candidate_sizes)

        target_size_list = list(target_size)
        target_size_list.sort()
        target_size = torch.tensor(target_size_list)

        size_difference = abs(candidate_sizes - target_size).mean(axis=1) / 100
        size_difference = size_difference.tolist()

        candidates_with_size_difference = []
        for i, (uid, score) in enumerate(candidates):
            candidates_with_size_difference.append(
                (uid, score - size_difference[i] * 10)
            )

        # sort the candidates by score
        candidates_with_size_difference = sorted(
            candidates_with_size_difference, key=lambda x: x[1], reverse=True
        )

        return candidates_with_size_difference

def load_info_for_retrieval(
    scene_files_dir = "/cpfs01/user/zhongweipeng/Projects/layout/embodiedscene/scan",
    semantic_label_file_path = '/cpfs01/user/zhongweipeng/Projects/layout/semantic.json',
    cate_dict_new_path = "/cpfs01/user/zhongweipeng/Projects/layout/cate_dict_new.json",
    partnetmobility_cate_addto_embodiedscan_path = "/cpfs01/user/zhongweipeng/Projects/layout/partnetmobility_cate_addto_embodiedscan.json",
    partnetmobility_cate_dict_path = "/cpfs01/user/zhongweipeng/Projects/layout/partnetmobility_cate_dict.json",
    Cap3D_csv_file_path = '/cpfs01/shared/landmark_3dgen/xuxudong_group/zhongweipeng/objaverse_categorize/Cap3D_Objaverse_batch_14_1.csv',
    # new add 
    uid_bbox_info_dict_path = "/cpfs01/user/zhongweipeng/Projects/layout/json_files/AssetSizeReader/uid_bbox_info_dict_total.json",

    # uid_size_dict_path = "/cpfs01/user/zhongweipeng/Projects/layout/cate_dict_new_size.json",
    # partnetmobility_size_dict_path = "/cpfs01/user/zhongweipeng/Projects/layout/partnetmobility_size_dict.json",
    useful_uid_rotate_dict_path = "/cpfs01/user/zhongweipeng/Projects/layout/useful_uid_rotate_dict.json",
    cate_with_apparent_direction_path = "/cpfs01/user/zhongweipeng/Projects/layout/cate_with_apparent_direction.json",
    gen_assets_cate_dict_path = "/cpfs01/user/zhongweipeng/Projects/layout/gen_assets_cate_dict.json",
    # gen_assets_size_dict_path = "/cpfs01/user/zhongweipeng/Projects/layout/gen_assets_size_dict.json",
    
    gen_assets_rotate_dict_path = "/cpfs01/user/zhongweipeng/Projects/layout/gen_assets_rotate_dict.json",
    use_xysort_in_retrieval = False
):

    t0 = time.time()
    add_gapartnet_asset_candidate = True


    # 提取原始cate_dict
    with open(cate_dict_new_path, 'r', encoding='utf-8') as file:
        cate_dict = json.load(file)

    # 提取 gen_assets_cate_dict
    with open(gen_assets_cate_dict_path, 'r') as file:
        gen_assets_cate_dict = json.load(file)

    # 提取加入partnetmobility 的 cate_dict
    if add_gapartnet_asset_candidate:
        with open(partnetmobility_cate_addto_embodiedscan_path, "r") as f:
            cates_need_addto_embodiedscan_dict = json.load(f)
        with open(partnetmobility_cate_dict_path, "r") as f:
            partnetmobility_cate_dict = json.load(f)

    # 将 gen_assets 中的资产根据cate划分给  partnetmobility 或者 cate_dict
    for cate, uids in gen_assets_cate_dict.items():
        sample_uid = uids[0]
        if cate in partnetmobility_cate_dict.keys() and "partnet_mobility/" in sample_uid:
            #### 这里特殊处理，把 partnetmobility_cate_dict的 storagefurniture 全部替换为 gen_assets 中的
            # partnetmobility_cate_dict[cate].extend(gen_assets_cate_dict[cate])
            partnetmobility_cate_dict[cate] = copy.deepcopy(gen_assets_cate_dict[cate])
        elif cate in cate_dict.keys():
            cate_dict[cate].extend(gen_assets_cate_dict[cate])
        else:
            cate_dict[cate] = gen_assets_cate_dict[cate]
    
    # 将 partnetmobility 中的资产根据 addto_embodiedscan_dict 加入 cate_dict
    for cate in cate_dict.keys():
        if len(cates_need_addto_embodiedscan_dict[cate]) != 0:
            for cate_add in cates_need_addto_embodiedscan_dict[cate]:
                cate_dict[cate].extend(partnetmobility_cate_dict[cate_add])

    # with open("/cpfs01/user/zhongweipeng/Projects/layout/codes_ybq/base_process/end_cate_dict.json", "w")as f:
    #     json.dump(cate_dict, f, indent=4)
    # pdb.set_trace()
    # # 提取 input_instance_infos
    # with open(input_instance_infos_path, 'r', encoding='utf-8') as file:
    #     input_instance_infos = json.load(file)

    # 提取 Cap3D_csv_file
    with open(Cap3D_csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        csv_file_rows = list(reader)

    with open(uid_bbox_info_dict_path, "r") as f:
        uid_bbox_info_dict = json.load(f)
        uid_size_dict = {uid:bbox_info["bounding_box"] for uid, bbox_info in uid_bbox_info_dict.items()}

    # # 读取原始 uid_size_dict
    # with open(uid_size_dict_path, 'r') as file:
    #     uid_size_dict = json.load(file)

    # # 提取 gen_assets_size_dict
    # with open(gen_assets_size_dict_path, 'r') as file:
    #     gen_assets_size_dict = json.load(file)
    # uid_size_dict = {**uid_size_dict, **gen_assets_size_dict}
 
    # 提取加入partnetmobility 的 uid_size_dict 
    if add_gapartnet_asset_candidate:
        # with open(partnetmobility_size_dict_path, "r") as f:
        #     partnetmobility_size_dict = json.load(f)
        #如果类别为 phone、pen、remote，则要将size绕X轴旋转90度，即对换YZ轴长度(finish TODO)
        for uid in uid_size_dict.keys():
            if "partnet_mobility/" in uid:
                obj_cate = get_dataset_cate_from_uid(uid, cate_dict, partnetmobility_cate_dict)
                if obj_cate in ["pen", "remote", "phone"]:
                    size = copy.deepcopy(uid_size_dict[uid])
                    uid_size_dict[uid] = [size[0], size[2], size[1]]#[size[2], size[0], size[1]]
        # #合并两个dict
        # uid_size_dict = {**uid_size_dict, **partnetmobility_size_dict}

    # 提取9K+ Objects的旋转
    with open(useful_uid_rotate_dict_path, 'r') as file:
        useful_uid_rotate_dict = json.load(file)

    # 提取 gen_assets 的旋转
    with open(gen_assets_rotate_dict_path, 'r') as file:
        gen_assets_rotate_dict = json.load(file)
    useful_uid_rotate_dict = {**useful_uid_rotate_dict, **gen_assets_rotate_dict}
        
    # 加载有明显朝向的类别列表
    with open(cate_with_apparent_direction_path, 'r') as file:
        cate_with_apparent_direction = json.load(file)

    print("load data time:", time.time()-t0)

    retrieval_info_dict = {
        "scene_files_dir": scene_files_dir,
        # "objathor_clip_features": objathor_clip_features,
        # "clip_features": clip_features,
        # "categories": categories,
        "cate_dict": cate_dict,
        "partnetmobility_cate_dict": partnetmobility_cate_dict,
        "csv_file_rows": csv_file_rows,
        "uid_size_dict": uid_size_dict,
        "useful_uid_rotate_dict": useful_uid_rotate_dict,
        "cate_with_apparent_direction": cate_with_apparent_direction,
        "use_xysort_in_retrieval":use_xysort_in_retrieval
    }
    return retrieval_info_dict

def main(
        scene_name,
        scene_files_dir = "/cpfs01/user/zhongweipeng/Projects/layout/embodiedscene/scan",
        semantic_label_file_path = '/cpfs01/user/zhongweipeng/Projects/layout/semantic.json',
        cate_dict_new_path = "/cpfs01/user/zhongweipeng/Projects/layout/cate_dict_new.json",
        partnetmobility_cate_addto_embodiedscan_path = "/cpfs01/user/zhongweipeng/Projects/layout/partnetmobility_cate_addto_embodiedscan.json",
        partnetmobility_cate_dict_path = "/cpfs01/user/zhongweipeng/Projects/layout/partnetmobility_cate_dict.json",
        Cap3D_csv_file_path = '/cpfs01/shared/landmark_3dgen/xuxudong_group/zhongweipeng/objaverse_categorize/Cap3D_Objaverse_batch_14_1.csv',
        uid_size_dict_path = "/cpfs01/user/zhongweipeng/Projects/layout/cate_dict_new_size.json",
        partnetmobility_size_dict_path = "/cpfs01/user/zhongweipeng/Projects/layout/partnetmobility_size_dict.json",
        useful_uid_rotate_dict_path = "/cpfs01/user/zhongweipeng/Projects/layout/useful_uid_rotate_dict.json",
        cate_with_apparent_direction_path = "/cpfs01/user/zhongweipeng/Projects/layout/cate_with_apparent_direction.json",
        gen_assets_cate_dict_path = "/cpfs01/user/zhongweipeng/Projects/layout/gen_assets_cate_dict.json",
        gen_assets_size_dict_path = "/cpfs01/user/zhongweipeng/Projects/layout/gen_assets_size_dict.json",
        gen_assets_rotate_dict_path = "/cpfs01/user/zhongweipeng/Projects/layout/gen_assets_rotate_dict.json"
):
    t0 = time.time()
    add_gapartnet_asset_candidate = True

    # scene_img_features = compress_pickle.load("/cpfs01/user/yinbaiqiao/understand_3d/indoor/embodiedscene/scan/obj_img/{}/image_features.pkl".format(scene_name))
    objathor_clip_features_dict = compress_pickle.load(
                os.path.join(OBJATHOR_FEATURES_DIR, f"clip_features.pkl")
            )  
    objathor_uids = objathor_clip_features_dict["uids"]
    objathor_clip_features = objathor_clip_features_dict["img_features"].astype(
                np.float32
            )
    clip_features = torch.from_numpy(objathor_clip_features)
    clip_features = F.normalize(clip_features, p=2, dim=-1)  # 得到所有key的feature

    # img paths                                                                                         
    # folder_path = os.path.join(scene_files_dir, scene_name, "obj_img")
    #f'/cpfs01/user/zhongweipeng/Projects/layout/embodiedscene/scan/{scene_name}/obj_img'  

    # file paths
    input_instance_infos_path = os.path.join(scene_files_dir, scene_name, "instances_info.json")
    output_instance_infos_path = os.path.join(scene_files_dir, scene_name, "instances_info_with_retri_uid_withpartnetmobility.json")

    # initialize moels 
    clip_model, _, clip_preprocess = None, None, None #open_clip.create_model_and_transforms('ViT-L-14', pretrained='/cpfs01/user/zhongweipeng/model_weights/models--laion--CLIP-ViT-L-14-laion2B-s32B-b82K/open_clip_pytorch_model.bin')
    clip_tokenizer = None #open_clip.get_tokenizer("ViT-L-14")
    sbert_model = None #SentenceTransformer("all-mpnet-base-v2",device ="cpu")     # initialize sentence transformer
    retrieval_threshold = 0.2

    # 提取category list
    with open(semantic_label_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        categories = [item["category"] for item in data if "category" in item]

    # 提取原始cate_dict
    with open(cate_dict_new_path, 'r', encoding='utf-8') as file:
        cate_dict = json.load(file)

    # 提取 gen_assets_cate_dict
    with open(gen_assets_cate_dict_path, 'r') as file:
        gen_assets_cate_dict = json.load(file)
    for cate in gen_assets_cate_dict.keys():
        if cate in cate_dict.keys():
            cate_dict[cate].extend(gen_assets_cate_dict[cate])
        else:
            cate_dict[cate] = gen_assets_cate_dict[cate]
        

    # 提取加入partnetmobility 的 cate_dict
    if add_gapartnet_asset_candidate:
        with open(partnetmobility_cate_addto_embodiedscan_path, "r") as f:
            cates_need_addto_embodiedscan_dict = json.load(f)
        with open(partnetmobility_cate_dict_path, "r") as f:
            partnetmobility_cate_dict = json.load(f)

        for cate in cate_dict.keys():
            if len(cates_need_addto_embodiedscan_dict[cate]) != 0:
                for cate_add in cates_need_addto_embodiedscan_dict[cate]:
                    cate_dict[cate].extend(partnetmobility_cate_dict[cate_add])

    # 提取 input_instance_infos
    with open(input_instance_infos_path, 'r', encoding='utf-8') as file:
        input_instance_infos = json.load(file)

    # 提取 Cap3D_csv_file
    with open(Cap3D_csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        csv_file_rows = list(reader)

    # 读取原始 uid_size_dict
    with open(uid_size_dict_path, 'r') as file:
        uid_size_dict = json.load(file)

    # 提取 gen_assets_size_dict
    with open(gen_assets_size_dict_path, 'r') as file:
        gen_assets_size_dict = json.load(file)
    uid_size_dict = {**uid_size_dict, **gen_assets_size_dict}
 
    # 提取加入partnetmobility 的 uid_size_dict 
    if add_gapartnet_asset_candidate:
        with open(partnetmobility_size_dict_path, "r") as f:
            partnetmobility_size_dict = json.load(f)
        #如果类别为 phone、pen、remote，则要将size绕X轴旋转90度，即对换YZ轴长度(finish TODO)
        for uid in partnetmobility_size_dict.keys():
            obj_cate = get_dataset_cate_from_uid(uid, cate_dict, partnetmobility_cate_dict)
            if obj_cate in ["pen", "remote", "phone"]:
                size = partnetmobility_size_dict[uid]
                partnetmobility_size_dict[uid] = [size[2], size[0], size[1]]
        #合并两个dict
        uid_size_dict = {**uid_size_dict, **partnetmobility_size_dict}

    # 提取9K+ Objects的旋转
    with open(useful_uid_rotate_dict_path, 'r') as file:
        useful_uid_rotate_dict = json.load(file)

    # 提取 gen_assets 的旋转
    with open(gen_assets_rotate_dict_path, 'r') as file:
        gen_assets_rotate_dict = json.load(file)
    useful_uid_rotate_dict = {**useful_uid_rotate_dict, **gen_assets_rotate_dict}
        
    # 加载有明显朝向的类别列表
    with open(cate_with_apparent_direction_path, 'r') as file:
        cate_with_apparent_direction = json.load(file)

    print("load data time:", time.time()-t0)

    '''
    # obj_num = len(input_instance_infos) # query的数量
    # instance_sizes = [] #用于存放每个query的bbox
    # queries = ['' for _ in range(obj_num)] #用于存放每个query的类别
    # queries_cat = ['' for _ in range(obj_num)] # 用于存放每个query的所需要的数据集的类别，例如 akb，partnet,objaverse

    
    # # 读取 datalist
    # file = '/cpfs01/user/zhongweipeng/Projects/layout/EmbodiedScan/data/embodiedscan_infos_train.pkl'
    # with open(file, 'rb') as f:
    #     data = pickle.load(f)
    # data_list = data['data_list']

    # for scene in data_list:
    #     if scene_name.split('/')[1] in scene['sample_idx']: #如果是 scene_name = 'scannet/scene0000_00' 这个场景
    #         instance_info_dict_list = [] #加入一个info_list，用于后面的object这一类的替换 
    #         instances = scene['instances']
    #         assert obj_num == len(instances)
    # for i, instance in enumerate(input_instance_infos):
    #     info = {}
    #     instance_sizes.append(instance['bbox_3d']); info["bbox_3d"] = instance['bbox_3d']

    #     cate = categories[int(instance['bbox_label_3d']) - 1] ;  info["category"] = cate ######代码里的id有0，实际的id没有##############!!!!!!!!!!!!!!!!!!!!!!!
        
    #     queries[i] = cate
        
    #     # if cate in ["box", "bucket"]:
    #     #     queries_cat[i]='akb_part'
    #     # elif cate in [ "camera","cabinet", "coffee maker","dishwasher","door","keyboard","laptop","luggage","microwave","oven","remote control","toaster","printer","refrigerator","table","toilet","washing machine"]:
    #     #     queries_cat[i]='part'
    #     # elif cate in ["drawer"]:
    #     #     queries_cat[i]='akb'
    #     # else:
    #     #     queries_cat[i]='obja'

    #     # info["retri_dateset"] = queries_cat[i]
    #     # instance_info_dict_list.append(info)
    '''

    retriever = ObjathorRetriever(clip_model,
            clip_preprocess,
            clip_tokenizer,
            sbert_model,
            retrieval_threshold)

    for i in tqdm(range(len(input_instance_infos))):

        instance_info = input_instance_infos[i]
        category = instance_info["category"]
        target_size = instance_info["bbox"]

        #如果没有类别，就跳过
        if category=='':
            continue

        candidate_uids, replace_type = get_candidate_uids(i, cate_dict, input_instance_infos, scene_name, object_cate_replace=True) # 候选的所有uids
        
        # # 分析函数每一行的耗时
        # from line_profiler import LineProfiler
        # lp = LineProfiler()
        # lp.add_function(retriever.retrieve_size_new)
        # result = lp(retriever.retrieve_size_new)(candidate_uids, target_size, uid_size_dict)
        # lp.print_stats()
        # pdb.set_trace()

        if len(candidate_uids) == 0:
            print(i, category, "no candidate uid")
            continue

        ret = retriever.retrieve_size_new(
            candidates = candidate_uids, 
            target_size=target_size, 
            uid_size_dict = uid_size_dict, 
            replace_type = replace_type,
            partnetmobility_sim_add = 0.1, 
            partnetmobility_sim_multi = 1.0,
            useful_uid_rotate_dict=useful_uid_rotate_dict, 
            cate_with_apparent_direction=cate_with_apparent_direction,
            cate_dict = cate_dict,
            partnetmobility_cate_dict=partnetmobility_cate_dict,
            target_category=category
        )

        if ret==None:
            # 没有retrieval结果,所有的candidate uid在 holodeck里面找不到
            print(i, category, "no retrieval result") 
             
            ret=[(candidate_uids[0], 0)] # 用第一个candidate uid 代替, sim = 0

        # 保存结果
        dataset_uid_cate = get_dataset_cate_from_uid(ret[0][0], cate_dict, partnetmobility_cate_dict)
        input_instance_infos[i]["3d_model"]= (ret[0][0], 
                              f"sim: {ret[0][1]}",
                              f"uid_caption: {get_caption(ret[0][0], csv_file_rows)}", 
                              f"cate before replace: {category}", 
                              f"cate after replace: {dataset_uid_cate}", 
                              f"replace_type: {replace_type}",
                              f"index: {i}")
        input_instance_infos[i]["dataset_uid_cate"] = dataset_uid_cate
        input_instance_infos[i]["embodiedscan_uid_cate"] = get_embodiedscan_cate_from_uid(ret[0][0], cate_dict)
        input_instance_infos[i]["embodiedscan_uid_cate_before_replace"] = category


    # save the result
    os.makedirs(os.path.dirname(output_instance_infos_path), exist_ok=True)
    with open(output_instance_infos_path, mode='w', encoding='utf-8') as jsonfile:
            jsonfile.write(json.dumps(input_instance_infos, ensure_ascii=False, indent=4))    
  
    

def test():
    uid = "2617962b3fcf4427ade4d873e8e73e3e"
    glb_folder = "/oss/lvzhaoyang/chenkaixu/objaverse_mesh/glbs/"
    gobja_to_obja_json_path = "/cpfs01/user/zhongweipeng/Projects/layout/gobjaverse_index_to_objaverse.json"

    with open(gobja_to_obja_json_path, 'r') as file:
        gobja_to_obja = json.load(file)
    
    exist_uid_path_dict = {path.split("/")[-1].split(".")[0]:path for path in gobja_to_obja.values()}
    glb_path = os.path.join(glb_folder, exist_uid_path_dict[uid])
    
    import trimesh
    mesh = trimesh.load(glb_path, force='mesh')
    current_size = mesh.bounding_box_oriented.extents

    print(current_size )

def test_bbox_in():
    # 示例
    bbox1 = (np.array([0, 0, 0, 2, 2, 2, 0, 0, 0]))
    bbox2 = (np.array([0, 0, 0, 4, 4, 4, 0, 0, 0]))

    print(is_bbox1_inside_bbox2(bbox1, bbox2))  # 输出: True

def retrieval_one_scene(
    scene_name,
    retrieval_info_dict
):
    for key in retrieval_info_dict:
        if key == "scene_files_dir":
            scene_files_dir = retrieval_info_dict[key]
        # elif key == "objathor_clip_features":
        #     objathor_clip_features = retrieval_info_dict[key]
        # elif key == "clip_features":
        #     clip_features = retrieval_info_dict[key]
        # elif key == "categories":
        #     categories = retrieval_info_dict[key]
        elif key == "cate_dict":
            cate_dict = retrieval_info_dict[key]
        elif key == "partnetmobility_cate_dict":
            partnetmobility_cate_dict = retrieval_info_dict[key]
        elif key == "csv_file_rows":
            csv_file_rows = retrieval_info_dict[key]
        elif key == "uid_size_dict":
            uid_size_dict = retrieval_info_dict[key]
        elif key == "useful_uid_rotate_dict":
            useful_uid_rotate_dict = retrieval_info_dict[key]
        elif key == "cate_with_apparent_direction":
            cate_with_apparent_direction = retrieval_info_dict[key]
        elif key == "use_xysort_in_retrieval":
            use_xysort_in_retrieval = retrieval_info_dict[key]

    # file paths
    input_instance_infos_path = os.path.join(scene_files_dir, scene_name, "instances_info.json")
    output_instance_infos_path = os.path.join(scene_files_dir, scene_name, "instances_info_with_retri.json")

    # initialize moels 
    clip_model, _, clip_preprocess = None, None, None 
    clip_tokenizer = None 
    sbert_model = None 
    retrieval_threshold = 0.2

    # 提取 input_instance_infos
    with open(input_instance_infos_path, 'r', encoding='utf-8') as file:
        input_instance_infos = json.load(file)

    retriever = ObjathorRetriever(clip_model,
            clip_preprocess,
            clip_tokenizer,
            sbert_model,
            retrieval_threshold)

    for i in tqdm(range(len(input_instance_infos)), desc="Retrieving scene..."):

        instance_info = input_instance_infos[i]
        category = instance_info["category"]
        target_size = instance_info["bbox"]

        #如果没有类别，就跳过
        if category=='':
            continue

        candidate_uids, replace_type = get_candidate_uids(i, cate_dict, input_instance_infos, scene_name ,object_cate_replace=True) # 候选的所有uids
        

        if len(candidate_uids) == 0:
            print(i, category, "no candidate uid")
            continue

        ret = retriever.retrieve_size_new(
            candidates = candidate_uids, 
            target_size=target_size, 
            uid_size_dict = uid_size_dict, 
            replace_type = replace_type,
            partnetmobility_sim_add = 0.1, 
            partnetmobility_sim_multi = 1.0,
            useful_uid_rotate_dict=useful_uid_rotate_dict, 
            cate_with_apparent_direction=cate_with_apparent_direction,
            cate_dict = cate_dict,
            partnetmobility_cate_dict=partnetmobility_cate_dict,
            target_category=category,
            use_xysort_in_retrieval = use_xysort_in_retrieval
        )

        # # 分析函数每一行的耗时
        # from line_profiler import LineProfiler
        # lp = LineProfiler()
        # lp.add_function(retriever.retrieve_size_new)
        # result = lp(retriever.retrieve_size_new)(
        #     candidates = candidate_uids, 
        #     target_size=target_size, 
        #     uid_size_dict = uid_size_dict, 
        #     replace_type = replace_type,
        #     partnetmobility_sim_add = 0.1, 
        #     partnetmobility_sim_multi = 1.0,
        #     useful_uid_rotate_dict=useful_uid_rotate_dict, 
        #     cate_with_apparent_direction=cate_with_apparent_direction,
        #     cate_dict = cate_dict,
        #     partnetmobility_cate_dict=partnetmobility_cate_dict,
        #     target_category=category
        # )
        # lp.print_stats()
        # pdb.set_trace()

        if ret==None:
            # 没有retrieval结果,所有的candidate uid在 holodeck里面找不到
            print(i, category, "no retrieval result") 
             
            ret=[(candidate_uids[0], 0)] # 用第一个candidate uid 代替, sim = 0

        # 保存结果
        dataset_uid_cate = get_dataset_cate_from_uid(ret[0][0], cate_dict, partnetmobility_cate_dict)
        input_instance_infos[i]["3d_model"]= (ret[0][0], 
                              f"sim: {ret[0][1]}",
                              f"uid_caption: {get_caption(ret[0][0], csv_file_rows)}", 
                              f"cate before replace: {category}", 
                              f"cate after replace: {dataset_uid_cate}", 
                              f"replace_type: {replace_type}",
                              f"index: {i}")
        input_instance_infos[i]["dataset_uid_cate"] = dataset_uid_cate
        input_instance_infos[i]["embodiedscan_uid_cate"] = get_embodiedscan_cate_from_uid(ret[0][0], cate_dict)
        input_instance_infos[i]["embodiedscan_uid_cate_before_replace"] = category


    # save the result
    os.makedirs(os.path.dirname(output_instance_infos_path), exist_ok=True)
    with open(output_instance_infos_path, mode='w', encoding='utf-8') as jsonfile:
            jsonfile.write(json.dumps(input_instance_infos, ensure_ascii=False, indent=4))

def load_cate_candidate_dict():
    retrieval_info_dict = load_info_for_retrieval()
    for key in retrieval_info_dict:
        if key == "cate_dict":
            cate_dict = retrieval_info_dict[key]


    cate_candidate_dict = {}

    for cate in cate_dict.keys():
        if  cate == 'object':

            object_replace_cate_list = []
            replace_dict = { "floor_":["bin", "bag", "backpack", "basket", "shoe", "ball"],
                    "bed_couch" : ["toy", "pillow", "bag", "book", "backpack", "hat"],
                    "table_desk":["book", "plant", "lamp", "bottle", "socket", "cup", "vase", "bowl", "plate", "fruit", "teapot"],
                    "washroom_sink":["cup", "box", "bottle", "towel", "case", "soap","soap dish", "soap dispenser"],
                    "kitchen_stove":["bowl", "cup", "knife", "plate", "can", "fruit", "food"],
                        "cabinet_": ["box", "toy", "book", "hat", "bag", "cup", "shoe"],
                        "wall_high_type": ["picture"],
                        "wall_low_type": ["socket"]}
            for cate_list in replace_dict.values():
                object_replace_cate_list.extend(cate_list)

            object_replace_cate_list.append("object")

            uids = []
            for category in object_replace_cate_list:
                uids.extend(cate_dict[category])

            cate_candidate_dict[cate] = copy.deepcopy(uids)
            
        elif cate == 'couch':
            cate_candidate_dict[cate] = copy.deepcopy(cate_dict[cate])

        elif cate == 'stool':
            candidate_uids = []
            for uid in cate_dict[cate]:
                if "stable_stool" in uid:
                    candidate_uids.append(uid)
            cate_candidate_dict[cate] = copy.deepcopy(candidate_uids)

        elif cate == 'heater':
            # heater里面加入暖气片
            candidate_uids = cate_dict['heater'] + cate_dict["radiator"]
            cate_candidate_dict[cate] = copy.deepcopy(candidate_uids)

        elif cate == "light":
            # light 里面加入 chandelier
            candidate_uids = cate_dict['light'] + cate_dict["chandelier"]
            cate_candidate_dict[cate] = copy.deepcopy(candidate_uids)
        
        elif cate == "clothes dryer":
            # light 里面加入 chandelier
            candidate_uids = cate_dict['clothes dryer'] + cate_dict["washing machine"]
            cate_candidate_dict[cate] = copy.deepcopy(candidate_uids)

        else:
            cate_candidate_dict[cate] = copy.deepcopy(cate_dict[cate])

    # 筛选 uid，只要有的uid
    with open("/cpfs01/user/zhongweipeng/Projects/layout/codes_ybq/assets_process/total_uids.json", "r")as f:
        exist_uids = json.load(f)
    cate_candidate_dict_final = {}
    for cate, uids in cate_candidate_dict.items():
        candidates = []
        for uid in uids:
            if uid not in exist_uids:
                continue
            candidates.append(uid)
        cate_candidate_dict_final[cate] = copy.deepcopy(candidates)

    with open("/cpfs01/user/zhongweipeng/Projects/layout/codes_ybq/assets_process/cate_candidate_dict.json", "w")as f:
        json.dump(cate_candidate_dict_final, f, indent=4)


def check_cate_candidate_dict():
    with open("/cpfs01/user/zhongweipeng/Projects/layout/codes_ybq/assets_process/cate_candidate_dict.json", "r")as f:
        cate_candidate_dict_final = json.load(f)

    with open("/cpfs01/shared/landmark_3dgen/xuxudong_group/zhongweipeng/projects/layout/asset_process/scene_name_list_final.json", "r") as f:
        scene_name_list_final = json.load(f)

    for scene_name in tqdm(scene_name_list_final):
        scene_dir = os.path.join("/cpfs01/user/zhongweipeng/Projects/layout/embodiedscene/scan", scene_name)
        json_path = os.path.join(scene_dir, "final_layout.json")
        with open(json_path, "r") as f:
            final_layout = json.load(f)
        for instance in final_layout:
            cate = instance["category"]
            uid = instance["model_uid"]
            if uid not in cate_candidate_dict_final[cate]:
                print("error:", scene_name, cate, uid)

def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_name", required=True, type=str, help = "scene name, e.g. scannet/scene0000_00")
    parser.add_argument("--scene_files_dir", required=True, type=str, help = "")
    parser.add_argument("--semantic_label_file_path", required=True, type=str, help = "")
    parser.add_argument("--cate_dict_new_path", required=True, type=str, help = "")
    parser.add_argument("--partnetmobility_cate_addto_embodiedscan_path", required=True, type=str, help = "")
    parser.add_argument("--partnetmobility_cate_dict_path", required=True, type=str, help = "")
    parser.add_argument("--Cap3D_csv_file_path", required=True, type=str, help = "")
    parser.add_argument("--uid_size_dict_path", required=True, type=str, help = "")
    parser.add_argument("--partnetmobility_size_dict_path", required=True, type=str, help = "")
    parser.add_argument("--useful_uid_rotate_dict_path", required=True, type=str, help = "uid到旋转角度的映射字典")
    parser.add_argument("--cate_with_apparent_direction_path", required=True, type=str, help = "有明显朝向的类别列表")

    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    # load_cate_candidate_dict()
    check_cate_candidate_dict()


    # args = parse_arguments()
    # main(
    #     args.scene_name,
    #     args.scene_files_dir,
    #     args.semantic_label_file_path,
    #     args.cate_dict_new_path,
    #     args.partnetmobility_cate_addto_embodiedscan_path,
    #     args.partnetmobility_cate_dict_path,
    #     args.Cap3D_csv_file_path,
    #     args.uid_size_dict_path,
    #     args.partnetmobility_size_dict_path,
    #     args.useful_uid_rotate_dict_path,
    #     args.cate_with_apparent_direction_path,
    # )

    # main("scannet/scene0002_00")

    # test()
    # test_bbox_in()
    



