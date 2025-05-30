
import sys
import json
import copy
import argparse
import numpy as np


sys.path.extend(["./optimiz_large_bbox", "./Sapien_Scene_Sim/optimiz_large_bbox"])
from .bbox_utils import euler_to_matrix, get_obb_vertices, is_bbox1_inside_bbox2, is_point_inside_bbox, get_totated_bbox_iou
from .bbox_utils import get_point_face_distance, get_bbox_Z_faces
 


def get_object_shift(small_index, large_index_list, instance_infos,
                             floor_distance_threshold = 0.15, distance_threshold=4.0):
    '''
    small_index: int, small object index
    large_index_list: list, large object index list
    instance_infos: list, instance info list

    把小物体和大物体进行关联，然后返回小物体的 平移 translation
    '''

    # 小物体信息
    try:
        small_cate = instance_infos[small_index]["category"]
    except Exception as e:
        print(f"Error in instance_info index:{small_index}: lack of key _category_")
        return [0.0, 0.0, 0.0], "Error in instance_info type"
    small_bbox = instance_infos[small_index]["bbox"]
    small_loc = np.array(instance_infos[small_index]["bbox"][0:3])
    small_size = np.array(instance_infos[small_index]["bbox"][3:6])
    small_euler = np.array(instance_infos[small_index]["bbox"][6:9])

    # 小物体信息列表
    large_cate_list = [instance_infos[large_index]["category"] for large_index in large_index_list]
    large_bbox_list = [instance_infos[large_index]["bbox"] for large_index in large_index_list]
    large_loc_list = [np.array(instance_infos[large_index]["bbox"][0:3]) for large_index in large_index_list]
    large_size_list = [np.array(instance_infos[large_index]["bbox"][3:6]) for large_index in large_index_list]
    large_euler_list = [np.array(instance_infos[large_index]["bbox"][6:9]) for large_index in large_index_list]
    large_shift_list = [np.array(instance_infos[large_index]["delta_translation_after_large_obj_optimization"]) 
                        for large_index in large_index_list]
    
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
    # 0. 地毯：抬高10cm后仿真
    if small_cate == "carpet":
        return [0.0, 0.0, 0.1], "carpet type"

    # 1. 特定语义关联：椅子中心和桌子侧面距离在 1.2 米之内，椅子和桌子关联
    # 近似判断策略：把桌子的bbox的size增加1.2，然后判断椅子的bbox中心点是否在桌子的bbox内部
    if small_cate == "chair":

        min_i, min_distance = -1, float('inf')

        for i, large_cate in enumerate(large_cate_list):
            if large_cate in ["table", "desk"]:
                enlarged_size = large_size_list[i] + np.array([1.2, 1.2, 0])
                enlarged_bbox = np.concatenate([large_loc_list[i], enlarged_size, large_euler_list[i]])
                if is_point_inside_bbox(small_loc, enlarged_bbox):
                    distance = np.linalg.norm(small_loc - large_loc_list[i])
                    if distance < min_distance:
                        min_distance = distance
                        min_i = i      

        if min_i != -1: # 找到了关联的桌子的情况下
            return large_shift_list[min_i].tolist(), "chair_table type"

    # 1. 特定语义关联 # 门框和门进行绑定
    # 近似判断策略：把door的 bbox 的 size 增加 1.2，然后判断 doorframe 的 bbox 中心点是否在 door 的 bbox 内部
    if small_cate == "doorframe":
        min_i, min_distance = -1, float('inf')
        for i, large_cate in enumerate(large_cate_list):
            if large_cate in ["door"]:
                enlarged_size = large_size_list[i] + np.array([1.2, 1.2, 0])
                enlarged_bbox = np.concatenate([large_loc_list[i], enlarged_size, large_euler_list[i]])
                if is_point_inside_bbox(small_loc, enlarged_bbox):
                    distance = np.linalg.norm(small_loc - large_loc_list[i])
                    if distance < min_distance:
                        min_distance = distance
                        min_i = i      

        if min_i != -1: # 找到了关联的桌子的情况下
            return large_shift_list[min_i].tolist(), "doorframe door type"

    # 2. 小物体bbox所有顶点在大型家具的bbox内部直接判定关联 iou = 1，考虑到有沙发桌子这类大型家具bbox重叠的情况，优先选择最小的大型家具
    min_i, min_volume = -1, float('inf')
    for i, large_index in enumerate(large_index_list):
        iou = get_totated_bbox_iou(small_bbox, large_bbox_list[i])
        if iou > (1 - 1e-4):
            large_volume = np.prod(large_size_list[i])
            if large_volume < min_volume:
                min_volume = large_volume
                min_i = i
    if min_i != -1:
        return large_shift_list[min_i].tolist(), "inside_large type"

    # 3. 小物体bbox中心点在大型家具上下柱体内部的时候，选择离小物体最近的表面所属的大型家具作为关联家具。
    # 同时判断是否为支撑类型：两个bbox iou大于0时，小物体抬高 15 cm后再物理仿真。TODO
    min_i, min_point_face_distance = -1, float('inf')

    for i, large_index in enumerate(large_index_list):
        enlarged_size = large_size_list[i] +  np.array([0,0,1e6]) # 扩充为无限高
        enlarged_bbox = np.concatenate([large_loc_list[i], enlarged_size, large_euler_list[i]])

        if is_point_inside_bbox(small_loc, enlarged_bbox):

            large_faces = get_bbox_Z_faces(large_bbox_list[i])
            face_dis = min([get_point_face_distance(small_loc, face) for face in large_faces])

            if face_dis < min_point_face_distance:
                min_point_face_distance = face_dis
                min_i = i
            
    if min_i != -1:
        enlarged_size = large_size_list[min_i] +  np.array([0,0,0.03])  # 增加三厘米
        enlarged_bbox = np.concatenate([large_loc_list[min_i], enlarged_size, large_euler_list[min_i]])
        iou = get_totated_bbox_iou(small_bbox, enlarged_bbox)
        if iou > 0: # 判断是否为支撑类型：两个bbox iou大于0时，小物体抬高 15 cm后再物理仿真。
            return (large_shift_list[min_i] + np.array([0, 0, 0.15])).tolist(), "support type"
        else:
            return large_shift_list[min_i].tolist(), "over type"
        

    # 4. 小物体bbox所有顶点在 大型家具bbox扩大为原来的2倍大小之后的bbox内部，判定关联
    min_i, min_volume = -1, float('inf')
    for i, large_index in enumerate(large_index_list):
        enlarged_size = large_size_list[i] * 2
        enlarged_bbox = np.concatenate([large_loc_list[i], enlarged_size, large_euler_list[i]])
        if is_bbox1_inside_bbox2(small_bbox, enlarged_bbox):
            large_volume = np.prod(large_size_list[i])
            if large_volume < min_volume:
                min_volume = large_volume
                min_i = i

    if min_i != -1:
        return large_shift_list[min_i].tolist(),"inside_large_enlarge type"


    # 5. 其余物体不判断关联，shift = 0.
    return [0.0, 0.0, 0.0], "no type"




 
def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_instance_infos_dir", required=True, type=str, help = "")
    parser.add_argument("--output_instance_infos_dir", required=True, type=str, help = "")
    parser.add_argument("--cate_with_large_size_path", required=True, type=str, help = "")

    args = parser.parse_args()
    
    return args

def bind_one_scene(
        input_instance_infos_dir,
        output_instance_infos_dir,
        cate_with_large_size
):
    with open(input_instance_infos_dir, "r") as f:
        instance_infos = json.load(f)

    large_index_list = []
    small_index_list = []
    for i, instance_info in enumerate(instance_infos):
        try:
            cate = instance_info["category"]
        except Exception as e:
            print(f"Error in instance_info index:{i}: lack of key _category_")
            continue
        if cate in cate_with_large_size:
            large_index_list.append(i)
        else:  
            small_index_list.append(i)

    from tqdm import tqdm
    for small_index in tqdm(small_index_list, desc = "binding small to large..."):

        object_shift, shift_type = get_object_shift(small_index, large_index_list, copy.deepcopy(instance_infos))
        instance_infos[small_index]["delta_translation_after_large_obj_optimization"] = object_shift
        instance_infos[small_index]["shift_type"] = shift_type

    with open(output_instance_infos_dir, "w") as f:
        json.dump(instance_infos, f, indent=4)


if __name__ == "__main__":
    args = parse_arguments()

    
    with open(args.input_instance_infos_dir, "r") as f:
        instance_infos = json.load(f)
    with open(args.cate_with_large_size_path, "r") as f:
        cate_with_large_size = json.load(f) 
        cate_with_large_size = list(cate_with_large_size.keys())

    large_index_list = []
    small_index_list = []
    for i, instance_info in enumerate(instance_infos):
        try:
            cate = instance_info["category"]
        except Exception as e:
            print(f"Error in instance_info index:{i}: lack of key _category_")
            continue
        if cate in cate_with_large_size:
            large_index_list.append(i)
        else:  
            small_index_list.append(i)

    from tqdm import tqdm
    for small_index in tqdm(small_index_list, desc = "binding small to large..."):
        # # 分析函数每一行的耗时
        # from line_profiler import LineProfiler
        # lp = LineProfiler()
        # lp.add_function(get_object_shift)
        # result = lp(get_object_shift)(small_index, large_index_list, copy.deepcopy(instance_infos))
        # lp.print_stats()
        # import pdb;pdb.set_trace()

        object_shift, shift_type = get_object_shift(small_index, large_index_list, copy.deepcopy(instance_infos))
        instance_infos[small_index]["delta_translation_after_large_obj_optimization"] = object_shift

    with open(args.output_instance_infos_dir, "w") as f:
        json.dump(instance_infos, f, indent=4)


        
        



