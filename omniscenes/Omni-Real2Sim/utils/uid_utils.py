
import os
import json

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

    elif "gen_assets/" in uid: # 生成资产
        mesh_path = os.path.join(gen_assets_glb_folder, uid + ".glb")
        
    else:  # partnet-mobility 资产
        if "partnet_mobility_part/" in uid: #gapartnet 资产
            mesh_path = os.path.join(gapartnet_glb_folder, "PartNet", uid, "total_with_origin_rotation.glb")
        elif "partnet_mobility/" in uid: #partnet_mobility 资产
            mesh_path = os.path.join(partnet_mobility_glb_folder, uid,  "whole.obj")
        
    
    return mesh_path


def get_mapped_scene_name(scene_name):
    '''
    返回 mapping 之后的scenename 和 原始 scenename 对应的dict
    '''
    # load scene_mapping json
    mp3d_mapping = "/cpfs01/user/zhongweipeng/Projects/layout/EmbodiedScan/data/mp3d_mapping.json"
    threerscan_mapping = "/cpfs01/user/zhongweipeng/Projects/layout/EmbodiedScan/data/3rscan_mapping.json"
    with open(mp3d_mapping, "r") as f:
        mp3d_mapping = json.load(f)
    with open(threerscan_mapping, "r") as f:
        threerscan_mapping = json.load(f)

    if "arkitscenes" in scene_name:
        mapped_scene_name = "_".join(scene_name.split("/")[1:])
    elif "scannet" in scene_name:
        mapped_scene_name = scene_name.split("/")[-1]
    elif "3rscan" in scene_name:
        mapped_scene_name = scene_name.split("/")[-1]
        mapped_scene_name = threerscan_mapping[mapped_scene_name]
    elif "matterport3d" in scene_name:
        slices = scene_name.split("/")
        mapped_scene_name = mp3d_mapping[slices[1]] + "_" + slices[2]

    return mapped_scene_name

def get_origin_scene_name_wall_vertex(mapped_scene_name):

    if "scene" in mapped_scene_name:
        origin_scene_name = "scannet/" + mapped_scene_name
    elif "1mp3d_" in mapped_scene_name:
        mp3d_mapping = "/cpfs01/user/zhongweipeng/Projects/layout/EmbodiedScan/data/mp3d_mapping.json"
        with open(mp3d_mapping, "r") as f:
            mp3d_mapping = json.load(f)
        mapping_dict = {value:key for key, value in mp3d_mapping.items()}
        origin_scene_name = "matterport3d/" + mapping_dict["_".join(mapped_scene_name.split("_")[:2])] + "/" + mapped_scene_name.split("_")[-1]

    elif "-" in mapped_scene_name:
        origin_scene_name = "3rscan/" + mapped_scene_name

    elif "Validation_" in mapped_scene_name or "Training_" in mapped_scene_name:
        origin_scene_name = "arkitscenes/" + mapped_scene_name.split("_")[0] + "/" + mapped_scene_name.split("_")[1]

    return origin_scene_name

if __name__ == "__main__":
    ans = get_origin_scene_name_wall_vertex("02b33e03-be2b-2d54-9129-5d28efdd68fa")
    print(ans)