import sys
sys.path.extend(["./", "./codes_ybq", "./codes_ybq/base_process", "./codes_ybq/fuction", "./Sapien_Scene_Sim"])

import os
import pdb
import json
import time
import copy
import glob
import pickle
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import argparse


from codes_ybq.base_process.d_retrieval import retrieval_one_scene, load_info_for_retrieval
from codes_ybq.base_process.f_import_a_scene import compose_one_scene, load_info_for_compose_scene

from Sapien_Scene_Sim.simulate import simulate_one_scene, load_info_for_simulation
from codes_ybq.fuction.create_bbox_scene import compose_scene as compose_bbox_scene




def make_info_json(scene_name, pkl_data_list, scene_files_dir, categories):

    # path to make json file
    json_file_path = os.path.join(scene_files_dir, scene_name, "instances_info.json")

    for scene in pkl_data_list:
        select_scene = None
        if scene_name in scene['sample_idx']:
            select_scene = scene
            instance_list = []

            # process bbox
            if "arkitscenes" in scene_name:
                min_z = float("inf")
                for instance in select_scene['instances']:
                    cate = categories[int(instance['bbox_label_3d']) - 1]
                    if cate in ["floor", "ceiling", "wall"]:
                        continue
                    bottom_z = instance['bbox_3d'][2] - instance['bbox_3d'][5] / 2
                    if bottom_z < min_z:
                        min_z = bottom_z
                print(f"min_z = {min_z}")
                # pdb.set_trace()
                for instance in select_scene['instances']:
                    bbox = np.array(instance['bbox_3d']) - np.array([0,0,min_z,0,0,0,0,0,0])
                    instance_list.append({
                        'id':instance['bbox_id'],
                        'label':instance['bbox_label_3d'],
                        'category':categories[int(instance['bbox_label_3d']) - 1],
                        "origin_bbox": instance['bbox_3d'],
                        'bbox':bbox.tolist(),
                        '3d_model':''
                    })

            else:
                for instance in select_scene['instances']:
                    instance_list.append({
                        'id':instance['bbox_id'],
                        'label':instance['bbox_label_3d'],
                        'category':categories[int(instance['bbox_label_3d']) - 1],
                        'bbox':instance['bbox_3d'],
                        '3d_model':''
                    })

            os.makedirs(os.path.dirname(json_file_path), exist_ok=True)

            with open(json_file_path, 'w') as file:
                    json.dump(instance_list, file, indent=4)

            return
    
    raise Exception(f"scene {scene_name} not found in pkl file")

            # with open(json_file_path, mode='w', encoding='utf-8') as jsonfile:
            #         jsonfile.write(json.dumps(instance_list, ensure_ascii=False, indent=4))   

def process_one_scene_wo_try(
        scene_name, 
        pkl_data_list, 
        retrieval_info_dict, 
        compose_scene_info_dict,
        simulation_info_dict
):

    # 各种路径的设置
    scene_files_dir = "/cpfs01/user/zhongweipeng/Projects/layout/embodiedscene/scan"
    semantic_label_file_path = '/cpfs01/user/zhongweipeng/Projects/layout/semantic.json'

    # 1. 加载语义标签文件
    with open(semantic_label_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        categories = [item["category"] for item in data if "category" in item]
    
    # 创建instances_info.json文件
    make_info_json(scene_name, pkl_data_list, scene_files_dir, categories)

    # 2. 检索资产
    retrieval_one_scene(scene_name, retrieval_info_dict)

    # 3. 组合场景
    compose_one_scene(scene_name, compose_scene_info_dict)

    # 4. 模拟
    simulate_one_scene(scene_name, simulation_info_dict)

    # 5.output bbox scene
    compose_bbox_scene(os.path.join(scene_files_dir, scene_name, "instances_info_with_opti_with_shift_with_simu.json"), "embodiedscan")
    compose_bbox_scene(os.path.join(scene_files_dir, scene_name, "instances_info_with_opti_with_shift_with_simu.json"), "embodiedscan_aft_simu")
    
    return True, scene_name, None


def process_one_scene(
        scene_name, 
        pkl_data_list, 
        retrieval_info_dict, 
        compose_scene_info_dict,
        simulation_info_dict
):
    # paths
    # pkl_file = '/cpfs01/user/zhongweipeng/Projects/layout/EmbodiedScan/data/embodiedscan_infos_train.pkl'
    # cate_dict_new_path = "/cpfs01/user/zhongweipeng/Projects/layout/cate_dict_new.json"
    # partnetmobility_cate_addto_embodiedscan_path = "/cpfs01/user/zhongweipeng/Projects/layout/partnetmobility_cate_addto_embodiedscan.json"
    # partnetmobility_cate_dict_path = "/cpfs01/user/zhongweipeng/Projects/layout/partnetmobility_cate_dict.json"
    # Cap3D_csv_file_path = '/cpfs01/shared/landmark_3dgen/xuxudong_group/zhongweipeng/objaverse_categorize/Cap3D_Objaverse_batch_14_1.csv'
    # uid_size_dict_path = "/cpfs01/user/zhongweipeng/Projects/layout/cate_dict_new_size.json"
    # partnetmobility_size_dict_path = "/cpfs01/user/zhongweipeng/Projects/layout/partnetmobility_size_dict.json"
    # useful_uid_rotate_dict_path = "/cpfs01/user/zhongweipeng/Projects/layout/useful_uid_rotate_dict.json"
    # cate_with_apparent_direction_path = "/cpfs01/user/zhongweipeng/Projects/layout/cate_with_apparent_direction.json"
    try:
        # 各种路径的设置
        scene_files_dir = "/cpfs01/user/zhongweipeng/Projects/layout/embodiedscene/scan"
        semantic_label_file_path = '/cpfs01/user/zhongweipeng/Projects/layout/semantic.json'

        # 1. 加载语义标签文件
        with open(semantic_label_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            categories = [item["category"] for item in data if "category" in item]
        
        # 创建instances_info.json文件
        make_info_json(scene_name, pkl_data_list, scene_files_dir, categories)

        # 2. 检索资产
        retrieval_one_scene(scene_name, retrieval_info_dict)

        # 3. 组合场景
        compose_one_scene(scene_name, compose_scene_info_dict)

        # 4. 模拟
        simulate_one_scene(scene_name, simulation_info_dict)
        
        # 5.output bbox scene
        compose_bbox_scene(os.path.join(scene_files_dir, scene_name, "instances_info_with_opti_with_shift_with_simu.json"), "embodiedscan")
        compose_bbox_scene(os.path.join(scene_files_dir, scene_name, "instances_info_with_opti_with_shift_with_simu.json"), "embodiedscan_aft_simu")

        return True, scene_name, None
    except Exception as e:
        return False, scene_name, str(e)

def process_scenes_parallel(scene_names, pkl_data_list, retrieval_info_dict, compose_scene_info_dict, simulation_info_dict, max_workers):
    """
    并行处理多个场景
    """
    results = []
    
    # 使用ThreadPoolExecutor进行并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有场景处理任务
        futures = []
        for scene_name in scene_names:
            future = executor.submit(
                process_one_scene,
                scene_name,
                pkl_data_list,
                retrieval_info_dict,
                compose_scene_info_dict,
                simulation_info_dict
            )
            futures.append(future)
        
        # 收集所有任务的结果
        for future in tqdm(futures, desc="处理场景"):
            try:
                success, scene_name, error = future.result()
                if success:
                    print(f"场景 {scene_name} 处理成功")
                    results.append((scene_name, True))
                else:
                    print(f"场景 {scene_name} 处理失败: {error}")
                    results.append((scene_name, False, error))
            except Exception as e:
                print(f"处理任务失败: {str(e)}")
                results.append((None, False, str(e)))
    
    return results

def main():
    exp_num = 14
    read_exp_nums = [] # [10, 11]# [1, 2, 3]
    need_texture = False
    use_xysort_in_retrieval = False
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # read last exp succuss info
    successed_scene_names = []
    for read_exp_num in read_exp_nums:
        json_dir = f"/cpfs01/user/zhongweipeng/Projects/layout/codes_ybq/base_process/logs/logs_exp_{read_exp_num}/json_log/success_scenes_*.json"
        succuss_json_paths = glob.glob(json_dir)
        for succuss_json_path in succuss_json_paths:
            with open(succuss_json_path, 'r') as f:
                names = json.load(f)
                successed_scene_names.extend(names)

    # with open("/cpfs01/shared/landmark_3dgen/xuxudong_group/zhongweipeng/projects/layout/asset_process/scene_name_list.json", "w") as f:
    #     json.dump(successed_scene_names, f, indent=4)
    # pdb.set_trace()

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser = argparse.ArgumentParser(description="Process a calculated slice of 3D scenes sequentially.")
    parser.add_argument('--batch_index', type=int, required=True, help='The index of this batch (0-based).')
    parser.add_argument('--total_batches', type=int, required=True, help='The total number of batches.')
    args = parser.parse_args()

    batch_index = args.batch_index
    total_batches = args.total_batches
    process_id = os.getpid() # 用于日志
    if batch_index < 0 or batch_index >= total_batches:
        print(f"[PID {process_id}] ERROR: Invalid batch_index {batch_index}. Must be between 0 and {total_batches-1}.")
        sys.exit(1)
    print(f"[PID {process_id}] Starting Batch {batch_index+1}/{total_batches}.")
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


    # 0. load pkl_files
    pkl_files = [
         '/cpfs01/user/zhongweipeng/Projects/layout/EmbodiedScan/data/embodiedscan_infos_train.pkl',
         '/cpfs01/user/zhongweipeng/Projects/layout/EmbodiedScan/data/embodiedscan_infos_val.pkl',
         "/cpfs01/user/zhongweipeng/Projects/layout/EmbodiedScan/data/embodiedscan_ark_infos_test_div_fix.pkl"
    ]
    pkl_data_list = []
    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        pkl_data_list += data['data_list']

    # 1. load info dict for retrieval
    retrieval_info_dict = load_info_for_retrieval(use_xysort_in_retrieval=use_xysort_in_retrieval)

    # 2. load info dict for compose scene
    compose_scene_info_dict = load_info_for_compose_scene(use_xysort_in_retrieval=use_xysort_in_retrieval)

    # 3. load info dict for simulation
    simulation_info_dict = load_info_for_simulation(
        not_visualize = True,
        render = False,
        make_large_bbox_ground_aligned = True,
        need_texture = need_texture
    )

    # load scene_names
    scene_name_list = []
    for scene in pkl_data_list:
        if "instances" not in scene.keys():
            continue
        scene_name = scene['sample_idx']
        scene_name_list.append(scene_name)
    print(f"number of scenes: {len(scene_name_list)}")

    # 过滤掉已经成功处理的场景
    new_scene_name_list = []
    for scene_name in scene_name_list:
        if scene_name in successed_scene_names:
            continue
        else:
            new_scene_name_list.append(scene_name)
    scene_name_list = new_scene_name_list

    # 单个场景测试
    # 单个场景测试
    scene_name = 'scannet/scene0000_00'#"arkitscenes/Training/47331603"#"scannet/scene0457_01"#"matterport3d/Z6MFQCViBuw/region4" #"arkitscenes/Training/47331603"# 'scannet/scene0182_01'# 'scannet/scene0119_00' 'scannet/scene0000_00'
    # scene_name =  # "3rscan/751a559f-fe61-2c3b-8cec-258075450954" # # "arkitscenes/Training/48018233"  "arkitscenes/Training/42899811" "arkitscenes/Training/42444574" 
    process_one_scene_wo_try(
                scene_name=scene_name,
                pkl_data_list=pkl_data_list, # 需要完整的列表来查找场景信息
                retrieval_info_dict=retrieval_info_dict,
                compose_scene_info_dict=compose_scene_info_dict,
                simulation_info_dict=simulation_info_dict
            )
    # scene_files_sample_dir = "/cpfs01/user/zhongweipeng/Projects/layout/embodiedscene/50_samples_250417/"
    # output_files_path = os.path.join(scene_files_sample_dir, scene_name)
    # output_files_path = "/".join(output_files_path.split("/")[:-1])
    # os.makedirs(output_files_path, exist_ok=True)
    # os.system(f"cp -r /cpfs01/user/zhongweipeng/Projects/layout/embodiedscene/scan/{scene_name} {output_files_path}")
    pdb.set_trace() # --batch_index 0 --total_batches 8

    # # 多个场景测试
    # # 多个场景测试
    # with open("/cpfs01/user/zhongweipeng/Projects/layout/embodiedscene/20_samples_250416/20_samples_250416.json", "r")as f:
    #     scene_name_list = json.load(f)

    # # 筛选 arkitscenes
    # new_scene_name_list_wo_arkitscenes = []
    # for scene_name in scene_name_list:
    #     if "arkitscenes" not in scene_name:
    #         continue
    #     else:
    #         new_scene_name_list_wo_arkitscenes.append(scene_name)
    # scene_name_list = new_scene_name_list_wo_arkitscenes


    # 计算这个批次负责的场景切片
    total_scenes = len(scene_name_list)
    # 使用整数除法计算每个批次的大小（向上取整）
    chunk_size = (total_scenes + total_batches - 1) // total_batches
    start_index = batch_index * chunk_size
    end_index = min(start_index + chunk_size, total_scenes) # 防止越界
    scenes_to_process = scene_name_list[start_index:end_index]

    print(f"[PID {process_id}] Processing {len(scenes_to_process)} scenes in this batch: {start_index} - {end_index}.")

    # pdb.set_trace()

    #  time
    t_batch_start = time.time()
    failed_scenes = []
    success_scenes = []

    for scene_name in tqdm(scenes_to_process, desc=f"[PID {process_id}] Batch {batch_index+1}/{total_batches}", unit="scene"):
        t_scene_start = time.time()
        try:
            success, _, error = process_one_scene(
                scene_name=scene_name,
                pkl_data_list=pkl_data_list, # 需要完整的列表来查找场景信息
                retrieval_info_dict=retrieval_info_dict,
                compose_scene_info_dict=compose_scene_info_dict,
                simulation_info_dict=simulation_info_dict
            )
            if not success:
                print(f"\n[PID {process_id}] ERROR: Scene '{scene_name}' failed: {error}")
                failed_scenes.append(scene_name)
                with open(f"/cpfs01/user/zhongweipeng/Projects/layout/codes_ybq/base_process/logs/logs_exp_{exp_num}/json_log/failed_scenes_{batch_index}.json", "w") as f:
                    json.dump(failed_scenes, f, indent=4)

            else:
                success_scenes.append(scene_name)
                with open(f"/cpfs01/user/zhongweipeng/Projects/layout/codes_ybq/base_process/logs/logs_exp_{exp_num}/json_log/success_scenes_{batch_index}.json", "w") as f:
                    json.dump(success_scenes, f, indent=4)
                t_scene_end = time.time()
                print(f"\n[PID {process_id}] Scene '{scene_name}' succeeded in {t_scene_end - t_scene_start:.2f}s.")
        except Exception as e:
            print(f"\n[PID {process_id}] CRITICAL ERROR processing scene '{scene_name}': {e}")
            failed_scenes.append(scene_name)
            with open(f"/cpfs01/user/zhongweipeng/Projects/layout/codes_ybq/base_process/logs/logs_exp_{exp_num}/json_log/failed_scenes_{batch_index}.json", "w") as f:
                    json.dump(failed_scenes, f, indent=4)

    t_batch_end = time.time()
    print(f"\n[PID {process_id}] Batch {batch_index+1}/{total_batches} finished in {t_batch_end - t_batch_start:.2f} seconds.")

    if failed_scenes:
        print(f"[PID {process_id}] WARNING: {len(failed_scenes)} scene(s) failed in this batch: {', '.join(failed_scenes)}")
        sys.exit(1) # 批次中有失败，以失败状态退出
    else:
        print(f"[PID {process_id}] All {len(scenes_to_process)} scenes in this batch processed successfully.")
        sys.exit(0) # 批次成功完成

def filter_scene_with_extream_huge_object():
    scene_name_list = "/cpfs01/shared/landmark_3dgen/xuxudong_group/zhongweipeng/projects/layout/asset_process/scene_name_list.json"
    with open(scene_name_list, "r") as f:
        scene_name_list = json.load(f)

    scene_with_huge_obj = []
    for scene_name in tqdm(scene_name_list):
        scene_json_path = os.path.join("/cpfs01/user/zhongweipeng/Projects/layout/embodiedscene/scan", scene_name, "instances_info_with_retri_with_compose.json")
        with open(scene_json_path, "r") as f:
            scene_info = json.load(f)
        for instance in scene_info:
            if "geom_name" not in instance.keys():
                continue
            obj_size = instance["bbox"][3:6]

            if obj_size[0] > 15 or obj_size[1] > 15 or obj_size[2] > 15 or obj_size[0]*obj_size[1]*obj_size[2] > 100:
                if scene_name not in scene_with_huge_obj:
                    scene_with_huge_obj.append(scene_name)

    with open("/cpfs01/user/zhongweipeng/Projects/layout/ebdscan_scene_with_huge_obj_100.json", "w")as f:
        json.dump(scene_with_huge_obj, f, indent=4)

if __name__ == "__main__":
    # filter_scene_with_extream_huge_object()
    main()
    