import os 
import sys
import re
from tqdm import tqdm
import argparse
import json
import shutil
from collections import defaultdict

from pxr import Usd, UsdGeom
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
import open3d as o3d

from scene_utils.file_tools import get_json_data, recovery
from scene_utils.usd_tools import compute_bbox, merge_bbox, compute_bbox_volume, calculate_bbox_iou, is_bbox_nearby
from scene_utils.usd_tools import remove_empty_prims, fix_mdls, IsNestedXform, get_transform_from_prim, is_all_light_xform
from scene_utils.usd_tools import sample_points_from_prim, norm_coords, filter_free_noise, convert_usd_to_points, get_prims_from_xyz_bounding
from scene_utils.visualization_tools import visualize_bboxes, visualize_clusters
from scene_utils.file_tools import parse_scene
from scene_utils.usd_tools import move_merged_group


parser = argparse.ArgumentParser()
parser.add_argument("--scene_start_index", type=int, default=1)
parser.add_argument("--scene_end_index", type=int, default=2)
parser.add_argument("--scene_id", type=int, default=1)
parser.add_argument("--data_path", help="Path to the grutopia file", default='/ailab/user/caopeizhou/data/GRScenes')
parser.add_argument("--output_dir", help="Path to where the data should be saved", default="/ailab/user/caopeizhou/projects/SceneCleaning/output/instances")
parser.add_argument("--part", type=int, default=1)
parser.add_argument("--usd_folder", type=int, default=1)
parser.add_argument("--debug", type=bool, required=True)
parser.add_argument("--mdl_path", help="Path to the mdl file", default='/ailab/user/caopeizhou/projects/GRRegion/mdls/default.mdl')
parser.add_argument("--use_scene_str", type=bool, required=True)
parser.add_argument("--scene_str", type=str)

args = parser.parse_known_args()[0]


def find_path(dir, endswith):
    paths = sorted(os.listdir(dir))
    for p in paths:
        if endswith in p:
            target_path = os.path.join(dir,p)
            return target_path
        
def find_all_usd_files(dir):
    paths = sorted(os.listdir(dir))
    usd_files = []
    for p in paths:
        if p.endswith('.usd') or p.endswith('.usda'):
            usd_files.append(os.path.join(dir,p))
    return usd_files


def move_prim(source_path, target_path, editor):
    editor.MovePrimAtPath(source_path, target_path)
    editor.ApplyEdits()

def reparent_prim(source_prim, target_prim, editor):
    
    editor.ReparentPrim(source_prim, target_prim)
    editor.ApplyEdits()

def clean_scene(usd_path, annotation_json_path, duplicates_json_path, filter_noise=False):
    structure_list = ['wall', 'floor', 'ceiling', 'bg_wall']
    instance_list = ['complete', 'combined', 'part']

    # fix materials
    default_mdl_path = args.mdl_path
    # print(os.path.exists(default_mdl_path))
    fix_mdls(usd_path, default_mdl_path)

    stage = Usd.Stage.Open(usd_path)
    remove_empty_prims(stage)
    world_prim = stage.GetPrimAtPath("/World")
    prims_all = world_prim.GetAllChildren()
    # remove_bad_prims(stage)
    if filter_noise:
        meters_per_unit = stage.GetMetadata('metersPerUnit')
        scene_pcd_without_labeled_noise, prim_mesh = convert_usd_to_points(stage, meters_per_unit, json_data, False)
        scene_pcd_without_xy_free_noise = filter_free_noise(scene_pcd_without_labeled_noise, plane='xy')    
        scene_pcd_without_free_noise = filter_free_noise(scene_pcd_without_xy_free_noise, plane='xz')     
        scene_pcd = scene_pcd_without_free_noise
        points = np.array(scene_pcd.points)
        points_max = np.max(points, axis=0) / meters_per_unit
        points_min = np.min(points, axis=0) / meters_per_unit

        prims_in, prims_out = get_prims_from_xyz_bounding(prim_mesh, points_min, points_max)


    # editor = Usd.NamespaceEditor(stage)
    annotation_data = get_json_data(annotation_json_path)
    duplicates_data = get_json_data(duplicates_json_path)
    recovery_info = recovery(annotation_data, duplicates_data)
    


    light_scope_path = world_prim.GetPath().AppendChild("Lights")
    camera_scope_path = world_prim.GetPath().AppendChild("Cameras")
    structure_scope_path = world_prim.GetPath().AppendChild("Structure")
    wall_scope_path = structure_scope_path.AppendChild("Wall")
    floor_scope_path = structure_scope_path.AppendChild("Floor")
    ceiling_scope_path = structure_scope_path.AppendChild("Ceiling")
    bg_wall_scope_path = structure_scope_path.AppendChild("BgWall")
    instance_scope_path = world_prim.GetPath().AppendChild("Instances")
    complete_scope_path = instance_scope_path.AppendChild("Complete")
    combined_scope_path = instance_scope_path.AppendChild("Combined")
    part_scope_path = instance_scope_path.AppendChild("Part")
    noise_scope_path = world_prim.GetPath().AppendChild("Noise")
    
    stage.DefinePrim(light_scope_path, "Scope")
    stage.DefinePrim(camera_scope_path, "Scope")
    stage.DefinePrim(structure_scope_path, "Scope")
    stage.DefinePrim(instance_scope_path, "Scope")
    stage.DefinePrim(noise_scope_path, "Scope")
    stage.DefinePrim(complete_scope_path, "Scope")
    stage.DefinePrim(combined_scope_path, "Scope")
    stage.DefinePrim(part_scope_path, "Scope")
    stage.DefinePrim(wall_scope_path, "Scope")
    stage.DefinePrim(floor_scope_path, "Scope")
    stage.DefinePrim(ceiling_scope_path, "Scope")
    stage.DefinePrim(bg_wall_scope_path, "Scope")

    editor = Usd.NamespaceEditor(stage)
    print(len(prims_all))
    
    # part_prims = []
    print("processing all prims which have been annotated...")

    for prim in prims_all:
        if prim.GetTypeName() in ['CylinderLight', 'DistantLight', 'DomeLight', 'DiskLight', 'GeometryLight', 'RectLight', 'SphereLight']:
            move_prim(prim.GetPath(), light_scope_path.AppendChild(prim.GetName()), editor)
        elif prim.GetTypeName() == 'Camera':
            move_prim(prim.GetPath(), camera_scope_path.AppendChild(prim.GetName()), editor)
        elif prim.GetTypeName() in ['Mesh', 'Xform']:
            # Determine whether an xform is full of light
            if prim.GetTypeName() == 'Xform' and is_all_light_xform(prim):
                move_prim(prim.GetPath(), light_scope_path.AppendChild(prim.GetName()), editor)
                continue
            prim_name = prim.GetName()
            for item in recovery_info:
                if item['name'] == prim_name:
                    if item['feedback'] in instance_list:
                        if item['feedback'] == 'complete':
                            move_prim(prim.GetPath(), complete_scope_path.AppendChild(prim.GetName()), editor)
                        elif item['feedback'] == 'combined':
                            move_prim(prim.GetPath(), combined_scope_path.AppendChild(prim.GetName()), editor)
                        elif item['feedback'] == 'part':
                            
                            move_prim(prim.GetPath(), part_scope_path.AppendChild(prim.GetName()), editor)
                            # part_prims.append(prim)
                        # move_prim(prim.GetPath(), instance_scope_path.AppendChild(prim.GetName()), editor)
                    elif item['feedback'] in structure_list:
                        if item['feedback'] == 'wall':
                            move_prim(prim.GetPath(), wall_scope_path.AppendChild(prim.GetName()), editor)
                        elif item['feedback'] == 'floor':
                            move_prim(prim.GetPath(), floor_scope_path.AppendChild(prim.GetName()), editor)
                        elif item['feedback'] == 'ceiling':
                            move_prim(prim.GetPath(), ceiling_scope_path.AppendChild(prim.GetName()), editor)
                        elif item['feedback'] == 'bg_wall':
                            move_prim(prim.GetPath(), bg_wall_scope_path.AppendChild(prim.GetName()), editor)
                        # move_prim(prim.GetPath(), structure_scope_path.AppendChild(prim.GetName()), editor)
                    elif item['feedback'] == 'confirm' and (prim in prims_out if filter_noise else True):
                        move_prim(prim.GetPath(), noise_scope_path.AppendChild(prim.GetName()), editor)




    print("Processing the part prims...")
    part_scope = stage.GetPrimAtPath("/World/Instances/Part")
    part_prims = [prim for prim in part_scope.GetAllChildren()]
    merge_groups = merge_parts(part_prims)
    for i, group in enumerate(merge_groups):
        group_path = part_scope_path.AppendChild(f"Group{i}_merged")
        stage.DefinePrim(group_path, "Xform")
        # print(group)
        group = set(group)
        group = list(group)
        # print(len(group))
        # if i == 9:
        #     visualize_bboxes(group)
        for group_prim in group:
            move_prim(group_prim.GetPath(), group_path.AppendChild(group_prim.GetName()), editor)
        # stage.Save()

    print("Processing the combined prims...")
    combined_scope = stage.GetPrimAtPath("/World/Instances/Combined")
    combined_prims_before = [prim for prim in combined_scope.GetAllChildren()]

    for prim in combined_prims_before:
        
        prim_children = prim.GetChildren()
        # new_path = None
        child_types = [child.GetTypeName() for child in prim_children]

        if all(child_type =="Xform" for child_type in child_types):
            for child in prim_children:
                new_child = IsNestedXform(child)
                if child != new_child:
                    new_path = combined_scope_path.AppendChild(new_child.GetName())
                    local2word_transform = get_transform_from_prim(new_child) #
                    move_prim(new_child.GetPath(), new_path, editor)
                    xformable = UsdGeom.Xformable(stage.GetPrimAtPath(new_path))
                    xformable.MakeMatrixXform().Set(local2word_transform)

    # remove_empty_prims(stage)
    combined_prims = [prim for prim in combined_scope.GetAllChildren()]
    # print(combined_prims)
    print(combined_prims)

    groups = defaultdict(list)
    label_num = None
    for prim in combined_prims:
        if prim.GetTypeName() == 'Mesh':
            continue
        if label_num is None:
            label_num = 0
        else:
            label_num += len(set(mesh_prim_labels)) + len(xform_prims)
        mesh_prim, mesh_prim_labels, xform_prims = split_group(prim, light_scope_path, editor)
        for prim_mesh, prim_mesh_label in zip(mesh_prim, mesh_prim_labels):
            groups[label_num + prim_mesh_label].append(prim_mesh)
        current_idx = len(set(mesh_prim_labels))
        for idx_xform, xform_prim in enumerate(xform_prims):
            groups[label_num + current_idx + idx_xform].append(xform_prim)

        
    print(groups)

    for label, group in groups.items():
        # print(group)
        if len(group) == 1:
            # print(group[0].GetPath(), combined_scope_path.AppendChild(group[0].GetName()))
            local2word_transform = get_transform_from_prim(group[0])
            group_parent_name = group[0].GetParent().GetName()
            group_new_name = f"{group[0].GetName()}_from_{group_parent_name}"
            new_path = combined_scope_path.AppendChild(group_new_name)
            move_prim(group[0].GetPath(), new_path, editor)
            xformable = UsdGeom.Xformable(stage.GetPrimAtPath(new_path))
            xformable.MakeMatrixXform().Set(local2word_transform)
        else:
            group_splited_path = combined_scope_path.AppendChild(f"Group{label}_splited")
            stage.DefinePrim(group_splited_path, "Xform")
            group_splited = stage.GetPrimAtPath(group_splited_path)
            # print(group)
            for prim_in_group in group:
                # print(prim_in_group.GetPath(), group_splited_path.AppendChild(prim_in_group.GetName()))
                local2word_transform = get_transform_from_prim(prim_in_group)
                new_path = group_splited_path.AppendChild(prim_in_group.GetName())
                move_prim(prim_in_group.GetPath(), group_splited_path.AppendChild(prim_in_group.GetName()), editor)
                xformable = UsdGeom.Xformable(stage.GetPrimAtPath(new_path))
                xformable.MakeMatrixXform().Set(local2word_transform)
                # print(label, group)
            
    remove_empty_prims(stage)
    stage.Save()

def buid_relationship_matrix(part_prims):
    relationship_matrix = {}
    for i, prim_a in enumerate(part_prims):
        bbox_a = compute_bbox(prim_a)
        relationships = {'intersect': [], 'nearby': [], 'all': []}
        for j, prim_b in enumerate(part_prims):
            if i == j:
                continue
            bbox_b = compute_bbox(prim_b)
            is_intersect, iou = calculate_bbox_iou(bbox_a, bbox_b)
            is_nearby, distance = is_bbox_nearby(bbox_a, bbox_b)
            if is_intersect:
                relationships['intersect'].append((iou, prim_b))
                continue
            if is_nearby:
                relationships['nearby'].append((distance, prim_b))
        relationships['intersect'].sort(key=lambda x: x[0], reverse=True)
        relationships['nearby'].sort(key=lambda x: x[0])
        relationships['all'] = [x[1] for x in relationships['intersect']] if relationships['intersect'] else [x[1] for x in relationships['nearby']]
        relationship_matrix[prim_a.GetName()] = relationships
    return relationship_matrix

def merge_parts(part_prims):

    relationship_matrix = buid_relationship_matrix(part_prims)
    sorted_part_prims = sorted(part_prims, key=lambda prim: (len(relationship_matrix[prim.GetName()]['all']), compute_bbox_volume(compute_bbox(prim))), reverse=True)
    # for prim in sorted_part_prims:
    #     print(prim.GetName(), len(relationship_matrix[prim.GetName()]['intersect']), relationship_matrix[prim.GetName()]['all'])
        # if len(relationship_matrix[prim.GetName()]['intersect']) == 0:
        #     print(prim.GetName())
            # continue
    
    merged_group = []
    visited_prims = set()
    sorted_part_prims_copy = sorted_part_prims.copy()
    # print(sorted_part_prims_copy)
    for current_prim in sorted_part_prims_copy:
        if current_prim in visited_prims:
            continue

        group = [current_prim]
        

        related_prims = relationship_matrix[current_prim.GetName()]['all']
        if len(related_prims) == 1 and related_prims[0] in visited_prims:
            for group in merged_group:
                if related_prims[0] in group:
                    group.append(current_prim)
                    visited_prims.add(current_prim)
                    break
            continue


        for related_prim in related_prims:
            if related_prim in visited_prims:
                continue

            relationships = relationship_matrix[related_prim.GetName()]['all']
            if len(relationships) == 1:
                if relationships[0].GetName() == current_prim.GetName():
                    group.append(related_prim)
                    visited_prims.add(related_prim)
                    continue

            else:
                if relationships[0].GetName() == current_prim.GetName():
                    group.append(related_prim)
                    visited_prims.add(related_prim)
                elif relationships[0] in related_prims:
                    group.append(related_prim)
                    visited_prims.add(related_prim)
                elif calculate_bbox_iou(compute_bbox(relationships[0]), merge_bbox(group))[0]:
                    group.append(related_prim)
                    visited_prims.add(related_prim)
                    group.append(relationships[0])
                    visited_prims.add(relationships[0])

        visited_prims.add(current_prim)
        merged_group.append(group)

    return merged_group

def dbscan_cluster_custom(points, labels, eps=0.04, min_samples=5):

    # calulate distance matrix
    dist_matrix = pairwise_distances(points)

    # compute prim centers
    # unique_labels = np.unique(labels)
    # centers = {label: np.mean(points[labels == label], axis=0) for label in unique_labels}
    # center_array = np.array([centers[label] for label in labels])
    # centers_dist_matrix = pairwise_distances(center_array)

    # dist_matrix += centers_dist_matrix
    label_matrix = labels[:, None] == labels[None, :]
    dist_matrix[label_matrix] = 0

    clusting = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    cluster_labels = clusting.fit_predict(dist_matrix)
    # labels = clusting.labels_
    return cluster_labels

def split_group(group_prim, light_scope_path, editor):
    
    point_number = 100
    # group must be a xform type 
    while len(group_prim.GetChildren()) == 1 and group_prim.GetChildren()[0].GetTypeName() == "Xform":
        group_prim = group_prim.GetChildren()[0]
    print(f"Splitting {group_prim.GetName()}")
    if len(group_prim.GetChildren()) == 1:
        return [group_prim], [0], []
    else:
        prims_in_group = group_prim.GetChildren()
        pcs_all = []
        prim_meshes = []
        prim_xforms = []
        # pcs_need_to_merge
        labels_all = []
        for prim_id, prim in enumerate(prims_in_group):
            if prim.GetTypeName() == "Xform":
                # if len(prim.GetChildren()) == 1 and prim.GetChildren()[0].GetTypeName() == "Xform":
                #     split_group(prim)
                #     continue
                #     # while len(prim.GetChildren()) == 1 and prim.GetChildren()[0].GetTypeName() == "Xform":
                #     #     prim = prim.GetChildren()[0]
                # else:
                prim_xforms.append(prim)
                continue
            # continue
            # print(f"Processing {prim.GetName()}")
            if prim.GetTypeName() in ['CylinderLight', 'DistantLight', 'DomeLight', 'DiskLight', 'GeometryLight', 'RectLight', 'SphereLight']:
                move_prim(prim.GetPath(), light_scope_path.AppendChild(prim.GetName()), editor)
                continue
            current_prim_pcs, _ = sample_points_from_prim(prim, point_number)
            pcs_all.append(current_prim_pcs)
            prim_meshes.append(prim)
            labels_all.extend([prim_id]*point_number)

        if len(pcs_all) != 0:
            labels_all = np.array(labels_all)
            pcs_all = np.concatenate(pcs_all, axis=0) 
            # normalize points
            pcs_all = norm_coords(pcs_all)
            labels = dbscan_cluster_custom(pcs_all, labels_all)
            # print(labels)
            # visualize_clusters(pcs_all, labels)
            prim__meshes_labels = []
            for i in range(len(prim_meshes)):
                prim__meshes_labels.append(labels[i*point_number])
            # print(prim__meshes_labels)

            return prim_meshes, prim__meshes_labels, prim_xforms
        else:
            return [], [], prim_xforms
            

if __name__ == '__main__':

    usd_folder = f"{args.usd_folder}_usd"
    part = args.part
    part_folder = f"part{part}"
    data_path = os.path.join(args.data_path, part_folder, usd_folder)
    output_path = os.path.join(args.output_dir, part_folder, usd_folder)
    # house_id = sorted(os.listdir(data_path))[args.scene_id]

    if args.use_scene_str:
        house_id = args.scene_str
    else:
        house_id = sorted(os.listdir(data_path))[args.scene_id]

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    mdl_path = args.mdl_path
    # for house_id in house_ids:
    print(house_id)

    if args.debug:
        check_path = os.path.join(data_path, house_id, 'models')
    else:
        check_path = os.path.join(output_path, house_id)

    if not os.path.exists(check_path):
        # usd_path = find_path(os.path.join(data_path, house_id), endswith='.usd')
        # usd_file_name = os.path.split(usd_path)[-1].replace(".usd", "")
        scene_path = os.path.join(data_path, house_id)
        json_path = find_path(scene_path, endswith='.json')
        usd_paths = find_all_usd_files(scene_path)
        json_data = None
        if json_path is not None:
            print(json_path)
            if len(usd_paths) == 1:
                usd_path = usd_paths[0]
                usd_name = os.path.basename(usd_path)
                usd_copy_name = os.path.basename(usd_path).replace(".usd", "_copy.usd")
            else:
                pattern = f'_part{part}_{usd_folder}.*_check\.json'
                json_name = os.path.basename(json_path)
                usd_name = re.sub(pattern, '.usd', json_name)
                # usd_name = os.path.basename(json_path).replace("_check.json", ".usd")
                usd_copy_name = usd_name.replace(".usd", "_copy.usd")
                usd_path = os.path.join(scene_path, usd_name)
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            if os.path.exists(usd_path):
                shutil.copy(usd_path, os.path.join(scene_path, usd_copy_name))
                usd_path = os.path.join(scene_path, usd_copy_name)
                dup_folder_path = os.path.join(scene_path, "pcs/duplicate_records")
                duplicates_json_path = find_path(dup_folder_path, endswith='.json')
                models_path = os.path.join(scene_path, "models")
                print(f"Cleaning part{part}/{usd_folder}/{house_id}.")
                clean_scene(usd_path, json_path, duplicates_json_path, filter_noise=True)
                print(f"Cleaning part{part}/{usd_folder}/{house_id} done.")

                print(f"Creating model usd files for part{part}/{usd_folder}/{house_id}.")
                output_instances_path = os.path.join(output_path, house_id)
                # if not os.path.exists(output_instances_path):
                #     os.makedirs(output_instances_path)
                # original_path = True
                if args.debug:
                    parse_scene(usd_path, scene_path)
                else:
                    parse_scene(usd_path, output_instances_path)
                print(f"Creating model usd files for part{part}/{usd_folder}/{house_id} done.")

                if os.path.exists(models_path):
                    for model_name in os.listdir(models_path):
                        if model_name.endswith("_merged"):
                            model_usd_path = os.path.join(models_path, f'{model_name}/instance.usd')
                            move_merged_group(model_usd_path)
                if not os.path.exists(models_path) and os.path.exists(usd_path):
                    shutil.remove(usd_path)

        if not args.debug:
            shutil.remove(usd_path)





    # # print(folder_part_usd)
    # for usd in folder_part_usd:
    #     usd_path = os.path.join(part_path, usd)
    #     scene_folders = [f for f in os.listdir(usd_path) if os.path.isdir(os.path.join(usd_path, f))]
    #     scene_folders = sorted(scene_folders)
    #     # scene_folders.remove("0006")
    #     # scene_folders.remove("0007")
    #     # scene_folders.remove("0008")
    #     # scene_folders.remove("0009")
    #     for folder in scene_folders:
    #         folder_path = os.path.join(usd_path, folder)
    #         dup_folder_path = os.path.join(folder_path, "pcs/duplicate_records")
    #         usd_files = [f for f in os.listdir(folder_path) if f.endswith(".usd")]
    #         usd_file_path = os.path.join(folder_path, usd_files[0])
    #         annotation_json = [f for f in os.listdir(folder_path) if f.endswith(".json")][0]
    #         annotation_json_path = os.path.join(folder_path, annotation_json)
    #         duplicates_json = [f for f in os.listdir(dup_folder_path) if f.endswith(".json")][0]
    #         duplicates_json_path = os.path.join(dup_folder_path, duplicates_json)
    #         models_path = os.path.join(folder_path, "models")
    #         if os.path.exists(models_path):
    #             continue
    #         print(f"Cleaning part{part}/{usd}/{folder}.")
    #         clean_scene(usd_file_path, annotation_json_path, duplicates_json_path)
    #         print(f"Cleaning part{part}/{usd}/{folder} done.")

    #         # create the model usd files
    #         print(f"Creating model usd files for part{part}/{usd}/{folder}.")
    #         parse_scene(usd_file_path, folder_path)
    #         print(f"Creating model usd files for part{part}/{usd}/{folder} done.")
            
    #         # This is because there are no objects in some scenes. 
    #         # It may be that when rendering, all objects are a whole xform, so they are not rendered successfully.
    #         if os.path.exists(models_path):
    #             for model_name in os.listdir(models_path):
    #                 if model_name.endswith("_merged"):
    #                     model_usd_path = os.path.join(models_path, f'{model_name}/instance.usd')
    #                     move_merged_group(model_usd_path)


    # test_folder_path = os.path.join(part_path, "110_usd/0002")
    # test_dup_folder_path = os.path.join(test_folder_path, "pcs/duplicate_records")

    # test_usd = [f for f in os.listdir(test_folder_path) if f.endswith("Test.usd")][0]
    # test_usd_path = os.path.join(test_folder_path, test_usd)

    # test_annotation_json = [f for f in os.listdir(test_folder_path) if f.endswith(".json")][0]
    # test_annotation_json_path = os.path.join(test_folder_path, test_annotation_json)

    # test_duplicates_json = [f for f in os.listdir(test_dup_folder_path) if f.endswith(".json")][0]
    # test_duplicates_json_path = os.path.join(test_dup_folder_path, test_duplicates_json)
    # clean_scene(test_usd_path, test_annotation_json_path, test_duplicates_json_path)

    # stage = Usd.Stage.Open(test_usd_path)
    # test_prim_path = "/World/Instances/Combined/___2146586954_1/___2146586953_1/___2146586943_1/___2146586944_1/___E_model_202354125940424171_1"
    # test_prim = stage.GetPrimAtPath(test_prim_path)
    # split_group(test_prim)