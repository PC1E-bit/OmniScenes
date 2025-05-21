import os
import sys
import gradio as gr
from viser import ViserServer
import time
import socket
import numpy as np
import json
import trimesh
from collections import defaultdict
from utils.instance_utils import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--user_id", type=int, default=0)

args = parser.parse_args()

dir_name = os.path.dirname(os.path.abspath(sys.argv[0]))
refix_json_path = os.path.join(dir_name, f"data/refix1_jsons/refix_user{args.user_id}.json")
if not os.path.exists(refix_json_path):
    with open(refix_json_path, "w") as f:
        json.dump({}, f, indent=4)
all_folder_id_names = []
for part in ['1', '2']:
    data_path = os.path.join(dir_name, f"data/part{part}")
    usd_folders = sorted(os.listdir(data_path))
    usd_paths = [os.path.join(f"part{part}", f) for f in usd_folders]
    all_folder_id_names.extend(usd_paths)




all_scene_id_names = []
all_scene_ply_names = []
all_objects_npy_dirpath = []
all_scene_ply_path = []
all_check_json_paths = []

def get_scene_info(part, usd_folder):

    global all_scene_id_names, all_scene_ply_names, all_objects_npy_dirpath, all_scene_ply_path, all_check_json_paths

    data_path = os.path.join(dir_name, f"data/{part}")

    usd_path = os.path.join(data_path, usd_folder)
    scene_id_names = [f for f in os.listdir(usd_path) if os.path.isdir(os.path.join(usd_path, f))]
    # print(usd_folder, scene_id_names)
    scene_id_names = sorted(scene_id_names, key=lambda x: int(x))

    objects_npy_dirpath = [os.path.join(usd_path, f, "pcs") for f in scene_id_names]
    scene_ply_dirpath = [os.path.join(usd_path, f, "pcs/mesh_scene") for f in scene_id_names]
    check_json_paths = []
    scene_ply_names = []
    new_scene_id_names = scene_id_names.copy()
    for scene_id_name in scene_id_names:
        scen_id_path = os.path.join(usd_path, scene_id_name)
        # print([f for f in os.listdir(scen_id_path) if f.endswith("_check.json")])
        check_json = [f for f in os.listdir(scen_id_path) if f.endswith("_check.json")]
        # print(check_json)
        if check_json:
            check_json = check_json[0]
            scene_name = check_json.replace("_check.json", "")
            check_json_path = os.path.join(scen_id_path, check_json)
            check_json_paths.append(check_json_path)
            scene_ply_names.append(scene_name)
        else:
            print(f"Warning: {scene_id_name} has no check.json file")
            new_scene_id_names.remove(scene_id_name)
    scene_id_names = new_scene_id_names
    objects_npy_dirpath = [os.path.join(usd_path, f, "pcs") for f in scene_id_names]
    scene_ply_dirpath = [os.path.join(usd_path, f, "pcs/mesh_scene") for f in scene_id_names]
    # scene_ply_names = [os.listdir(scene_ply_dir)[0].split(".")[0] for scene_ply_dir in scene_ply_dirpath]
    # print(scene_ply_names)
    # name_count = defaultdict(int)
    # unique_scene_ply_names = []
    # for name in scene_ply_names:
    #     name_count[name] += 1
    #     if name_count[name] > 1:
    #         print(f"Warning: {name} is repeated in {scene_ply_dirpath}")
    #         unique_name = f"{name}_part{part}_{usd_folder}_{name_count[name]}"
    #         print(unique_name)
    #         unique_scene_ply_names.append(unique_name)
    #     else:
    #         unique_scene_ply_names.append(name)

    scene_ply_path = [os.path.join(scene_ply_dir, f"{name}.ply") for scene_ply_dir, name in zip(scene_ply_dirpath, scene_ply_names)]

    # for idx, ply_path in enumerate(scene_ply_path):
    #     if scene_ply_names[idx] != unique_scene_ply_names[idx]:
    #         print(f"Rename {scene_ply_names[idx]} to {unique_scene_ply_names[idx]}")
    #         os.rename(ply_path, os.path.join(os.path.dirname(ply_path), os.path.basename(ply_path).replace(scene_ply_names[idx], unique_scene_ply_names[idx])))
    scene_ply_path = [os.path.join(scene_ply_dir, f"{name}.ply") for scene_ply_dir, name in zip(scene_ply_dirpath, scene_ply_names)]
    check_json_paths = [os.path.join(usd_path, f, f"{name}_check.json") for f, name in zip(scene_id_names, scene_ply_names)]

    all_scene_id_names = scene_id_names
    all_scene_ply_names = scene_ply_names
    all_objects_npy_dirpath = objects_npy_dirpath
    all_scene_ply_path = scene_ply_path
    all_check_json_paths = check_json_paths
    # print(unique_scene_ply_names)

    return scene_ply_names


def update_refix_json(folder_id, scene_name):
    with open(refix_json_path, "r") as f:
        data = json.load(f)
    if folder_id not in data:
        key = f'{folder_id}, {scene_name}'
        data[key] = True
    else:
        pass
    with open(refix_json_path, "w") as f:
        json.dump(data, f, indent=4)
    
def check_is_refix(folder_id, scene_name):
    with open(refix_json_path, "r") as f:
        data = json.load(f)
    key = f'{folder_id}, {scene_name}'
    if key in data:
        return data[key]
    else:
        return False



ip_share = "0.0.0.0"
# ip_gradio = "127.0.0.1"
ip_gradio = "139.196.152.150"

css = """
.row {
}
.left {
    height: 750px;
}
.right{

    height: 400px;
}
.custom-textbox textarea {
    font-size: 20px;
}
.custom-slider{

}
"""

recheck_info_box = "#### Determine which category the object belongs to: Objects in the room/Structure of the house/Abandoned materials that do not belong to the room (noise)"
def info_mapping(info):
    mapping = {
        "confirm": "Noise",
        "complete": "Complete",
        "combined": "Grouped",
        "part": "Partial",
        "ceiling": "Ceiling",
        "floor": "Floor",
        "wall": "Wall",
        "bg_wall": "Background"
    }
    return mapping[info]




def get_check_len(check_data):
    check_len = 0
    for data in check_data:
        if 'feedback' in data:
            check_len += 1
    return check_len
            

scene_centroid_mesh = None
scene_scale_mesh = None 

current_scene_index = 0
current_file_index = -100

def start_annotation(user_index=0, default_port=7864):
    print(f"[INFO] The user_id is {user_index}")

    viser_port_1 = default_port + 3 * user_index
    server1 = ViserServer(ip_share, port = viser_port_1)
    server1.gui.configure_theme(control_layout="collapsible")
    server1.scene.world_axes.visible = False
    server1.scene.reset()

    viser_port_2 = viser_port_1 + 1
    server2 = ViserServer(ip_share, port = viser_port_2)
    server2.gui.configure_theme(control_layout="collapsible")
    server2.scene.world_axes.visible = True
    server2.scene.reset()

    gradio_port = viser_port_2 + 1
    # local_version
    #  <iframe src="http://{ip}:{viser_port1}" class="children"></iframe>
    # public_version
    # <iframe src="{share_url1}" class="children"></iframe>

    # Viser service iframe
    html_str=f"""
        <head>
        <meta charset="utf-8">
        <title></title>
        <style>
            .parent {{
                display: flex;
                height: 100%;
            }}
        
            .children {{
                margin: 0;
                width: 100%;
                height: 100%;
                background-color: black;
            }}
        </style>
        </head>
        <iframe src="http://{ip_gradio}:{viser_port_1}" class="children"></iframe>
        """
    
    html_str2=f"""
        <head>
        <meta charset="utf-8">
        <title></title>
        <style>
            .parent {{
                display: flex;
                height: 100%;
            }}
        
            .children {{
                margin: 0;
                width: 100%;
                height: 100%;
                background-color: black;
            }}
        </style>
        </head>
        <iframe src="http://{ip_gradio}:{viser_port_2}" class="children"></iframe>
        """


    global all_scene_id_names, all_scene_ply_names, all_objects_npy_dirpath, all_scene_ply_path, all_scene_ply_path
    global current_scene_index, current_file_index

    def load_scene_pcd(scene_name):

        global scene_centroid_mesh, scene_scale_mesh, current_file_index, current_scene_index
        current_scene_index = all_scene_ply_names.index(scene_name)

        current_file_index = 0
        scene_centroid_mesh = None
        scene_scale_mesh = None
        server1.scene.reset()
        server2.scene.reset()
        scene_mesh_path = all_scene_ply_path[all_scene_ply_names.index(scene_name)]
        # current_scene_name = all_scene_ply_names[all_scene_ply_names.index(scene_name)]
        original_size_mb = os.path.getsize(scene_mesh_path) / (1024 * 1024)
        print(f"Original file size is {original_size_mb} MB")
        scene_ori_mesh = trimesh.load(scene_mesh_path)
        scene_mesh, scene_centroid_mesh, scene_scale_mesh = mesh_norm(scene_ori_mesh)
        # server1.scene.add_point_cloud(f"{scene_name}", scene[:, :3], scene[:, 3:6], point_size=0.0001)
        server1.scene.add_mesh_trimesh(
            name=f"/{scene_name}",
            mesh = scene_mesh,
            position=(0.0, 0.0, 0.0),
        )
        # load check json file
        check_json_path = all_check_json_paths[current_scene_index]
        print(check_json_path)
        file_in_scene = os.listdir(all_objects_npy_dirpath[current_scene_index])
        npy_files = sorted([f for f in file_in_scene if f.endswith(".npy")])
        if not os.path.exists(check_json_path):
            progress = 0.0
            pass
        else:
            check_data = load_json(check_json_path)
            check_len = get_check_len(check_data)
            progress = check_len / len(npy_files)
        part_info= os.path.abspath(os.path.join(scene_mesh_path, '../../..')).split('/')[-3:]
        folder_id = '/'.join(part_info)
        is_checked = check_is_refix(folder_id, scene_name)
        return round(progress*100, 2), gr.update(value=len(npy_files)), update_submit_statue(is_checked)


    def update_scene_statue(scene_name):

        global scene_centroid_mesh, scene_scale_mesh, current_file_index, current_scene_index

        scene_mesh_path = all_scene_ply_path[all_scene_ply_names.index(scene_name)]
        # current_scene_name = all_scene_ply_names[all_scene_ply_names.index(scene_name)]
        part_info= os.path.abspath(os.path.join(scene_mesh_path, '../../..')).split('/')[-3:]
        folder_id = '/'.join(part_info)
        print(folder_id)
        update_refix_json(folder_id, scene_name)


    def load_prim_pcd(prim_name, scene_name):

        # global scene_centroid, scene_scale, current_scene_index
        global scene_centroid_mesh, scene_scale_mesh, current_scene_index
        check_status = ["UnChecked", "Checked"]
        ischecked = check_status[False]
        checked_info = ""
        if scene_centroid_mesh is None or scene_scale_mesh is None:
            return '', ''
        server2.scene.reset()
        check_json_path = all_check_json_paths[current_scene_index]
        print("Pay attention!", prim_name, scene_name)
        if not os.path.exists(check_json_path):
            pass
        else:
            check_info = load_json(all_check_json_paths[current_scene_index])
            for check_dict in check_info:
                if check_dict['name'] == prim_name.split(".")[0] and 'feedback' in check_dict:
                    ischecked = check_status[True]
                    checked_info = check_dict['feedback']
                    break

        prim_path = os.path.join(all_objects_npy_dirpath[all_scene_ply_names.index(scene_name)], f"{prim_name}")
        if not os.path.exists(prim_path) or not os.path.isfile(prim_path):
            return '', ''
        prim_ori = np.load(prim_path)
        prim, _, _ = pc_norm(prim_ori)
        prim_bbox_max = np.max(prim[:, :3], axis=0)
        prim_bbox_min = np.min(prim[:, :3], axis=0)
        prim_bbox_center = (prim_bbox_min + prim_bbox_max) / 2
        camera_position = np.array(prim_bbox_center)
        camera_position += 0.8 * (prim_bbox_max - prim_bbox_min).max()

        data_hightlight = pc_norm_use_info(prim_ori, scene_centroid_mesh, scene_scale_mesh)
        hightlight_pcd_in_scene(data_hightlight)
        server2.scene.add_point_cloud(f"{prim_name}", prim[:, :3], prim[:, 3:6], point_size=0.005)



        clients = server2.get_clients()
        for id, client in clients.items():
            client.camera.position = camera_position
            client.camera.look_at = prim_bbox_center
        
        return prim_name, update_text_with_color(ischecked, checked_info)


    def hightlight_pcd_in_scene(pc_normalized_by_scene):

        bbox_max = np.max(pc_normalized_by_scene[:, :3], axis=0)
        bbox_min = np.min(pc_normalized_by_scene[:, :3], axis=0)
        bbox = (bbox_min, bbox_max)

        bbox_center = (bbox_min + bbox_max) / 2
        
        # version 1
        # bbox_size = np.linalg.norm(bbox_max - bbox_min)
        # camera_distance = bbox_size * 1.1
        # camera_position = bbox_center + np.array([0, 0, camera_distance])
        # version 2
        camera_position = np.array(bbox_center)
        bbox_size = np.linalg.norm(bbox_max - bbox_min)
        print(f"bbox_size is {bbox_size}")
        if bbox_size < 0.01:    # small bbox
            camera_position += 20 * (bbox_max - bbox_min).max()
        else:
            camera_position += 0.55 * (bbox_max - bbox_min)

        alpha = 0.1
        ori_color = pc_normalized_by_scene[:, 3:6]
        points_num = pc_normalized_by_scene.shape[0]
        highlight_color = np.array([(1, 0.714, 0)] * points_num).astype(np.float32) * (1-alpha) + alpha * ori_color
        try:
            server1.scene.remove_by_name("bbox")
            server1.scene.remove_by_name("hightlight")
        except:
            pass
        server1.scene.add_point_cloud("hightlight", pc_normalized_by_scene[:, :3], highlight_color, point_size=0.0003)
        draw_bbox(server1, bbox, color=(1, 0, 0), thickness=0.00001)
        clients = server1.get_clients()
        for id, client in clients.items():
            client.camera.position = camera_position
            client.camera.look_at = bbox_center
            # client.camera.fov = 65
        pass


    def update_files_list(scene_name):

        # global current_file_index
        files_in_scene = sorted([f for f in os.listdir(all_objects_npy_dirpath[all_scene_ply_names.index(scene_name)]) if f.endswith(".npy")])
        current_value = files_in_scene[0]
        return gr.update(choices=files_in_scene, value=current_value)

    def update_scenes_list(folder_id_name):

        part, folder_idx = folder_id_name.split("/")
        scene_names = get_scene_info(part, folder_idx)
        return gr.update(choices=scene_names, value=scene_names[0])



    def next_scene():

        global current_scene_index
        if current_scene_index < len(all_scene_ply_names) - 1:
            current_scene_index += 1
        return all_scene_ply_names[current_scene_index]

    def prev_scene():

        global current_scene_index
        if current_scene_index > 0:
            current_scene_index -= 1
        return all_scene_ply_names[current_scene_index]
    

    def next_file(file_name):
        global current_file_index, current_scene_index

        # print(f"Current file index: {current_file_index}")
        file_in_scene = os.listdir(all_objects_npy_dirpath[current_scene_index])
        file_in_scene.remove("mesh_scene")
        file_in_scene.remove("duplicate_records")
        # print(file_in_scene)
        check_json_path = all_check_json_paths[current_scene_index]
        npy_files = sorted(file_in_scene)
        time.sleep(0.3)


        check_data = load_json(check_json_path) if os.path.exists(check_json_path) else []
        checked_names = [d["name"] for d in check_data]
        unchecked_files = sorted([f for f in npy_files if f.split(".")[0] not in checked_names])
        print(unchecked_files)
        target_files = unchecked_files if len(unchecked_files)!=0 else npy_files


        current_file_index = target_files.index(file_name) if file_name in target_files else -1
        print(f"current index is {current_file_index}")
        current_file_index = (current_file_index + 1) % len(npy_files)
        print(f"next index is {current_file_index}")

        return npy_files[current_file_index]

    def prev_file(file_name):

        global current_file_index
        file_in_scene = os.listdir(all_objects_npy_dirpath[current_scene_index])
        file_in_scene.remove("mesh_scene")
        file_in_scene.remove("duplicate_records")
        # print(file_in_scene)
        npy_files = sorted(file_in_scene)
        time.sleep(0.3)
        # current_file_index = npy_files.index(file_name)
        print(current_file_index)
        if current_file_index > 0:
            current_file_index -= 1
        elif current_file_index <= 0:
            current_file_index = len(npy_files) - 1
        print(f"current_file_index in next_file is {current_file_index}")
        return npy_files[current_file_index]


    def first_check(file_name, feedback):
        global current_file_index, current_scene_index
        check_json_path = all_check_json_paths[current_scene_index]
        file_in_scene = os.listdir(all_objects_npy_dirpath[current_scene_index])
        npy_files = sorted([f for f in file_in_scene if f.endswith(".npy")])
        if current_file_index == -100:
            current_file_index = 0
        current_file_index = npy_files.index(file_name)
        if not os.path.exists(check_json_path):
            print("Create new check_json file")
            check_data = []
            check_data.append({"name": file_name.split(".")[0], "feedback": feedback})
            with open(check_json_path, "w") as f:
                json.dump(check_data, f, indent=4)
        else:
            with open(check_json_path, "r") as f:
                print("Load existing check_json file")
                check_data = json.load(f)
                updated = False
                for index, data in enumerate(check_data):
                    if data["name"] == file_name.split(".")[0]:
                        print(f"find {file_name.split('.')[0]}")
                        check_data[index].update({"feedback": feedback})
                        updated = True
                        break
                if not updated:
                    check_data.append({"name": file_name.split(".")[0], "feedback": feedback})
        with open(check_json_path, "w") as f:
            json.dump(check_data, f, indent=4)

        check_len = get_check_len(check_data)
        with open(check_json_path, "w") as f:
            json.dump(check_data, f, indent=4)
        
        return next_file(file_name), round(check_len / len(npy_files) * 100, 2)

        

    def update_text_with_color(status, checked_info):

        common_style = '''
        <div style="
            border: 3px solid #ccc;
            padding: 3px;
            font-size: 20px;
            text-align: center; 
            font-weight: bold;
        ">{}</div>
        '''
        
        if status == "Checked":
            specific_content = '<div style="background-color: #c9f5bf;"><span style="color:green;">' + f"Checked, label is: {info_mapping(checked_info)}" + '</span>'
        elif status == "UnChecked":
            specific_content = '<div style="background-color: #ebbbb4;"><span style="color:red;">' + "Unchecked" + '</span>'

        return common_style.format(specific_content)
    
    def update_submit_statue(is_checked):

        common_style = '''
        <div style="
            border: 3px solid #ccc;
            padding: 3px;
            font-size: 20px;
            text-align: center; 
            font-weight: bold;
        ">{}</div>
        '''
        
        if is_checked:
            specific_content = '<div style="background-color: #c9f5bf;"><span style="color:green;">' + "The scene is checked" + '</span>'
        else:
            specific_content = '<div style="background-color: #ebbbb4;"><span style="color:red;">' + "The scene is not checked" + '</span>'

        return common_style.format(specific_content)




    with gr.Blocks(css=css) as demo:

        
        with gr.Row(equal_height=True, elem_classes=["row"]):
            with gr.Column(scale=3):
                with gr.Row(equal_height=True):
                    folder_id_name = gr.Dropdown(all_folder_id_names, label="Select a folder index")
                    scene_name = gr.Dropdown(all_scene_ply_names, label="Select a scene")
                gr.HTML(html_str,elem_classes=["left"])
                show_unfixed_box = gr.HTML()
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        prev_scene_buttom = gr.Button("< Previous Scene", size="sm")
                    with gr.Column(scale=1):
                        next_scene_buttom = gr.Button("Next Scene >", size="sm")


            with gr.Column(scale=3):
                
                file_name = gr.Dropdown([], label="Select an object in the scene")
                gr.HTML(html_str2, elem_classes=["right"])
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        prev_buttom = gr.Button("< Previous", size="sm")
                    with gr.Column(scale=1):
                        next_buttom = gr.Button("Next >", size="sm")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        current_files_number = gr.Textbox(value="", label="The number of objects in the scene", interactive=False)
                    with gr.Column(scale=8):
                        progress_bar = gr.Slider(minimum=0, maximum=100, label="Checked (%)", interactive=False)
                
                # show_unchecked_box = gr.Textbox(value="", label="Checked/Unckecked")
                show_unchecked_box = gr.HTML()
                
                # gr.Markdown(category_info_box)
                # info_category = gr.Textbox(value="", label="Category(Instances/Structures)", interactive=False, elem_classes=["custom-textbox"])
                
                with gr.Row():
                    gr.Markdown(recheck_info_box)
                    # reset_buttom = gr.Button("Reset First Choice", visible=True)
                    # reset_first_buttom = gr.Button("Reset First", visible=False)
                # with gr.Row():
                #     instance_buttom = gr.Button("物体")
                #     structure_buttom = gr.Button("房间结构")
                #     noise_buttom = gr.Button("噪声")
                # gr.HTML(custom_css)
                # gr.HTML("<hr style='border: 2px solid #00b4fa;'>")
                with gr.Row():
                    # gr.Markdown("**物体**")
                    instance_part_buttom = gr.Button("Part", visible=True, size="sm")
                    instance_complete_buttom = gr.Button("Complete", visible=True, size="sm")
                    instance_combined_buttom = gr.Button("Grouped", visible=True, size="sm")
                # 
                gr.HTML("<hr style='border: 2px solid #00b4fa;'>")
                with gr.Row():
                    # gr.Markdown("**结构**")
                    structure_ceiling_buttom = gr.Button("Ceiling", visible=True, size="sm")
                    structure_floor_buttom = gr.Button("Floor", visible=True, size="sm")
                    structure_wall_buttom = gr.Button("Wall", visible=True, size="sm")
                    structure_bg_wall_buttom = gr.Button("Background", visible=True, size="sm")
                gr.HTML("<hr style='border: 2px solid #00b4fa;'>")
                with gr.Row():
                    # gr.Markdown("**噪声**")
                    noise_confirm_buttom = gr.Button("noise", visible=True, size="sm")

                gr.HTML("<hr style='border: 2px solid #b409f2;'>")
                with gr.Row():
                    submit_buttom = gr.Button("Submit", visible=True, size="sm")
                # with gr.Row():
                #     back_buttom = gr.Button("重新选择一级选项", visible=False)
        folder_id_name.change(update_scenes_list, inputs=[folder_id_name], outputs=[scene_name])
        scene_name.change(update_files_list, inputs=[scene_name], outputs=[file_name])
        scene_name.change(load_scene_pcd, inputs=[scene_name], outputs=[progress_bar, current_files_number, show_unfixed_box])
        file_name.change(load_prim_pcd, inputs=[file_name, scene_name], outputs=[file_name, show_unchecked_box])
        
        next_scene_buttom.click(
            next_scene, 
            inputs=[], 
            outputs=[scene_name]
        )

        prev_scene_buttom.click(
            prev_scene, 
            inputs=[], 
            outputs=[scene_name]
        )
        next_buttom.click(
            fn=next_file, 
            inputs=[file_name], 
            outputs=[file_name]
        )
        prev_buttom.click(
            fn=prev_file, 
            inputs=[file_name], 
            outputs=[file_name]
        )
        
        
        def setup_mark_as_checked_buttom(buttom, feedback):
            buttom.click(
                first_check,
                inputs=[file_name, gr.State(feedback)],
                outputs=[file_name, progress_bar]
            )


        submit_buttom.click(update_scene_statue, inputs=[scene_name], outputs=[])
        setup_mark_as_checked_buttom(instance_part_buttom, "part")
        setup_mark_as_checked_buttom(instance_complete_buttom, "complete")
        setup_mark_as_checked_buttom(instance_combined_buttom, "combined")
        setup_mark_as_checked_buttom(structure_ceiling_buttom, "ceiling")
        setup_mark_as_checked_buttom(structure_floor_buttom, "floor")
        setup_mark_as_checked_buttom(structure_wall_buttom, "wall")
        setup_mark_as_checked_buttom(structure_bg_wall_buttom, "bg_wall")
        setup_mark_as_checked_buttom(noise_confirm_buttom, "confirm")
        


    demo.queue()
    demo.launch(server_name='0.0.0.0',server_port=gradio_port)
    pass


if __name__ == "__main__":

    start_annotation(args.user_id)
