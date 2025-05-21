import os
import pickle
import cv2
import gradio as gr  # Now: gradio==3.50.2 Deprecated: gradio==3.44.0
import numpy as np
from math import ceil
import json
import open3d as o3d

# from render_bev_for_all import load_mesh, _render_2d_bev, take_bev_screenshot, process_mesh
from utils.region_utils import get_data, get_position_in_mesh_render_image, is_in_poly
import matplotlib
import copy
import os
import argparse

sourcedir = "/root/projects/GRRegion_annotation/data"
output_dir = '/root/projects/GRRegion_annotation/output'

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=7921)
parser.add_argument("--part", type=int, required=True)
args = parser.parse_known_args()[0]

global init_item_dict
init_item_dict = {}

init_item_dict["out_image"] = None
init_item_dict["poly_done"] = False

init_item_dict["vertex_list"] = []
init_item_dict["annotation_list"] = []



global SCENE_LIST
SCENE_LIST = []
part_id = f"part{args.part}"
# for part_id in os.listdir(sourcedir):
part_path = os.path.join(sourcedir, part_id)
for usd_folder in sorted(os.listdir(part_path)):
    usd_path = os.path.join(part_path, usd_folder)
    for scene_id in sorted(os.listdir(usd_path)):
        scan_id = f"{part_id}/{usd_folder}/{scene_id}"
        SCENE_LIST.append(scan_id)


# SCENE_LIST = [scan_id for scan_id in os.listdir(sourcedir)]
# SCENE_LIST.sort()
def get_prev_scene_id(current_scene_id):
    if current_scene_id == None or current_scene_id == 'None':
        return None
    else:
        index = SCENE_LIST.index(current_scene_id)
        if index == 0:
            return None
        else:
            return SCENE_LIST[index - 1]

def get_next_scene_id(current_scene_id):
    if current_scene_id == None or current_scene_id == 'None':
        return SCENE_LIST[0]
    else:
        index = SCENE_LIST.index(current_scene_id)
        if index == len(SCENE_LIST) - 1:
            return None
        else:
            return SCENE_LIST[index + 1]
            
REGIONS = {
    "living region": "living region",    
    "study region": "study region",          
    "resting region": "resting region",        
    "dining region": "dining region",         
    "cooking region": "cooking region",        
    "storage region": "storage region",       
    "toilet region": "toilet region",        
    "balcony region": "balcony region",        
    "retail region": "retail region",         
    "entertainment region": "entertainment region",  
    "education region": "education region",      
    "corridor region": "corridor region",       
    "elevator stairsregion": "elevator_stairs_region", 
    "public service region": "public_service region",  

    "other region": "other region"                                   
}
    
REGIONS_COLOR = {
    "living region": (255, 0, 255),         
    "study region": (0, 255, 0),           
    "resting region": (0, 0, 255),         
    "dining region": (0, 255, 255),        
    "cooking region": (255, 255, 0),       
    "storage region": (122, 255, 122),     
    "toilet region": (69, 139, 0),        
    "balcony region": (255, 182, 193),     
    "retail region": (255, 0, 0),          
    "entertainment region": (139, 69, 19), 
    "education region": (135, 206, 235),   
    "corridor region": (139, 105, 20),     
    "elevator_stairs_region": (211, 211, 211), 
    "public_service region": (176, 196, 222),  
    "other region": (122, 122, 0)        
}
    
REGIONS_COLOR = {k: np.array(v) for k, v in REGIONS_COLOR.items()}


def get_camera_info(coord_info, bev_image_shape):
    camera_info_dict = {}
    img_coord = np.array(coord_info['sample_points'])

    origin_coord = np.array(coord_info['bev_camera_translation'])
    offset_coord = img_coord-origin_coord

    gt_coord = offset_coord*100
    for test_id in range(len(gt_coord)):
        ex,ey = int(gt_coord[test_id][1]),int(gt_coord[test_id][0])

        img_ex = bev_image_shape[0]//2-ex
        img_ey = bev_image_shape[1]//2+ey
        camera_info_dict[test_id] = (img_ex, img_ey)
    return camera_info_dict

def translate_region_name_cn_en(region_name):
    return REGIONS.get(region_name, None)

def translate_region_name_en_cn(region_name):
    for k, v in REGIONS.items():
        if v == region_name:
            return k
    return None

with gr.Blocks() as demo:
    detail_img_list = gr.State([])
    detail_img_index = gr.State(0)
    click_evt_list = gr.State([])
    vertex_list = gr.State([])
    annotation_list = gr.State([])
    poly_done = gr.State(False)
    poly_image = gr.State()
    item_dict_list = gr.State([])
    enable_undo = gr.State(False)
    anno_result = gr.State([])
    store_vertex_list = gr.State([])
    to_rotate_clockwise_90 = gr.State(False)
    to_show_areas = gr.State(False)
    user_name_locked = gr.State(False)

    with gr.Row():
        user_name = gr.Textbox(label="user name", value="", placeholder="Please enter your name, the first character must be an alphabet, and do not include spaces.")
        view_only = gr.Checkbox(label="view only mode", value=False)
        confirm_user_name_btn = gr.Button(value="confirm and lock user name (refresh the page to reset the name)", label="confirm user name")
        check_annotated_btn = gr.Button(value="check the number of unannotated scenes", label="check annotated")
    with gr.Row():
        scene_id = gr.Dropdown(SCENE_LIST, label="select the scene to be annotated")
        scene_anno_info = gr.Textbox(label="annotation information", value="", visible=True, interactive=False)
    anno_result_img = gr.Image(label="annotation result", interactive=True, tool=[], height=400)
    show_label_box = gr.Textbox(label="click the region type")

    total_vertex_num = gr.Slider(
        visible=False,
        label="Vertex Number",
        info="",
        minimum=3,
        maximum=20,
        step=1,
        value=4,
    )


    def check_user_name_validity(user_name):
        if len(user_name) == 0 or ' ' in user_name or not user_name[0].isalpha():
            gr.Warning("User name is invalid, please check and try again.")
            return False
        return True


    def lock_user_name(user_name, user_name_locked):
        if check_user_name_validity(user_name):
            user_name = user_name.strip()
            user_name = gr.Textbox(label="user name", value=user_name, interactive=False)
            user_name_locked = True
        return user_name, user_name_locked
    confirm_user_name_btn.click(lock_user_name, inputs=[user_name, user_name_locked],
                                outputs=[user_name, user_name_locked])


    def check_annotated_scenes(user_name):
        missing_scenes = []
        global output_dir
        for scene_id in SCENE_LIST:
            file_path_to_save = os.path.join(f"{output_dir}/{scene_id}", f'region_segmentation_{user_name}.txt')
            if not os.path.exists(file_path_to_save):
                missing_scenes.append(scene_id)
        num_in_total = len(SCENE_LIST)
        num_missing = len(missing_scenes)
        missing_scenes.sort()
        gr.Info(f"unannotated scenes: {num_missing}/{num_in_total}")
        return missing_scenes
    check_annotated_btn.click(check_annotated_scenes, inputs=[user_name], outputs=[scene_anno_info])

    def get_file(scene_id, user_name, user_name_locked):
        annotation_list = []
        click_evt_list = []
        vertex_list = []
        poly_done = False
        item_dict_list = [init_item_dict]
        anno_result = []
        to_show_areas = False

        if not user_name_locked:
            gr.Warning("Please confirm your user name first.")
            return None, None, None, None, None, '', None, annotation_list, click_evt_list, vertex_list, poly_done, item_dict_list, anno_result, to_show_areas,0,[]

        if scene_id == None or scene_id == 'None':
            return None, None, None, None, None, '', None, annotation_list, click_evt_list, vertex_list, poly_done, item_dict_list, anno_result, to_show_areas,0,[]

        if "floor" in scene_id:
            floor_id = int((scene_id.split("/")[-1]).split("_")[-1]) - 1
            bev_path = f"bevmap_{floor_id}"
            # print(scene_id)
        else:
            bev_path = "bevmap_0"

        render_image_path = os.path.join(sourcedir, scene_id, bev_path, "bev_map.jpg")
        file_path_to_save = os.path.join(output_dir, scene_id, f'region_segmentation_{user_name}.txt')
        anno_result_img_path = gr.update(value=render_image_path)
        if os.path.exists(file_path_to_save):
            scene_anno_state = scene_id + ' has been annotated ! ! ! ! !'
            gr.Info("This scene has been annotated.")
            anno_result = get_data(file_path_to_save)
            to_show_areas = True
        else:
            scene_anno_state = scene_id + ' need to be annotated'
            anno_result = []

        return render_image_path, render_image_path.replace('bev_map.jpg', 'pos.jpg'), \
               None, None, None, scene_anno_state, anno_result_img_path, \
               annotation_list, click_evt_list, vertex_list, poly_done, item_dict_list, anno_result, to_show_areas, 0,[]


    with gr.Row():
        input_img = gr.Image(label="annotation image", tool=[], height=600, width=600)
        output_img = gr.Image(label="", tool=[], height=600, width=600)

    btn_finish = gr.Button("finish drawing")
    vertex_list = gr.State([])
    poly_done = gr.State(False)
    store_vertex_list = gr.State([])
    item_dict_list = gr.State([{}])

    label = gr.Radio(REGIONS.keys(),
                     label="label",
                     info="Select the region type here",
                     )

    custom_label_input = gr.Textbox(
        label="custom region type",
        visible=False,
        placeholder="Please enter the custom region type here",
        interactive=True
    )

    def toggle_custom_input(selected_label):
        return gr.Textbox(visible=(selected_label == "other region"))

    label.change(
        fn=toggle_custom_input,
        inputs=label,
        outputs=custom_label_input
    )

    with gr.Row():
        annotate_btn = gr.Button("Label a single area")
        undo_btn = gr.Button("withdraw")
        save_btn = gr.Button("All areas have been annotated. Save it.")
        prev_scene_btn = gr.Button("The previous scene")
        next_scene_btn = gr.Button("The next scene")
        clear_btn = gr.Button("Clear all annotations in the current scene (with caution).")

    with gr.Row():
        object_postion_img = gr.Image(label="Assist in viewing the selection area.", interactive=True, tool=[], show_label=False, height=600, width=600)
        detail_show_img = gr.Image(label="View the picture for assistance. ", tool=[], show_label=False, height=600, width=800)
    with gr.Row():
        rotate_button_increase = gr.Button("Switch among the four perspectives.")
        # rotate_button_decrease = gr.Button("视角顺时针旋转约45度 (初始相机视角朝向正北方向 ⬆️)")

    rotate_times = gr.State(0)
    show_json = gr.JSON(label="Annotate History")

    def view_only_mode(view_only):
        input_img = gr.Image(label="Image", visible=not view_only)
        output_img = gr.Image(label="Selected Polygon", visible=not view_only)
        label = gr.Radio(REGIONS.keys(), label="label", visible=not view_only)
        total_vertex_num = gr.Slider(label="Vertex Number", visible=not view_only,
                                     minimum=3, maximum=20, step=1, value=4)
        annotate_btn = gr.Button("annotate a single region", visible=not view_only)
        undo_btn = gr.Button("undo the last annotation", visible=not view_only)
        save_btn = gr.Button("all regions have been annotated. save it.", visible=not view_only)
        clear_btn = gr.Button("clear all annotations in the current scene (with caution).", visible=not view_only)
        return input_img, output_img, label, total_vertex_num, annotate_btn, undo_btn, save_btn, clear_btn
    view_only.change(fn=view_only_mode, inputs=[view_only], outputs=[input_img, output_img, label, total_vertex_num, annotate_btn, undo_btn, save_btn, clear_btn])


    def get_coverage_mask(h, w, poly):
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x, y = x.flatten(), y.flatten()
        pixel_points = np.vstack((y, x)).T
        p = matplotlib.path.Path(poly)
        mask = p.contains_points(pixel_points).reshape((h, w))
        return mask


    def draw_polygon(img, vertex_list, alpha=0.4):
        img = img.copy()
        coverage_mask = get_coverage_mask(img.shape[0], img.shape[1], vertex_list)
        img[coverage_mask] = img[coverage_mask] * (1 - alpha) + np.array([255, 0, 0]) * alpha
        return img
    
    def draw_dot(scene_id, img, out_img, 
                click_evt_list, poly_done, poly_image, vertex_list, annotation_list, 
                enable_undo, item_dict_list, store_vertex_list,
                evt: gr.SelectData):
        w, h, c = img.shape
        size = ceil(max([w, h]) * 0.01)
        out = img.copy()
        
        # 记录历史状态（支持撤销）
        new_item_dict = copy.deepcopy(item_dict_list[-1])
        new_item_dict.update({
            "poly_done": poly_done,
            "out_image": out_img,
            "annotation_list": copy.deepcopy(annotation_list),
            "vertex_list": copy.deepcopy(vertex_list)
        })
        item_dict_list.append(new_item_dict)
        
        if not poly_done:
            vertex_list.append([evt.index[1], evt.index[0]]) 
            
            for vertex in vertex_list:
                x, y = vertex
                sx = max(x - size, 0)
                ex = min(x + size, out.shape[0] - 1)
                sy = max(y - size, 0)
                ey = min(y + size, out.shape[1] - 1)
                out[sx:ex, sy:ey] = np.array([255, 0, 0]).astype(np.uint8)
            
            for i in range(len(vertex_list)):
                x1, y1 = vertex_list[i]
                x2, y2 = vertex_list[(i + 1) % len(vertex_list)]
                cv2.line(out, (y1, x1), (y2, x2), (255, 0, 0), 2)
        
        return out, click_evt_list, poly_done, poly_image, vertex_list, annotation_list, enable_undo, item_dict_list, store_vertex_list

    def finish_polygon(poly_done, vertex_list, store_vertex_list, img):
        if not poly_done and len(vertex_list) >= 3:  
            poly_done = True
            store_vertex_list = copy.deepcopy(vertex_list)
            filled = draw_polygon(img.copy(), store_vertex_list)
            return filled, poly_done, store_vertex_list, []
        return img, poly_done, store_vertex_list, vertex_list

    input_img.select(draw_dot, [scene_id, input_img, output_img,
                                click_evt_list, poly_done, poly_image, vertex_list, annotation_list, enable_undo,
                                item_dict_list, store_vertex_list,
                                ],
                     [output_img, click_evt_list, poly_done, poly_image, vertex_list, annotation_list, enable_undo,
                      item_dict_list, store_vertex_list])
    btn_finish.click(
        finish_polygon,
        inputs=[poly_done, vertex_list, store_vertex_list, input_img],
        outputs=[output_img, poly_done, store_vertex_list, vertex_list]
    )

    def visualize_object_pos(scene_id, img, to_rotate_clockwise_90, evt: gr.SelectData):
        # import pdb; pdb.set_trace()
        out = img.copy()
        min_distance = np.inf
        min_object_id = 0
        m_x = evt.index[1]
        m_y = evt.index[0]
        if m_x is None or m_y is None:
            return out, None
        w, h, c = img.shape
        size = ceil(max([w, h]) * 0.01)
        
        if "floor" in scene_id:
            floor_id = int((scene_id.split("/")[-1]).split("_")[-1]) - 1
        else:
            floor_id = 0
        info_file_path = os.path.join(sourcedir, scene_id, f"camera_info_{floor_id}.json")
        
        bev_img_size = img.shape
        camera_info_dict = get_camera_info(json.load(open(info_file_path)),bev_img_size)
        
        min_distance = np.inf
        min_camera_id = -1
        for camera_id in camera_info_dict:
            img_x, img_y = camera_info_dict[camera_id]
            distance = (img_x - m_x) ** 2 + (img_y - m_y) ** 2
            if distance < min_distance:
                min_distance = distance
                min_camera_id = camera_id
        if min_camera_id == -1:
            return out, None
        for camera_id in camera_info_dict:
            if camera_id == min_camera_id:
                color_ = [255, 0, 0]
            else:
                color_ = [0, 255,0]
            p = camera_info_dict[camera_id]
            out[
            max(p[0] - size, 0): min(p[0] + size, out.shape[0] - 1),
            max(p[1] - size, 0): min(p[1] + size, out.shape[1] - 1),
            ] = np.array(color_).astype(np.uint8)
        
        detail_img_index = 0
        detail_img_list = []
        # for i in range(8):
        #     img_path = os.path.join(sourcedir, scene_id, f"regions/sample_{min_camera_id}/render_{i}.jpg")
        #     detail_img_list.append(img_path)
        if "floor" in scene_id:
            floor_id = int((scene_id.split("/")[-1]).split("_")[-1]) - 1
            region_path = f"regions_{floor_id}"
            # print(scene_id)
        else:
            region_path = "regions_0"
        for i in range(2):
            img_path = os.path.join(sourcedir, scene_id, region_path, f"sample_{min_camera_id}/combine_{i+1}.jpg")
            detail_img_list.append(img_path)
        detail_img = gr.update(value = detail_img_list[detail_img_index])

        to_rotate_clockwise_90 = True

        return out, detail_img, to_rotate_clockwise_90,detail_img_list,detail_img_index


    def rotate_clockwise_90(detail_img, to_rotate_clockwise_90,rotate_times):

        if to_rotate_clockwise_90:
            to_rotate_clockwise_90 = False
            for time_ in range(rotate_times):
              
                detail_img = cv2.rotate(detail_img, cv2.ROTATE_90_CLOCKWISE)
        return detail_img, to_rotate_clockwise_90
    
 


    def visualize_annotation_result(input_img, to_show_areas, anno_result):

        if to_show_areas:
            anno_result_img = 0.6 * input_img.copy()
            for anno_ in anno_result:
                label_ = anno_["label"]
                poly_ = anno_["vertex"]
                if label_ not in REGIONS:
                    label_ = "other region"
                color_ = REGIONS_COLOR[label_]
                coverage_mask = get_coverage_mask(anno_result_img.shape[0], anno_result_img.shape[1], poly_)
                anno_result_img[coverage_mask] = anno_result_img[coverage_mask] + 0.4 * color_
            return anno_result_img.astype(np.uint8), to_show_areas, anno_result
        return input_img, to_show_areas


    def show_labels(anno_result_img, anno_result, evt: gr.SelectData):

        m_x = evt.index[1]
        m_y = evt.index[0]
        for anno_ in anno_result:
            label_ = anno_["label"]
            chinese_label_ = translate_region_name_en_cn(label_)
            poly_ = anno_["vertex"]
            if is_in_poly((m_x, m_y), poly_):
                return f"{label_}:{chinese_label_}", anno_result
        return "no annotation", anno_result


    def annotate(scene_id, label, custom_label, output_img, poly_done, vertex_list, click_evt_list,
                annotation_list, store_vertex_list, to_show_areas, anno_result, item_dict_list):

        if scene_id is None or scene_id == 'None':
            return None, None, None

        if poly_done and label is not None:
            if label == "other region":
                if not custom_label.strip():
                    gr.Warning("Please enter the custom region type.")
                    return label, output_img, annotation_list, poly_done, vertex_list, click_evt_list, \
                        annotation_list, store_vertex_list, to_show_areas, anno_result, item_dict_list
                final_label = custom_label.strip()
            else:
                final_label = translate_region_name_cn_en(label)

            new_item_dict = copy.deepcopy(item_dict_list[-1])
            new_item_dict.update({
                "poly_done": poly_done,
                "out_image": output_img,
                "annotation_list": copy.deepcopy(annotation_list),
                "vertex_list": copy.deepcopy(vertex_list)
            })
            item_dict_list.append(new_item_dict)

            annotation = {
                "id": len(annotation_list),
                "label": final_label,
                "vertex": copy.deepcopy(store_vertex_list)
            }
            annotation_list.append(annotation)
            
            vertex_list = []
            poly_done = False
            to_show_areas = True
            anno_result = copy.deepcopy(annotation_list)

            return None, None, annotation_list, poly_done, vertex_list, click_evt_list, \
                annotation_list, store_vertex_list, to_show_areas, anno_result, item_dict_list
        else:
            gr.Info('Polygen has not been finished yet. Please finish the polygon first.')
            return label, output_img, annotation_list, poly_done, vertex_list, click_evt_list, \
                annotation_list, store_vertex_list, to_show_areas, anno_result, item_dict_list


    def clear_all(scene_id, output_img, annotation_list, poly_done, click_evt_list, vertex_list, item_dict_list):

        if scene_id == None or scene_id == 'None':
            return [None] * 6

        new_item_dict = copy.deepcopy(item_dict_list[-1])
        new_item_dict["poly_done"] = poly_done
        new_item_dict["out_image"] = output_img
        new_item_dict["annotation_list"] = copy.deepcopy(annotation_list)
        new_item_dict["vertex_list"] = copy.deepcopy(vertex_list)
        item_dict_list.append(new_item_dict)
        annotation_list = []
        vertex_list = []

        poly_done = False
        return None, annotation_list, annotation_list, poly_done, click_evt_list, vertex_list


    def undo(output_img, annotation_list, poly_done, vertex_list, click_evt_list, item_dict_list):
        # print('undo!!!!')

        # print(len(item_dict_list), item_dict_list[-1]["annotation_list"], item_dict_list[-1]['show_json'])
        if len(item_dict_list) == 1:
            return output_img, annotation_list, annotation_list, poly_done, vertex_list, click_evt_list
        else:
            output_img = item_dict_list[-1]["out_image"]
            annotation_list = copy.deepcopy(item_dict_list[-1]["annotation_list"])
            poly_done = item_dict_list[-1]["poly_done"]
            vertex_list = item_dict_list[-1]["vertex_list"]
            del item_dict_list[-1]

            return output_img, annotation_list, annotation_list, poly_done, vertex_list, click_evt_list


    undo_btn.click(fn=undo,
                   inputs=[output_img, annotation_list, poly_done, vertex_list, click_evt_list, item_dict_list],
                   outputs=[output_img, show_json, annotation_list, poly_done, vertex_list, click_evt_list])


    def save_to_file(scene_id, annotation_list, click_evt_list, vertex_list, poly_done, item_dict_list, user_name):
        if scene_id == None or scene_id == 'None' or annotation_list is None or len(annotation_list) == 0:
            return None, None, None, None, None, None, None, None, annotation_list, click_evt_list, vertex_list, poly_done, item_dict_list

        os.makedirs(os.path.join(output_dir,scene_id), exist_ok=True)
        print(os.path.join(output_dir,scene_id))
        with open(os.path.join(output_dir,scene_id,f'region_segmentation_{user_name}.txt'), 'w') as file:
            file.write(str(annotation_list))
        annotation_list = []
        click_evt_list = []
        vertex_list = []
        poly_done = False
        item_dict_list = [init_item_dict]

        return scene_id, None, None, None, None, None, None, None, annotation_list, click_evt_list, vertex_list, poly_done, item_dict_list
    
    
    def increase_rotate_times(rotate_times,detail_img):
        if detail_img is None:
            return None,0
        rotate_times = rotate_times +1 
        detail_img = cv2.rotate(detail_img, cv2.ROTATE_90_CLOCKWISE)
        return detail_img, rotate_times
    
    def increase_rotate_angle(detail_img_index,detail_img_list,detail_show_img):
        if detail_img_list is None or len(detail_img_list) == 0:
            return None,0
        detail_img_index = (detail_img_index + 1) % len(detail_img_list)
        detail_show_img = gr.update(value = detail_img_list[detail_img_index])
        return detail_img_index,detail_show_img
    
    def decrease_rotate_angle(detail_img_index,detail_img_list,detail_show_img):
        if detail_img_list is None or len(detail_img_list) == 0:
            return None,0
        detail_img_index = (detail_img_index - 1) % len(detail_img_list)
        detail_show_img = gr.update(value = detail_img_list[detail_img_index])
        return detail_img_index,detail_show_img
    
    scene_id.change(
        get_file, inputs=[scene_id, user_name, user_name_locked],
        outputs=[input_img, object_postion_img, output_img, show_json, detail_show_img, scene_anno_info,
                 anno_result_img, annotation_list, click_evt_list, vertex_list, poly_done, item_dict_list,
                 anno_result, to_show_areas,rotate_times,detail_img_list]
    )

    object_postion_img.select(
        visualize_object_pos, [scene_id, input_img, to_rotate_clockwise_90],
        [object_postion_img, detail_show_img, to_rotate_clockwise_90,detail_img_list,detail_img_index]
    )
    anno_result_img.select(
        show_labels, [anno_result_img, anno_result], [show_label_box, anno_result]
    )
    detail_show_img.change(
        rotate_clockwise_90, [detail_show_img, to_rotate_clockwise_90,rotate_times], [detail_show_img, to_rotate_clockwise_90]
    )
    clear_btn.click(fn=clear_all, inputs=[scene_id, output_img, annotation_list, poly_done, click_evt_list, vertex_list,
                                      item_dict_list],
                    outputs=[output_img, show_json, annotation_list, poly_done, click_evt_list, vertex_list])
    annotate_btn.click(
        fn=annotate, inputs=[scene_id, label, custom_label_input, output_img, poly_done, vertex_list, click_evt_list,
                             annotation_list, store_vertex_list, to_show_areas, anno_result, item_dict_list],
        outputs=[label, output_img, show_json, poly_done, vertex_list, click_evt_list,
                 annotation_list, store_vertex_list, to_show_areas, anno_result, item_dict_list]
    )
    annotate_btn.click(
        visualize_annotation_result, [input_img, to_show_areas, anno_result], [anno_result_img, to_show_areas]
    )
    input_img.change(
        visualize_annotation_result, [input_img, to_show_areas, anno_result], [anno_result_img, to_show_areas]
    )

    save_btn.click(
        fn=save_to_file,
        inputs=[scene_id, annotation_list, click_evt_list, vertex_list, poly_done, item_dict_list, user_name],
        outputs=[
            scene_id,
            output_img,
            show_json,
            object_postion_img,
            detail_show_img,
            scene_anno_info,
            anno_result_img,
            show_label_box,
            annotation_list, click_evt_list, vertex_list, poly_done, item_dict_list

        ],
    )
    prev_scene_btn.click(
        fn=get_prev_scene_id,
        inputs=[scene_id],
        outputs=[scene_id]
    )
    next_scene_btn.click(
        fn=get_next_scene_id,
        inputs=[scene_id],
        outputs=[scene_id]
    )
    rotate_button_increase.click(increase_rotate_angle, [detail_img_index,detail_img_list,detail_show_img], [detail_img_index,detail_show_img])
    # rotate_button_decrease.click(decrease_rotate_angle, [detail_img_index,detail_img_list,detail_show_img], [detail_img_index,detail_show_img])
demo.queue(concurrency_count=20)

    
def shrink_scene_info(keep_num=20):
    # avoid too much memory usage, delete "old" scenes
    global scene_info
    num_scene = len(scene_info)
    if num_scene > keep_num:
        scene_info = {k: v for k, v in sorted(scene_info.items(), key=lambda item: item[1]["timestamp"])[-keep_num:]}
    import gc
    gc.collect()


if __name__ == "__main__":



    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    scene_info = {}
    
    demo.launch(server_port=args.port, server_name='0.0.0.0')