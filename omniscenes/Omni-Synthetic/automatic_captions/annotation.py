import math
import argparse
import json
import requests
import os
import time

import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import cv2

# 
from petrel_client.client import Client
client = Client(enable_multi_cluster=True)

parser = argparse.ArgumentParser()
# parser.add_argument("--scene_start_index", type=int, default=1)
# parser.add_argument("--scene_end_index", type=int, default=2)
# parser.add_argument("--scene_id", type=int, default=1)
parser.add_argument("--part", type=int, default=1)
parser.add_argument("--usd", type=int, default=1)
parser.add_argument("--process_mode", type=str, default='single', required=True)
parser.add_argument("--start_usd", type=int, default=1)
parser.add_argument("--end_usd", type=int, default=1)
args = parser.parse_known_args()[0]




def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def get_image_from_url(url):
    presigned_url = client.generate_presigned_url(url, client_method ='get_object', expires_in=3600)
    image = Image.open(requests.get(presigned_url, stream=True).raw)
    return image

def load_image(image_url, input_size=448, max_num=12, load_method="local"):

    if load_method == "local":
        image = Image.open(image_url).convert('RGB')
    elif load_method == "ceph":
        image = get_image_from_url(image_url).convert('RGB')

    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values



path = "/path/to/the/InternVL2_5-78B-MPO"
device_map = split_model('InternVL2_5-78B')
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map=device_map).eval()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

generation_config = dict(max_new_tokens=1024, do_sample=True)


query = "This image is made up of 6 renderings of an indoor scene object. Please identify the object in this image using only one word or phrase."
embodiedscan_classes = ['adhesive tape', 'air conditioner', 'alarm', 'album', 'arch', 'backpack', 'bag', 'balcony', 'ball', 'banister', 'bar', 'barricade', 'baseboard', 'basin', 'basket', 'bathtub', 'beam', 'beanbag', 'bed', 'bench', 'bicycle', 'bidet', 'bin', 'blackboard', 'blanket', 'blinds', 'board', 'body loofah', 'book', 'boots', 'bottle', 'bowl', 'box', 'bread', 'broom', 'brush', 'bucket', 'cabinet', 'calendar', 'camera', 'can', 'candle', 'candlestick', 'cap', 'car', 'carpet', 'cart', 'case', 'ceiling', 'chair', 'chandelier', 'cleanser', 'clock', 'clothes', 'clothes dryer', 'coat hanger', 'coffee maker', 'coil', 'column', 'commode', 'computer', 'conducting wire', 'container', 'control', 'copier', 'cosmetics', 'couch', 'counter', 'countertop', 'crate', 'crib', 'cube', 'cup', 'curtain', 'cushion', 'decoration', 'desk', 'detergent', 'device', 'dish rack', 'dishwasher', 'dispenser', 'divider', 'door', 'door knob', 'doorframe', 'doorway', 'drawer', 'dress', 'dresser', 'drum', 'duct', 'dumbbell', 'dustpan', 'dvd', 'eraser', 'excercise equipment', 'fan', 'faucet', 'fence', 'file', 'fire extinguisher', 'fireplace', 'floor', 'flowerpot', 'flush', 'folder', 'food', 'footstool', 'frame', 'fruit', 'furniture', 'garage door', 'garbage', 'glass', 'globe', 'glove', 'grab bar', 'grass', 'guitar', 'hair dryer', 'hamper', 'handle', 'hanger', 'hat', 'headboard', 'headphones', 'heater', 'helmets', 'holder', 'hook', 'humidifier', 'ironware', 'jacket', 'jalousie', 'jar', 'kettle', 'keyboard', 'kitchen island', 'kitchenware', 'knife', 'label', 'ladder', 'lamp', 'laptop', 'ledge', 'letter', 'light', 'luggage', 'machine', 'magazine', 'mailbox', 'map', 'mask', 'mat', 'mattress', 'menu', 'microwave', 'mirror', 'molding', 'monitor', 'mop', 'mouse', 'napkins', 'notebook', 'object', 'ottoman', 'oven', 'pack', 'package', 'pad', 'pan', 'panel', 'paper', 'paper cutter', 'partition', 'pedestal', 'pen', 'person', 'piano', 'picture', 'pillar', 'pillow', 'pipe', 'pitcher', 'plant', 'plate', 'player', 'plug', 'plunger', 'pool', 'pool table', 'poster', 'pot', 'price tag', 'printer', 'projector', 'purse', 'rack', 'radiator', 'radio', 'rail', 'range hood', 'refrigerator', 'remote control', 'ridge', 'rod', 'roll', 'roof', 'rope', 'sack', 'salt', 'scale', 'scissors', 'screen', 'seasoning', 'shampoo', 'sheet', 'shelf', 'shirt', 'shoe', 'shovel', 'shower', 'sign', 'sink', 'soap', 'soap dish', 'soap dispenser', 'socket', 'speaker', 'sponge', 'spoon', 'stairs', 'stall', 'stand', 'stapler', 'statue', 'step', 'stick', 'stool', 'stopcock', 'stove', 'structure', 'sunglasses', 'support', 'switch', 'table', 'tablet', 'teapot', 'telephone', 'thermostat', 'tissue', 'tissue box', 'toaster', 'toilet', 'toilet paper', 'toiletry', 'tool', 'toothbrush', 'toothpaste', 'towel', 'toy', 'tray', 'treadmill', 'trophy', 'tube', 'tv', 'umbrella', 'urn', 'utensil', 'vacuum cleaner', 'vanity', 'vase', 'vent', 'ventilation', 'wall', 'wardrobe', 'washbasin', 'washing machine', 'water cooler', 'water heater', 'window', 'window frame', 'windowsill', 'wine', 'wire', 'wood', 'wrap']
multi_image_query = "These 6 images contain 3 top-down views and 3 bottom-up views of the same indoor object. Please identify the object in this image using only one word or phrase."
close_prompt = f'''
    {multi_image_query} Follow these Rules to answer:
    **Rules**:
    - Answer must be a SINGLE WORD/PHRASE from the list: {embodiedscan_classes}.
    - If uncertain, select the most generic matching category (e.g., "furniture" if ambiguous).
    - No explanations, only the final answer.

    Answer: CATEGORY
    '''
open_prompt = '''
    This image shows 6 renderings of the same indoor object. 
    Identify the object in ONE WORD or SHORT PHRASE (e.g., "chair", "desk lamp"). 
    Focus on its primary function and shape. Do not explain.

    Answer: OBJECT
'''



def single_image(img_path, prompt_type="close", load_method="local"):
    if prompt_type == "close":
        question = f'<image>\n{close_prompt}'
    elif prompt_type == "open":
        question = f'<image>\n{open_prompt}'
    pixel_values = load_image(img_path, max_num=12, load_method=load_method).to(torch.bfloat16).cuda()
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    # print(f'Assistant: {response}')
    return response
    


def multi_image(img_paths, prompt_type="close", load_method="local"):

    pixel_values_list = []
    for img_path in img_paths:
        pixel_values_i = load_image(img_path, max_num=12, load_method=load_method).to(torch.bfloat16).cuda()
        pixel_values_list.append(pixel_values_i)
    
    if pixel_values_list:
        pixel_values = torch.cat(pixel_values_list, dim=0)
        
    if prompt_type == "close":
        question = f'<image>\n{close_prompt}'
    elif prompt_type == "open":
        question = f'<image>\n{open_prompt}'

    response = model.chat(tokenizer, pixel_values, question, generation_config)
    # print(f'Assistant: {response}')
    return response

def get_images(url):
    presigned_url = client.generate_presigned_url(url, client_method ='get_object', expires_in=3600)
    image = Image.open(requests.get(presigned_url, stream=True).raw)
    return image

def upload_images(image_url, big_image):
    image_ext = os.path.splitext(image_url)[-1]
    big_image_cv = cv2.cvtColor(np.array(big_image), cv2.COLOR_RGB2BGR)
    success, img_encoded = cv2.imencode(image_ext, big_image_cv)
    assert(success)
    img_bytes = img_encoded.tostring()
    client.put(image_url, img_bytes)

def get_client_contents(url):
    directory_contents = []
    object_contents = []
    contents = client.list(url)
    for content in contents:
        if content.endswith('/'):
            directory_contents.append(content)
        else:
            object_contents.append(content)
            # new_url = f"{url}/{content}"
    return directory_contents, object_contents


if __name__ == "__main__":

    part = args.part
    usd_idx = args.usd
    process_mode = args.process_mode
    usd_list = []
    if process_mode == 'single':
        usd_list.append(f"{usd_idx}_usd")
    elif process_mode == 'loop':
        start = args.start_usd
        end = args.end_usd
        for i in range(start, end+1):
            usd_list.append(f"{i}_usd")


    ceph_dir = f'path/to/the/images/'
    directory_contents, _ = get_client_contents(ceph_dir)
    print(usd_list)
    for usd in usd_list:
        
        if usd+'/' in directory_contents:
            whole_usd_time_start = time.time()
            ceph_usd_path = f'{ceph_dir}/{usd}'
            print(ceph_usd_path)
            scene_names, _ = get_client_contents(ceph_usd_path)
            for scene_name in scene_names:
                print(scene_name)
                ceph_scene_path = f'{ceph_usd_path}/{scene_name}models_rendered/'
                ceph_multi_views_33_path = f'{ceph_scene_path}multi_views_33/'
                objects, _ = get_client_contents(ceph_multi_views_33_path)
                _, contents = get_client_contents(ceph_scene_path)
                if 'multi_mode_result_simplify_prompt.json' in contents:
                    print(f"The scene {scene_name} has been annotated")
                    continue
                multi_mode_result = {}
                object_time = time.time()
                for object_name in objects:
                    ceph_object_path = f'{ceph_multi_views_33_path}{object_name}'
                    _, image_names = get_client_contents(ceph_object_path)
                    image_paths = []
                    for image_name in image_names:
                        image_path = f'{ceph_object_path}{image_name}'
                        image_paths.append(image_path)
                    start_time = time.time()
                    print(image_paths)
                    response = multi_image(image_paths, load_method='ceph')
                    print(f"Annotation time: {time.time() - start_time}s")
                    model_name = object_name[:-1]    # remove '/'
                    multi_mode_result[model_name] = response
                print(f"Annotating the {len(objects)} objects cost {time.time() - object_time}s")
                result_json_path = f'{ceph_scene_path}multi_mode_result_simplify_prompt.json'
                client.put(result_json_path, json.dumps(multi_mode_result, indent=4).encode('utf-8'))
            print(f"Annotating the whole usd cost {time.time() - whole_usd_time_start}s")
                

        
    


