import trimesh
import copy
import pdb
from tqdm import tqdm
import torch
import argparse
import numpy as np

import json
from .bbox_utils import check_obb_overlap_trimesh_with_sphere

def visualize_obbs(bboxes, colors=None, window_name="OBB Visualization"):
    """
    可视化带旋转的 OBB
    
    参数:
        bboxes (np.ndarray): (N, 9) 的数组，每行格式为 [x, y, z, w, h, d, rot_z, rot_x, rot_y]
        colors (list): 每个OBB的颜色,格式为 [(R, G, B)],范围0~1,长度需与bboxes一致
        window_name (str): 窗口名称
    """
    import open3d as o3d
    bboxes = bboxes.detach().numpy()
    # 创建Open3D可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=800, height=600)
    
    # 默认颜色：随机生成
    if colors is None:
        np.random.seed(42)
        colors = np.random.rand(len(bboxes), 3)
    
    # 遍历每个OBB，生成对应的线框立方体
    for i, bbox in enumerate(bboxes):
        # 解析参数
        loc = bbox[0:3]
        size = bbox[3:6]
        zxy_euler = bbox[6:9]
        
        # --- 步骤1: 生成局部坐标的立方体顶点（未旋转）---
        half_extents = np.array(size) / 2
        vertices_local = np.array([
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
        ]) * half_extents
        
        # --- 步骤2: 应用旋转（ZXY欧拉角顺序）---
        # 欧拉角 → 旋转矩阵
        R = euler_to_matrix(zxy_euler).numpy()
        
        # 旋转顶点
        # pdb.set_trace()
        vertices_rotated = vertices_local @ R.T
        
        # --- 步骤3: 平移至世界坐标系 ---
        vertices_world = vertices_rotated + np.array(loc)
        
        # --- 步骤4: 创建线框立方体 ---
        # 定义立方体的12条边（连接顶点的索引对）
        lines = [
            [0, 1], [0, 2], [0, 4],
            [1, 3], [1, 5],
            [2, 3], [2, 6],
            [3, 7],
            [4, 5], [4, 6],
            [5, 7],
            [6, 7]
        ]
        
        # 创建LineSet
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(vertices_world)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color(colors[i])
        
        # 添加到可视化窗口
        vis.add_geometry(line_set)
    
    # --- 添加坐标轴和设置视角 ---
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))
    
    # 设置视角
    view_ctl = vis.get_view_control()
    view_ctl.set_front([0, -1, 0.5])  # 相机朝向
    view_ctl.set_up([0, 0, 1])        # 相机的上方向
    view_ctl.set_zoom(0.5)            # 缩放级别
    
    # 运行可视化
    vis.run()
    vis.destroy_window()


def euler_to_matrix(zxy_euler):
    # 输入: [rot_z, rot_x, rot_y] (弧度)
    if type(zxy_euler) != torch.Tensor:
        zxy_euler = torch.tensor(zxy_euler)
    ai, aj, ak = zxy_euler[0], zxy_euler[1], zxy_euler[2]

    _AXES2TUPLE = {
    "sxyz": (0, 0, 0, 0),
    "sxyx": (0, 0, 1, 0),
    "sxzy": (0, 1, 0, 0),
    "sxzx": (0, 1, 1, 0),
    "syzx": (1, 0, 0, 0),
    "syzy": (1, 0, 1, 0),
    "syxz": (1, 1, 0, 0),
    "syxy": (1, 1, 1, 0),
    "szxy": (2, 0, 0, 0),
    "szxz": (2, 0, 1, 0),
    "szyx": (2, 1, 0, 0),
    "szyz": (2, 1, 1, 0),
    "rzyx": (0, 0, 0, 1),
    "rxyx": (0, 0, 1, 1),
    "ryzx": (0, 1, 0, 1),
    "rxzx": (0, 1, 1, 1),
    "rxzy": (1, 0, 0, 1),
    "ryzy": (1, 0, 1, 1),
    "rzxy": (1, 1, 0, 1),
    "ryxy": (1, 1, 1, 1),
    "ryxz": (2, 0, 0, 1),
    "rzxz": (2, 0, 1, 1),
    "rxyz": (2, 1, 0, 1),
    "rzyz": (2, 1, 1, 1),
    }
    _NEXT_AXIS = [1, 2, 0, 1]
    firstaxis, parity, repetition, frame = _AXES2TUPLE["rzxy"] # (1, 1, 0, 1)
    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    
    sin, cos = torch.sin, torch.cos
    M = torch.eye(4)

    si, sj, sk = sin(ai), sin(aj), sin(ak)
    ci, cj, ck = cos(ai), cos(aj), cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    if repetition:
        M[i, i] = cj
        M[i, j] = sj * si
        M[i, k] = sj * ci
        M[j, i] = sj * sk
        M[j, j] = -cj * ss + cc
        M[j, k] = -cj * cs - sc
        M[k, i] = -sj * ck
        M[k, j] = cj * sc + cs
        M[k, k] = cj * cc - ss
    else:
        M[i, i] = cj * ck
        M[i, j] = sj * sc - cs
        M[i, k] = sj * cc + ss
        M[j, i] = cj * sk
        M[j, j] = sj * ss + cc
        M[j, k] = sj * cs - sc
        M[k, i] = -sj
        M[k, j] = cj * si
        M[k, k] = cj * ci
    return M[:3, :3]

    return  #trimesh.transformations.euler_matrix(z, x, y, axes='rzxy')[:3, :3]

def get_obb_vertices(bbox, device):
    # 解析参数
    position = bbox[:3]          # [x, y, z]
    size = bbox[3:6]             # [w, h, d]
    euler_angles = bbox[6:9]     # [rot_z, rot_x, rot_y]
    
    # 生成局部坐标系下的顶点（未旋转）
    half_size = size / 2
    local_vertices = torch.tensor([
        [-1, -1, -1], 
        [-1, -1, 1], 
        [-1, 1, -1], 
        [-1, 1, 1],
        [1, -1, -1], 
        [1, -1, 1], 
        [1, 1, -1], 
        [1, 1, 1]
    ], device=device) * half_size
    
    # 应用旋转矩阵
    Rot = euler_to_matrix(euler_angles)
    Rot = Rot.to(device)
    rotated_vertices = torch.mm(local_vertices, Rot.T) # 这里转置是因为 位置坐标是行向量而不是列向量
    
    # 平移至世界坐标系
    world_vertices = rotated_vertices + position
    return world_vertices


def obb_iou(bbox1, bbox2, device,return_intersection=False):
    '''
    简略版本的 iou计算，把OBB转换为AABB，计算AABB的交集和并集
    '''
    # 获取两个OBB的顶点
    vertices1 = get_obb_vertices(bbox1, device)  # Shape: (8, 3)
    vertices2 = get_obb_vertices(bbox2, device)  # Shape: (8, 3)
    
    # 计算两个OBB的最小包围AABB
    min1, max1 = torch.min(vertices1, dim=0)[0], torch.max(vertices1, dim=0)[0]
    min2, max2 = torch.min(vertices2, dim=0)[0], torch.max(vertices2, dim=0)[0]

    # 快速AABB相交检测
    if (max1[0] < min2[0] or min1[0] > max2[0] or
        max1[1] < min2[1] or min1[1] > max2[1] or
        max1[2] < min2[2] or min1[2] > max2[2]):
        return torch.tensor(0.0, device=device) if not return_intersection else (torch.tensor(0.0, device=device), torch.tensor(0.0, device=device))
    
    # 计算AABB的交集和并集
    overlap_min = torch.max(min1, min2)
    overlap_max = torch.min(max1, max2)
    overlap_dims = torch.clamp(overlap_max - overlap_min, min=0)
    intersection = torch.prod(overlap_dims)
    
    volume1 = torch.prod(max1 - min1)
    volume2 = torch.prod(max2 - min2)
    union = volume1 + volume2 - intersection
    
    # IoU = intersection / union
    iou = intersection / (union + 1e-6)


    if return_intersection:
        return iou, intersection
    else:
        return iou
    
def box1_center_in_box2(box1, box2):
    from trimesh.transformations import euler_matrix
    
    box1, box2 = box1.detach().cpu().numpy(),  box2.detach().cpu().numpy()

    point = box1[0:3]
    center, size, euler_angles = box2[0:3], box2[3:6], box2[6:9]
    rotation = euler_matrix(*euler_angles, axes='rzxy')[:3, :3]

    displacement = point - center
    local_point = rotation.T.dot(displacement)

    half_size = size / 2.0
    return np.all(np.abs(local_point) <= half_size + 1e-6)
    # # 获取两个OBB的顶点
    # vertices2 = get_obb_vertices(box2)  # Shape: (8, 3)

    # center1 = box1[0:3]

    # # 判断 center1 是否在 vertices2内部
    



class OBBCollisionOptimizer:
    def __init__(
            self, 
            initial_bboxes, 
            cate_list,
            cate_need_touch_ground, 
            make_large_bbox_ground_aligned, 
            lr=0.02, 
            max_iter=100, 
            lambda_reg=0.1, 
            lambda_gravity = 1, 
            lambda_rotation=10
    ):
        # settings
        self.make_large_bbox_ground_aligned = make_large_bbox_ground_aligned
        self.cate_need_touch_ground = cate_need_touch_ground
        self.cate_list = cate_list

        # device
        self.device = "cpu" # if torch.cuda.is_available() else "cpu"
        if type(initial_bboxes) != torch.Tensor:
            initial_bboxes = torch.tensor(initial_bboxes, dtype=torch.float32, device=self.device)

        # 初始化可优化参数：位置(x,y,z) 和 旋转(rot_z, rot_x, rot_y)
        self.positions = torch.nn.Parameter(initial_bboxes[:, :3].clone())
        # self.rotations = torch.nn.Parameter(initial_bboxes[:, 6:9].clone())
        self.rotations = initial_bboxes[:, 6:9].clone()
        
        # 固定参数：尺寸(w,h,d)
        self.sizes = initial_bboxes[:, 3:6].clone()
        
        # 优化器和超参数
        # self.optimizer = torch.optim.Adam([self.positions, self.rotations], lr=lr)
        self.optimizer = torch.optim.Adam([self.positions], lr=lr)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max_iter, eta_min=5e-3)
        self.lambda_reg = lambda_reg
        self.lambda_rotation = lambda_rotation
        self.lambda_gravity = lambda_gravity
        self.max_iter = max_iter

        # 初始位置和旋转
        self.initial_positions = initial_bboxes[:, :3].clone()
        self.initial_rotations = initial_bboxes[:, 6:9].clone()

        # 定义 collisoin_free pairs 
        self.collision_free_pairs = [("countertop", "sink"),
                                     ("cabinet", "oven"),
                                     ("cabinet", "stove")]

        

    def compute_avg_iou(self):
        bboxes = self.get_current_bboxes()
        total_iou = 0.0
        count = 0
        for i in range(len(bboxes)):
            for j in range(i+1, len(bboxes)):
                iou = obb_iou(bboxes[i], bboxes[j], device=self.device)
                total_iou += iou
                count += 1
        return total_iou / count if count > 0 else 1.0
        
    def get_current_bboxes(self):
        # 组合成完整bbox参数：[x,y,z, w,h,d, rot_z,rot_x,rot_y]
        return torch.cat([
            self.positions,
            self.sizes,
            self.rotations
        ], dim=1)
    
    def compute_loss(self, device):
        bboxes = self.get_current_bboxes()
        N = bboxes.shape[0]

        # 计算平均体积
        volumes = torch.prod(bboxes[:, 3:6], dim=1)
        average_volume = torch.mean(volumes)
        
        
        # 碰撞损失
        collision_loss = 0.0
        set_iou_zero_num = 0 # 记录有多少个iou被设置为0
        for i in range(N):
            for j in range(i+1, N):
                # 精确判断两个oriented bbox是否有重叠，有的话用AABB计算iou
                # 判断：
                overlap = check_obb_overlap_trimesh_with_sphere(bboxes[i].detach().numpy(), bboxes[j].detach().numpy())
                if not overlap:
                    continue
                else:

                    iou, intersection = obb_iou(bboxes[i], bboxes[j], device = device, return_intersection = True)

                    # 处理微波炉、沙发、炉子 和其它物体的的碰撞
                    # 处理 collision_free_pairs 
                    pairs = ((self.cate_list[i], self.cate_list[j]), (self.cate_list[j], self.cate_list[i]))
                    flag = False
                    for pair in pairs:
                        if pair in self.collision_free_pairs:   
                            flag = True
                            break
                        
                    if flag == True:
                        iou = 1e-1
                        set_iou_zero_num += 1
                    # else:
                    # 中心判断
                    if box1_center_in_box2(box1 = bboxes[i], box2 = bboxes[j]) or box1_center_in_box2(box1 = bboxes[j], box2 = bboxes[i]):
                        # iou判断
                        threshold = 0.2
                        if intersection/volumes[i] > threshold or intersection/ volumes[j] > threshold:
                            iou = 0
                            set_iou_zero_num += 1
                    
                    if iou > 1e-6:
                        collision_loss += iou *  (volumes[i] * volumes[j] ) / (average_volume**2 + 1e-6)

        # 重力损失
        gravity_loss = 0.0
        if self.make_large_bbox_ground_aligned:
            for index in range(N):
                box = bboxes[index]
                bottom_height = box[2] - box[5] / 2

                # 如果物体底面离地面比较近, 或者物体类别是cate_need_touch_ground 的话，增加重力损失
                if self.cate_list[index] not in ["blinds", "jalousie", "window"]:
                    if torch.abs(bottom_height) < box[5] * 0.4 or torch.abs(bottom_height) < 0.4 or self.cate_list[index] in self.cate_need_touch_ground:
                        gravity_loss += (volumes[index] / (average_volume + 1e-6)) * (bottom_height/(box[5] + 1e-6))** 2 

        # 正则化损失（保持初始位置和旋转）
        #rot_reg = torch.sum((self.rotations - self.initial_rotations)**2)
        pos_reg = torch.sum((self.positions - self.initial_positions)**2 * volumes.reshape(-1,1) / average_volume)
        reg_loss = pos_reg #+ rot_reg * self.lambda_rotation
        
        total_loss = collision_loss + self.lambda_reg * reg_loss + self.lambda_gravity * gravity_loss
        return total_loss, set_iou_zero_num
    
    def optimize(self):
        
        for i in tqdm(range(self.max_iter), desc="Optimizing..."):
            self.optimizer.zero_grad()
            loss, set_iou_zero_num = self.compute_loss(device=self.device)
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            
            # 可选：约束旋转角在 [-π, π] 内
            self.rotations.data = torch.remainder(self.rotations.data + np.pi, 2*np.pi) - np.pi

            # print(f"iter:{i+1}, Loss: {loss.item():.4f},"+ 
            #       f"set_iou_zero_num: {set_iou_zero_num}, lr: {self.optimizer.param_groups[0]['lr']:.6f}")

            #  Avg IoU: {self.compute_avg_iou():.8f},



def parse_arguments():

    parser = argparse.ArgumentParser()
    # parser.add_argument("--scene_name", required=True, type=str, help = "scene name, e.g. scannet/scene0000_00")
    # parser.add_argument("--base_dir", required=True, type=str, help = "")
    parser.add_argument("--input_instance_infos_dir", required=True, type=str, help = "")
    parser.add_argument("--output_instance_infos_dir", required=True, type=str, help = "")
    parser.add_argument("--cate_with_large_size_path", required=True, type=str, help = "")
    parser.add_argument("--cate_need_touch_ground_path", required=True, type=str, help = "")
    parser.add_argument("--not_visualize", action="store_true", help="不启用可视化")
    parser.add_argument("--make_large_bbox_ground_aligned", action="store_true", help="启用地面对齐bbox")

    args = parser.parse_args()
    
    return args


def optimize_one_scene(
        input_instance_infos_dir,
        output_instance_infos_dir,
        cate_with_large_size,
        cate_need_touch_ground,
        not_visualize,
        make_large_bbox_ground_aligned# 保持bbox的底面与地面平行
):
    with open(input_instance_infos_dir, "r") as f: 
        instance_infos = json.load(f)

    # load bboxes
    bbox_list = []
    cate_list = []
    for instance_info in instance_infos:
        try:
            cate = instance_info["category"]
        except Exception as e:
            print(f"Error in instance_info: lack of key _category_")
            continue

        if cate not in cate_with_large_size:
            continue

        bbox = copy.deepcopy(instance_info["bbox"])

        # 保持bbox的底面与地面平行
        if make_large_bbox_ground_aligned:
            bbox[7] = 0.0
            bbox[8] = 0.0

        bbox_list.append(bbox)
        cate_list.append(cate)

    # 场景中没有大家居的情形
    if len(bbox_list) == 0:
        print("No large bbox to optimize")


    else:
        bbox_list = torch.tensor(bbox_list)
        origin_bbox_list = copy.deepcopy(bbox_list)

        if not not_visualize:
            visualize_obbs(bbox_list)

        # instancing optimizer
        optimizer = OBBCollisionOptimizer(
            bbox_list,
            cate_list, 
            cate_need_touch_ground, 
            make_large_bbox_ground_aligned
        )#, lr=0.012, max_iter=50, lambda_reg=0.5, lambda_rotation=10)
        optimizer.optimize()

        bbox_list = optimizer.get_current_bboxes()
        if not not_visualize:
            visualize_obbs(bbox_list)   

        # calculate delta_translation
        delta_translation = bbox_list[:, :3] - origin_bbox_list[:, :3]

        # record instance_infos
        i = 0
        for instance_info in instance_infos:
            try:
                cate = instance_info["category"]
            except Exception as e:
                print(f"Error in instance_info: lack of key _category_")
                continue
            if cate not in cate_with_large_size:
                continue
            instance_info["delta_translation_after_large_obj_optimization"] = delta_translation[i].tolist()

            # 保持bbox的底面与地面平行
            if make_large_bbox_ground_aligned:
                instance_info["bbox"][7] = 0.0
                instance_info["bbox"][8] = 0.0

            i += 1

    # save instances_info
    with open(output_instance_infos_dir, "w") as f: # "./instances_info.json"
        json.dump(instance_infos, f, indent=4)
    

def main(
        # scene_name,
        # base_dir,
        input_instance_infos_dir,
        output_instance_infos_dir,
        cate_with_large_size_path,
        cate_need_touch_ground_path,
        not_visualize,
        make_large_bbox_ground_aligned = True # 保持bbox的底面与地面平行
    ):

    with open(input_instance_infos_dir, "r") as f: # "./scene_glbs/embodiedscene/scannet/scene0000_00/instances_info_with_retri_uid_withpartnetmobility.json"
        instance_infos = json.load(f)

    with open(cate_with_large_size_path, "r") as f: # "./optimiz_large_bbox/cate_with_large_size.json"
        cate_with_large_size = json.load(f) 
        cate_with_large_size = list(cate_with_large_size.keys())
    
    with open(cate_need_touch_ground_path, "r") as f: # "./optimiz_large_bbox/cate_with_large_size.json"
        cate_need_touch_ground = json.load(f) 
        cate_need_touch_ground = list(cate_need_touch_ground.keys())
    
    # load bboxes
    bbox_list = []
    cate_list = []
    for instance_info in instance_infos:
        try:
            cate = instance_info["category"]
        except Exception as e:
            print(f"Error in instance_info index: lack of key _category_")
            continue
        if cate not in cate_with_large_size:
            continue

        bbox = copy.deepcopy(instance_info["bbox"])

        # 保持bbox的底面与地面平行
        if make_large_bbox_ground_aligned:
            bbox[7] = 0.0
            bbox[8] = 0.0

        bbox_list.append(bbox)
        cate_list.append(cate)
        
    bbox_list = torch.tensor(bbox_list)
    origin_bbox_list = copy.deepcopy(bbox_list)

    if not not_visualize:
        visualize_obbs(bbox_list)

    # instancing optimizer
    optimizer = OBBCollisionOptimizer(
        bbox_list,
        cate_list, 
        cate_need_touch_ground, 
        make_large_bbox_ground_aligned
    )#, lr=0.012, max_iter=50, lambda_reg=0.5, lambda_rotation=10)
    optimizer.optimize()

    bbox_list = optimizer.get_current_bboxes()
    if not not_visualize:
        visualize_obbs(bbox_list)   

    # calculate delta_translation
    delta_translation = bbox_list[:, :3] - origin_bbox_list[:, :3]

    # record instance_infos
    i = 0
    for instance_info in instance_infos:
        try:
            cate = instance_info["category"]
        except Exception as e:
            print(f"Error in instance_info index: lack of key _category_")
            continue
        if cate not in cate_with_large_size:
            continue
        instance_info["delta_translation_after_large_obj_optimization"] = delta_translation[i].tolist()

        # 保持bbox的底面与地面平行
        if make_large_bbox_ground_aligned:
            instance_info["bbox"][7] = 0.0
            instance_info["bbox"][8] = 0.0

        i += 1

    # save instances_info
    with open(output_instance_infos_dir, "w") as f: # "./instances_info.json"
        json.dump(instance_infos, f, indent=4)

        

if __name__ == "__main__":

    args = parse_arguments()
    main(
        # args.scene_name,
        # args.base_dir,
        args.input_instance_infos_dir,
        args.output_instance_infos_dir,
        args.cate_with_large_size_path,
        args.cate_need_touch_ground_path,
        args.not_visualize,
        args.make_large_bbox_ground_aligned
    )
    


    # main(
    #     "./scene_glbs/embodiedscene/scannet/scene0000_00/instances_info_with_retri_uid_withpartnetmobility.json",
    #     "./scene_glbs/embodiedscene/scannet/scene0000_00/instances_info_with_opti.json",
    #     "./optimiz_large_bbox/cate_with_large_size.json"
    # )