import trimesh
import numpy as np
from tqdm import tqdm

import threading
from concurrent.futures import ThreadPoolExecutor



def get_obb_vertices(bbox):
    # 解析参数
    position = np.array(bbox[:3])          # [x, y, z]
    size = np.array(bbox[3:6])             # [w, h, d]
    euler_angles = np.array(bbox[6:9])     # [rot_z, rot_x, rot_y]
    
    # 生成局部坐标系下的顶点（未旋转）
    half_size = size / 2
    local_vertices = np.array([
        [-1, -1, -1], 
        [-1, -1, 1], 
        [-1, 1, -1], 
        [-1, 1, 1],
        [1, -1, -1], 
        [1, -1, 1], 
        [1, 1, -1], 
        [1, 1, 1]
    ]) * half_size
    
    # 应用旋转矩阵
    R = trimesh.transformations.euler_matrix(*euler_angles, axes='rzxy')[:3, :3]
    rotated_vertices = (R @ local_vertices.T).T # 这里转置是因为 位置坐标是行向量而不是列向量
    
    # 平移至世界坐标系
    world_vertices = rotated_vertices + position
    return world_vertices

def is_point_inside_bbox(point, bbox):
    """
    判断 point 是否在 bbox 内部
    bbox 的格式为 (center, size, euler_angles)
    """
    center, size, euler = bbox[0:3], bbox[3:6], bbox[6:9]

    # 创建 trimesh 的 Box 对象
    bbox_mesh = trimesh.primitives.Box(extents=size, transform=trimesh.transformations.euler_matrix(*euler, axes='rzxy'))
    bbox_mesh.apply_translation(center)

    # 检查 point 是否在 bbox 内部
    return bbox_mesh.contains([point])


def is_bbox1_inside_bbox2(bbox1, bbox2):
    '''
    判断 bbox1 是否在 bbox2 内部
    bbox: (centerx, centery, centerz, size_x, size_y, size_z, rot_z, rot_x, rot_y)
    '''
    c2, s2, e2 = np.array(bbox2[0:3]), np.array(bbox2[3:6]), bbox2[6:9]
    c1, s1, e1 = np.array(bbox1[0:3]), np.array(bbox1[3:6]), bbox1[6:9]

    # 首先进行简单判断：    
    if np.linalg.norm(c2 - c1) > np.linalg.norm(s2/2) + np.linalg.norm(s1/2):
        return False

    # bbox1
    bbox1_vertices = get_obb_vertices(bbox1)

    # bbox2
    box2 = trimesh.primitives.Box(extents=s2, transform=trimesh.transformations.euler_matrix(*e2, axes='rzxy'))
    box2.apply_translation(c2)

    flag = True
    for vertex in bbox1_vertices:
        if not box2.contains([vertex]):
            flag = False
            break
    return flag

def is_bbox1_inside_bbox2_fast(bbox1, bbox2):
    '''
    判断 bbox1 是否在 bbox2 内部,  把bbox1的顶点变换到bbox2的坐标系中，然后进行判断，不使用trimesh
    bbox: (centerx, centery, centerz, size_x, size_y, size_z, rot_z, rot_x, rot_y)
    '''
    c2, s2, e2 = np.array(bbox2[0:3]), np.array(bbox2[3:6]), bbox2[6:9]
    c1, s1, e1 = np.array(bbox1[0:3]), np.array(bbox1[3:6]), bbox1[6:9]

    # 首先进行简单判断：    
    if np.linalg.norm(c2 - c1) > np.linalg.norm(s2/2) + np.linalg.norm(s1/2):
        return False
    
    # 计算 bbox1 的 8 个顶点
    vertices = get_obb_vertices(bbox1)

    # 得到bbox2的旋转矩阵 ,世界坐标到局部坐标的transform
    transform_2 = np.linalg.inv(trimesh.transformations.euler_matrix(*e2, axes='rzxy')[:3,:3])

    # 把bbox1的顶点变换到bbox2的坐标系中
    vertices_1_local = (transform_2 @ (np.array(vertices) - c2).T).T
    
    # 判断bbox1的顶点是否在bbox2的内部
    half2 = s2 * 0.5
    inside_mask = np.abs(vertices_1_local) <= half2  # 形状：(8,3)
    return bool(np.all(inside_mask))
    
    

    

def get_rotated_bbox_iou(bbox1, bbox2):
    '''
    计算两个旋转bbox的iou
    bbox1: dict, bbox1 信息，包含 location, size, euler
    bbox2: dict, bbox2 信息，包含 location, size, euler
    '''
    def bbox_to_mesh(bbox):
        loc = bbox[0:3]
        size = bbox[3:6]
        euler = bbox[6:9]

        # 创建一个原点在中心，size为bbox大小的立方体
        box = trimesh.creation.box(extents=size)

        # 计算旋转矩阵
        rotation_matrix = trimesh.transformations.euler_matrix(*euler, axes='rzxy')
        transform = np.eye(4)
        transform[:3,:3] = rotation_matrix[:3,:3]
        transform[:3,3] = loc

        # 应用变换
        box.apply_transform(transform)
        return box    

    # 首先进行简单判断：   
    c2, s2, e2 = np.array(bbox2[0:3]), np.array(bbox2[3:6]), bbox2[6:9]
    c1, s1, e1 = np.array(bbox1[0:3]), np.array(bbox1[3:6]), bbox1[6:9] 
    if np.linalg.norm(c2 - c1) > np.linalg.norm(s2/2) + np.linalg.norm(s1/2):
        return 0.0

    mesh1 = bbox_to_mesh(bbox1)
    mesh2 = bbox_to_mesh(bbox2)

    # 计算交集
    try:
        intersection_mesh = mesh1.intersection(mesh2)
        intersection_volume = intersection_mesh.volume
    except:
        print("不是有效的体积网格.")
        return 0.0

    # 计算并集 (体积(A) + 体积(B) - 体积(交集))
    # print( mesh1.volume, mesh2.volume, intersection_volume)
    union_volume = mesh1.volume + mesh2.volume - intersection_volume

    if union_volume == 0:
        return 0.0  # 防止除零错误

    iou = intersection_volume / union_volume
    return iou

def is_bboxes_collision(bbox1, bbox2):
    '''
    判断两个bbox是否碰撞, 使用 SAT 分离轴实现
    bbox: (centerx, centery, centerz, size_x, size_y, size_z, rot_z, rot_x, rot_y)
    '''
    # 获取两个包围盒的顶点
    vertices1 = get_obb_vertices(bbox1)
    vertices2 = get_obb_vertices(bbox2)
    
    # 获取两个包围盒的中心点
    center1 = np.array(bbox1[:3])
    center2 = np.array(bbox2[:3])
    
    # 计算两个包围盒的旋转矩阵
    R1 = trimesh.transformations.euler_matrix(*bbox1[6:9], axes='rzxy')[:3, :3]
    R2 = trimesh.transformations.euler_matrix(*bbox2[6:9], axes='rzxy')[:3, :3]
    
    # 获取两个包围盒的局部坐标轴
    axes1 = [R1[:, i] for i in range(3)]
    axes2 = [R2[:, i] for i in range(3)]
    
    # 需要检测的所有分离轴: 
    # 1. 两个包围盒的3个面的法向量: 6个
    # 2. 两个包围盒所有棱的叉积: 3*3=9个
    test_axes = []
    
    # 添加包围盒面的法向量
    for axis in axes1:
        test_axes.append(axis)
    for axis in axes2:
        test_axes.append(axis)
    
    # 添加所有棱的叉积
    for i in range(3):
        for j in range(3):
            cross_axis = np.cross(axes1[i], axes2[j])
            # 确保叉积不是零向量
            if np.any(np.abs(cross_axis) > 1e-6):
                test_axes.append(cross_axis / np.linalg.norm(cross_axis))
    
    # 对每个轴进行投影测试
    for axis in test_axes:
        # 将顶点投影到当前轴上
        proj1 = np.dot(vertices1, axis)
        proj2 = np.dot(vertices2, axis)
        
        # 获取投影的最小和最大值
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # 检查是否有重叠，如果没有重叠则返回False
        if max1 < min2 or max2 < min1:
            return False
    
    # 所有轴都有重叠，两个包围盒碰撞
    return True

if __name__ == "__main__":
    bbox1 = [0,0,0,0.999,0.999,0.999,0.01,0,0]
    bbox2 = [0,0,0,1,1,1,0,0,0]
    print(is_bbox1_inside_bbox2_fast(bbox1, bbox2))
