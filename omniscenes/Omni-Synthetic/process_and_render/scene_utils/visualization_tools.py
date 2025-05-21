import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .usd_tools import compute_bbox
import numpy as np
import open3d as o3d

# ============================
# Visualize two bounding boxes
# ============================

def visualize_2bboxes(bbox_a, bbox_b):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    a_min, a_max = bbox_a.min, bbox_a.max
    ax.bar3d(a_min[0], a_min[1], a_min[2], a_max[0] - a_min[0], a_max[1] - a_min[1], a_max[2] - a_min[2], color='b', alpha=0.5)


    b_min, b_max = bbox_b.min, bbox_b.max
    ax.bar3d(b_min[0], b_min[1], b_min[2], b_max[0] - b_min[0], b_max[1] - b_min[1], b_max[2] - b_min[2], color='r', alpha=0.5)


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim([min(a_min[0], b_min[0]), max(a_max[0], b_max[0])])
    ax.set_ylim([min(a_min[1], b_min[1]), max(a_max[1], b_max[1])])
    ax.set_zlim([min(a_min[2], b_min[2]), max(a_max[2], b_max[2])])

    plt.show()

def visualize_bboxes(prims):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    prim_bboxes = []
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for prim in prims:
        bbox = compute_bbox(prim)  
        prim_bboxes.append(bbox)
        b_min, b_max = bbox.min, bbox.max
        ax.bar3d(b_min[0], b_min[1], b_min[2], b_max[0] - b_min[0], b_max[1] - b_min[1], b_max[2] - b_min[2], color=colors.pop(0), alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


    x_min, x_max = min(bbox.min[0] for bbox in prim_bboxes), max(bbox.max[0] for bbox in prim_bboxes)
    y_min, y_max = min(bbox.min[1] for bbox in prim_bboxes), max(bbox.max[1] for bbox in prim_bboxes)
    z_min, z_max = min(bbox.min[2] for bbox in prim_bboxes), max(bbox.max[2] for bbox in prim_bboxes)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])

    plt.show()


def visualize_clusters(points, labels):

    unique_labels = set(labels)
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for label, color in zip(unique_labels, colors):
        if label == -1:  
            color = 'k'  
        cluster_points = points[labels == label]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], color=color, label=f'Cluster {label}')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()

def draw_3d_bbox(corners) -> o3d.geometry.LineSet:
    # 创建一个 LineSet 对象
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)

    # 定义边界框的边
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    line_set.lines = o3d.utility.Vector2iVector(lines)

    # 定义边的颜色（这里使用红色）
    colors = [[0, 0, 0] for _ in range(len(lines))]
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set
    # o3d.visualization.draw_geometries([line_set])

def draw_transformed_bbox(corners, rotation_matrix) -> o3d.geometry.LineSet:

    rotated_corners = np.dot(corners, rotation_matrix.T)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(rotated_corners)

    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    line_set.lines = o3d.utility.Vector2iVector(lines)

    colors = [[1, 0, 0] for _ in range(len(lines))]
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def visulize_region_result(offset_coord, current_floor):
    offset_coord_np = []
    for coord in offset_coord:
        region_3d_coord = np.array([coord[0], coord[1], current_floor + 0.2]) 
        offset_coord_np.append(region_3d_coord)
    offset_coord_np = np.array(offset_coord_np)
    lines = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0]
    ]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(offset_coord_np)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    return line_set