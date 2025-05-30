from math import radians, degrees, sin, cos, atan2


def angle_diff(angle1, angle2, angle_range = 360):
    """
    计算两个角度之间的最小差值 (考虑 0-360 循环).

    Args:
        angle1: 第一个角度 (0-360).
        angle2: 第二个角度 (0-360).

    Returns:
        float: 两个角度之间的最小差值 (0-180).
    """
    diff = abs(angle1 - angle2)
    return min(diff, angle_range - diff)

def circular_mean(angles):
    """
    计算角度列表的向量平均值 (0-360 度).

    Args:
        angles: 角度列表 (0-360).

    Returns:
        float: 向量平均角度 (0-360).
    """
    x_sum = sum(cos(radians(angle)) for angle in angles)
    y_sum = sum(sin(radians(angle)) for angle in angles)
    avg_x = x_sum / len(angles)
    avg_y = y_sum / len(angles)
    mean_angle_radians = atan2(avg_y, avg_x)
    mean_angle_degrees = degrees(mean_angle_radians)
    # 调整角度范围到 0-360
    return (mean_angle_degrees + 360) % 360

def calculate_circular_average_with_outliers(data, close_threshold=35, view_num = 4):
    """
    使用投票方法去除离群点,并计算0-360度分布数据的平均值.

    Args:
        data: 包含四个数字的列表 (0-360). 

    Returns:
        float: 根据离群值情况计算出的平均角度值 (0-360).
        ValueError: 如果四个数字互相离得都比较远.
    """
    if len(data) != view_num:
        raise ValueError(f"输入数据必须包含{view_num}个数字。")

    for angle in data:
        if not 0 <= angle <= 360:
            raise ValueError("输入角度值必须在 0-360 度之间。")

    neighbors_count = []
    for i in range(view_num):
        count = 0
        for j in range(view_num):
            if i != j and angle_diff(data[i], data[j]) < close_threshold:
                count += 1
        neighbors_count.append(count)

    max_neighbors = max(neighbors_count)

    if max_neighbors == 0: # 新增的判断条件：max_neighbors 为 0 时报错
        print("数据异常，离群值数量超过预期, 数字互相离得都比较远。")
        return "None"

    outlier_indices = [i for i, count in enumerate(neighbors_count) if count < max_neighbors]
    remaining_data = [data[i] for i in range(view_num) if i not in outlier_indices]
    remaining_count = len(remaining_data)

    if remaining_count <= 1:
        print("数据异常，无有效角度可计算平均值。")
        return "None"
    
    elif remaining_count == 2:
        if angle_diff(remaining_data[0], remaining_data[1]) > close_threshold:
                print("剩余的两个角度相差过远，无法计算有效平均值。")
                return "None"
        else:
            return circular_mean(remaining_data)
    
    else:
            # 直接计算剩余角度的向量平均
            return circular_mean(remaining_data)