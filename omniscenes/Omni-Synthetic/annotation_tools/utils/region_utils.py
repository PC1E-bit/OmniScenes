import numpy as np
import json
import cv2
import matplotlib


def is_in_poly(ps, poly):
    """
        ps: a numpy array of shape (N, 2)
        poly: a polygon represented as a list of (x, y) tuples
    """
    if isinstance(ps, tuple):
        ps = np.array([ps])
    if len(ps.shape) == 1:
        ps = np.expand_dims(ps, axis=0)
    assert ps.shape[1] == 2
    assert len(ps.shape) == 2
    path = matplotlib.path.Path(poly)
    return path.contains_points(ps)


def get_position_in_mesh_render_image(points, center_x, center_y, num_pixels_per_meter, photo_pixel,force_list=False):
    """
        Args:
        points: a numpy array of shape (N, 2) IN WORLD COORDINATE
        phote_pixel: a tuple of (h, w)
        return: a numpy array of shape (N, 2) IN RENDER IMAGE COORDINATE, in y, x order
    """
    if isinstance(points, tuple):
        points = np.array([points])
    if len(points.shape) == 1:
        points = np.expand_dims(points, axis=0)
    assert points.shape[1] == 2
    assert len(points.shape) == 2
    dxs = points[:, 0] - center_x
    dys = points[:, 1] - center_y
    xs = (dxs * num_pixels_per_meter + photo_pixel[1] // 2).astype(int)
    ys = (photo_pixel[0] // 2 - dys * num_pixels_per_meter).astype(int)
    if len(xs) == 1 and not force_list:
        xs = xs[0]
        ys = ys[0]
    return ys, xs


def get_data(input_annotation_path):
    '''extract data from input files'''
    with open(input_annotation_path, "r", encoding="UTF-8") as f:
        lines = f.readlines()
    s = ''
    for l in lines[0]:
        if l == "'":
            s += '"'
        else:
            s += l
    region_with_label = json.loads(s)
    return region_with_label

def enlarge_poly(poly,ratio):
    m_x,m_y = (np.array(poly)[:,0]).mean(),(np.array(poly)[:,1]).mean()
    poly_new = []
    for (x,y) in poly:
        poly_new.append((m_x+(1.0+ratio)*(x-m_x),m_y+(1.0+ratio)*(y-m_y)))
    return poly_new

def process_data(region_with_label, scene_id, object_ids, bboxes, center_x, center_y, num_pixels_per_meter,enlarge_ratio=0.3 ,neglect_class=['wall'],img_save_root='.'):
    '''
    process the annotation data
    get the corresponding object ids
    '''

    img = cv2.imread(f'data/{scene_id}/render.png')

    original_region_cover_list = []

    # 首先在原始region anno上做一遍

    for region in region_with_label:

        poly = region['vertex']
        region['object_ids']=[]

        for _index in range(len(object_ids)):
            object_x = bboxes[_index][0]
            object_y = bboxes[_index][1]
            x, y = get_position_in_mesh_render_image((object_x, object_y), center_x, center_y, num_pixels_per_meter, img.shape[:2])

            if is_in_poly((x, y), poly):
                region['object_ids'].append(object_ids[_index])
                original_region_cover_list.append(object_ids[_index])


    # 接着在增扩后的region anno上做一遍(扩增不能占用其它区域的object)

    for region in region_with_label:

        poly = enlarge_poly(region['vertex'], enlarge_ratio)


        for _index in range(len(object_ids)):
            object_x = bboxes[_index][0]
            object_y = bboxes[_index][1]
            x, y = get_position_in_mesh_render_image((object_x, object_y), center_x, center_y, num_pixels_per_meter,
                                                     img.shape[:2])

            if is_in_poly((x, y), poly) and object_ids[_index] not in original_region_cover_list:
                region['object_ids'].append(object_ids[_index])

    return region_with_label


if __name__ == "__main__":

    pass

