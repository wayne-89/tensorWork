import tensorlayer as tl
import sys
import os
import glob
import xml.etree.ElementTree as ET
import cv2
import math
import numpy as np

jitter = 0.2
im_size = [416, 416]  # 输出图的大小


# {xxx:[[filename,width,height,class,xmin,ymin,xmax,ymax]]}
def xml_cfg_map(path):
    cfg_map = {}
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = root.find('filename').text
        size = root.find('size')
        width = size.find('width').text
        height = size.find('height').text
        list = []
        for member in root.findall('object'):
            # class x y x1 y1
            value = [
                filename,
                width,
                height,
                member[0].text,  # class
                int(member[4][0].text),  # xmin
                int(member[4][1].text),  # ymin
                int(member[4][2].text),  # xmax
                int(member[4][3].text)  # ymax
            ]
            list.append(value)
        cfg_map[filename] = list
    print('xml_cfg_map', cfg_map)
    return cfg_map


def csv_cfg_map(path):
    cfg_map = {}
    with open(path, "r") as f:
        i = 0
        for line in f:
            if i != 0:
                list = line.strip().split(",")
                if list[0] not in cfg_map:
                    cfg_map[list[0]] = []
                cfg_map[list[0]].append(list)
            i = i + 1
    # print('csv_cfg_map', cfg_map)
    return cfg_map


def _rotate_coord(shape, newxy, point, angle):
    angle = -1 * angle / 180.0 * math.pi
    ox, oy = shape
    px, py = point
    ox /= 2
    oy /= 2
    qx = math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    new_x, new_y = newxy
    qx += ox - new_x
    qy += oy - new_y
    return int(qx + 0.5), int(qy + 0.5)


def _largest_rotated_rect(w, h, angle):
    """
    Get largest rectangle after rotation.
    http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """
    angle = angle / 180.0 * math.pi
    if w <= 0 or h <= 0:
        return 0, 0

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2. * sin_a * cos_a * side_long:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a
    return int(np.round(wr)), int(np.round(hr))


def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2]  # image shape has 3 dimensions
    image_center = (
        width / 2,
        height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def _data_pre_aug_fn(data):
    img, coords = data
    img_shape = np.shape(img)
    height = img_shape[0]
    width = img_shape[1]
    center = (width * 0.5, height * 0.5)  # x, y
    res = []
    for deg in [90, 180, 270]:
        neww, newh = _largest_rotated_rect(width, height, deg)
        newx = int(center[0] - neww * 0.5)
        newy = int(center[1] - newh * 0.5)
        coords_new = []
        for coord in coords:
            point = [coord[4], coord[5], coord[6], coord[7]]
            x, y = _rotate_coord((width, height), (newx, newy), (int(point[0]), int(point[1])), deg)
            x2, y2 = _rotate_coord((width, height), (newx, newy), (int(point[2]), int(point[3])), deg)
            suffix = coord[0].split('.')[-1]
            name = coord[0][0:-len(suffix) - 1]
            img_name = "{0}_{1}.{2}".format(name, deg, suffix)
            coords_new.append(
                [img_name, str(neww), str(newh), coord[3], str(min(x, x2)), str(min(y, y2)), str(max(x, x2)),
                 str(max(y, y2))])
        sub_img = rotate_image(img, deg)
        # print(img_shape, deg, coords, coords_new)
        res.append([sub_img, coords_new])
    return res


PATH_TO_IMAGE = sys.argv[1]
PATH_TO_LABELS = sys.argv[2]
cfg_map = csv_cfg_map(PATH_TO_LABELS)

b_im_name = [name for name in os.listdir(PATH_TO_IMAGE)
             if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]

b_im_path = []
ann_list = []
for name in b_im_name:
    b_im_path.append(os.path.join(PATH_TO_IMAGE, name))
    cfg = []
    if name in cfg_map:
        cfg = cfg_map[name]
    ann_list.append(cfg)

# 多线程处理
b_images = tl.prepro.threading_data(b_im_path, fn=tl.vis.read_image)

data = tl.prepro.threading_data([_ for _ in zip(b_images, ann_list)],
                                _data_pre_aug_fn)

with open(PATH_TO_LABELS, 'a+') as f:
    i = 1
    for img_datas in data:
        if i <= 3:
            for img_data in img_datas:
                img = img_data[0]
                coords = img_data[1]
                tl.vis.save_image(img, os.path.join(PATH_TO_IMAGE, coords[0][0]))
                # cv2.imwrite(os.path.join(PATH_TO_IMAGE, coords[0][0]), img)
                for coord in coords:
                    f.write(','.join(coord) + '\n')
        i = i + 1

