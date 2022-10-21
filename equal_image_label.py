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

def csv_cfg_map(path):
    cfg_map = {}
    line0=''
    with open(path, "r") as f:
        i = 0
        for line in f:
            if i != 0:
                list = line.strip().split(",")
                if list[0] not in cfg_map:
                    cfg_map[list[0]] = []
                cfg_map[list[0]].append(list)
            else:
                line0=line.strip()
            i = i + 1
    # print('csv_cfg_map', cfg_map)
    return [cfg_map,line0]


PATH_TO_IMAGE = sys.argv[1]
PATH_TO_LABELS = sys.argv[2]
PATH_TO_SAVE = sys.argv[3]

[cfg_map,line0] = csv_cfg_map(PATH_TO_LABELS)

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

os.system('mkdir -p {0}'.format(PATH_TO_SAVE))
new_ann_list = [line0]
for ann in ann_list:
    ann=ann[0]
    w = int(ann[1])
    h = int(ann[2])
    if w > h:
        # 补充高
        h = w
    if w < h:
        # 补充宽
        w = h
    img = cv2.imread(os.path.join(PATH_TO_IMAGE, ann[0]), 1)
    outImg = cv2.copyMakeBorder(img,0,h-int(ann[2]),0,w-int(ann[1]),cv2.BORDER_CONSTANT,value=[255,255,255])
    cv2.imwrite(os.path.join(PATH_TO_SAVE, ann[0]), outImg)
    ann[1]=w
    ann[2]=h
    new_ann_list.append('\n'+(",".join([str(a) for a in ann])))

fo = open('{0}/labels.csv'.format(PATH_TO_SAVE), "w")
fo.writelines(new_ann_list)
fo.close()
