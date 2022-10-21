# coding=utf-8
######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/15/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on an image.
# It draws boxes and scores around the objects of interest in the image.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import json
import ast
import urllib.parse

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt


# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
IMAGE_NAME = 'test_pot/4.jpg'

# Grab path to current working directory
CWD_PATH = os.getcwd()
IMAGE_SHOW = False
PATH_TO_CKPT = sys.argv[1]
PATH_TO_LABELS = sys.argv[2]
PATH_TO_IMAGE = sys.argv[3]
labelNameMap = {}
# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
if PATH_TO_CKPT is None:
    PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'saved_model')

# Path to label map file
if PATH_TO_LABELS is None:
    PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt')

# Path to image
if PATH_TO_IMAGE is None:
    PATH_TO_IMAGE = os.path.join(CWD_PATH, IMAGE_NAME)

# Number of classes the object detector can identify
if sys.argv[4] is None:
    NUM_CLASSES = 6
else:
    NUM_CLASSES = int(sys.argv[4])

if sys.argv[5] is not None:
    loaded = urllib.parse.unquote(sys.argv[5])
    loaded = json.loads(loaded)
    labelNameMap = loaded
# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)
# Load the Tensorflow model into memory.

model_c=tf.saved_model.load(PATH_TO_CKPT) #加载模型

IMAGE_PATHS = []
PATH_TO_IMAGE_DIR = PATH_TO_IMAGE
if not os.path.isfile(PATH_TO_IMAGE):
    for filename in os.listdir(PATH_TO_IMAGE):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            IMAGE_PATHS.append(os.path.join(PATH_TO_IMAGE, filename))
else:
    IMAGE_PATHS.append(PATH_TO_IMAGE)
    PATH_TO_IMAGE_DIR = os.path.dirname(PATH_TO_IMAGE)
for IMAGE_PATH in IMAGE_PATHS:
    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    print('path####### {0}'.format(IMAGE_PATH))
    image = cv2.imread(IMAGE_PATH)
    image_expanded = np.expand_dims(image, axis=0)

    # Perform the actual detection by running the model with the image as input
        #运行model
    y = model_c(image_expanded)
    boxes = y['detection_boxes']
    scores = y['detection_scores']
    classes = y['detection_classes']
    num = y['num_detections']

    # Draw the results of the detection (aka 'visulaize the results')

    # print('image params ',labelNameMap,category_index)
    for key in category_index:
        _label = category_index[key]
        _name = _label['name']
        if _name in labelNameMap:
            _label['name'] = labelNameMap[_name]
    print('image params after', category_index)
    v_res = vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.60)
    image_rec_res = []
    # print('v_res 识别数量: ', scores, classes, num, image_rec_res)
    if len(scores) > 0:
        print('v_res 识别数量: ', scores[0])
        scoresNumpy=scores[0].numpy()
        classesNumpy=classes[0].numpy()
        for idx in range(0, len(scoresNumpy)):
            if scoresNumpy[idx] > 0.60:
                image_rec_res.append(
                    {'score': float(scoresNumpy[idx]), 'id': int(classesNumpy[idx]),
                     'name': category_index[classesNumpy[idx]]['name']})
    # All the results have been drawn on image. Now display the image.
    FULL_NAME = IMAGE_PATH.split("/")
    SHOW_NAME = FULL_NAME[-1]
    with open(os.path.join(PATH_TO_IMAGE_DIR, 'result', '{0}.out'.format(SHOW_NAME)), 'w', encoding='utf8') as f:
        json.dump(image_rec_res, f, ensure_ascii=False)
    # plt.figure(SHOW_NAME)
    # plt.imshow(image)
    if IMAGE_SHOW:
        cv2.imshow(SHOW_NAME, image)
    else:
        write_path = os.path.join(PATH_TO_IMAGE_DIR, 'result', SHOW_NAME)
        print('valid write path: {0}'.format(write_path))
        cv2.imwrite(write_path, image)

    # Press any key to close the image
if IMAGE_SHOW:
    while (1):
        c = cv2.waitKey(0)
        print('loop wait esc')
        if c == 27:
            print('will close！！！！')
            cv2.destroyAllWindows()
            break

    # # Clean up
    # cv2.destroyAllWindows()
