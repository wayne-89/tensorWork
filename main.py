# coding=utf-8
import os
import time
import sys
import configparser
import tensorflow.compat.v2 as tf

from tflite_support.metadata_writers import object_detector
from tflite_support.metadata_writers import writer_utils
from tflite_support import metadata
from object_detection.utils import label_map_util
ObjectDetectorWriter = object_detector.MetadataWriter



# os.system('conda activate tensorflow1')
# init
print("###&###|train_prepare")
curPath = sys.path[0]
basePath = curPath
workPath = curPath
defaultTrainCheckPoint = os.path.join(basePath, "ssd_mobilenet_v2_320x320_coco17_tpu-8/checkpoint/ckpt-0")
dstPath = sys.argv[1]
cfgPath = os.path.join(dstPath, "config.conf")
labelMapPath = os.path.join(dstPath, "labelmap.pbtxt")
baseSection = "base"
labelSection = "label"
imageOperates=[]
cf = configparser.ConfigParser()
cf.read(cfgPath)
numClasses = cf.get(baseSection, "num_classes")
print('baseSection', baseSection)

if cf.has_option(baseSection, "rotate_image") and cf.get(baseSection, "rotate_image")=='true':
    imageOperates.append('rotate')
if cf.has_option(baseSection, "crop_image") and cf.get(baseSection, "crop_image")=='true':
    imageOperates.append('crop')
if cf.has_option(baseSection, "scale_image") and cf.get(baseSection, "scale_image")=='true':
    imageOperates.append('scale')
if cf.has_option(baseSection, "visual_image") and cf.get(baseSection, "visual_image")=='true':
    imageOperates.append('visual')

restartTrain = 'false'
if cf.has_option(baseSection, "restart_train"):
    restartTrain = cf.get(baseSection, "restart_train")
numExamples = len([name for name in os.listdir(os.path.join(dstPath, 'images/test')) if
                   name.lower().endswith((".jpg",".jpeg", ".png", ".bmp"))])
trainCheckPoint = None
try:
    trainCheckPoint = cf.get(baseSection, "train_check_point")
except Exception as ex:
    print(ex, ", use this instead: ")
if trainCheckPoint is None:
    trainCheckPoint = defaultTrainCheckPoint
    tcpfPath = os.path.join(dstPath, "training/checkpoint")
    if os.path.exists(tcpfPath):
        tcpf = open(tcpfPath, 'r')
        chkpoit = tcpf.readline()
        if chkpoit is not None:
            chkpoit = chkpoit.replace("model_checkpoint_path:", "").replace("\"", "").strip()
            print(chkpoit)
            if os.path.exists(chkpoit + '.meta'):
                trainCheckPoint = chkpoit

print("train check point use: ", trainCheckPoint)
# loss=cf.get(baseSection,"loss")
numSteps = cf.get(baseSection, "num_steps")
learningRate = cf.get(baseSection, "learning_rate")
batchSize = cf.get(baseSection, "batch_size")
if batchSize is None:
    batchSize = 1
if learningRate is None:
    learningRate = 0.0002
labels = cf.items(labelSection)
print('num_classes num_examples', numClasses, numExamples)
print('label_map', labels)

if restartTrain == 'true':
    os.system('python {0}/clean.py {1}'.format(basePath, dstPath))

# write label map
f = open(labelMapPath, 'w')
for label in labels:
    f.write('item {{\n\tid:{0}\n\tname:\'{1}\'\n\tdisplay_name:\'{1}\'\n}}\n\n'.format(label[0], label[1]))
f.close()

# generate csv
os.system('python {0}/xml_to_csv.py {1}'.format(basePath, dstPath))
# expand image to be equal
os.system('rm -rf {0}/images/train_out'.format(dstPath))
operatesArg='None'
if len(imageOperates)>0:
    operatesArg=','.join(imageOperates)
os.system('python {0}/img_enhance.py --PATH_TO_IMAGE_DIR {1}/images/train --PATH_TO_LABELS {1}/images/train_labels.csv  --NEW_PATH_TO_IMAGE_DIR {1}/images/train_out --DEBUG_ON False --OPERATES {2}'.format(basePath, dstPath,operatesArg))
# os.system('python {0}/equal_image_label.py {1}/images/train {1}/images/train_labels.csv  {1}/images/train_out'.format(basePath, dstPath))
os.system('rm -rf {0}/images/test_out'.format(dstPath))
os.system('python {0}/img_enhance.py --PATH_TO_IMAGE_DIR {1}/images/test --PATH_TO_LABELS {1}/images/test_labels.csv  --NEW_PATH_TO_IMAGE_DIR {1}/images/test_out --DEBUG_ON False --OPERATES {2}'.format(basePath, dstPath,operatesArg))
# os.system('python {0}/equal_image_label.py {1}/images/test {1}/images/test_labels.csv  {1}/images/test_out'.format(basePath, dstPath))


# rotate image
# if rotateImage == 'true':
#     os.system('python {0}/rotate_image_label.py {1}/images/train_out {1}/images/train_out/labels.csv'.format(basePath, dstPath))
#     os.system('python {0}/rotate_image_label.py {1}/images/test_out {1}/images/test_out/labels.csv'.format(basePath, dstPath))

# generate TFrecord
os.system(
    'python {0}/generate_tfrecord.py --csv_input={1}/images/train_out/labels.csv --image_dir={1}/images/train_out --output_path={1}/train.record --label_map_path={2}'.format(
        basePath, dstPath, cfgPath))
os.system(
    'python {0}/generate_tfrecord.py --csv_input={1}/images/test_out/labels.csv --image_dir={1}/images/test_out --output_path={1}/test.record --label_map_path={2}'.format(
        basePath, dstPath, cfgPath))

# generate train config
with open(os.path.join(workPath, "ssd_mobilenet_v2_default.config"), "r") as cfgFile:
    cfgData = cfgFile.read()
cfgData = cfgData.replace('${num_classes}', str(numClasses)).replace('${num_examples}', str(numExamples)).replace(
    '${train_record_path}', os.path.join(dstPath, 'train.record')).replace('${test_record_path}', os.path.join(dstPath,
                                                                                                               'test.record')).replace(
    '${label_map_path}', labelMapPath).replace('${num_steps}', numSteps).replace('${train_check_point}',
                                                                                 trainCheckPoint).replace(
    '${learning_rate}', learningRate).replace('${batch_size}', batchSize)
# write label map
dstConfigPath = os.path.join(dstPath, "train.config")
f = open(dstConfigPath, 'w')
f.write(cfgData)
f.close()
print("###&###|train_starting")
# start train
# os.system(
#     'python {0}/train.py --logtostderr --train_dir={1}/training/ --pipeline_config_path={2} --model_config_path={1}'.format(basePath, dstPath,
#                                                                                                     dstConfigPath))
trainCmd='python {0}/model_main_tf2.py --pipeline_config_path={2} --model_dir={1} --alsologtostderr --checkpoint_every_n=100'.format(
        basePath, dstPath, dstConfigPath)
if trainCheckPoint != defaultTrainCheckPoint:
    trainCmd='{0} --checkpoint_dir={1}'.format(trainCmd,os.path.join(dstPath, "training"))
print("trainCmd",trainCmd)
os.system(trainCmd)
os.system('mkdir -p {0}/training/'.format(dstPath))
os.system('cp -rf {0}/ckpt-* {0}/training/'.format(dstPath))
os.system('cp -rf {0}/checkpoint {0}/training/'.format(dstPath))
print("###&###|train_model_finish")
# clean output
os.system('rm -rf {0}/inference_graph'.format(dstPath))
# export Inference Graph
os.system(
    'python {0}/exporter_main_v2.py --input_type image_tensor --pipeline_config_path {1} --trained_checkpoint_dir {2}/training/ --output_directory {2}/inference_graph'.format(
        basePath, dstConfigPath, dstPath))
print("###&###|train_pb_export_finish")

# export tflite
tfCmd='python {0}/export_tflite_graph_tf2.py --pipeline_config_path {1} --trained_checkpoint_dir {2}/training/ --output_directory {2}/inference_graph/tf'.format(
        basePath, dstConfigPath, dstPath);
print("tfCmd",tfCmd)
os.system(tfCmd)
print("###&###|train_tflite_pb_export_finish")
converter = tf.lite.TFLiteConverter.from_saved_model('{0}/inference_graph/tf/saved_model'.format(dstPath)) # path to the SavedModel directory
tflite_model = converter.convert()
with open('{0}/inference_graph/tf/tf_model.tflite'.format(dstPath), 'wb') as f:
  f.write(tflite_model)
print("###&###|train_tflite_model_export_finish")
# Normalization parameters is required when reprocessing the image. It is
# optional if the image pixel values are in range of [0, 255] and the input
# tensor is quantized to uint8. See the introduction for normalization and
# quantization parameters below for more details.
# https://www.tensorflow.org/lite/convert/metadata#normalization_and_quantization_parameters)
# _MODEL_PATH = "yolo_without_metadata.tflite"
# write label map

_LABEL_FILE = os.path.join(dstPath, "label_tflite.pbtxt")
category_index = label_map_util.create_category_index_from_labelmap(labelMapPath)

f = open(_LABEL_FILE, 'w')
for class_id in range(1, 91):
  if class_id not in category_index:
    f.write('???\n')
    continue
  name = category_index[class_id]['name']
  f.write(name+'\n')
f.close()


_MODEL_PATH = '{0}/inference_graph/tf/tf_model.tflite'.format(dstPath)
# Task Library expects label files that are in the same format as the one below.
# _SAVE_TO_PATH = "yolo_with_metadata.tflite"
_SAVE_TO_PATH = '{0}/inference_graph/tf/tf_model_meta.tflite'.format(dstPath)

_INPUT_NORM_MEAN = 127.5
_INPUT_NORM_STD = 127.5
 
# Create the metadata writer.
writer = ObjectDetectorWriter.create_for_inference(
    writer_utils.load_file(_MODEL_PATH), [_INPUT_NORM_MEAN], [_INPUT_NORM_STD],
    [_LABEL_FILE])
 
# Verify the metadata generated by metadata writer.
print(writer.get_metadata_json())
 
# Populate the metadata into the model.
writer_utils.save_file(writer.populate(), _SAVE_TO_PATH)


print("###&###|train_finish")

