import tensorlayer as tl
import sys
import os
import glob
import xml.etree.ElementTree as ET
import cv2
import math
import numpy as np

PATH_TO_MODEL = sys.argv[1]

os.system('rm -rf {0}/checkpoint'.format(PATH_TO_MODEL))
os.system('rm -rf {0}/ckpt*'.format(PATH_TO_MODEL))
os.system('rm -rf {0}/*.record'.format(PATH_TO_MODEL))
os.system('rm -rf {0}/train.config'.format(PATH_TO_MODEL))
os.system('rm -rf {0}/train'.format(PATH_TO_MODEL))
os.system('rm -rf {0}/training'.format(PATH_TO_MODEL))
os.system('rm -rf {0}/images/test_out'.format(PATH_TO_MODEL))
os.system('rm -rf {0}/images/train_out'.format(PATH_TO_MODEL))
os.system('rm -rf {0}/labelmap.pbtxt'.format(PATH_TO_MODEL))
os.system('rm -rf {0}/label_tflite.pbtxt'.format(PATH_TO_MODEL))
