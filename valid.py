# coding=utf-8
import os
import time
import sys
import configparser
import json
import codecs
# os.system('conda activate tensorflow1')
# init
curPath = sys.path[0]
basePath = curPath
workPath = curPath
dstPath = sys.argv[1]
labelNameMap={}
print(sys.stdin.encoding, sys.stdout.encoding)
if len(sys.argv) >= 3:
	dstImage=sys.argv[2]
else:
	dstImage=os.path.join(dstPath,"images/valid")
if len(sys.argv) >= 4:
	labelNameMap=sys.argv[3]
	print('mmmmm',labelNameMap)
	# labelNameMap = json.dumps(sys.argv[3])
cfgPath=os.path.join(dstPath,"config.conf")
labelMapPath=os.path.join(dstPath,"labelmap.pbtxt")    
baseSection="base"
labelSection="label"

cf = configparser.ConfigParser()
cf.read(cfgPath)
numClasses = cf.get(baseSection,"num_classes")
MODEL_NAME = 'inference_graph'
PATH_TO_CKPT=os.path.join(dstPath,MODEL_NAME,'frozen_inference_graph.pb')

os.system('python {0}/Object_detection_image.py {1} {2} {3} {4} {5}'.format(basePath,PATH_TO_CKPT,labelMapPath,dstImage,numClasses,labelNameMap))


# if os.path.isfile(dstImage):
#     os.system('python {0}/Object_detection_image.py {1} {2} {3} {4}'.format(basePath,PATH_TO_CKPT,labelMapPath,dstImage,numClasses))
# else:
#     cnt=0
#     for filename in os.listdir(dstImage):
#         if filename.endswith(".JPG") or filename.endswith(".jpg"):
#             cnt=cnt+1
#     for filename in os.listdir(dstImage):
#         print(filename)
#         if filename.endswith(".JPG") or filename.endswith(".jpg"):
#             filepath = os.path.join(dstImage,filename)
#             cnt=cnt-1
#             os.system('python {0}/Object_detection_image.py {1} {2} {3} {4} {5}'.format(basePath,PATH_TO_CKPT,labelMapPath,filepath,numClasses,cnt==0))






