# coding=utf-8
import os
import time
import sys
import configparser
# os.system('conda activate tensorflow1')
# init
print("###&###|train_prepare")
curPath = sys.path[0]
basePath = curPath
workPath = curPath
defaultTrainCheckPoint = os.path.join(basePath,"faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt")
dstPath=sys.argv[1]
cfgPath=os.path.join(dstPath,"config.conf")	
labelMapPath=os.path.join(dstPath,"labelmap.pbtxt")	
baseSection="base"
labelSection="label"

cf = configparser.ConfigParser()
cf.read(cfgPath)
numClasses = cf.get(baseSection,"num_classes")
numExamples = len([name for name in os.listdir(os.path.join(dstPath,'images/test')) if name.endswith(".JPG")])
trainCheckPoint = None
try:
	trainCheckPoint = cf.get(baseSection,"train_check_point")
except Exception as ex:
	print(ex,", use this instead: ")
if trainCheckPoint is None:
	trainCheckPoint = defaultTrainCheckPoint
	tcpfPath=os.path.join(dstPath,"training/checkpoint")
	if os.path.exists(tcpfPath):
		tcpf = open(tcpfPath,'r')
		chkpoit=tcpf.readline()
		if chkpoit is not None:
			chkpoit=chkpoit.replace("model_checkpoint_path:","").replace("\"","").strip()
			print(chkpoit)
			if os.path.exists(chkpoit+'.meta'):
				trainCheckPoint=chkpoit	
print("train check point use: ", trainCheckPoint)
# loss=cf.get(baseSection,"loss")
numSteps=cf.get(baseSection,"num_steps")
labels=cf.items(labelSection)
print('num_classes num_examples',numClasses,numExamples)
print('label_map',labels)

# write label map
f = open(labelMapPath,'w')
for label in labels:
	f.write('item {{\n\tid:{0}\n\tname:\'{1}\'\n}}\n\n'.format(label[0],label[1]))
f.close()

# generate csv
os.system('python {0}/xml_to_csv.py {1}'.format(basePath,dstPath))

# generate TFrecord
os.system('python {0}/generate_tfrecord.py --csv_input={1}/images/train_labels.csv --image_dir={1}/images/train --output_path={1}/train.record --label_map_path={2}'.format(basePath,dstPath,cfgPath))
os.system('python {0}/generate_tfrecord.py --csv_input={1}/images/test_labels.csv --image_dir={1}/images/test --output_path={1}/test.record --label_map_path={2}'.format(basePath,dstPath,cfgPath))

# generate train config
with open (os.path.join(workPath,"faster_rcnn_inception_default.config"), "r") as cfgFile:
	cfgData = cfgFile.read()
cfgData = cfgData.replace('${num_classes}',str(numClasses)).replace('${num_examples}',str(numExamples)).replace('${train_record_path}',os.path.join(dstPath,'train.record')).replace('${test_record_path}',os.path.join(dstPath,'test.record')).replace('${label_map_path}',labelMapPath).replace('${num_steps}',numSteps).replace('${train_check_point}',trainCheckPoint)
# write label map
dstConfigPath=os.path.join(dstPath,"train.config")
f = open(dstConfigPath,'w')
f.write(cfgData)
f.close()
print("###&###|train_starting")
# start train
os.system('python {0}/train.py --logtostderr --train_dir={1}/training/ --pipeline_config_path={2}'.format(basePath,dstPath,dstConfigPath))
# clean output
os.system('rm -rf {0}/inference_graph'.format(dstPath))
# export Inference Graph
os.system('python {0}/export_inference_graph.py --input_type image_tensor --pipeline_config_path {1} --trained_checkpoint_prefix {2}/training/model.ckpt-{3} --output_directory {2}/inference_graph'.format(basePath,dstConfigPath,dstPath,numSteps))
print("###&###|train_finish")



