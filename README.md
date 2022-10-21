使用的系统：Ubuntu 20.04

## 安装
1. 安装conda, 需要输入的地方输入yes
```
curl -O https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
bash Anaconda3-5.0.1-Linux-x86_64.sh
```

创建conda环境, 记得新开窗口

```conda create -n tensorWork pip python=3.5```

进入conda环境 

```source activate tensorWork```

2. 安装tensorflow `pip install tensorflow`

3. 安装其他组件
  ```shell
  conda install -c anaconda protobuf
  pip install pillow lxml Cython contextlib2 jupyter matplotlib pandas opencv-python ConfigParser nets imgaug
  ```

4. 安装物体识别
```
git clone https://github.com/tensorflow/models.git
cd models/research
protoc object_detection/protos/*.proto --python_out=.

python setup.py build

python setup.py install

vim ~/.bashrc
添加环境变量。相关路径修改成自己的
export PYTHONPATH="/home/wayne/Work/models:/home/wayne/Work/models/research:/home/wayne/Work/models/research/slim:$PYTHONPATH"
打开新窗口，重新接入
source activate tensorWork
```

5. 安装标注工具
```
git clone https://github.com/tzutalin/labelImg.git
sudo apt-get install pyqt5-dev-tools
pip install pyqt5==5.10.1 lxml==4.2.4
cd labelImg/
make qt5py3
```

## 训练
1. 进入环境

```source activate tensorWork```

2. 创建训练项目example

```mkdir tensorWork/datas/example```

3. 图片预处理
进行图片压缩

```python tensorWork/resizer.py path/to/image/dir```

第二个参数为图片所在文件夹

4. 添加图片
```
mkdir tensorWork/datas/example/images/train
mkdir tensorWork/datas/example/images/test
mkdir tensorWork/datas/example/images/valid
```
train 下放需要训练的图片 80%， 需要标注
test 下放需要训练测试的图片 20%， 需要标注
valid 下放用于验证训练结果的图片，不同于train test的图片

5. 图片标注
```python labelImg/labelImg.py```

选择对应文件夹 train  test 进行标注

6. 添加配置文件
```tensorWork/datas/example/config.conf```

配置内容
```
[base]
num_classes=2
num_steps=100
batch_size=1
learning_rate=0.0002
rotate_image=true
crop_image=false
scale_image=true
visual_image=true
restart_train=true
[label]
1=pan
2=luoshuan
```
num_classes 表示要训练的分类数量， 与 label中的数量相等
num_steps 表示训练的步数
label中为图片标注时用的label名称，按序填写
rotate_image 是否旋转图片(每45度旋转一次)
crop_image 是否剪切图片
scale_image 是否缩放图片
visual_image 是否视觉处理(噪声、模糊、扭曲等)
restart_train 是否重新开始训练

7. 进行训练
```python tensorWork/main.py tensorWork/datas/example```

8. 如果训练完成1000次后， 想继续训练至2000次，只需修改num_steps=2000，再启动训练

## 识别
1. 需要识别的图片放到tensorWork/datas/images/valid中
2. ```python tensorWork/valid.py tensorWork/datas/example```
3. esc退出
