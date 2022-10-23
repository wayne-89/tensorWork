from absl import app
from absl import flags
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import cv2
import sys
import os


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'PATH_TO_IMAGE_DIR', None,
    '待处理的图片文件夹地址')
flags.DEFINE_string('PATH_TO_LABELS', None,
                    '待处理的图片标注地址')
flags.DEFINE_string('NEW_PATH_TO_IMAGE_DIR', None, '处理结果的存放地址,图片标注也会输出到该文件夹下')
flags.DEFINE_string(
    'OPERATES', None, '操作项，多个请用,分割,取值范围: rotate,crop,scale,visual')
flags.DEFINE_string(
    'DEBUG_ON', 'False', '是否开启调试模式,默认关闭')



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



def compose(path,name,anns,new_path,method,params):
    DEBUG_ON = FLAGS.DEBUG_ON is not None and FLAGS.DEBUG_ON=='True'
    image = cv2.imread(os.path.join(path, name), 1)
    bbs = BoundingBoxesOnImage([
            BoundingBox(x1=int(ann[4]), y1=int(ann[5]), x2=int(ann[6]), y2=int(ann[7]),label=ann[3]) for ann in anns
        ]
        , shape=image.shape)
    w=image.shape[1]
    h=image.shape[0]
    piplines=[]
    newName=name
    if method=='rotate':
       piplines.append(iaa.Affine(rotate=params['rotate']))
       newName='r{0}_'.format(params['rotate'])+newName
    elif method=='pad':
       piplines.append(iaa.Pad(px=(0,max(w,h)-w,max(w,h)-h,0),keep_size=False,pad_mode='constant',pad_cval=0))
       newName='pad_'+newName
       w=max(w,h)
       h=max(w,h)
    elif method=='crop':
       l=int(min(w,h)/2)
       piplines.append(iaa.CropToFixedSize(width=l,height=l,position="uniform"))
       newName='crop_'+newName
       w=l
       h=l
    elif method=='scale':
       piplines.append(iaa.Pad(px=(0,w,h,0),keep_size=True,pad_mode='constant',pad_cval=0))
       newName='scale_'+newName
    elif method=='visual':
        #
        # 每个图像执行以下0到3个（不太重要）增强器。不要全部执行，因为这通常会太过强烈。
        #
        #Sometimes（0.5，…）在50%的情况下应用给定的增强器，
        #例如，Sometimes（0.5，GaussianBlur（0.3））大约每秒都会模糊图像。
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        piplines.append(
            iaa.SomeOf((1, 3),
                [
                    # 将一些图像转换为其超像素表示，每个图像采样20到200个超像素，
                    # 但不要用其平均值替换所有超像素，只替换其中的一些（p_replace）。
                    # sometimes(
                    #     iaa.Superpixels(
                    #         p_replace=(0, 1.0),
                    #         n_segments=(20, 200)
                    #     )
                    # ),
     
                    #使用不同的强度模糊每个图像
                    #高斯模糊（sigma介于0和3.0之间）
                    #平均/均匀模糊（内核大小在2x2和7x7之间）
                    #中值模糊（内核大小在3x3和11x11之间）。
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)),
                        iaa.AverageBlur(k=(2, 7)),
                        iaa.MedianBlur(k=(3, 11)),
                    ]),
     
                    # 锐化每个图像，使用介于0（无锐化）和1（完全锐化效果）之间的alpha将结果与原始图像覆盖。
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
     
                    # 与锐化相同，但用于浮雕效果。
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
     
                    # 在某些图像中搜索所有边缘或定向边缘。
                    # 然后在黑白图像中标记这些边缘，并使用0到0.7的alpha与原始图像叠加。
                    sometimes(iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0, 0.7)),
                        iaa.DirectedEdgeDetect(
                            alpha=(0, 0.7), direction=(0.0, 1.0)
                        ),
                    ])),
     
                    # 在一些图像中添加高斯噪声。在其中50%的情况下，噪声是按通道和像素随机采样的。
                    # 在其他50%的情况下，每像素采样一次（即亮度变化）。
                    iaa.AdditiveGaussianNoise(
                        loc=0, scale=(0.0, 0.05*255), per_channel=0.5
                    ),
     
                    # 要么随机删除所有像素的1%到10%（即将其设置为黑色），
                    # 要么将其放置在原始大小的2%到5%的图像上，从而导致大矩形的删除。
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.1), per_channel=0.5),
                        iaa.CoarseDropout(
                            (0.03, 0.15), size_percent=(0.02, 0.05),
                            per_channel=0.2
                        ),
                    ]),
     
                    # 以5%的概率反转每个图像的通道
                    # 这将每个像素值设置为255-v
                    # iaa.Invert(0.05, per_channel=True), # 反转颜色通道
     
                    # 为每个像素添加-10到10的值。
                    iaa.Add((-10, 10), per_channel=0.5),
     
                    # 更改图像亮度（原始值的50-150%）。
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
     
                    # 改善或恶化图像的对比度。
                    iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
     
                    #将每个图像转换为灰度，然后用随机alpha将结果与原始图像叠加。去除不同强度的颜色。
                    iaa.Grayscale(alpha=(0.0, 1.0)),
     
                    # 在某些图像中，局部移动像素（具有随机强度）。
                    sometimes(
                        iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                    ),
     
                    # 在一些图像中，局部区域的扭曲程度不同。
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
                ],
            # 按随机顺序执行上述所有增强
            random_order=True
            )
        )
        newName='visual_'+newName

    seq = iaa.Sequential(piplines)

    # Augment BBs and images.
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)

    # print coordinates before/after augmentation (see below)
    # use .x1_int, .y_int, ... to get integer coordinates
    bbs_aug = bbs_aug.clip_out_of_image()
    new_ann_list = []
    for i in range(len(bbs.bounding_boxes)):
        before = bbs.bounding_boxes[i]
        if i>=len(bbs_aug.bounding_boxes):
            break
        after = bbs_aug.bounding_boxes[i]
        if DEBUG_ON:
            print("BB %d: (%d, %d, %d, %d) -> (%d, %d, %d, %d)" % (
                i,
                before.x1_int, before.y1_int, before.x2_int, before.y2_int,
                after.x1_int, after.y1_int, after.x2_int, after.y2_int)
            )
        new_ann=[newName,w,h,after.label,after.x1_int, after.y1_int, after.x2_int, after.y2_int]
        new_ann_list.append('\n'+(",".join([str(a) for a in new_ann])))

    # image with BBs before/after augmentation (shown below)
    if DEBUG_ON:
        image_before = bbs.draw_on_image(image, size=2)
        image_after = bbs_aug.draw_on_image(image_aug, size=2, color=[0, 0, 255])
        #cv2.imwrite(os.path.join(new_path,'res', name), image_before)
        cv2.imwrite(os.path.join(new_path,'res', newName), image_after)
    cv2.imwrite(os.path.join(new_path, newName), image_aug)
    fo = open('{0}/labels.csv'.format(new_path), "a")
    fo.writelines(new_ann_list)
    fo.close()

def main(argv):
    del argv  # Unused.
    flags.mark_flag_as_required('PATH_TO_IMAGE_DIR')
    flags.mark_flag_as_required('PATH_TO_LABELS')
    flags.mark_flag_as_required('NEW_PATH_TO_IMAGE_DIR')
    PATH_TO_IMAGE_DIR = FLAGS.PATH_TO_IMAGE_DIR
    PATH_TO_LABELS = FLAGS.PATH_TO_LABELS
    NEW_PATH_TO_IMAGE_DIR = FLAGS.NEW_PATH_TO_IMAGE_DIR
    OPERATES = FLAGS.OPERATES
    os.system('rm -rf'.format(NEW_PATH_TO_IMAGE_DIR))
    os.system('mkdir -p {0}'.format(NEW_PATH_TO_IMAGE_DIR))
    os.system('mkdir -p {0}/res'.format(NEW_PATH_TO_IMAGE_DIR))
    operates=['pad']
    if OPERATES is not None:
        for e in OPERATES.strip().split(","):
            operates.append(e) 
    [cfg_map,line0] = csv_cfg_map(PATH_TO_LABELS)
    fo = open('{0}/labels.csv'.format(NEW_PATH_TO_IMAGE_DIR), "w")
    fo.writelines([line0])
    fo.close()
    for operate in operates:
        if operate not in['pad','rotate','crop','scale','visual']:
            continue
        [cfg_map,line0] = csv_cfg_map(PATH_TO_LABELS)
        b_im_name = [name for name in os.listdir(PATH_TO_IMAGE_DIR)
                     if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        b_im_path = []
        ann_list = []
        for name in b_im_name:
            b_im_path.append(os.path.join(PATH_TO_IMAGE_DIR, name))
            if name in cfg_map:
                ann_list = cfg_map[name]
                if len(ann_list) > 0:
                    if operate=='rotate':
                        compose(PATH_TO_IMAGE_DIR,name,ann_list,NEW_PATH_TO_IMAGE_DIR,operate,{"rotate":45})
                        compose(PATH_TO_IMAGE_DIR,name,ann_list,NEW_PATH_TO_IMAGE_DIR,operate,{"rotate":90})
                        compose(PATH_TO_IMAGE_DIR,name,ann_list,NEW_PATH_TO_IMAGE_DIR,operate,{"rotate":135})
                        compose(PATH_TO_IMAGE_DIR,name,ann_list,NEW_PATH_TO_IMAGE_DIR,operate,{"rotate":180})
                        compose(PATH_TO_IMAGE_DIR,name,ann_list,NEW_PATH_TO_IMAGE_DIR,operate,{"rotate":225})
                        compose(PATH_TO_IMAGE_DIR,name,ann_list,NEW_PATH_TO_IMAGE_DIR,operate,{"rotate":270})
                        compose(PATH_TO_IMAGE_DIR,name,ann_list,NEW_PATH_TO_IMAGE_DIR,operate,{"rotate":315})
                    else:
                        compose(PATH_TO_IMAGE_DIR,name,ann_list,NEW_PATH_TO_IMAGE_DIR,operate,{})
        PATH_TO_IMAGE_DIR = NEW_PATH_TO_IMAGE_DIR
        PATH_TO_LABELS=os.path.join(NEW_PATH_TO_IMAGE_DIR, 'labels.csv')

if __name__ == '__main__':
  app.run(main)
