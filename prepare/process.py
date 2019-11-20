import os
import cv2
from imgaug import augmenters as iaa


"""定义14种扩增方法"""
# 1: 水平翻转
aug1 = iaa.Fliplr(1.0)

# 2: 上下翻转
aug2 = iaa.Flipud(1.0)

# 3: 按10%的比例裁剪
aug3 = iaa.Crop(percent=0.1, keep_size=False)

# 4: 旋转180度
aug4 = iaa.Affine(rotate=180)

# 5: 旋转10度
aug5 = iaa.Affine(rotate=10)

# 6: 旋转-10度
aug6 = iaa.Affine(rotate=-10)

# 7: 弹性变换
aug7 = iaa.ElasticTransformation(alpha=50.0, sigma=5.0)

# 8: 按10%的比例裁剪+水平翻转
aug8 = iaa.Sequential([
    iaa.Crop(percent=0.1, keep_size=False),
    iaa.Fliplr(1.0)
    ])

# 9: 按10%的比例裁剪+垂直翻转
aug9 = iaa.Sequential([
    iaa.Crop(percent=0.1, keep_size=False),
    iaa.Flipud(1.0)
    ])

# 10: 按10%的比例裁剪+旋转180度
aug10 = iaa.Sequential([
    iaa.Crop(percent=0.1, keep_size=False),
    iaa.Affine(rotate=180)
    ])

# 11: 水平翻转+旋转180度
aug11 = iaa.Sequential([
    iaa.Crop(percent=0.1, keep_size=False),
    iaa.Affine(rotate=180)
    ])

# 12: 垂直翻转+旋转180度
aug12 = iaa.Sequential([
    iaa.Crop(percent=0.1, keep_size=False),
    iaa.Affine(rotate=180)
    ])

# 13: 按10%的比例裁剪+水平翻转+旋转180度
aug13 = iaa.Sequential([
    iaa.Crop(percent=0.1, keep_size=False),
    iaa.Fliplr(1.0),
    iaa.Affine(rotate=180)
    ])

# 14: 按10%的比例裁剪+垂直翻转+旋转180度
aug14 = iaa.Sequential([
    iaa.Crop(percent=0.1, keep_size=False),
    iaa.Flipud(1.0),
    iaa.Affine(rotate=180)
    ])

auglist = [aug1,aug2,aug3,aug4,aug5,aug6,aug7,aug8,aug9,aug10,aug11,aug12,aug13,aug14]


# 读取normal文件名
with open("F:\\data\\chest_pro_data\\normal.txt", "r") as f:
    lines = f.readlines() # 读取全部内容 ，并以列表方式返回
    # 将文件按6个分为一批
    f = lambda s, step:[s[i:i+step] for i in range(0, len(s), step)]
    lines_six = f(lines, 6)




# 将一张图片增强为14张,并保存
def augment_one(imagepath):
    # 读取为(H,W,C=3)的彩色图,（灰度图单纯复制3遍）
    src_img = cv2.imread(imagepath)
    for i,aug in enumerate(auglist):
        image_aug = aug.augment_image(src_img)





aug.augment_images(images)