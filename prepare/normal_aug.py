import os
import cv2
from tqdm import tqdm
from imgaug import augmenters as iaa

"""定义14种扩增方法"""
# 1: 水平翻转
aug1 = iaa.Fliplr(1.0)

# 2: 上下翻转
aug2 = iaa.Flipud(1.0)

# 3: 按10%的比例裁剪
aug3 = iaa.Crop(percent=0.05, keep_size=False)

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

if __name__ == '__main__':

    # 原始normal图片的路径txt文件
    filetxt = "F:\\data\\chest_pro_data\\normal.txt"
    # 增强后的存放目录
    augdir = "F:\\data\\chest_pro_data\\normal_aug"

    # 按6个一个batch扩增图片
    with open(filetxt, "r") as f:
        lines = f.read().splitlines()
        # 将文件按6个分为一批
        fu = lambda s, step:[s[i:i+step] for i in range(0, len(s), step)]
        lines_six = fu(lines, 6)

        # load images with different sizes
        for six_line in lines_six:
            # 读取为[(Hi,Wi,C=3)]的彩色图,（灰度图单纯复制3遍）
            images_different_sizes = [cv2.imread(line) for line in six_line]

            # different augment them as one batch
            for i in tqdm(range(len(auglist))):
                images_aug = auglist[i].augment_images(images_different_sizes)

                # 保存
                for j in range(6):
                    new_basename = os.path.basename(six_line[j]).split(".")[0] + "_" + str(i+1).zfill(2) + ".jpg"
                    new_path = os.path.join(augdir, new_basename)
                    cv2.imwrite(new_path, images_aug[j])