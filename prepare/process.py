import os
import cv2
from multiprocessing import Pool
from functools import partial

# 数据处理步骤
def pro(id, filepaths):
    """
    path: 图片路径
    """
    srcimg = cv2.imread(filepaths[id], cv2.IMREAD_GRAYSCALE)

    # 1、对比度增强
    srcimg_equalize = cv2.equalizeHist(srcimg)

    # 2、缩放至256 * 256 
    resize_img = cv2.resize(srcimg_equalize, (256,256), cv2.INTER_AREA)

    # 3、变换为3通道
    bgr_img = cv2.cvtColor(resize_img, cv2.COLOR_GRAY2BGR)

    # 4、保存
    cv2.imwrite(filepaths[id], bgr_img)

# 处理所有数据
def preprocess():
    train_abnormal_dir = r"F:\data\chest_data\train\abnormal"
    train_normal_dir = r"F:\data\chest_data\train\normal"

    val_abnormal_dir = r"F:\data\chest_data\val\abnormal"
    val_normal_dir = r"F:\data\chest_data\val\normal"

    test_abnormal_dir = r"F:\data\chest_data\test\abnormal"
    test_normal_dir = r"F:\data\chest_data\test\normal"

    procdir = {0:train_abnormal_dir, 1:train_normal_dir, 2:val_abnormal_dir,
               3:val_normal_dir, 4:test_abnormal_dir, 5:test_normal_dir}

    print("Start preprocessing chest_data......")
    # 开启线程池
    pool = Pool()

    # 6份数据
    for i in range(6):
        print("process " + procdir[i].split("\\",2)[-1])
        filepaths = [os.path.join(procdir[i], file) for file in os.listdir(procdir[i])]

        # 函数修饰
        partial_pro = partial(pro, filepaths=filepaths)

        N = len(filepaths)
        _ = pool.map(partial_pro, range(N))
    pool.close() # 关闭线程池
    pool.join()

if __name__ == '__main__':
    preprocess()