from imgaug import augmenters as iaa

# 1: 水平翻转
aug1 = iaa.Fliplr(1.0)

# 2: 上下翻转
aug2 = iaa.Flipud(1.0)

# 3: 裁剪
aug3 = iaa.Crop()

# 4: 旋转90度
aug4 = iaa.Affine(rotate=90)

# 5: 旋转180度
aug5 = iaa.Affine(rotate=180)

# 6: 旋转-90度(270度)
aug6 = iaa.Affine(rotate=-90)

# 7: 旋转10度
aug7 = iaa.Affine(rotate=10)

# 8: 旋转-10度
aug8 = iaa.Affine(rotate=-10)

# 9: 平移20个像素+旋转10度
aug9 = iaa.Affine(rotate=10, translate_px=20)

# 10: 平移20个像素+旋转-10度
aug10 = iaa.Affine(rotate=-10, translate_px=20)

# 11: 水平翻转+旋转90度
aug11 = iaa.Sequential([
    iaa.Fliplr(1.0),
    iaa.Affine(rotate=90),
    ])

# 12: 水平翻转+旋转-90度
aug12 = iaa.Sequential([
    iaa.Fliplr(1.0),
    iaa.Affine(-90)
    ])

# 13: 弹性变换
aug13 = iaa.ElasticTransformation()

# 14:  


aug.augment_images(images)