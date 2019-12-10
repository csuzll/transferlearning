import torch.nn as nn
from torchvision import models

# 设置模型参数的".requires_grad"属性
def set_parameter_requires_grad(model, feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

"""
按不同方式获取训练模型
1、use_pretrained=True, feature_extract=True(A部分迁移权重初始化固定，B部分pytorch默认初始化微调)

2、use_pretrained=True, feature_extract=False(A部分迁移权重初始化微调，B部分pytorch默认初始化微调)

3、use_pretrained=False, feature_extract=False(A部分预训练规定的初始化，B部分pytorch默认初始化微调)

4、use_pretrained=False, feature_extract=True(不允许存在)
"""
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    """
    model_name: 使用的模型名称，从此列表中选择[vgg16, resnet50, resnet101, resnext101, densenet161, inception]
    num_classes: 数据集的类别数
    feature_extract: 特征提取(True)，微调(False)
    use_pretrained: 预训练(True)，从头开始训练(False)

    """
    model_ft = None
    input_size = 0

    if model_name == "vgg16":

        """
        vgg16_bn
        重塑整个分类层，包括3个全连接层。 
        """
        model_ft = models.vgg16_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # alter classifier
        model_ft.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
            )
        input_size = 224

    elif model_name == "resnet50":
        """
        resnet50 
        重塑分类层，也就是fc层。
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet101":
        """
        resnet101
        重塑分类层，也就是fc层。
        """
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnext101":
        """
        resnext101_32x8d
        重塑分类层，也就是fc层。
        """
        model_ft = models.resnetxt101_32x8d(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "densenet161":
        """
        densenet161
        重塑分类层，也就是fc层。
        """
        model_ft = models.densenet161(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ 
        inception v3
        训练时重塑辅助输出和主输出的分类层。        
        验证和测试时只考虑主输出。
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # 处理辅助网络AuxLogits部分
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # 处理主要网络
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 229

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size