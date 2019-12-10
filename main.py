# -*- coding: utf-8 -*-
import argparse
import json
import sys
import random
import warnings
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from model.model import initialize_model
from train.train import train
from utils.util import init_logger

# 可选模型
model_names = ["vgg16", "resnet50", "resnet101", "resnext101", "densenet161", "inception"]

# 接收bool变量字符串转换为Bool变量
def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"

parser = argparse.ArgumentParser(description='PyTorch ChestData Training')
arg = parser.add_argument

# 数据集地址(--data transferlearing/chest_data)
arg("--data", metavar="DIR", type=str, default="chest_data", help="path to dataset.") 
# 模型名称(--arch vgg16)
arg("--arch", metavar="ARCH", choices=model_names, help="model architecture: " + " | ".join(model_names) + " (default: vgg16)") 
# 特征提取布尔值(--feature False即为False)
arg("--feature", default="True", type=boolean_string, help="train feature_extract or not.")
# 预训练布尔值(--pretrained False即为False)
arg("--pretrained", default="True", type=boolean_string, help="train pretrained or not.")
# 数据加载时工作进程数(--workers 16)
arg("--workers", type=int, default=16, metavar="N", help="number of data loading workers (default: 16)")
# 训练周期数(--epochs 25)
arg("--epochs", type=int, default=25, metavar='N', help="number of total epochs to run.")
# 数据批大小(--batch-size 16)(不用考虑除不尽，如果最后剩下不完全的batch,不丢弃。)
arg("--batch-size", type=int, default=16, metavar="N", help="mini-batch size (default: 16), this is the total batch size of all GPUs on the current node when using Data Parallel")
# 学习率(--lr 0.001)
arg("--lr", type=float, default=0.001, metavar="LR", help="initial learning rate.", dest="lr")
# 使用的GPU设备(--device-ids 0,1)
arg("--device-ids", type=str, default="0", help="For example 0,1 to run on two GPUs.")
# 初始化种子随机数(--seed 42)
arg("--seed", default=42, type=int, help="seed for initializing training.")

# arg("--weight-decay", default=1e-4, type=float, metavar="W", help="weight decay (default: 1e-4)")
# arg("--print-freq", default=10, type=int, metavar="N", help="print frequency (default: 10)")

# 模型保存位置(--output output)
arg("--output", default="output", help="model and test result storage location.")

def main():
    # 只能在命令行模式下运行
    args = parser.parse_args()

    # 创建模型保存的地址
    if args.pretrained and args.feature:
        mode = "Feature_extractor" # pretrained=True, feature=True
    elif args.pretrained and not args.feature:
        mode = "Fine_tuning" # pretrained=True, feature=False
    else:
        mode = "From_scratch" # pretrained=False, feature=False
    modelpath = Path(args.output) / args.arch / mode
    modelpath.mkdir(exist_ok=True, parents=True)

    # 初始化日志
    logger = init_logger(log_name=args.arch, log_path=str(modelpath))
    if args.seed is not None:
        """
        设置随机种子用来保证模型初始化的参数一致
        同时pytorch中的随机种子也能够影响dropout的作用
        """
        random.seed(args.seed) 
        torch.manual_seed(args.seed) # 设置随机种子，保证实验的可重复性
        cudnn.deterministic = True # 保证重复性
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.device_ids is not None:
        print("Use GPU: {} for training".format(args.device_ids))
    
    # 创建模型
    model_ft = initialize_model(args.arch, num_classes=2, feature_extract=args.feature, use_pretrained=args.use_pretrained)
    
    # gpu实现模型parallel
    if torch.cuda.is_available():
        if args.device_ids:
            device_ids = list(map(int, args.device_ids.split(','))) # [0,1,2,...]
        else:
            device_ids=None
        model_ft = nn.DataParallel(model_ft, device_ids=device_ids).cuda() # 模型并行
    else:
        raise SyntaxError("GPU device not found")

    # 开启benchmark，加速训练
    cudnn.benchmark = True

    # 数据扩充和训练标准化
    if args.arch == "inception":
        input_size = 229
    else:
        input_size = 224

    """
    train: 随机裁剪输出input_size，50%的概率水平翻转，转换为Tensor，标准化
    val: 中心裁剪输出input_size，转换为Tensor，标准化
    """
    data_transforms = {
        "train":transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        "val":transforms.Compose([
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
    }

    print("Initializing Datasets and Dataloaders...")

    # 创建训练和验证数据集
    data_path = Path(args.data)
    image_datasets = {x: datasets.ImageFolder(str(data_path / x), data_transforms[x]) for x in ["train","val"]}
    # pin_memory=True可以将dataloader放到固定内存中
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=torch.cuda.is_available()) for x in ["train","val"]}

    print("num train = {}, num_val = {}".format(len(dataloaders["train"].dataset), len(dataloaders["val"].dataset)))

    # 将args写入json
    modelpath.joinpath("params.json").write_text(json.dumps(vars(args), indent=True, sort_keys=True))

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 收集要更新的参数,由于前面将model并行了，模型参数名称都加上了module
    if args.feature:
        params_to_update = []
        for name, param in mode_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                # print("\t", name)
    else:
        params_to_update = model_ft.parameters()

    # 优化器
    optimizer = torch.optim.SGD(params_to_update, lr=args.lr, momentum=0.9)

    # 学习率调整器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    logger.info("training model...")
    train(args = args, 
        model = model_ft, 
        dataloaders = dataloaders,
        criterion = criterion,
        optimizer = optimizer, 
        scheduler = scheduler,
        logger = logger,
        epochs = args.epochs,
        is_inception = (args.arch=="inception")
        )

if __name__ == '__main__':
    main()