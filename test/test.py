# -*- coding: utf-8 -*-
import sys
import args
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
import torchvision.models as models, datasets, transforms
from model.model import initialize_model
from torchnet import meter
from torchnet.utils import ResultsWriter

# 获取测试模型
def get_testmodel(model_path, model_name, num_classes):
    model = initialize_model(model_name=model_name,
        num_classes=num_classes,
        feature_extract=False,
        use_pretrained=False)

    state = torch.load(str(model_path))
    # 去掉模型参数关键字中的module
    state = {key.replace("module.", ""):value for key,value in state["model"].items()}
    # 加载模型参数
    model.load_state_dict(state)
    # 放入GPU中测试（因为模型也是在GPU中训练的）
    if torch.cuda.is_available():
        model.cuda()

    # 进入验证模式
    model.eval()

    return model

# 测试函数
def test(model, dataloader, num_workers, batch_size, resultpath):
    print("num test = {}".format(len(dataloader.dataset)))

    """
    测试指标：
    1、 准确率(Accuracy): 模型预测正确样本数占总样本数的比例。test_acc
    2、 各个类的精度: 模型对各个类别的预测准确率。
    3、 AUC
    4、 混淆矩阵: 用于计算各种指标（包括灵敏性，特异性等）
    """
    # 整个测试数据集的准确率
    test_acc = meter.ClassErrorMeter(topk=[1], accuracy=True)
    # 每一类的精度
    test_ap = meter.APMeter() 
    # AUC指标，AUC要求输入样本预测为正例的概率
    """根据我的数据集文件命名，0表示阴性，1表示阳性（即1表示正例）"""
    test_auc = meter.AUCMeter() 
    # 混淆矩阵
    test_conf = meter.ConfusionMeter(k=2, normalized=False)

    result_writer = ResultsWriter(str(resultpath), overwrite=False)

    with torch.no_grad():

        for inputs, labels in tqdm(dataloader, desc="Test"):

            # inputs[B,C,H,W]
            inputs = inputs.cuda() if torch.cuda.is_available() else inputs
            # labes[B,numclasses]
            labels = labels.cuda() if torch.cuda.is_available() else labels

            # outputs[B,numclasses]
            outputs = model(inputs)

            # 计算指标
            pred_proc = F.softmax(outputs.detach(), dim=1)
            test_acc.add(pred_proc, labels.detach())
            test_ap.add(pred_proc, labels.detach())
            # 取出正例即1（患病）的概率
            test.auc.add(pred_proc[:1], labels.detach())
            test_conf.add(pred_proc, labels.detach())

    # 记录保存, 便于evaluate.py计算和画图一些结果
    result_writer.update("test", 
        {"acc": test_acc.value(), 
         "ap":test_ap.value(), 
         "test_auc":test_auc.value()[0],
         "test_tpr":test_auc.value()[1],
         "test_fpr":test_auc.value()[2],
         "test_conf":test_conf.value()
         })

    return test_acc, test_ap, test_auc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument()

    model_names = ["vgg16", "resnet50", "resnet101", "resnext101", "densenet161", "inception"]

    # 测试数据集地址(--data transferlearning)
    arg("--data", metavar="DIR", type=str, default="chest_data", help="path to dataset")
    # 模型名称(--arch vgg16)
    arg("--arch", metavar="ARCH", choice=model_names, help="model architecture: " + " | ".join(model_names) + "(default: vgg16)")
    # 模型地址(--model-path 地址)
    arg("--model-path", type="str", help="path to model folder")
    # 数据加载时工作进程数(--workers 4)
    arg("--workers", type=int, default=4, metavar="N", help="number of data loading workers (default: 4)")
    # 数据批大小(--batch-size 2)
    arg("--batch-size", type=int, default=2, metavar="N", 
        help="mini-batch size (default: 2), this is the total batch size of all GPUs on the current node when using Data Parallel")

    model_path = Path(args.model_path)
    # 模型地址不是一个文件 或者模型地址与模型架构不一致退出
    if not model_path.is_file() or model_path.parent.parent.name != args.arch:
        print("Please check you command")
        sys.exit()

    # 获取测试模型
    model = get_testmodel(args.model_path, args.arch, 2)

    # 数据扩充和训练标准化
    if args.arch == "inception":
        input_size = 229
    else:
        input_size = 224
    """中心裁剪输出input_size，转换为Tensor，标准化"""
    test_transforms = transforms.Compose([
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    print("Initializing Datasets and Dataloaders...")
    # 创建测试数据集
    datapath = Path(args.data)
    testdataset = datasets.ImageFolder(str(datapath / "test"), test_transforms)
    dataloader = DataLoader(dataset=testdataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=torch.cuda.is_available()
        )

    # 输出结果保存地址
    resultpath = model_path.parent / "test_result.pkl"

    # 执行测试
    test_acc, test_ap = test(model, dataloader, args.num_workers, args.batch_size, resultpath)
    print("test_acc", test_acc.value())
    print("test_ap", test_ap.value())
    print("test_auc", test_auc.value()[0])