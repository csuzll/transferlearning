# -*- coding: utf-8 -*-
import time
import copy
import tqdm
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torchnet import meter
from torchnet.utils import ResultsWriter

"""
处理给定模型的训练和验证
"""
def train(args, model, dataloaders, criterion, optimizer, scheduler, logger, epochs=25, is_inception=False):
    """
    args: 从键盘接收的参数
    model: 将被训练的模型
    dataloaders: 数据加载器
    criterion: 优化准则（loss）
    optimizer: 训练时的优化器
    scheduler: 学习率调整机制
    logger: 日志
    epochs: 训练周期数
    is_inception: 是否为inception模型的标志
    """

    # 模型保存地址
    if args.pretrained and args.feature:
        mode = "Feature_extractor" # pretrained=True, feature=True
    elif args.pretrained and not args.feature:
        mode = "Fine_tuning" # pretrained=True, feature=False
    else:
        mode = "From_scratch" # pretrained=False, feature=False
    modelpath = Path(args.output) / args.arch / mode / "model.pt"

    # 断点训练
    if (modelpath.exists()):
        state = torch.load(str(modelpath))
        epoch = state["epoch"]
        model.load_state_dict(state["model"])

        logger.info("Loading epoch {} checkpoint ...".format(epoch))
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        epoch = 0

    # save匿名函数，使用的时候就调用save(ep)
    save = lambda epoch: torch.save({
        'model':model.state_dict(),
        'epoch':epoch,
        }, str(modelpath))


    # best_model_wts = copy.deepcopy(model.state_dict())
    # best_acc = 0.0

    # since = time.time()
    # meters

    running_loss_meter = meter.AverageValueMeter() # 平均值loss
    running_corrects_meter = meter.mAPMeter() # 所有类的平均正确率
    # running_corrects = meter.ClassErrorMeter(topk=[1], accuracy=True) # 每个类的正确率
    # data_time_meter = meter.TimeMeter(unit=True)  # 测量数据加载时间
    batch_time_meter = meter.TimeMeter(unit=True) # 测量一个batch的处理时间

    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch, epochs-1))
        print("-" * 10)

        # 每个epoch重置
        running_loss_meter.reset()
        running_corrects_meter.reset()
        batch_time_meter.reset()

        # 每个epoch都有一个训练和验证阶段
        for phase in ["train", "val"]:
            if phase == "train":
                model.train() # Set model to training mode
            else:
                model.eval() # Set model to evaluate mode

            # 迭代数据
            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                # 将输入和标签放入gpu或者cpu中
                inputs = inputs.cuda() if torch.cuda.is_available() else inputs
                labels = labels.cuda() if torch.cuda.is_available() else labels

                # 零参数梯度
                optimizer.zero_grad()

                # 前向
                # track history if only in train
                with torch.set_grad_enabled(phase=="train"):
                    # inception的训练和验证有区别
                    if is_inception and phase == "train":
                        outputs, aux_outputs = model(inpus)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels) # 计算loss

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # 一次迭代的统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # 学习率调整
            if phase == "train":
                scheduler.step()

            # 1个epoch的统计
            epoch_loss = running_loss / len(dataloaders[phase].datasets)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].datasets)

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == "val":
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best val Acc: {:.4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history