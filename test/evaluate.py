"""
根据混淆矩阵(p患病，n未患病)：
1、灵敏性Sensitivity（召回率Recall）: sensitivity = TPR = tp/(tp+fn) 真实患癌并预测出患癌的人数在真实患癌总人数中的占比。
2、特异性Specificity: specificity = TNR = tn/(fp+tn) = 1-FPR 无病并预测出无病的人数在真实无病总人数中的占比。
3、精准率Precision: precision = tp/(tp+fp) 真实患癌并预测出患癌的人数在所有预测出患癌人数中的占比。
4、F1 Score: F1 = （2 * precision * recall) / (precision * recall)
5、假阳性率: FPR = fp/(tn+fp) = 1 - specificity
"""

# 精准率
def precision(tp, fp):
    try:
        return tp / (tp + fp) # 避免分母为0报错
    except:
        return 0.0

# 灵敏性，召回率
def recall(tp, fp):
    try:
        return tp / (tp + fn) # 避免分母为0报错
    except:
        return 0.0

# 特异性
def specificity(tn, fp):
    try: 
        return tn / (fp +tn) # 避免分母为0报错
    except:
        return 0.0

# f1 score
def f1_score(precision, recall):
    try:
        return 2 * precision * recall / (precision * recall)
    except:
        return 0.0

# 绘制ROC曲线（横轴fpr, 纵轴tpr）
# plt.plot(fpr, tpr)

from pathlib import Path
import pickle as pk

if __name__ == '__main__':
    
    # 读取文件
    paths = ""
    with open(paths, "rb") as f:
        data = pk.load(f)