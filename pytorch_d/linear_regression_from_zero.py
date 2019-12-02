import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random


def use_svg_display():
    # 用矢量图显示
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(6.5, 5.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i:min(i + batch_size, num_examples)])
        yield features.index_select(0, j), labels.index_select(0, j)


def linreg(X, w, b):
    """
    定义模型
    @param X: 输入
    @param w: 权重
    @param b: 偏置
    @return: 模型
    """
    return torch.mm(X, w) + b


def squared_loss(y_hat, y):
    """
    损失函数
    @param y_hat: 预测值
    @param y: 真实值
    @return: 损失
    """
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


def sgd(params, lr, batch_size):
    """
    优化算法：小批量梯度下降算法
    :param params:
    :param lr:
    :param batch_size:
    :return:
    """
    for param in params:
        param.data -= lr * param.grad / batch_size


if __name__ == "__main__":
    """1. 生成数据集"""
    num_inputs = 2
    num_examples = 1000
    true_w = [2, -3.4]
    true_b = 4.2
    features = torch.randn(num_examples, num_inputs, dtype=torch.float32)
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)
    # print(features[0], labels[0])
    set_figsize()
    # plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
    # plt.show()

    """2. 读取数据"""

    """3. 初始化模型参数"""
    w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
    b = torch.zeros(1, dtype=torch.float32)
    w.requires_grad_(requires_grad=True)
    b.requires_grad_(requires_grad=True)

    """训练模型"""
    batch_size = 10
    lr = 0.03
    num_epochs = 3
    net = linreg
    loss = squared_loss

    for epoch in range(num_epochs):  # 训练模型一共需要num_epochs个迭代周期
        # 在每一个迭代周期中，会使用训练数据集中所有样本一次(假设样本数能够被批量大小整除)
        # X 和y 分别是小批量样本的特征和标签
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y).sum()  # l是有关小批量X和y的损失
            l.backward()  # 小批量的损失对模型参数求梯度
            sgd([w, b], lr, batch_size)  # 使用小批量随机梯度下降迭代模型参数

            # 梯度清零
            w.grad.data.zero_()
            b.grad.data.zero_()

        train_l = loss(net(features, w, b), labels)
        print("epoch %d, loss %f" % (epoch + 1, train_l.mean().item()))

    print(true_w, '\n', w)
    print(true_b, '\n', b)
