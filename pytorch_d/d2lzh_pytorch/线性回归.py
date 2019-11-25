import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
import torch.utils.data as Data
from torch import nn

torch.manual_seed(1)
print(torch.__version__)
torch.set_default_tensor_type('torch.FloatTensor')


class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    def forward(self, x):
        y = self.linear(x)
        return y


if __name__ == "__main__":
    """1. 生成数据集"""
    num_inputs = 2
    num_examples = 1000
    true_w = [2, -3.4]
    true_b = 4.2
    features = torch.randn(num_examples, num_inputs, dtype=torch.float32)
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)

    """2. 读取数据"""
    batch_size = 10
    # 将训练数据的特征和标签组合
    dataset = Data.TensorDataset(features, labels)
    # 随机读取小批量
    data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

    """ 3. 定义模型"""
    net = nn.Sequential(
        LinearNet(num_inputs)
        # 此处还可以传入其他层
    )

    """4. 初始化模型参数"""
    from torch.nn import init

    init.normal_(net[0].linear.weight, mean=0, std=0.01)
    init.constant_(net[0].linear.bias, val=0)  # 也可以直接修改bias的data: net[0].bias.data.fill_(0)

    """5. 定义损失函数"""
    loss = nn.MSELoss()

    """6. 定义优化算法"""
    import torch.optim as optim

    optimizer = optim.SGD(net.parameters(), lr=0.03)

    num_epochs = 3
    for epoch in range(1, num_epochs + 1):
        for X, y in data_iter:
            output = net(X)
            l = loss(output, y.view(-1, 1))
            optimizer.zero_grad()  # 梯度清零，等价于net.zero_grad()
            l.backward()
            optimizer.step()
        print('epoch %d, loss: %f' % (epoch, l.item()))
    dense = net[0]
    print(true_w, dense.linear.weight)
    print(true_b, dense.linear.bias)
