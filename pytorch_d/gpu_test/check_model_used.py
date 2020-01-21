import numpy as np

import torch
from torch import nn

model = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    nn.ReLU(inplace=True),
    nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
    nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
)
para = sum([np.prod(list(p.size())) for p in model.parameters()])
type_size = 4
print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

# model是我们加载的模型
# input是实际中投入的input（Tensor）变量

input = torch.rand((1, 3, 224, 224))
# 利用clone()去复制一个input，这样不会对input造成影响
input_ = input.clone()
# 确保不需要计算梯度，因为我们的目的只是为了计算中间变量而已
input_.requires_grad_(requires_grad=False)

mods = list(model.modules())
out_sizes = []

for i in range(1, len(mods)):
    m = mods[i]
    # 注意这里，如果relu激活函数是inplace则不用计算
    if isinstance(m, nn.ReLU):
        if m.inplace:
            continue
    out = m(input_)
    out_sizes.append(np.array(out.size()))
    input_ = out

total_nums = 0
for i in range(len(out_sizes)):
    s = out_sizes[i]
    nums = np.prod(np.array(s))
    total_nums += nums
# 打印两种，只有 forward 和 foreward、backward的情况
print('Model {} : intermedite variables: {:3f} M (without backward)'
      .format(model._get_name(), total_nums * type_size / 1000 / 1000))
# print('Model {} : intermedite variables: {:3f} M (with backward)'
#       .format(model._get_name(), total_nums * type_size * 2 / 1000 / 1000))
