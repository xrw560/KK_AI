import torch
from torch import nn
import torch as t
from torch.nn import Module
from torch import nn


class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 3, padding=1)
        self.conv2 = nn.Conv2d(3, 3, 3, padding=1)
        self.maxpooling = nn.MaxPool2d(2, 2)
        self.trans_conv = nn.ConvTranspose2d(3, 16, 3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpooling(x)
        x = self.trans_conv(x)
        return x


model = Net()
x = t.ones((1, 3, 12, 12))
print(model(x).size())

# """目的：先实现卷积，再实现转置卷积，输入与输出大小不变"""
# """method:1"""
# input = torch.randn(1, 16, 12, 12)
# downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
# upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
# h = downsample(input)
# print(h.size())
# output = upsample(h, output_size=input.size())
# print(output.size())
#
# """method:2"""
# input2 = torch.randn(1, 16, 12, 12)
# downsample2 = nn.Conv2d(16, 16, 3, stride=2, padding=1)
# upsample2 = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1, output_padding=1)
# h2 = downsample2(input2)
# print(h2.size())
# output2 = upsample2(h2)
# print(output2.size())
