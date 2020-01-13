import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
import matplotlib.pyplot as plt


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1):
        super(ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride)

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        return out


class Bottleneck(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Bottleneck, self).__init__()
        assert out_ch % 4 == 0
        self.block1 = ResBlock(in_ch, int(out_ch / 4), kernel_size=1, padding=0)
        self.block2 = ResBlock(int(out_ch / 4), int(out_ch / 4), kernel_size=3, padding=1)
        self.block3 = ResBlock(int(out_ch / 4), out_ch, kernel_size=1, padding=0)

    def forward(self, x):
        identity = x
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out += identity
        return out


class DownBottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2):
        super(DownBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, stride=stride)
        self.block1 = ResBlock(in_ch, int(out_ch / 4), kernel_size=1, padding=0, stride=stride)
        self.block2 = ResBlock(int(out_ch / 4), int(out_ch / 4), kernel_size=3, padding=1)
        self.block3 = ResBlock(int(out_ch / 4), out_ch, kernel_size=1, padding=0)

    def forward(self, x):
        identity = self.conv1(x)
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out += identity
        return out


def make_layers(in_channels, layer_list):
    layers = []
    layers += [DownBottleneck(in_channels, layer_list[0])]
    in_channels = layer_list[0]
    for v in layer_list[1:]:
        layers += [Bottleneck(in_channels, v)]
        in_channels = v
    return nn.Sequential(*layers)


class Layer(nn.Module):
    def __init__(self, in_ch, layer_list):
        super(Layer, self).__init__()
        self.layer = make_layers(in_ch, layer_list)

    def forward(self, x):
        out = self.layer(x)
        return out


class ResNet101(nn.Module):
    def __init__(self):
        super(ResNet101, self).__init__()
        self.conv1 = Block(3, 64, 7, 3, 2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.conv2_1 = DownBottleneck(64, 256, stride=1)
        self.conv2_2 = Bottleneck(256, 256)
        self.conv2_3 = Bottleneck(256, 256)
        self.layer3 = Layer(256, [512] * 2)
        self.layer4 = Layer(512, [1024] * 23)
        self.layer5 = Layer(1024, [2048] * 3)

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2_3(self.conv2_2(self.conv2_1(f1)))
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        f5 = self.layer5(f4)
        return [f2, f3, f4, f5]



net = ResNet101()
x = torch.ones((1, 3, 224, 224))
from torchsummaryX import summary

summary(net, x)
