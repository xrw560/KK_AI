import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, pad=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=pad, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """
    resnet 50 层以下
    """

    def __init__(self, in_planes, planes, stride=1, norm_layer=None, downsample=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d  # 如果没有自定义的BN，则使用系统自带的
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample  # 调整feature map的维度
        self.stride = stride

    def forward(self, x):
        identity = x  # 保存输入，和FCN类似

        out = self.conv1(x)
        out = self.bn1(out)  # batch normalization
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample
        out += identity
        out = self.relu(out)

        return out


class BottleBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, norm_layer=None, downsample=None):
        super(BottleBlock, self).__init__()
        if norm_layer:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)

        self.conv3 = conv1x1(planes, in_planes)
        self.bn3 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x  # same input

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, nnum_class=1000, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3,
                               bias=False)  # bias=False由于后面是BN层，因此不需要bias，也可以节省存储空间
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # (1,1) is global average pooling
        self.fc = nn.Linear(512, nnum_class)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample, norm_layer))
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, norm_layer=norm_layer))
        return nn.Sequential(*layers)  # 把模块拼装成一个Sequential层，调用函数直接可以加入到网络中

    def forwad(self, x):
        # conv1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # conv2 - conv5
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)  # use one fc layer
        x = self.fc(x)

        return x


def resnet34(pretrained=False):
    return ResNet('resnet18', BasicBlock, [3, 4, 6, 3])


def resnet50():
    return ResNet('resnet50', BottleBlock, [3, 4, 6, 3])


def resnet101():
    return ResNet('resnet101', BottleBlock, [3, 4, 23, 3])
