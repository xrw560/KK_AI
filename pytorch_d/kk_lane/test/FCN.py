import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.init as init
import matplotlib.pyplot as plt

"""建立block"""


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
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=2):
        super(ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride)

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_chans, out_chans):
        super(Bottleneck, self).__init__()
        assert out_chans % 4 == 0
        self.block1 = ResBlock(in_chans, int(out_chans / 4), kernel_size=1, padding=0)
        self.block2 = ResBlock(int(out_chans / 4), int(out_chans / 4), kernel_size=3, padding=1)
        self.block3 = ResBlock(int(out_chans / 4), out_chans, kernel_size=1, padding=0)

    def forward(self, x):
        identity = x
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out += identity
        return out


def make_layers(in_channels, layer_list):
    layers = []
    for v in layer_list:
        layers += [Block(in_channels, v)]
        in_channels = v
    return nn.Sequential(*layers)


class Layer(nn.Module):
    def __init__(self, in_channels, layer_list):
        super(Layer, self).__init__()
        self.layer = make_layers(in_channels, layer_list)

    def forward(self, x):
        out = self.layer(x)
        return out


class VGG(nn.Module):
    '''
    VGG model
    '''

    def __init__(self):
        super(VGG, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=100)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.layer1 = Layer(64, [64])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = Layer(64, [128, 128])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer3 = Layer(128, [256, 256, 256, 256])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer4 = Layer(256, [512, 512, 512, 512])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer5 = Layer(512, [512, 512, 512, 512])
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        f0 = self.relu1(self.bn1(self.conv1(x)))
        f1 = self.pool1(self.layer1(f0))
        f2 = self.pool2(self.layer2(f1))
        f3 = self.pool3(self.layer3(f2))
        f4 = self.pool4(self.layer4(f3))
        f5 = self.pool5(self.layer5(f4))
        return [f3, f4, f5]


def bilinear_kernel(in_channels, out_channels, kernel_size):
    # 保证权重之和为1，使用转置卷积且stride=2时，数值中间插入0，
    factor = (kernel_size + 1) // 2
    center = kernel_size / 2
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)  # 广播；离中心越大，权重越大
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)


# x = plt.imread('1.jpg')
# print(x.shape)
# x = torch.from_numpy(x.astype('float32')).permute(2, 0, 1).unsqueeze(0)
# conv_trans = nn.ConvTranspose2d(3, 3, 4, 2, 1)
# # 将其定义为bilinear kernel
# conv_trans.weight.data = bilinear_kernel(3, 3, 4)
# y = conv_trans(x).data.squeeze().permute(1, 2, 0).numpy()
# plt.imshow(y.astype('uint8'))
# print(y.shape)
# plt.show()

class VGG_19bn_8s(nn.Module):
    def __init__(self, n_class):
        super(VGG_19bn_8s, self).__init__()
        self.encode = VGG()

        self.fc6 = nn.Conv2d(512, 4096, 7)  # padding=7
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.trans_p3 = nn.Conv2d(256, n_class, 1)
        self.trans_p4 = nn.Conv2d(512, n_class, 1)

        self.up2time = nn.ConvTranspose2d(n_class, n_class, 4, stride=2, bias=False)
        self.up4time = nn.ConvTranspose2d(n_class, n_class, 4, stride=2, bias=False)
        self.up32time = nn.ConvTranspose2d(n_class, n_class, 16, stride=8, bias=False)
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data = bilinear_kernel(n_class, n_class, m.kernel_size[0])

    def forward(self, x):
        feature_list = self.encode(x)
        p3, p4, p5 = feature_list
        f6 = self.drop6(self.relu6(self.fc6(p5)))
        f7 = self.score_fr(self.drop7(self.relu7(self.fc7(f6))))

        up2_feat = self.up2time(f7)
        h = self.trans_p4(p4)
        h = h[:, :, 5:5 + up2_feat.size()[2], 5:5 + up2_feat.size()[3]]
        h = h + up2_feat

        up4_feat = self.up4time(h)
        h = self.trans_p3(p3)
        h = h[:, :, 9:9 + up4_feat.size()[2], 9:9 + up4_feat.size()[3]]
        h = h + up4_feat

        h = self.up32time(h)
        final_scores = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()
        return final_scores


# x = torch.randn(1, 3, 256, 256)
# model = VGG_19bn_8s(21)
# model.eval()
# y = model(x)
# print(y.size())
#
# from torchsummaryX import summary
#
# summary(model, x)


class ResNet101(nn.Module):
    """
    ResNet101 model
    
    """

    def __init__(self):
        super(ResNet101, self).__init__()
        self.conv1 = Block(3, 64, 7, 3, 2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2_1 = Bottleneck(64, 256)
        self.conv2_2 = Bottleneck(256, 256)
        self.conv2_3 = Bottleneck(256, 256)
        self.layer3 = Layer(256, [512, 512, 512, 512])
        self.layer4 = Layer(512, [1024] * 23)
        self.layer5 = Layer(1024, [2048] * 3)

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2_3(self.conv2_2(self.conv2_1(self.pool1(f1))))
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        f5 = self.layer5(f4)
        return [f3, f4, f5]


class Resnet101_8s(nn.Module):

    def __init__(self, n_class):
        super(Resnet101_8s, self).__init__()
        self.encode = ResNet101()
        self.score_fr = nn.Conv2d(2048, n_class, 1)
        self.trans_p3 = nn.Conv2d(512, n_class, 1)
        self.trans_p4 = nn.Conv2d(1024, n_class, 1)

        self.up2time = nn.ConvTranspose2d(n_class, n_class, 2, stride=2, bias=False)
        self.up4time = nn.ConvTranspose2d(n_class, n_class, 2, stride=2, bias=False)
        self.up32time = nn.ConvTranspose2d(n_class, n_class, 8, stride=8, bias=False)

    def forward(self, x):
        feature_list = self.encode(x)
        p3, p4, p5 = feature_list
        f7 = self.score_fr(p5)

        up2_feat = self.up2time(f7)
        h = self.trans_p4(p4)
        h = h + up2_feat

        up4_feat = self.up4time(h)
        h = self.trans_p3(p3)
        h = h + up4_feat

        h = self.up32time(h)
        return h



# x = torch.randn(1, 3, 256, 256)
# model = Resnet101_8s(21)
# model.eval()
# y = model(x)
# print(y.size())
#
# from torchsummaryX import summary
#
# summary(model, x)
