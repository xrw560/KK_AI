import time
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import models
from tensorboardX import SummaryWriter
from torchviz import make_dot
import sys

sys.path.append("..")
import d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if stride != 1 or in_channels != out_channels:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.resnet_block1 = resnet_block(64, 64, 2, first_block=True)
        self.resnet_block2 = resnet_block(64, 128, 2)
        self.resnet_block3 = resnet_block(128, 256, 2)
        self.resnet_block4 = resnet_block(256, 512, 2)
        self.global_avg_pool = d2l.GlobalAvgPool2d()
        self.fc = nn.Sequential(d2l.FlattenLayer(), nn.Linear(512, 10))

    def forward(self, x):
        h = x
        h = self.relu1(self.bn1(self.conv1(h)))
        h = self.pool1(h)
        h = self.resnet_block1(h)
        h = self.resnet_block2(h)
        h = self.resnet_block3(h)
        h = self.resnet_block4(h)
        h = self.global_avg_pool(h)
        h = self.fc(h)
        return h


class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.resnet_block1 = resnet_block(64, 64, 3, first_block=True)
        self.resnet_block2 = resnet_block(64, 128, 4)
        self.resnet_block3 = resnet_block(128, 256, 6)
        self.resnet_block4 = resnet_block(256, 512, 3)
        self.global_avg_pool = d2l.GlobalAvgPool2d()
        self.fc = nn.Sequential(d2l.FlattenLayer(), nn.Linear(512, 10))

    def forward(self, x):
        h = x
        h = self.relu1(self.bn1(self.conv1(h)))
        h = self.pool1(h)
        h = self.resnet_block1(h)
        h = self.resnet_block2(h)
        h = self.resnet_block3(h)
        h = self.resnet_block4(h)
        h = self.global_avg_pool(h)
        h = self.fc(h)
        return h


net = ResNet34()
X = torch.rand((1, 1, 224, 224))
vis_graph = make_dot(net(X), params=dict(net.named_parameters()))
vis_graph.view()
# with SummaryWriter(comment='resnet18') as w:
#     w.add_graph(net, input_to_model=X)
for name, layer in net.named_children():
    X = layer(X)
    print(name, "output shape: ", X.shape)

# batch_size = 256
# import utils
#
# train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96, root=utils.get_fashion_mnist_path())
# lr, num_epochs = 0.001, 5
# optimizer = torch.optim.Adam(net.parameters(), lr=lr)
# d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
