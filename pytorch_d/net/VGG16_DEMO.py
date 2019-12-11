import torch
import torch.nn as nn


class VGG16(nn.Module):
    def __init__(self, n_class=21):
        super(VGG16, self).__init__()
        # conv1
        # padding=1与tensorflow='same'类似
        self.conv1_1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.relu1_1 = torch.nn.ReLU(inplace=True)
        self.conv1_2 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = torch.nn.ReLU(inplace=True)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu2_1 = torch.nn.ReLU(inplace=True)
        self.conv2_2 = torch.nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = torch.nn.ReLU(inplace=True)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.relu3_1 = torch.nn.ReLU(inplace=True)
        self.conv3_2 = torch.nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = torch.nn.ReLU(inplace=True)
        self.conv3_3 = torch.nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = torch.nn.ReLU(inplace=True)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.relu4_1 = torch.nn.ReLU(inplace=True)
        self.conv4_2 = torch.nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = torch.nn.ReLU(inplace=True)
        self.conv4_3 = torch.nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = torch.nn.ReLU(inplace=True)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.relu5_1 = torch.nn.ReLU(inplace=True)
        self.conv5_2 = torch.nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = torch.nn.ReLU(inplace=True)
        self.conv5_3 = torch.nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = torch.nn.ReLU(inplace=True)
        self.pool5 = torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = torch.nn.Linear(in_features=512 * 7 * 7, out_features=4096)
        self.fc7 = torch.nn.Linear(in_features=4096, out_features=4096)
        self.fc8 = torch.nn.Linear(in_features=4096, out_features=n_class)

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.fc6(h.view(x.shape[0], -1))
        h = self.fc7(h)
        h = self.fc8(h)
        return h


if __name__ == "__main__":
    x = torch.randn((10, 3, 224, 224))
    model = VGG16()
    print(model)
    y = model(x)
    print(y.size())
