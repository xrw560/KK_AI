import torch
import torch.nn as nn


class FCN8s(nn.Module):
    def __init__(self, n_class=21):
        super(FCN8s, self).__init__()
        # conv1
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
        self.fc6 = torch.nn.Conv2d(512, 4096, kernel_size=7)  # full-size kernel
        self.relu6 = torch.nn.ReLU(inplace=True)
        self.drop6 = torch.nn.Dropout2d()

        # fc7
        self.fc7 = torch.nn.Conv2d(4096, 4096, kernel_size=1)  # 1x1 convolution
        self.relu7 = torch.nn.ReLU(inplace=True)
        self.drop7 = torch.nn.Dropout2d()

        # coarse output
        self.score = torch.nn.Conv2d(4096, n_class, kernel_size=1)  # 1x1 convolution

        # FCN-8s
        self.upscore2 = torch.nn.ConvTranspose2d(n_class, n_class, 4, stride=2)  # x2
        self.score_pool4 = torch.nn.Conv2d(512, n_class, 1)
        self.upscore8 = torch.nn.ConvTranspose2d(n_class, n_class, 16, stride=8)  # x8

        self.score_pool3 = torch.nn.Conv2d(256, n_class, 1)
        self.upscore4 = torch.nn.ConvTranspose2d(n_class, n_class, 4, stride=2)  # x2

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
        pool3 = h

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score(h)

        # FCN-8s
        upscore2 = self.upscore2(h)  # x2
        pool4 = self.score_pool4(pool4)  # same channel
        score_pool4c = pool4[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]  # same spatial
        h = score_pool4c + upscore2

        upscore4 = self.upscore4(h)  # x2
        pool3 = self.score_pool3(pool3)  # same channel
        score_pool3c = pool3[:, :, 9:9 + upscore4.size()[2], 9:9 + upscore4.size()[3]]  # same spatial
        h = score_pool3c + upscore4
        h = self.upscore8(h)  # x8
        return h


if __name__ == "__main__":
    x = torch.randn((10, 3, 256, 256))
    model = FCN8s()
    from torchsummaryX import summary
    summary(model,x)
    y = model(x)
    print(y.size())
