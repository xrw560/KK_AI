import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=4):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places * self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(places * self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class FCN8s(nn.Module):
    def __init__(self, n_class=21):
        super(FCN8s, self).__init__()
        # conv1
        self.conv1_1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=100, bias=False)
        self.bn1_1 = torch.nn.BatchNorm2d(64)
        self.relu1_1 = torch.nn.ReLU(inplace=True)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(in_places=64, places=64, block=3, stride=1)
        self.layer2 = self.make_layer(in_places=256, places=128, block=4, stride=2)
        self.layer3 = self.make_layer(in_places=512, places=256, block=23, stride=2)
        self.layer4 = self.make_layer(in_places=1024, places=512, block=3, stride=2)

        # fc6
        self.fc6 = torch.nn.Conv2d(2048, 4096, kernel_size=7)  # full-size kernel
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
        self.score_pool4 = torch.nn.Conv2d(1024, n_class, 1)
        self.upscore8 = torch.nn.ConvTranspose2d(n_class, n_class, 16, stride=8)  # x8

        self.score_pool3 = torch.nn.Conv2d(512, n_class, 1)
        self.upscore4 = torch.nn.ConvTranspose2d(n_class, n_class, 4, stride=2)  # x2

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places, stride, downsampling=True))
        for i in range(1, block):
            layers.append(Bottleneck(places * 4, places))

        return nn.Sequential(*layers)

    def forward(self, x):
        h = x
        h = self.relu1_1(self.bn1_1(self.conv1_1(h)))
        h = self.pool1(h)

        h = self.layer1(h)
        h = self.layer2(h)
        lay2 = h
        h = self.layer3(h)
        lay3 = h
        h = self.layer4(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score(h)

        # FCN-8s
        upscore2 = self.upscore2(h)  # x2
        pool4 = self.score_pool4(lay3)  # same channel
        score_pool4c = pool4[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]  # same spatial
        h = score_pool4c + upscore2

        upscore4 = self.upscore4(h)  # x2
        pool3 = self.score_pool3(lay2)  # same channel
        score_pool3c = pool3[:, :, 9:9 + upscore4.size()[2], 9:9 + upscore4.size()[3]]  # same spatial
        h = score_pool3c + upscore4
        h = self.upscore8(h)  # x8
        return h


if __name__ == "__main__":
    x = torch.randn((1, 3, 256, 256))
    model = FCN8s()
    from torchsummaryX import summary

    summary(model, x)
    y = model(x)
    print(y.size())
