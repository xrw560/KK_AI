import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
import d2lzh_pytorch as d21


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):
    d21.use_svg_display()
    # 这里的_表示我们忽略(不使用的)变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


if __name__ == "__main__":
    mnist_train = torchvision.datasets.FashionMNIST(root='F:/data/pytorch_data/FashionMNIST', train=True, download=True,
                                                    transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root='F:/data/pytorch_data/FashionMNIST', train=False, download=True,
                                                   transform=transforms.ToTensor())
    print(type(mnist_train))
    print(len(mnist_train), len(mnist_test))

    feature, lable = mnist_train[0]
    print(feature.shape, lable)

    X, y = [], []
    for i in range(10):
        X.append(mnist_train[i][0])
        y.append(mnist_train[i][1])
    show_fashion_mnist(X, get_fashion_mnist_labels(y))
