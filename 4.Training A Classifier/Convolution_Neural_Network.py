#!/usr/bin/env python
# coding: utf-8

import torch

# 畳み込みニューラルネットワークの定義
import torch.nn as nn
import torch.nn.functional as F

# 損失関数と最適化アルゴリズムを定義する
import torch.optim as optim

# 行列計算
import matplotlib.pyplot as plt
import numpy as np

# データ読み込み
import torchvision
import torchvision.transforms as transforms

#並列処理のライブラリ(windowsで必要)
from multiprocessing import freeze_support

########################################################################
# 初期設定

# deviceの設定(GPUを使う場合)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# transform の定義
transform = transforms.Compose([
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# トレーニング用データセット(型)
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

# トレーニング用データローダ
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=False, num_workers=2)

# テスト用データセット(型)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

# テスト用データローダ
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

# クラス分類の一覧
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

########################################################################
# ニューラルネットワークの定義

class Net(nn.Module):

    # コンストラクタ
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)       # 2次元の畳み込み
        self.pool = nn.MaxPool2d(2, 2)        # プーリング
        self.conv2 = nn.Conv2d(6, 16, 5)      # 2次元の畳み込み
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # 全結合(線型変換)
        self.fc2 = nn.Linear(120, 84)         # 全結合(線型変換)
        self.fc3 = nn.Linear(84, 10)          # 全結合(線型変換)

    # 順伝播
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 畳み込み -> 活性化 -> プーリング
        x = self.pool(F.relu(self.conv2(x)))  # 畳み込み -> 活性化 -> プーリング
        x = x.view(-1, 16 * 5 * 5)            # 行列の形状を変換
        x = F.relu(self.fc1(x))               # 全結合(線型変換) -> 活性化
        x = F.relu(self.fc2(x))               # 全結合(線型変換) -> 活性化
        x = self.fc3(x)                       # 全結合(線型変換)
        return x

# 画像を表示する関数
def imshow(img):
    img = img / 2 + 0.5      # 標準化を戻す
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()               # 描画

########################################################################
# メイン関数

if __name__ == "__main__":
    freeze_support()        #並列処理(windowsで必要)

    net = Net()

    net.to(device)          # GPUを使う場合
    print(device)           # GPUを使う場合

    criterion = nn.CrossEntropyLoss()                                # 損失の計算
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # パラメータの更新

    ########################################################################
    # トレーニングデータで学習する
    for epoch in range(2):  # エポック数(ループする回数)

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # トレーニングデータを取得する
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)      # GPUを使う場合

            # 勾配を初期化する
            optimizer.zero_grad()

            # ニューラルネットワークにデータを通し、順伝播を計算する
            outputs = net(inputs)

            # 誤差の計算
            loss = criterion(outputs, labels)

            # 逆伝播の計算
            loss.backward()

            # 重みの計算
            optimizer.step()

            # 状態を表示する
            running_loss += loss.item()
            if i % 2000 == 1999:    # 2,000 データずつ
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    ########################################################################
    # テスト用データでテストする

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images))

    images, labels = images.to(device), labels.to(device)            # GPUを使う場合
    
    print('正解ラベル: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    outputs = net(images)
    print(outputs[0,:])

    axis = 1                                                          #axis=0:col, axis=1:row
    _, predicted = torch.max(outputs, axis)

    print('分類結果: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)    # GPUを使う場合
            outputs = net(images)
            axis = 1                                                 #axis=0:col, axis=1:row
            _, predicted = torch.max(outputs.data, axis)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)    # GPUを使う場合
            outputs = net(images)
            axis = 1                                                  #axis=0:col, axis=1:row
            _, predicted = torch.max(outputs, axis)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))