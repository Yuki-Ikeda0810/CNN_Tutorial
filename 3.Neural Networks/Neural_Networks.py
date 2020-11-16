#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

########################################################################
# ニューラルネットワークの定義
class Net(nn.Module):

    # コンストラクタ
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 3)                 # 2次元の畳み込み
        self.conv2 = nn.Conv2d(6, 16, 3)                # 2次元の畳み込み
        self.fc1 = nn.Linear(16 * 6 * 6, 120)           # 全結合(線型変換)
        self.fc2 = nn.Linear(120, 84)                   # 全結合(線型変換)
        self.fc3 = nn.Linear(84, 10)                    # 全結合(線型変換)

    # 順伝播
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # 畳み込み -> 活性化 -> プーリング
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)      # 畳み込み -> 活性化 -> プーリング
        x = x.view(-1, self.num_flat_features(x))       # 行列の形状を変換
        x = F.relu(self.fc1(x))                         # 全結合(線型変換) -> 活性化
        x = F.relu(self.fc2(x))                         # 全結合(線型変換) -> 活性化
        x = self.fc3(x)                                 # 全結合(線型変換)
        return x
    
    # 関数の定義
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# インスタンス化
net = Net()
print("\n――――ニューラルネットワークの定義――――")
print(net)

########################################################################
# パラメータの確認
params = list(net.parameters())
print("\n――――パラメータの確認――――")
print(len(params))
print(params[0].size())
print(params[1].size())

########################################################################
# データセットを入力する
input = torch.randn(1, 1, 32, 32, requires_grad=True)
out = net(input)
print("\n――――データセットの入力――――")
print(out)

########################################################################
# 損失の計算
output = net(input)
target = torch.randn(10)
target = target.view(1, -1)
criterion = nn.MSELoss()

loss = criterion(output, target)
print("\n――――損失の計算――――")
print(loss)

########################################################################
# パラメータの更新
import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=0.01)

optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()
print("\n")