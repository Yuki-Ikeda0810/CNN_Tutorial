#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import torch

print('\n')

x = torch.ones(5, 3, dtype=torch.int)
y = torch.ones(5, 3, dtype=torch.int)

# + 演算子での加算
print(x + y)

# torch.add での加算
print(torch.add(x, y))

# out 引数で、出力 Tensor を指定することができる
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

# add_ で入力 Tensor を計算結果で書き換える
y.add_(x)
print(y)

# 配列の一部をスライスできる
print(x[:, 1])

# torch.view は Tensor の形状を変換する
# -1を指定すると、他の次元を考慮して補完される
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

# 要素数1の Tensor に対して item() を利用すると、中身を取得できる
x = torch.randn(1)
print(x)
print(x.item())

# 要素数2
x = torch.randn(2)
print(x)
print(x[0].item())