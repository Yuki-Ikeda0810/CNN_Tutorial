#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import torch

print('\n')

# PyTorch の Tensor は、requires_grad 属性を True にすることで勾配が記録される
x = torch.ones(2, 2, requires_grad=True)
print(x)

# backward() で勾配を計算すると、Tensor の grad 属性に勾配が保持される
y = x + 2
print(y)

# print すると、出力に grad_fn がある
# これは、勾配を計算するための計算グラフが構築されていることを示している
print(y.grad_fn)

# y を利用してさらに計算グラフ（式） z、 out を作成する
z = y * y * 3
out = z.mean()

print(z, out)

# tensor.requires_grad_() で requires_grad 属性を変更することができる
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)