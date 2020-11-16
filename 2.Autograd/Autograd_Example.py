#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import torch

print('\n')

# ユークリッドノルムの計算(ユークリッド距離)
x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

print(x)
print(x.data.norm())

# y の勾配を y.backward() で計算したいが、y はスカラーではないため、そのまま計算することができない
# 適当なベクトルを設定することで、勾配が計算される
gradients = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(gradients)

print(x.grad)


print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)

print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())

