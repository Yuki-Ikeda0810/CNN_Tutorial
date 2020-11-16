#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import torch

print('\n')

x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()

# out.backward() で勾配を計算する
out.backward()
print(x.grad)
