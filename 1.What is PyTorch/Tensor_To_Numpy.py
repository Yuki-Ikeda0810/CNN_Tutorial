#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import torch

print('\n')

# Tensor から NumPy への変換は、torch.numpy() で行う
# メモリを共有するため、一方を変更すると、もう一方も変更される
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)

# NumPy からTensor への変換は、torch.from_numpy() で行う
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)