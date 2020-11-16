#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import torch

print('\n')

# torch.empty で初期化されていない Tensor を作成
x = torch.empty(5, 3)
print(x, '\n')

# torch.rand でランダムな値の Tensor を作成
x = torch.rand(5, 3)
print(x, '\n')

# torch.zeros で要素が0の Tensor を作成
x = torch.zeros(5, 3, dtype=torch.long)
print(x, '\n')

# torch.tensor にリストを渡すことで Tensor を作成
x = torch.tensor([5.5, 3])
print(x, '\n')

# new_ones で元の Tensor を要素1で書き換える
x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
print(x, '\n')

# randn_like で元の Tensor をランダム値で書き換る
# randn なので、標準化（平均0、標準偏差1）のランダム値になる
x = torch.randn_like(x, dtype=torch.float)    # override dtype!
print(x, '\n')

# size() で Tensor のサイズを取得
print(x.size(), '\n')

