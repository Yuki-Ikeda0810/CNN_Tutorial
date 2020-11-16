#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import torch

print('\n')

x = torch.zeros(5, 3, dtype=torch.int)

# torch.to() を利用して Tensor を様々なデバイスに移動できる
# 以下のコードでは CUDA デバイスに移動している
# CUDA は NVIDIA が提供している、GPU環境のプラットフォームである
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!