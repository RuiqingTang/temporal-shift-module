#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   check_gpu.py
@Time    :   2025/01/22 15:23:02
@Author  :   Ruiqing Tang 
@Contact :   tangruiqing123@gmail.com
'''

import torch

# 检查PyTorch是否支持CUDA
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")

# 如果CUDA可用，显示更多信息
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"当前GPU: {torch.cuda.current_device()}")
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("CUDA不可用，当前安装的是CPU版本")