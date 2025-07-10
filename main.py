# _*_ encoding: utf-8 _*_
# 文件: main
# 时间: 2025/7/10_14:47
# 作者: GuanXK

# system
import os

# third_party
import torch

# custom

if __name__ == "__main__":
    print("\n-------------- start --------------\n")

    print(torch.__version__)
    print(torch.cuda.is_available())

    print("\n--------------- end ---------------\n")
