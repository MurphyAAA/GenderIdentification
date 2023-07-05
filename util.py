# -*- coding: utf-8 -*-
"""
@Time ： 2023/7/5 11:23
@Auth ： Murphy
@File ：util.py
@IDE ：PyCharm
"""
from enum import Enum
class svm_kernel_type(Enum):
    rbf = "rbf"
    poly = "poly"


def vrow(v):
    return v.reshape((1, v.size))

def vcol(v):
    return v.reshape((v.size, 1))  # 变成列向量

