# -*- coding: utf-8 -*-
"""
@Time ： 2023/7/5 11:23
@Auth ： Murphy
@File ：util.py
@IDE ：PyCharm
"""
import pdb
from enum import Enum
import numpy as np
class svm_kernel_type(Enum):
    rbf = "rbf"
    poly = "poly"


def vrow(v):
    return v.reshape((1, v.size))

def vcol(v):
    return v.reshape((v.size, 1))  # 变成列向量

def minDcf(modelName, score, label, epiT, fusion):
    score = np.array(score).flatten()
    label = np.array(label).flatten()
    scoreArray = score.copy()
    scoreArray.sort()
    scoreArray = np.concatenate([np.array([-np.inf]), scoreArray, np.array([np.inf])])
    FPR = np.zeros(scoreArray.size)
    TPR = np.zeros(scoreArray.size)
    FNR = np.zeros(scoreArray.size)
    res = np.zeros(scoreArray.size)
    minDCF = 300
    minT = 2
    # {res[idx] : t}
    for idx, t in enumerate(scoreArray):
        Pred = np.int32(score > t)  # 强制类型转换为int32,True 变成1，False 变成0
        Conf = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                Conf[i, j] = ((Pred == i) * (label == j)).sum()
                TPR[idx] = Conf[1, 1] / (Conf[1, 1] + Conf[0, 1]) if (Conf[1, 1] + Conf[0, 1]) != 0.0 else 0
                FPR[idx] = Conf[1, 0] / (Conf[1, 0] + Conf[0, 0]) if ((Conf[1, 0] + Conf[0, 0]) != 0.0) else 0
                # FNR,FPR
                FNR[idx] = 1 - TPR[idx]

        # res[idx] = piT * Cfn * (1 - TPR[idx]) + (1 - piT) * Cfp * FPR[idx]
        res[idx] = epiT * (1 - TPR[idx]) + (1 - epiT) * FPR[idx]
        sysRisk = min(epiT, (1 - epiT))
        res[idx] = res[idx] / sysRisk  # 除 risk of an optimal system

        if res[idx] < minDCF:
            minDCF = res[idx]
    # pdb.set_trace()
    print("minDCF in {} is : {}".format(modelName, minDCF))
    if fusion:
        return minDCF, FNR, FPR
    else:
        return minDCF

def threthod(pi1,Cfn,Cfp):
    t = np.log(pi1 * Cfn)
    t = t - np.log((1 - pi1) * Cfp)
    t = -t
    return t

def normalizedDCF(modelName, score, label, epiT,Cfn, Cfp, fusion): # 传进来的就是计算好的(pi_tilde,1,1)
    score = np.array(score).flatten()
    label = np.array(label).flatten()

    t = threthod(epiT, Cfn, Cfp)
    Pred = np.int32(score > t)
    Conf = np.zeros((2, 2))
    FPR = 0
    TPR = 0
    FNR = 0
    for i in range(2):
        for j in range(2):
            Conf[i, j] = ((Pred == i) * (label == j)).sum()
            TPR = Conf[1, 1] / (Conf[1, 1] + Conf[0, 1]) if (Conf[1, 1] + Conf[0, 1]) != 0.0 else 0
            FPR = Conf[1, 0] / (Conf[1, 0] + Conf[0, 0]) if ((Conf[1, 0] + Conf[0, 0]) != 0.0) else 0
            # FNR,FPR
            FNR = 1 - TPR
    res = epiT*Cfn*FNR + (1-epiT)*Cfp*FPR # DCF

    sysRisk = min(epiT*Cfn,(1-epiT)*Cfp)
    res = res / sysRisk  #除 risk of an optimal system
    print("Actual normalized DCF in {} is : {} -- piT is :{}".format(modelName, res, epiT))
    if fusion:
        return res, FNR, FPR
    else:
        return res # actual DCF
