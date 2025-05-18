# -*- coding: utf-8 -*-
"""
@Time ： 2023/8/3 10:54
@Auth ： Murphy
@File ：ScoreCalibration.py
@IDE ：PyCharm
"""
import pdb

import util
import numpy as np
import scipy.linalg

class ScoreCalibration:
    def __init__(self, D, L):
        self.D = D # 之前训练得到的score，作为新的输入D
        self.L = L
        self.alpha = []
        self.beta = []
        self.gamma = []
        self.pi= self.D[:,self.L==1].shape[1] / self.D.shape[1]
    def logreg_object(self,v):  # loss function
        self.alpha = v[0:-1]
        self.gamma = v[-1]

        self.alpha = util.vcol(self.alpha)
        self.beta = self.gamma + np.log((self.pi / (1-self.pi)))
        negz = -1 * (2 * self.L - 1)
        # pdb.set_trace()
        fx = np.dot(self.alpha.T, self.D) + self.beta
        logJ = np.logaddexp(0, negz * fx)
        w=[]
        for nz in negz:
            if nz == -1:# z=1
                w.append(self.pi / self.D[:,self.L==1].shape[1])
            else: # z = -1
                w.append((1-self.pi) / self.D[:,self.L==1].shape[1])
        weight_logJ = w * logJ
        sum_weight_logJ = weight_logJ.sum()
        # print(mean_logJ)
        res = sum_weight_logJ
        res = res.reshape(res.size, )
        return res
    # 数据集的score 作为输入，得到校准后的score
    def KFoldCalibration(self):
        x, f, d = scipy.optimize.fmin_l_bfgs_b(self.logreg_object, np.zeros(self.D.shape[0] + 1),
                                                   approx_grad=True)
        self.alpha = x[0:-1]
        self.gamma = x[-1]
        new_score = np.dot(self.alpha.T, self.D) + self.gamma
        return new_score
