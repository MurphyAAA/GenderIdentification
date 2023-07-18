import numpy as np
import scipy
import pdb

import util


class LR:
    def __init__(self, DTR, LTR, DVAL, LVAL,  hyperPar):
        self.w = []
        self.b = []
        self.parameter = []
        self.predictList = []
        self.DTR = DTR
        self.LTR = LTR
        self.DVAL = DVAL
        self.LVAL = LVAL

        self.lam = hyperPar

    def logreg_object(self, v):  # loss function
        self.w = v[0:-1]
        self.b = v[-1]
        w_norm = np.linalg.norm(self.w)
        self.w = util.vcol(self.w)
        reg_term = (self.lam / 2) * (w_norm ** 2)
        negz = -1 * (2 * self.LTR - 1)
        fx = np.dot(self.w.T, self.DTR) + self.b
        logJ = np.logaddexp(0, negz * fx)
        mean_logJ = logJ.mean()
        # print(mean_logJ)
        res = reg_term + mean_logJ
        res = res.reshape(res.size, )
        return res

    def train(self):

        x, f, d = scipy.optimize.fmin_l_bfgs_b(self.logreg_object, np.zeros(self.DTR.shape[0] + 1),
                                               approx_grad=True)
        self.w = x[0:-1]
        self.b = x[-1]
        self.parameter = [{"w": self.w, "b": self.b}]

    # def bayes_decision_threshold(self, pi1, Cfn, Cfp):
    #     t = np.log(pi1 * Cfn)
    #     t = t - np.log((1 - pi1) * Cfp)
    #     t = -t
    #     return t

    ## output llr
    def score(self):
        s = np.dot(util.vrow(self.w), self.DVAL) + self.b
        s = s.reshape(s.size, )
        # print("s is : {}".format(s))
        return s


    # def minDcf(self, score, label,piTilde, fusion):
    #     score = np.array(score).flatten()
    #     label = np.array(label).flatten()
    #     scoreArray = score.copy()
    #     scoreArray.sort()
    #     scoreArray = np.concatenate([np.array([-np.inf]), scoreArray, np.array([np.inf])])
    #     FPR = np.zeros(scoreArray.size)
    #     TPR = np.zeros(scoreArray.size)
    #     FNR = np.zeros(scoreArray.size)
    #     res = np.zeros(scoreArray.size)
    #     minDCF = 300
    #     minT = 2
    #     #{res[idx] : t}
    #     for idx, t in enumerate(scoreArray):
    #         Pred = np.int32(score > t)  # 强制类型转换为int32,True 变成1，False 变成0
    #         Conf = np.zeros((2, 2))
    #         for i in range(2):
    #             for j in range(2):
    #                 Conf[i, j] = ((Pred == i) * (label == j)).sum()
    #                 TPR[idx] = Conf[1, 1] / (Conf[1, 1] + Conf[0, 1]) if (Conf[1, 1] + Conf[0, 1]) != 0.0 else 0
    #                 FPR[idx] = Conf[1, 0] / (Conf[1, 0] + Conf[0, 0]) if ((Conf[1, 0] + Conf[0, 0]) != 0.0) else 0
    #                 # FNR,FPR
    #                 FNR[idx] = 1 - TPR[idx]
    #
    #         #res[idx] = piT * Cfn * (1 - TPR[idx]) + (1 - piT) * Cfp * FPR[idx]
    #         res[idx] = piTilde * (1 - TPR[idx]) + (1-piTilde) * FPR[idx]
    #         sysRisk = min(piTilde, 1 - piTilde)
    #         res[idx] = res[idx] /sysRisk # 除 risk of an optimal system
    #
    #         if res[idx] < minDCF:
    #             minT = t
    #             minDCF = res[idx]
    #         print("minDCF in LR is : {}".format(minDCF))
    #     if fusion:
    #         return minDCF, FNR, FPR
    #     else:
    #         return minDCF

