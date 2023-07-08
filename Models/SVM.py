import numpy as np
import scipy
import pdb
import util


class SVM:
    def __init__(self, DTR, LTR, DVAL, LVAL,hyperParam):
        self.mu = []
        self.sigma = []
        self.parameter = hyperParam # { "C":1, "K":0, "loggamma":1, "d":2, "c":0}
        self.predictList = []
        self.DTR = DTR
        self.LTR = LTR
        self.DVAL = DVAL
        self.LVAL = LVAL
        self.Z = np.zeros(self.LTR.shape)
        self.Z[self.LTR == 1] = 1
        self.Z[self.LTR == 0] = -1


    def train_linear(self ):
        DTREXT = np.vstack([self.DTR, np.ones((1, self.DTR.shape[1])) * self.parameter['K']])
        H = np.dot(DTREXT.T, DTREXT)
        H = util.vcol(self.Z) * util.vrow(self.Z) * H

        def JDualv2(alpha):
            los_fun = -0.5 * np.dot(np.dot(alpha.T,H),alpha) + np.dot(alpha.T , np.ones(alpha.size))
            return los_fun, -np.dot(H,alpha) + np.ones(alpha.size)  # 损失函数，梯度
        def LDual(alpha):
            loss, grad =  JDualv2(alpha)
            return - loss, - grad

        def JPrimal(w):
            S = np.dot(w.T, DTREXT)
            loss = np.maximum(np.zeros(S.shape), 1 - self.Z * S).sum()
            return 0.5 * np.linalg.norm(w) ** 2 + self.parameter['C'] * loss

        alphaStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(
            LDual,
            np.zeros(self.DTR.shape[1]),
            bounds=[(0, self.parameter['C'])] * self.DTR.shape[1],
            factr=1.0,
            maxiter=100000,
            maxfun=100000)

        wStar = np.matmul(DTREXT,util.vcol(alphaStar * self.Z))
        #wStar = np.dot(np.dot(alphaStar,Z),DTREXT)

        #wStar = np.dot(DTREXT, util.vcol(alphaStar) * util.vcol(Z))  # wStar 为 (feature+K 行，1列) 的列向量

        print('my Dual loss ', JPrimal(wStar) + LDual(alphaStar)[0] )
        return wStar


    def score(self,wStar):
        xtilde_val = np.vstack([self.DVAL, np.ones((1, self.DVAL.shape[1])) * self.parameter['K']])
        res = np.dot(wStar.T,xtilde_val).reshape(-1)
        return res


    def minDcf(self, score, label, epiT):
        score = np.array(score).flatten()
        label = np.array(label).flatten()
        scoreArray = score.copy()
        scoreArray.sort()
        scoreArray = np.concatenate([np.array([-np.inf]), scoreArray, np.array([np.inf])])
        FPR = np.zeros(scoreArray.size)
        TPR = np.zeros(scoreArray.size)
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

            # res[idx] = piT * Cfn * (1 - TPR[idx]) + (1 - piT) * Cfp * FPR[idx]
            res[idx] = epiT * (1 - TPR[idx]) + (1 - epiT) * FPR[idx]
            sysRisk = min(epiT, (1 - epiT))
            res[idx] = res[idx] / sysRisk  # 除 risk of an optimal system

            if res[idx] < minDCF:
                minT = t
                minDCF = res[idx]

        print("minDCF with par {} in SVM is : {}".format(self.parameter["C"],minDCF))
        #print("minT in SVM is : {}".format(minT))
        return res.min()

    def train_nolinear(self, type):  # 非线性 使用 核函数
        # DTREXT = np.vstack([DTR, np.ones((1, DTR.shape[1])) * K])
        Z = np.zeros(self.LTR.shape)
        Z[self.LTR == 1] = 1
        Z[self.LTR == 0] = -1
        D1 = self.DTR
        D2 = self.DTR
        if type == util.svm_kernel_type.rbf:
            Dist = util.vcol((D1**2).sum(0)) + util.vrow((D2**2).sum(0)) - 2*np.dot(D1.T,D2)

            kernel = np.exp(-np.exp(self.parameter["loggamma"])* Dist) + self.parameter['K'] ** 0.5
            #kernel = np.exp(-self.parameter["loggamma"] * Dist) + self.parameter['K'] ** 0.5

        else: # polynomial
            kernel = (np.dot(D1.T, D2) + self.parameter["c"]) ** self.parameter["d"] + self.parameter["K"] ** 0.5

        H = util.vcol(Z) * util.vrow(Z) * kernel

        def JDualv2(alpha):
            los_fun = -0.5 * np.dot(np.dot(alpha.T, H), alpha) + np.dot(alpha.T, np.ones(alpha.size))
            return los_fun, -np.dot(H, alpha) + np.ones(alpha.size)  # 损失函数，梯度

        def LDual(alpha):
            loss, grad = JDualv2(alpha)
            return -loss, -grad

        alphaStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(
            LDual,
            np.zeros(self.DTR.shape[1]),
            bounds=[(0, self.parameter['C'])] * self.DTR.shape[1],
            factr=1.0,
            maxiter=100000,
            maxfun=100000)
        # wStar = np.dot(DTR, vcol(alphaStar) * vcol(Z))  # wStar 为 (feature+K 行，1列) 的列向量
        # pdb.set_trace()
        print('Dual loss ', JDualv2(alphaStar)[0])
        return alphaStar
    def score_nolinear(self, alphaStar, type):
        if type == util.svm_kernel_type.rbf:
            Dist = util.vcol((self.DVAL ** 2).sum(0)) + util.vrow((self.DTR ** 2).sum(0)) - 2 * np.dot(self.DVAL.T, self.DTR)
            # gamma not loggamma
            #kernel = np.exp(-self.parameter["loggamma"] * Dist) + self.parameter['K'] ** 0.5

            kernel = np.exp(-np.exp(self.parameter["loggamma"])* Dist) + self.parameter['K'] ** 0.5

        else: # polynomial
            # pdb.set_trace()
            kernel = (np.dot(self.DVAL.T, self.DTR) + self.parameter["c"]) ** self.parameter["d"] + self.parameter["K"] ** 0.5
        # print(kernel)
        S = np.matmul(kernel, util.vcol(alphaStar * self.Z)).flatten()

        return S

