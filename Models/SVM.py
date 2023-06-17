import numpy as np
import scipy
import pdb


class SVM:
    def __init__(self, DTR, LTR, DVAL, LVAL, hyperPar):
        self.mu = []
        self.sigma = []
        self.parameter = []
        self.predictList = []
        self.DTR = DTR
        self.LTR = LTR
        self.DVAL = DVAL
        self.LVAL = LVAL
        self.Z = np.zeros(self.LTR.shape)
        self.Z[self.LTR == 1] = 1
        self.Z[self.LTR == 0] = -1
    def vrow(self, v):
        return v.reshape((1, v.size))

    def mcol(self, v):
        return v.reshape((v.size, 1))  # 变成列向量



    def train_linear(self, C, K=1):
        DTREXT = np.vstack([self.DTR, np.ones((1, self.DTR.shape[1])) * K])

        H = np.dot(DTREXT.T, DTREXT)
        H = self.mcol(self.Z) * self.vrow(self.Z) * H

        def JDualv2(alpha):
            los_fun = -0.5 * np.dot(np.dot(alpha.T,H),alpha) + np.dot(alpha.T , np.ones(alpha.size))
            return los_fun, -np.dot(H,alpha) + np.ones(alpha.size)  # 损失函数，梯度
        def LDual(alpha):
            loss, grad =  JDualv2(alpha)
            return - loss, - grad

        def JPrimal(w):
            S = np.dot(w.T, DTREXT)
            loss = np.maximum(np.zeros(S.shape), 1 - self.Z * S).sum()
            return 0.5 * np.linalg.norm(w) ** 2 + C * loss

        alphaStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(
            LDual,
            np.zeros(self.DTR.shape[1]),
            bounds=[(0, C)] * self.DTR.shape[1],
            factr=1.0,
            maxiter=100000,
            maxfun=100000)
        wStar = np.matmul(DTREXT,self.mcol(alphaStar * self.Z))
        # pdb.set_trace()
        #wStar = np.dot(np.dot(alphaStar,Z),DTREXT)

        #wStar = np.dot(DTREXT, self.mcol(alphaStar) * self.mcol(Z))  # wStar 为 (feature+K 行，1列) 的列向量
        print(JPrimal(wStar))
        print('my Dual loss ', JPrimal(wStar) + LDual(alphaStar)[0] )
        return wStar


    def score(self,wStar,K):

        xtilde_val = np.vstack([self.DVAL, np.ones((1, self.DVAL.shape[1])) * K])

        res = np.dot(wStar.T,xtilde_val).reshape(-1)
        print(res.shape)
        return res


    def minDcf(self, score, label, epiT):
        label = np.concatenate(label).flatten()
        scoreArray = np.concatenate([arr for arr in score])
        scoreArray.sort()

        score = np.concatenate(score).flatten()
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

        print("minDCF in SVM is : {}".format(minDCF))
        print("minT in SVM is : {}".format(minT))
        return res.min()





    # def KFunc_rbf(g):
    #     def K(D1,D2):
    #        DIST = self.mcol((D1**2).sum(0)) + self.vrow((D2**2).sum(0)) - 2*np.dot(D1.T,D2)
    #        return np.exp(-g * DIST)
    def train_RBF(self, C, gamma, K=1):  # 非线性 使用 核函数
        # DTREXT = np.vstack([DTR, np.ones((1, DTR.shape[1])) * K])
        Z = np.zeros(self.LTR.shape)
        Z[self.LTR == 1] = 1
        Z[self.LTR == 0] = -1

        # H = np.dot(DTREXT.T, DTREXT)
        # Dist = np.zeros((self.DTR.shape[1], self.DTR.shape[1]))
        # for i in range(self.DTR.shape[1]):
        #     for j in range(self.DTR.shape[1]):
        #         xi = self.DTR[:, i]
        #         xj = self.DTR[:, j]
        #         Dist[i, j] = np.linalg.norm(xi - xj) ** 2
        D1 = self.DTR
        D2 = self.DTR
        Dist = self.mcol((D1**2).sum(0)) + self.vrow((D2**2).sum(0)) - 2*np.dot(D1.T,D2)
        kernel = np.exp(-gamma * Dist) + K ** 0.5
        H = self.mcol(Z) * self.vrow(Z) * kernel

        def JDualv2(alpha):
            los_fun = -0.5 * np.dot(np.dot(alpha.T, H), alpha) + np.dot(alpha.T, np.ones(alpha.size))
            return los_fun, -np.dot(H, alpha) + np.ones(alpha.size)  # 损失函数，梯度

        def LDual(alpha):
            loss, grad = JDualv2(alpha)
            return -loss, -grad

        alphaStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(
            LDual,
            np.zeros(self.DTR.shape[1]),
            bounds=[(0, C)] * self.DTR.shape[1],
            factr=1.0,
            maxiter=100000,
            maxfun=100000)
        # wStar = np.dot(DTR, vcol(alphaStar) * vcol(Z))  # wStar 为 (feature+K 行，1列) 的列向量

        #print('Dual loss ', JDual(alphaStar)[0])
        return alphaStar
    def score_rbf(self, alphaStar, gamma, K):
        Dist = self.mcol((self.DVAL ** 2).sum(0)) + self.vrow((self.DTR ** 2).sum(0)) - 2 * np.dot(self.DVAL.T, self.DTR)
        kernel = np.exp(-gamma * Dist) + K ** 0.5
        S = np.matmul(kernel, self.mcol(alphaStar * self.Z)).flatten()

        # pdb.set_trace()
        # print(S.shape)
        return S

