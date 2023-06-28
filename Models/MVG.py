import numpy as np
import scipy


class MVG:
    def __init__(self, DTR, LTR, DVAL, LVAL):
        self.mu = []
        self.sigma = []
        self.parameter = []
        self.predictList = []
        self.DTR = DTR
        self.LTR = LTR
        self.DVAL = DVAL
        self.LVAL = LVAL

    def vrow(self, v):
        return v.reshape((1, v.size))

    def mcol(self, v):
        return v.reshape((v.size, 1))  # 变成列向量

    def _logpdf_GAU_ND_fast(self, X, mu, C):
        XC = X - mu
        M = X.shape[0]
        const = -0.5 * M * np.log(2 * np.pi)
        logdet = np.linalg.slogdet(C)[1]
        L = np.linalg.inv(C)
        v = (XC * np.dot(L, XC)).sum(0)
        return const - 0.5 * logdet - 0.5 * v

    def train(self, tied=False, bayes=False):
        # ndarray(12,576)
        DTR0 = self.DTR[:, self.LTR == 0]  # 0类的所有Data
        DTR1 = self.DTR[:, self.LTR == 1]  # 1类的所有Data

        self.mu.append(self.mcol(DTR0.mean(1)))
        self.mu.append(self.mcol(DTR1.mean(1)))
        # 去中心化
        DTRc0 = DTR0 - self.mu[0]
        DTRc1 = DTR1 - self.mu[1]
        # print(DTR0.shape)
        # print(DTRc0.shape)
        # 协方差
        if tied:
            print("enter tied")
            self.sigma.append((np.dot(DTRc0, DTRc0.T) + np.dot(DTRc1, DTRc1.T)) / self.DTR.shape[1])
        elif bayes:
            print("enter native")
           ## C = (np.dot(DTRc0, DTRc0.T) + np.dot(DTRc1, DTRc1.T)) / self.DTR.shape[1]
            self.sigma.append(np.dot(DTRc0, DTRc0.T) / DTRc0.shape[1])
            self.sigma.append(np.dot(DTRc1, DTRc1.T) / DTRc1.shape[1])
            identity = np.identity(self.DTR.shape[0])
            self.sigma[0] = self.sigma[0] * identity
            self.sigma[1] = self.sigma[1] * identity
            # self.sigma.append(C)
        else:
            print("enter normal")
            self.sigma.append(np.dot(DTRc0, DTRc0.T) / DTRc0.shape[1])
            self.sigma.append(np.dot(DTRc1, DTRc1.T) / DTRc1.shape[1])
            self.parameter = [{"mu": self.mu}, {"sigma": self.sigma}]

    def bayes_decision_threshold(self, pi1, Cfn, Cfp):
        t = np.log(pi1 * Cfn)
        t = t - np.log((1 - pi1) * Cfp)
        t = -t
        return t

    ## output llr
    def score(self, tied=False, bayes=False):

        if tied:
            print("enter score tied ")
            tlogll0 = self._logpdf_GAU_ND_fast(self.DVAL, self.mu[0], self.sigma[0])
            tlogll1 = self._logpdf_GAU_ND_fast(self.DVAL, self.mu[1], self.sigma[0])
        elif bayes:
            print("enter score bayes ")
            tlogll0 = self._logpdf_GAU_ND_fast(self.DVAL, self.mu[0], self.sigma[0])
            tlogll1 = self._logpdf_GAU_ND_fast(self.DVAL, self.mu[1], self.sigma[1])

        else:
            print("enter score normal")
            tlogll0 = self._logpdf_GAU_ND_fast(self.DVAL, self.mu[0], self.sigma[0])
            tlogll1 = self._logpdf_GAU_ND_fast(self.DVAL, self.mu[1], self.sigma[1])
            print((tlogll1 - tlogll0).shape)
        return tlogll1 - tlogll0

    ##get score and compare with threshold
    ## output predictList np.array(1,0,0,1...)
    # def estimate(self,llr):
    #     Priori = 1 / 2
    #     t = self.bayes_decision_threshold(Priori, 1, 1)
    #     for r in llr:
    #         if r > t:
    #             self.predictList.append(1)
    #         else:
    #             self.predictList.append(0)

    # def evaluation(self, DTE):
    #     return
    #
    # def validation(self,DTE,LTE):
    #     ## confusionmatrix
    #     return

    # def computeAccuracy(self):
    #     res = []
    #     for i, pre in enumerate(self.predictList):
    #         if (pre == self.LVAL[i]):
    #             res.append(True)  # 预测正确
    #         else:
    #             res.append(False)
    #     corr = res.count(True)
    #     wrong = res.count(False)
    #     # print(f'\ncorrect number:{corr}\nwrong number:{wrong}\ntotal:{len(res)}')
    #     acc = corr / len(res)
    #     err = wrong / len(res)
    #     return acc, err

    # use effective_prior
    def minDcf(self, score, label, epiT):
        label = np.array(label).flatten()
        score = np.array(score).flatten()
        scoreArray = score
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

        print("minDCF in MVG is : {}".format(minDCF))
        print("minT in MVG is : {}".format(minT))
        return res.min()

    # use prior
    def minDcfPi(self, score, label, Cfn, Cfp, piT):
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

            res[idx] = piT * Cfn * (1 - TPR[idx]) + (1 - piT) * Cfp * FPR[idx]

            sysRisk = min(piT * Cfn, (1 - piT) * Cfp)
            res[idx] = res[idx] / sysRisk  # 除 risk of an optimal system

            if res[idx] < minDCF:
                minT = t
                minDCF = res[idx]


        return res.min()
