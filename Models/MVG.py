import numpy as np
import scipy


class MVG:
    def __init__(self, DTR, LTR, DVAL, LVAL, th):
        self.mu = []
        self.sigma = []
        self.parameter = []
        self.predictList = []
        self.DTR = DTR
        self.LTR = LTR
        self.DVAL = DVAL
        self.LVAL = LVAL
        self.th = th

    def vrow(self, v):
        return v.reshape((1, v.size))

    def mcol(self, v):
        return v.reshape((v.size, 1))  # 变成列向量

    def _logpdf_GAU_ND_fast(self,X, mu, C):
        XC = X - mu
        M = X.shape[0]
        const = -0.5 * M * np.log(2 * np.pi)
        logdet = np.linalg.slogdet(C)[1]
        L = np.linalg.inv(C)
        v = (XC * np.dot(L, XC)).sum(0)
        return const - 0.5 * logdet - 0.5 * v

    def train(self):
        # ndarray(12,480)
        DTR0 = self.DTR[:, self.LTR == 0]  # 0类的所有Data
        DTR1 = self.DTR[:, self.LTR == 1]  # 1类的所有Data

        self.mu.append(self.mcol(DTR0.mean(1)))
        self.mu.append(self.mcol(DTR1.mean(1)))
        # 去中心化
        DTRc0 = DTR0 - self.mu[0]
        DTRc1 = DTR1 - self.mu[1]
        # 协方差
        self.sigma.append(np.dot(DTRc0, DTRc0.T) / DTRc0.shape[1])
        self.sigma.append(np.dot(DTRc1, DTRc1.T) / DTRc1.shape[1])
        self.parameter = [{"mu": self.mu}, {"sigma": self.sigma}]

    def bayes_decision_threshold(self, pi1, Cfn, Cfp):
        t = np.log(pi1 * Cfn)
        t = t - np.log((1 - pi1) * Cfp)
        t = -t
        return t

    ## output llr
    def score(self):
        tlogll0 = self._logpdf_GAU_ND_fast(self.DVAL, self.mu[0], self.sigma[0])
        tlogll1 = self._logpdf_GAU_ND_fast(self.DVAL, self.mu[1], self.sigma[1])
        return tlogll1 - tlogll0

    ##get score and compare with threshold
    ## output predictList np.array(1,0,0,1...)
    def estimate(self,llr):
        Priori = 1 / 2
        t = self.bayes_decision_threshold(Priori, 1, 1)
        for r in llr:
            if r > t:
                self.predictList.append(1)
            else:
                self.predictList.append(0)

    # def evaluation(self, DTE):
    #     return
    #
    # def validation(self,DTE,LTE):
    #     ## confusionmatrix
    #     return

    def computeAccuracy(self):
        res = []
        for i, pre in enumerate(self.predictList):
            if (pre == self.LVAL[i]):
                res.append(True)  # 预测正确
            else:
                res.append(False)
        corr = res.count(True)
        wrong = res.count(False)
        # print(f'\ncorrect number:{corr}\nwrong number:{wrong}\ntotal:{len(res)}')
        acc = corr / len(res)
        err = wrong / len(res)
        return acc, err

    def main(self):
        print("it will run!")

    if __name__ == "__main__":
        main()
