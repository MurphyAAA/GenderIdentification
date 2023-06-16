import numpy as np
import scipy


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

    def mrow(self, v):
        return v.reshape((1, v.size))

    def mcol(self, v):
        return v.reshape((v.size, 1))  # 变成列向量

    def logreg_object(self, v):  # loss function
        self.w = v[0:-1]
        self.b = v[-1]
        w_norm = np.linalg.norm(self.w)
        self.w = self.mcol(self.w)
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
        # w = mcol(w)
        self.parameter = [{"w": self.w, "b": self.b}]

    # def bayes_decision_threshold(self, pi1, Cfn, Cfp):
    #     t = np.log(pi1 * Cfn)
    #     t = t - np.log((1 - pi1) * Cfp)
    #     t = -t
    #     return t

    ## output llr
    def score(self):
        s = np.dot(self.mrow(self.w), self.DVAL) + self.b
        s = s.reshape(s.size, )
        return s

    ##get score and compare with threshold
    ## output predictList np.array(1,0,0,1...)
    def estimate(self):
        s = self.score()
        for i in s:
            if i > 0:
                self.predictList.append(1)
            else:
                self.predictList.append(0)

    # def evaluation(self, DTE):
    #     return
    #
    # def validation(self,DTE,LTE):
    #     ## confusionmatrix
    #     return
    def minDcf(self, score, label, piT, Cfn, Cfp):
        label = np.concatenate(label).flatten()
        scoreArray = np.concatenate([arr for arr in score])
        scoreArray.sort()
        print(scoreArray)
        score = np.concatenate(score).flatten()
        scoreArray = np.concatenate([np.array([-np.inf]), scoreArray, np.array([np.inf])])
        FPR = np.zeros(scoreArray.size)
        TPR = np.zeros(scoreArray.size)
        res = np.zeros(scoreArray.size)
        minDCF = 300
        minT = 2
        #{res[idx] : t}
        for idx, t in enumerate(scoreArray):
            Pred = np.int32(score > t)  # 强制类型转换为int32,True 变成1，False 变成0
            Conf = np.zeros((2, 2))
            for i in range(2):
                for j in range(2):
                    Conf[i, j] = ((Pred == i) * (label == j)).sum()
                    TPR[idx] = Conf[1, 1] / (Conf[1, 1] + Conf[0, 1]) if (Conf[1, 1] + Conf[0, 1]) != 0.0 else 0
                    FPR[idx] = Conf[1, 0] / (Conf[1, 0] + Conf[0, 0]) if ((Conf[1, 0] + Conf[0, 0]) != 0.0) else 0


            #res[idx] = piT * Cfn * (1 - TPR[idx]) + (1 - piT) * Cfp * FPR[idx]
            res[idx] = 0.5 * (1 - TPR[idx]) + 0.5 * FPR[idx]
            sysRisk = min(piT * Cfn, (1 - piT) * Cfp)
            res[idx] = res[idx] / 0.01  # 除 risk of an optimal system

            if res[idx] < minDCF:
                minT = t
                minDCF = res[idx]

        print(minDCF)
        print(minT)
        return res.min()
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
