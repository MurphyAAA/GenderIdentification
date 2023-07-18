import numpy as np
import scipy
import pdb

import util


class GMM:
    def __init__(self, DTR, LTR, DVAL, LVAL, hyperPar):
        self.mu = []
        self.sigma = []
        self.parameter = None
        self.predictList = []
        self.DTR = DTR
        self.LTR = LTR
        self.DVAL = DVAL
        self.LVAL = LVAL
        self.n0 = hyperPar["n0"]
        self.n1 = hyperPar["n1"]


    def logpdf_GAU_ND(self,x, mu, C):  # mu，c都是某一个类别的样本的平均值和协方差矩阵
        M = x.shape[0]  # M 是特征数
        # print(M)
        a = M * np.log(2 * np.pi)
        _, b = np.linalg.slogdet(C)
        xc = (x - mu)
        # print("xc.shape", xc.shape)
        # xc应该每行循环列数次
        c = np.dot(xc.T, np.linalg.inv(C))
        # print(c.shape[0])
        c = np.dot(c, xc)

        c = np.diagonal(c)  # 点乘完了取对角线就ok
        return (-1.0 / 2.0) * (a + b + c)

    def LBG(self,initGmm, alpha, iterNum, X, diagCov, tiedCov, psi):  # initGmm: [(w, mu, C), (w, mu, C) ...]
        # for i in range(iterNum):
        if (iterNum <= 0):
            return initGmm
        GMM_new = []
        for w, mu, C in initGmm:
            U, s, Vh = np.linalg.svd(C)
            d = U[:, 0:1] * s[0] ** 0.5 * alpha
            # pdb.set_trace()
            # GMM_iteration(X,)
            GMM_new.append((w / 2.0, util.vcol(mu) + d, C))
            GMM_new.append((w / 2.0, util.vcol(mu) - d, C))
        GMM_new_gen = self.GMM_iteration(X, GMM_new, diagCov, tiedCov, psi)
        if iterNum == 1:
            return GMM_new_gen
        else:
            return self.LBG(GMM_new_gen, alpha, iterNum - 1, X, diagCov, tiedCov, psi)

    def constraintCov(self,C, psi):
        U, s, _ = np.linalg.svd(C)
        s[s < psi] = psi
        covNew = np.dot(U, util.vcol(s) * U.T)
        return covNew

    def GMM_iteration(self,X, gmm, diagCov, tiedCov, psi):  # 传入的gmm是初始值，训练过程中更新
        _llOld = None
        _ll = None
        _deltaLL = 10

        while _deltaLL >= 1e-6:
            lLL = []
            for (w, mu, C) in gmm:
                ll = self.logpdf_GAU_ND(X, mu, C) + np.log(w)
                lLL.append(util.vrow(ll))
                # print(i)
            LL = np.vstack(lLL)
            margin = scipy.special.logsumexp(LL, axis=0)  # 将log(w*N(x,mu,c)) 加起来，即log(∑w*N())
            # E-step
            post = np.exp(LL - margin)  # 3*1000   3组gmm， 1000个sample
            _ll = margin.sum() / X.shape[1]  # log-ll for previous M-step

            # M-step
            gmmUpd = []
            for g in range(post.shape[0]):
                Z = post[g].sum()
                F = util.vcol((post[g:g + 1, :] * X).sum(1))
                S = np.dot((post[g:g + 1, :] * X), X.T)

                wUpd = Z / X.shape[1]
                muUpd = F / Z
                CUpd = S / Z - np.dot(muUpd, muUpd.T)  # +np.eye(C.shape[0])*1e-9
                # CUpd = np.dot((post[g:g+1, :] * (D-muUpd)),(D-muUpd).T)
                if diagCov:
                    CUpd = CUpd * np.eye(CUpd.shape[0])
                gmmUpd.append((wUpd, muUpd, CUpd))  # 新的gmm

            if tiedCov:
                CTot = sum([w * C for w, mu, C in gmmUpd])
                gmmUpd = [(w, mu, CTot) for w, mu, C in gmmUpd]

            # print(gmmUpd[0])
            gmmUpd = [(w, mu, self.constraintCov(C, psi)) for w, mu, C in gmmUpd]

            gmm = gmmUpd
            lLL = []
            for (w, mu, C) in gmm:
                ll = self.logpdf_GAU_ND(X, mu, C) + np.log(w)
                lLL.append(util.vrow(ll))
            LL = np.vstack(lLL)
            margin_new = scipy.special.logsumexp(LL, axis=0)
            #print(margin_new.sum())
            _llOld = _ll
            _ll = margin_new.sum() / X.shape[1]

            if _llOld is not None:
                _deltaLL = _ll - _llOld
                # pdb.set_trace()
                # print(_deltaLL)

        return gmm

    def train(self, tied=False, bayes=False):
        # ndarray(12,576)
        DTR0 = self.DTR[:, self.LTR == 0]  # 0类的所有Data
        DTR1 = self.DTR[:, self.LTR == 1]  # 1类的所有Data

        self.mu.append(util.vcol(DTR0.mean(1)))
        self.mu.append(util.vcol(DTR1.mean(1)))
        # 去中心化
        DTRc0 = DTR0 - self.mu[0]
        DTRc1 = DTR1 - self.mu[1]
        # print(DTR0.shape)
        # print(DTRc0.shape)
        self.sigma.append(np.dot(DTRc0, DTRc0.T) / DTRc0.shape[1])
        self.sigma.append(np.dot(DTRc1, DTRc1.T) / DTRc1.shape[1])
        GMM0_init = [(1, self.mu[0], self.sigma[0])]
        GMM1_init = [(1, self.mu[1], self.sigma[1])]
        gmm_gen0 = self.LBG(GMM0_init, alpha=0.1, iterNum=self.n0, X=DTR0, diagCov=False, tiedCov=True, psi=0.01)
        gmm_gen1 = self.LBG(GMM1_init, alpha=0.1, iterNum=self.n1, X=DTR1, diagCov=False, tiedCov=True, psi=0.01)
        self.parameter = {"gmm0": gmm_gen0, "gmm1": gmm_gen1}

    def bayes_decision_threshold(self, pi1, Cfn, Cfp):
        t = np.log(pi1 * Cfn)
        t = t - np.log((1 - pi1) * Cfp)
        t = -t
        return t

    def logpdf_GMM(self, X, gmm):
        yList = []
        for w, mu, C in gmm:
            lc = self.logpdf_GAU_ND(X, mu, C) + np.log(w)
            yList.append(util.vrow(lc))
        return scipy.special.logsumexp(yList, axis=0)

    ## output llr
    def score(self, tied=False, bayes=False):
        logll0 = self.logpdf_GMM(self.DVAL, self.parameter["gmm0"])
        logll1 = self.logpdf_GMM(self.DVAL, self.parameter["gmm1"])

        # logS = np.vstack((logll0, logll1))
        # S = np.exp(logS)
        # predictLabel = np.argmax(S, axis=0)
        return logll1 - logll0


    # use effective_prior
    # def minDcf(self, score, label, epiT, fusion):
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
    #     # {res[idx] : t}
    #     for idx, t in enumerate(scoreArray):
    #         Pred = np.int32(score > t)  # 强制类型转换为int32,True 变成1，False 变成0
    #         Conf = np.zeros((2, 2))
    #         for i in range(2):
    #             for j in range(2):
    #                 Conf[i, j] = ((Pred == i) * (label == j)).sum()
    #                 TPR[idx] = Conf[1, 1] / (Conf[1, 1] + Conf[0, 1]) if (Conf[1, 1] + Conf[0, 1]) != 0.0 else 0
    #                 FPR[idx] = Conf[1, 0] / (Conf[1, 0] + Conf[0, 0]) if ((Conf[1, 0] + Conf[0, 0]) != 0.0) else 0
    #                 # FNR,FPR
    #                 FNR[idx] = 1-TPR[idx]
    #
    #         # res[idx] = piT * Cfn * (1 - TPR[idx]) + (1 - piT) * Cfp * FPR[idx]
    #         res[idx] = epiT * (1 - TPR[idx]) + (1 - epiT) * FPR[idx]
    #         sysRisk = min(epiT, (1 - epiT))
    #         res[idx] = res[idx] / sysRisk  # 除 risk of an optimal system
    #
    #         if res[idx] < minDCF:
    #             minT = t
    #             minDCF = res[idx]
    #
    #     print("minDCF in GMM is : {}".format(minDCF))
    #     if fusion:
    #         return minDCF, FNR, FPR
    #     else:
    #         return minDCF

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
