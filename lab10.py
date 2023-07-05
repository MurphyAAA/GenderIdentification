import json
import scipy.special
import sklearn.datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def vcol(v):
    return v.reshape((v.size, 1))


def vrow(v):
    return v.reshape((1, v.size))


def logpdf_GAU_ND(x, mu, C): #mu，c都是某一个类别的样本的平均值和协方差矩阵
    M = x.shape[0] # M 是特征数
    # print(M)
    a = M * np.log(2 * np.pi)
    _, b = np.linalg.slogdet(C)
    xc = (x - mu)
    # print("xc.shape", xc.shape)
    # xc应该每行循环列数次
    c = np.dot(xc.T, np.linalg.inv(C))
    # print(c.shape[0])
    c = np.dot(c,xc)

    c = np.diagonal(c) #点乘完了取对角线就ok
    return (-1.0 / 2.0) * (a + b + c)
def logpdf_GMM(X,gmm):
    yList = []
    for w, mu, C in gmm:
        lc = logpdf_GAU_ND(X, mu, C) + np.log(w)
        yList.append(vrow(lc))
    return scipy.special.logsumexp(yList, axis=0)


def logpdf_GMM2(X,gmm):
    # X (D,N) D 个特征， N个样本
    # D = X.shape[0] # 特征数
    S = np.empty((0,X.shape[1]))
    for (w, mu, C) in gmm:
        arr = np.array(logpdf_GAU_ND(X,mu,C))
        S = np.vstack((S,arr))

    for g in np.arange(len(gmm)):
        S[g,:] += np.log(gmm[g][0]) # joint log-density
    logdens = scipy.special.logsumexp(S, axis=0) # 将log(w*N(x,mu,c)) 加起来，即log(∑w*N())
    return logdens # (N,) 第i个样本的 log-density


def constraintCov(C, psi):
    U, s, _ = np.linalg.svd(C)
    s[s < psi] = psi
    covNew = np.dot(U, vcol(s) * U.T)
    return covNew
def GMM_iteration(X, gmm, diagCov, tiedCov, psi): # 传入的gmm是初始值，训练过程中更新
    _llOld = None
    _ll = None
    _deltaLL = 10

    while _deltaLL >= 1e-6:
        lLL = []
        for (w, mu, C) in gmm:
            ll = logpdf_GAU_ND(X, mu, C) + np.log(w)
            lLL.append(vrow(ll))
            # print(i)
        LL = np.vstack(lLL)
        margin = scipy.special.logsumexp(LL, axis=0)  # 将log(w*N(x,mu,c)) 加起来，即log(∑w*N())
        # E-step
        post = np.exp(LL - margin)  # 3*1000   3组gmm， 1000个sample
        _ll = margin.sum()/X.shape[1] # log-ll for previous M-step

        # M-step
        gmmUpd = []
        for g in range(post.shape[0]):
            Z = post[g].sum()
            F = vcol((post[g:g + 1, :] * X).sum(1))
            S = np.dot((post[g:g + 1, :] * X), X.T)

            wUpd = Z / X.shape[1]
            muUpd = F / Z
            CUpd = S / Z - np.dot(muUpd, muUpd.T) #+np.eye(C.shape[0])*1e-9
            # CUpd = np.dot((post[g:g+1, :] * (D-muUpd)),(D-muUpd).T)
            if diagCov:
                CUpd = CUpd * np.eye(CUpd.shape[0])
            gmmUpd.append((wUpd, muUpd, CUpd))  # 新的gmm

        if tiedCov:
            CTot = sum([w*C for w, mu, C in gmmUpd])
            gmmUpd = [(w, mu, CTot) for w, mu, C in gmmUpd]

        # print(gmmUpd[0])
        gmmUpd = [(w, mu, constraintCov(C, psi)) for w, mu, C in gmmUpd]


        gmm = gmmUpd
        lLL = []
        for (w, mu, C) in gmm:
            ll = logpdf_GAU_ND(X, mu, C) + np.log(w)
            lLL.append(vrow(ll))
        LL = np.vstack(lLL)
        margin_new = scipy.special.logsumexp(LL, axis=0)
        _llOld = _ll
        _ll = margin_new.sum()/X.shape[1]
        if _llOld is not None:
            _deltaLL = _ll - _llOld

    return gmm

def LBG(initGmm, alpha, iterNum, X, diagCov, tiedCov ,psi): # initGmm: [(w, mu, C), (w, mu, C) ...]
    # for i in range(iterNum):
    if(iterNum <= 0):
        return initGmm
    GMM_new = []
    for w,mu,C in initGmm:
        U,s,Vh = np.linalg.svd(C)
        d = U[:, 0:1] * s[0]**0.5 * alpha
        # pdb.set_trace()
        # GMM_iteration(X,)
        GMM_new.append((w/2.0, vcol(mu)+d, C))
        GMM_new.append((w/2.0, vcol(mu)-d, C))
    GMM_new_gen = GMM_iteration(X,GMM_new,diagCov,tiedCov,psi)
    if iterNum == 1:
        return GMM_new_gen
    else:
        return LBG(GMM_new_gen, alpha, iterNum -1, X, diagCov, tiedCov, psi)

def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L
def split_db_2tol(D, L, seed=0):
    nTrain = int(D.shape[1] * 2.0 / 3.0)  # 2/3的数据当做训练集，1/3当做测试
    np.random.seed(seed)  # 设置一个种子

    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]

    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

if __name__ == '__main__':

    D, L = load_iris()
    (DTR, LTR), (DTE, LTE) = split_db_2tol(D, L)
    DTR0 = DTR[:, LTR == 0]
    DTR1 = DTR[:, LTR == 1]
    DTR2 = DTR[:, LTR == 2]
    mu0 = vcol(DTR0.mean(1))
    mu1 = vcol(DTR1.mean(1))
    mu2 = vcol(DTR2.mean(1))
    C0 = np.dot(DTR0, DTR0.T) / DTR0.shape[1]
    C1 = np.dot(DTR1, DTR1.T) / DTR1.shape[1]
    C2 = np.dot(DTR2, DTR2.T) / DTR2.shape[1]
    w = 1.0
    # 每个类创建一个初始GMM
    GMM0_init = [(w, mu0, C0)]
    GMM1_init = [(w, mu1, C1)]
    GMM2_init = [(w, mu2, C2)]


    # LBG
    # mu = vcol(D.mean(1))
    # C = np.dot(D, D.T) / D.shape[1]
    # w = 1.0
    # GMM_init = [(w, mu, C)] # initial gmm
    # iterNum=n 则初始gmm分裂为2^n个components 即认为有2^n个高斯混合而成
    # 每个类的初始GMM生成各自的GMM
    gmm_gen0 = LBG(GMM0_init,alpha=0.1,iterNum=4, X=DTR0, diagCov=False, tiedCov=True, psi=0.01)
    gmm_gen1 = LBG(GMM1_init,alpha=0.1,iterNum=4, X=DTR1, diagCov=False, tiedCov=True, psi=0.01)
    gmm_gen2 = LBG(GMM2_init,alpha=0.1,iterNum=4, X=DTR2, diagCov=False, tiedCov=True, psi=0.01)
    # mygmm = GMM_iteration(D, gmm_gen)
    # 分别用各个类的GMM计算log-density 最后哪个类计算出来的结果最大，则预测为该类
    logll0 = logpdf_GMM(DTE,gmm_gen0)
    logll1 = logpdf_GMM(DTE,gmm_gen1)
    logll2 = logpdf_GMM(DTE,gmm_gen2)
    # print(myresll.mean()) # -7.253378442000956. 答案: -7.25337844
    logS =np.vstack((logll0,logll1,logll2))

    S = np.exp(logS)
    predictLabel = np.argmax(S, axis=0)
    res = []
    for i in range(predictLabel.size):
        if (predictLabel[i] == LTE[i]):
            res.append(True)
        else:
            res.append(False)
    corr = res.count(True)
    wrong = res.count(False)
    acc = corr / len(res)
    err = wrong / len(res)
    print("acc:", acc * 100, "%")
    print("err:", err * 100, "%")




