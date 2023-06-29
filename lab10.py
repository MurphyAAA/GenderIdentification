import json
import scipy.special
import GMM_load
import numpy as np
import pdb
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

# def GMM_iteration(X, gmm): # 传入的gmm是初始值，训练过程中更新
#     _llOld = None
#     _ll = None
#     _deltaLL = 1.0
#
#     while _deltaLL >= 1e-6:
#         LL = np.empty((0, X.shape[1]))
#         for (w, mu, C) in gmm:
#             arr = np.array(logpdf_GAU_ND(X, mu, C))
#             LL = np.vstack((LL, arr))
#
#         for g in np.arange(len(gmm)):
#             LL[g, :] += np.log(gmm[g][0])  # joint log-density
#             # print(i)
#         margin = scipy.special.logsumexp(LL, axis=0)  # 将log(w*N(x,mu,c)) 加起来，即log(∑w*N())
#         # E-step
#         post = np.exp(LL - margin) # 3*1000   3组gmm， 1000个sample
#         _llOld = _ll
#         _ll = margin.sum()
#         # print(_ll)
#         if _llOld is not None:
#             _deltaLL = _ll - _llOld
#
#         # M-step
#         gmmUpd = []
#         temp = []
#         Zsum =0
#         for g in range(post.shape[0]):
#             Z = post[g].sum()
#             F = vcol((post[g:g+1,:]*X).sum(1))
#             S = np.dot((post[g:g+1,:]*X), X.T)
#             temp.append((Z,F,S))
#             Zsum += Z
#         # pdb.set_trace()
#         for Z,F,S in temp:
#             wUpd = Z / Zsum
#             muUpd = F / Z
#             CUpd = S / Z - np.dot(muUpd, muUpd.T)
#             gmmUpd.append((wUpd,muUpd,CUpd)) # 新的gmm
#
#
#         gmm = gmmUpd
#
#     return gmm

def GMM_iteration(X, gmm): # 传入的gmm是初始值，训练过程中更新
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
        _llOld = _ll
        _ll = margin.sum() # log-ll for previous M-step
        # print(_ll)
        if _llOld is not None:
            _deltaLL = _ll - _llOld

        # M-step
        gmmUpd = []
        for g in range(post.shape[0]):
            Z = post[g].sum()
            F = vcol((post[g:g + 1, :] * X).sum(1))
            S = np.dot((post[g:g + 1, :] * X), X.T)
            pdb.set_trace()

            wUpd = Z / X.shape[1]
            muUpd = F / Z
            CUpd = S / Z - np.dot(muUpd, muUpd.T) #+np.eye(C.shape[0])*1e-9
            # CUpd = np.dot((post[g:g+1, :] * (D-muUpd)),(D-muUpd).T)
            gmmUpd.append((wUpd, muUpd, CUpd))  # 新的gmm

        gmm = gmmUpd

    return gmm

if __name__ == '__main__':
    gmm= GMM_load.load_gmm("data/GMM_4D_3G_init.json") # w, mu, c
    # gmm= GMM_load.load_gmm("data/GMM_1D_3G_init.json") # w, mu, c
    D = np.load("data/GMM_data_4D.npy")
    res = np.load("data/GMM_4D_3G_init_ll.npy")
    f = open("data/GMM_4D_3G_EM.json")
    gmmRes = json.load(f)

    mygmm = GMM_iteration(D, gmm)
    myres = logpdf_GMM(D,mygmm)

    print(myres.mean())
    # sns.distplot(myres,bins=25, hist=True)
    # plt.xticks(range(-10, 6)[::2])
    # plt.show()

    mu = D.mean(1)
    C = np.dot(D, D.T) / D.shape[1]
    GMM_1 = [(1.0, mu, C)]





