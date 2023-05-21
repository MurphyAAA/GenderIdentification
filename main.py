import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from scipy import stats
from DataPrepareShow import *



def split_data(D, L, seed=0):
    nTrain = int(D.shape[1] * 2.0 / 3.0)
    np.random.seed(seed) # 设置种子

    idx = np.random.permutation(D.shape[1]) # 将n个samples的索引顺序打乱
    idxTrain = idx[0:nTrain]
    idxVal = idx[nTrain:]
    DTR = D[:, idxTrain]
    DVAL = D[:, idxVal]
    LTR = L[idxTrain]
    LVAL = L[idxVal]
    return (DTR, LTR), (DVAL, LVAL)


def PCA(D,L, m):
    mu = mcol(D.mean(1))  # mean of each feature
    # centralized
    DC = D - mu
    C = np.dot(DC, DC.T)
    C = C / L.size  # 协方差矩阵
    U, s, Vh = np.linalg.svd(C)
    # print(f'U: {U.shape}')

    P = U[:, 0:m]
    DP = np.dot(P.T, D)
    # plot_scatter(DP, L)
    # print(f'DP: {DP.shape}')
    # print(f'L: {L.shape}')
    return DP

def LDA(D, L, m):
    N = D.shape[1]
    mu = mcol(D.mean(1))
    SWc = np.zeros((m, m))
    SB = np.zeros((m, m))
    for i in range(2):
        Dc = D[:, L == i]
        nc = Dc.shape[1]  # 样本数量
        muc = mcol(Dc.mean(1))
        DCc = Dc - muc
        C = np.dot(DCc, DCc.T)
        SWc += C
        M = muc - mu
        M = np.dot(M, M.T)
        M *= nc
        SB += M
    SW = SWc / N
    SB /= N
    s, U = scipy.linalg.eigh(SB, SW)
    # print(s)
    W = U[:, ::-1][:, 0:1]  # biggest eigenvector 2分类。can only reduced to 1D
    Dp = np.dot(W.T, D)
    # print(Dp.shape)
    return Dp


def logpdf_GAU_ND(x, mu, C):  # 概率密度、likelihood，x是未去中心化的原数据
    M = x.shape[0]
    a = M * np.log(2 * np.pi)
    _, b = np.linalg.slogdet(C)  # log|C| 矩阵C的绝对值的log  返回值第一个是符号，第二个是log|C|
    xc = (x - mu)
    # print(mu.shape)
    # xc应该每行循环列数次
    c = np.dot(xc.T, np.linalg.inv(C))  # np.linalg.inv求矩阵的逆

    c = np.dot(c, xc)

    c = np.diagonal(c)  # 点乘完了取对角线就ok
    return (-1.0 / 2.0) * (a + b + c) # 密度函数的log

def logpdf_GAU_ND_fast(X,mu,C):
    XC = X- mu
    M = X.shape[0]
    const = -0.5 * M * np.log(2*np.pi)
    logdet = np.linalg.slogdet(C)[1]
    L = np.linalg.inv(C)
    v = (XC * np.dot(L,XC)).sum(0)
    return const - 0.5 * logdet - 0.5 * v

def bayes_decision_threshold(pi1, Cfn, Cfp):
    t = np.log(pi1 * Cfn)
    t = t - np.log((1-pi1) * Cfp)
    t = -t
    return t
def MVG(DTR, LTR, DTE, method = "MVG"):
    DTR0 = DTR[:,LTR == 0] # 0类的所有Data
    DTR1 = DTR[:,LTR == 1] # 1类的所有Data
    mu0 = mcol(DTR0.mean(1))
    mu1 = mcol(DTR1.mean(1))
    # 去中心化
    DTRc0 = DTR0 - mu0
    DTRc1 = DTR1 - mu1
    # 协方差
    C0 = np.dot(DTRc0, DTRc0.T) / DTRc0.shape[1]
    C1 = np.dot(DTRc1, DTRc1.T) / DTRc1.shape[1]
    if method == "Bayes":
        identity = np.identity(DTR.shape[0])
        C0 = C0*identity
        C1 = C1*identity


    # log-likelihood
    tlogll0 = logpdf_GAU_ND(DTE, mu0, C0)
    tlogll1 = logpdf_GAU_ND(DTE, mu1, C1)

    Priori = 1/2
    # logS = np.vstack((tlogll0, tlogll1))
    # logSJoint = logS + np.log(Priori)
    # logSMarginal = mrow(scipy.special.logsumexp(logSJoint, axis=0))
    # logSPost = logSJoint - logSMarginal
    # SPost = np.exp(logSPost)
    #
    # predict = np.argmax(SPost, axis=0)

    llr = tlogll1 - tlogll0
    t = bayes_decision_threshold(Priori, 1, 1)
    predict = []
    for r in llr:
        if r>t:
            predict.append(1)
        else:
            predict.append(0)

    return np.array(predict)
    # DCF就是使用新的threshold 通过两个类likelihood的比值和这个新的t比较进行预测，而不是直接选最大的后验概率了！


def TiedMVG(DTR, LTR, DTE, method = "MVG"):
    DTR0 = DTR[:,LTR == 0] # 0类的所有Data
    DTR1 = DTR[:,LTR == 1] # 1类的所有Data
    mu0 = mcol(DTR0.mean(1))
    mu1 = mcol(DTR1.mean(1))
    # 去中心化
    DTRc0 = DTR0 - mu0
    DTRc1 = DTR1 - mu1
    # 协方差

    # DTR.shape:   (10,1600)
    # DTRc0.shape: (10, 491)
    # DTRc1.shape: (10, 1109)
    C = (np.dot(DTRc0, DTRc0.T)+np.dot(DTRc1, DTRc1.T)) / DTR.shape[1]
    if method == "Bayes":
        identity = np.identity(DTR.shape[0])
        C = C*identity

    # print(f'DTR.shape:{(np.dot(DTRc0, DTRc0.T)+np.dot(DTRc1, DTRc1.T)).shape}')
    # log-likelihood
    tlogll0 = logpdf_GAU_ND(DTE, mu0, C)
    tlogll1 = logpdf_GAU_ND(DTE, mu1, C)
    logS = np.vstack((tlogll0, tlogll1))
    Priori = 1/2
    logSJoint = logS + np.log(Priori)
    logSMarginal = mrow(scipy.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    SPost = np.exp(logSPost)

    predict = np.argmax(SPost, axis=0)
    return predict

def LOO_Gaussian(D, L, method = "MVG", Tied = False):
    predict = []
    LVAL = []
    for i in range(D.shape[1]):  # 不需要使用split函数划分验证集训练集了，使用k-fold( k=1 leave one out)
        DTR = np.delete(D.copy(), i, axis=1)
        LTR = np.delete(L.copy(), i, axis=0)
        DVAL = D[:, i:i + 1].copy()
        LVAL.append(L[i].copy())
        if Tied:
            pre = TiedMVG(DTR, LTR, DVAL,method)
        else:
            pre = MVG(DTR, LTR, DVAL,method)
        predict.append(pre)

    predict = np.array(predict).flatten().tolist()
    return predict, LVAL

def logreg_object(v,DTR, LTR, lam): # loss function
    w = v[0:-1]
    b = v[-1]
    w_norm = np.linalg.norm(w)

    w = mcol(w)
    reg_term = (lam/2) * (w_norm**2)
    negz = -1 * (2*LTR-1)
    fx = np.dot(w.T,DTR) + b
    logJ = np.logaddexp(0,negz * fx)
    mean_logJ = logJ.mean()
    # print(mean_logJ)
    res = reg_term + mean_logJ
    res = res.reshape(res.size,)
    return res
def BLR(DTR, LTR ,lam, DTE):
    x, f, d = scipy.optimize.fmin_l_bfgs_b(logreg_object,np.zeros(DTR.shape[0] + 1), args =(DTR,LTR,lam), approx_grad=True)
    w = x[0:-1]
    b = x[-1]
    # w = mcol(w)
    s = np.dot(mrow(w), DTE) + b
    s = s.reshape(s.size,)
    predict = []
    for i in s:
        if i>0:
            predict.append(1)
        else:
            predict.append(0)
    return predict

def plot_scatter(D, L):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    hFea = {
        0: 'male',
        1: 'female',
    }

    plt.figure()
    plt.xlabel(hFea[0])
    plt.ylabel(hFea[1])
    plt.scatter(D0[0, :], abs(D0[1, :]), label='male')  # X轴：D0[dIdx1, :]  Y轴： D0[dIdx2, :]
    plt.scatter(D1[0, :], abs(D1[1, :]), label='female')

    plt.legend()
    plt.tight_layout()
    plt.show()


def computeAccuracy(predictList, L):
    res = []
    for i, pre in enumerate(predictList):
        if (pre == L[i]):
            res.append(True)  # 预测正确
        else:
            res.append(False)
    corr = res.count(True)
    wrong = res.count(False)
    print(f'\ncorrect number:{corr}\nwrong number:{wrong}\ntotal:{len(res)}')
    acc = corr / len(res)
    err = wrong / len(res)
    return acc, err

def ConfusionMatrix(predictList, L):
    CM = np.zeros((2,2)) # 两个类
    # real class:    0   1
    # predict   : 0  TN  FN
    #             1  FP  TP
    #
    for i in range(predictList.size):
        CM[predictList[i], L[i]] += 1
    return CM


def main(hyperparameters):
    # D [ x0, x1, x2, x3, ...]  xi是列向量，每行都是一个feature
    D, L = load('./data/Train.txt')
    ## plot_hist(D,L)
    # mvg = Models.MVG()
    ## gaussianize the training data
    D_after = gaussianize(D)
    # plot_hist(D_after, L)
    # corrlationAnalysis(D)
    D = PCA(D_after, L, hyperparameters["m"])  # Dimensionality reduction  12D -> 10D
    # # DTR = LDA(DTR,LTR,m)
    (DTR, LTR), (DVAL, LVAL) = split_data(D, L)
    DTE, LTE = load('./data/Test.txt')
    # models
    method = ["MVG", "Bayes"]
    predict = MVG(DTR, LTR, DVAL,method[0]) # acc:90.0%
    # predict = MVG(DTR, LTR, DVAL, method[1]) # Bayes method: acc: 90.0%

    # predict = TiedMVG(DTR, LTR, DVAL, method[0]) # acc: 90.375%
    # predict = TiedMVG(DTR, LTR, DVAL, method[1]) # acc: 90.375%

    # predict, LVAL = LOO_Gaussian(D, L, method[0], Tied=False) # MVG acc: 90.66666666666666%
    # predict, LVAL = LOO_Gaussian(D, L, method[1], Tied=False) # Bayes acc: 90.54166666666667%
    # predict, LVAL = LOO_Gaussian(D, L, method[0], Tied=True) # TiedMVG acc: 90.54166666666667%
    # predict, LVAL = LOO_Gaussian(D, L, method[1], Tied=True)  # TiedBayes acc: 90.625%

    # predict = BLR(DTR,LTR,hyperparameters["l"],DVAL) # acc: 92%
    acc, err = computeAccuracy(predict, LVAL)

    CM = ConfusionMatrix(predict,LVAL)
    print(CM)
    print("-----------test-----------")
    print(f'|acc:{acc*100}%, err:{err*100}%|')
    print("--------------------------")



if __name__ == '__main__':

    # Hyperparameters
    hyperparameters = {"m" : 10, "l" : 0.001}
    main(hyperparameters)
