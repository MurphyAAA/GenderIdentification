import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from scipy import stats
from DataPrepareShow import *

# import sys
# sys.path.append('GenderIdentification/Models/')
from Models import MVG
from Models import LogisticRegression
from Models import SVM


def PCA(D, L, m):
    mu = mcol(D.mean(1))  # mean of each feature
    # centralized
    DC = D - mu
    C = np.dot(DC, DC.T)
    C = C / L.size  # 协方差矩阵
    U, s, Vh = np.linalg.svd(C)
    # print(f'U: {U.shape}')

    P = U[:, 0:m]
    DP = np.dot(P.T, D)

    D1x = DP[0, L == 0]
    D2x = DP[0, L == 1]

    D1y = DP[1, L == 0]
    D2y = DP[1, L == 1]

    # explain_variance = s/np.sum(s)
    # plt.plot(np.cumsum(explain_variance))
    # plt.xlabel('Number of Principal Components')
    # plt.ylabel('Explained Variance')
    # plt.title('Explained Variance by Principal Components')
    # plt.grid(True)
    # plt.show()


    # plt.scatter(D1y, D1x, alpha=0.4,label='male')
    # plt.scatter(D2y, D2x, alpha=0.4, label='female')
    # plt.hist(DP[0, L == 0], bins=30, density=True, alpha=0.4, label='male')
    # plt.hist(DP[0, L == 1], bins=30, density=True, alpha=0.4, label='female')
    # plt.hist(DP[0, L == 0], bins=30, density=True, alpha=0.4, label='male')
    # plt.hist(DP[0, L == 1], bins=30, density=True, alpha=0.4, label='female')
    # plt.legend()
    # plt.tight_layout()

    # plt.show()
    # print(f'DP: {DP.shape}')
    # print(f'L: {L.shape}')


    return DP


def LDA(D, L, m):
    N = D.shape[1]
    numFeature = D.shape[0]
    mu = mcol(D.mean(1))
    SWc = np.zeros((numFeature, numFeature))
    SB = np.zeros((numFeature, numFeature))
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
    W = U[:, ::-1][:, 0:m]  # biggest eigenvector 2分类。can only reduced to 1D
    Dp = np.dot(W.T, D)


    D1x = Dp[0, L == 0]
    D2x = Dp[0, L == 1]
    print(Dp.shape)
    plt.hist(Dp[0, L == 0], bins=30, density=True, alpha=0.4, label='male')
    plt.hist(Dp[0, L == 1], bins=30, density=True, alpha=0.4, label='female')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return Dp

#
# def logpdf_GAU_ND(x, mu, C):  # 概率密度、likelihood，x是未去中心化的原数据
#     M = x.shape[0]
#     a = M * np.log(2 * np.pi)
#     _, b = np.linalg.slogdet(C)  # log|C| 矩阵C的绝对值的log  返回值第一个是符号，第二个是log|C|
#     xc = (x - mu)
#     # print(mu.shape)
#     # xc应该每行循环列数次
#     c = np.dot(xc.T, np.linalg.inv(C))  # np.linalg.inv求矩阵的逆
#
#     c = np.dot(c, xc)
#
#     c = np.diagonal(c)  # 点乘完了取对角线就ok
#     return (-1.0 / 2.0) * (a + b + c)  # 密度函数的log


# def TiedMVG(DTR, LTR, DTE, method="MVG"):
#     DTR0 = DTR[:, LTR == 0]  # 0类的所有Data
#     DTR1 = DTR[:, LTR == 1]  # 1类的所有Data
#     mu0 = mcol(DTR0.mean(1))
#     mu1 = mcol(DTR1.mean(1))
#     # 去中心化
#     DTRc0 = DTR0 - mu0
#     DTRc1 = DTR1 - mu1
#     # 协方差
#
#     # DTR.shape:   (10,1600)
#     # DTRc0.shape: (10, 491)
#     # DTRc1.shape: (10, 1109)
#     C = (np.dot(DTRc0, DTRc0.T) + np.dot(DTRc1, DTRc1.T)) / DTR.shape[1]
#     if method == "Bayes":
#         identity = np.identity(DTR.shape[0])
#         C = C * identity
#
#     # print(f'DTR.shape:{(np.dot(DTRc0, DTRc0.T)+np.dot(DTRc1, DTRc1.T)).shape}')
#     # log-likelihood
#     tlogll0 = logpdf_GAU_ND(DTE, mu0, C)
#     tlogll1 = logpdf_GAU_ND(DTE, mu1, C)
#     logS = np.vstack((tlogll0, tlogll1))
#     Priori = 1 / 2
#     logSJoint = logS + np.log(Priori)
#     logSMarginal = mrow(scipy.special.logsumexp(logSJoint, axis=0))
#     logSPost = logSJoint - logSMarginal
#     SPost = np.exp(logSPost)
#
#     predict = np.argmax(SPost, axis=0)
#     return predict


##kfold for hyperparameter
## hyper:  dict{hypername:val}
## model: string
## K: int
## D: 12*2400
## L : array([1,0,0,0...]) 2400
def KFold(modelName, K, D, L,  piTilde,hyperPar):
    ## D0: 12 * 720
    D0 = D[:, L == 0]
    L0 = L[L == 0]
    ## D1: 12 * 1680
    D1 = D[:, L == 1]
    L1 = L[L == 1]
    ## error since will return to list
    # L1 = [x for x in L if x == 1]

    ## shuffle the sample
    np.random.seed(seed = 0)
    ind0 = np.random.permutation(D0.shape[1])
    ind1 = np.random.permutation(D1.shape[1])
    bestAcc = 0
    ## ind range for every fold
    sInFoldD0 = int(D0.shape[1] / K)
    sInFoldD1 = int(D1.shape[1] / K)
    score = []
    label = []
    for i in range(K):
        valD0Ind = ind0[i * sInFoldD0: (i + 1) * sInFoldD0]
        traD0Ind = [x for x in ind0 if x not in valD0Ind]
        D0VAL = D0[:, valD0Ind]
        D0TR = D0[:, traD0Ind]
        L0VAL = L0[valD0Ind]
        L0TR = L0[traD0Ind]

        valD1Ind = ind1[i * sInFoldD1: (i + 1) * sInFoldD1]
        traD1Ind = [x for x in ind1 if x not in valD1Ind]
        D1VAL = D1[:, valD1Ind]
        D1TR = D1[:, traD1Ind]
        L1VAL = L1[valD1Ind]
        L1TR = L1[traD1Ind]

        ## combine D0TR + D1TR
        DTR = np.concatenate((D0TR, D1TR), axis=1)
        LTR = np.concatenate((L0TR, L1TR))
        DVAL = np.concatenate((D0VAL, D1VAL), axis=1)
        LVAL = np.concatenate((L0VAL, L1VAL))
        ## paralist = {acc,[parameter]}
        piT = D0.shape[1] / D1.shape[1]

        if modelName == "MVG":
            model = MVG.MVG(DTR, LTR, DVAL, LVAL)
            model.train()
            score.append(model.score())
            label.append(LVAL)
            ##model.estimate(llr)
            minDCF = model.minDcf(score, label,piTilde)
            # Cfn = 1
            # Cfp = ((piT * Cfn) / 0.1 - (piT * Cfn)) / (1 - piT)
            # minDCF = model.minDcfPi(score, label, Cfn,Cfp,piT)



        if modelName == "LR":
            model = LogisticRegression.LR(DTR, LTR, DVAL, LVAL, hyperPar)
            model.train()
           ## model.estimate()
            score.append(model.score())
            label.append(LVAL)
            # Cfn = 1
            # Cfp = ((piT * Cfn) / 0.99 - (piT) * Cfn) / (1 - piT)
            minDCF = model.minDcf(score, label,piTilde)

        if modelName == "SVM":
            model = SVM.SVM(DTR, LTR, DVAL, LVAL, 1)
            wStar = model.train_linear(1)
            alphaStar = model.train_RBF(1,1,0)


            score.append(model.score(wStar,1))
            #score.append(model.score_rbf(alphaStar,1,1))
            label.append(LVAL)
            minDCF = model.minDcf(score, label,0.5)

    #print("piT is {}".format(piT))

    return model, minDCF


def split_data(D, L, seed=0):
    nTrain = int(D.shape[1] * 2.0 / 3.0)
    np.random.seed(seed)  # 设置种子

    idx = np.random.permutation(D.shape[1])  # 将n个samples的索引顺序打乱
    idxTrain = idx[0:nTrain]
    idxVal = idx[nTrain:]
    DTR = D[:, idxTrain]
    DVAL = D[:, idxVal]
    LTR = L[idxTrain]
    LVAL = L[idxVal]
    return (DTR, LTR), (DVAL, LVAL)


def LOO_Gaussian(D, L, method="MVG", Tied=False):
    predict = []
    LVAL = []
    for i in range(D.shape[1]):  # 不需要使用split函数划分验证集训练集了，使用k-fold( k=1 leave one out)
        DTR = np.delete(D.copy(), i, axis=1)
        LTR = np.delete(L.copy(), i, axis=0)
        DVAL = D[:, i:i + 1].copy()
        LVAL.append(L[i].copy())
        if Tied:
            pre = TiedMVG(DTR, LTR, DVAL, method)
        else:
            pre = MVG(DTR, LTR, DVAL, method)
        predict.append(pre)

    predict = np.array(predict).flatten().tolist()
    return predict, LVAL




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



def KFoldHyper(hyperParList, K, D, L, piTilde):
    bestminDCF = 1
    bestHyper=0
    x = hyperParList[0]["lam"]
    y = []
    for i in hyperParList[0]["lam"]:
        model,minDCF = KFold("LR", K, D, L, piTilde, i)
        y.append(minDCF)
        #print(minDCF)
        #print("lambda = {}:  minDCF:{} ".format(i,minDCF))

        if minDCF < bestminDCF:
            bestminDCF = minDCF
            bestHyper = i
    ## with norm
    ## x = [ 1e-05, 0.0001, 0.001, 0.01, 0.1, 1, 10]
    # print(x)
    print(y)
    return bestHyper,model,bestminDCF
def ConfusionMatrix(predictList, L):
    CM = np.zeros((2,2)) # 两个类
    # real class:    0   1
    # predict   : 0  TN  FN
    #             1  FP  TP
    #
    for i in range(predictList.size):
        CM[predictList[i], L[i]] += 1
    return CM


def main():
    # D [ x0, x1, x2, x3, ...]  xi是列向量，每行都是一个feature
    D, L = load('./data/Train.txt')
    ## plot_hist(D,L)

    ## gaussianize the training data
    D_gaussian = gaussianize(D)
    D_Znorm = Z_norm(D)

    # plot_hist(D_after, L)
    # corrlationAnalysis(D)
    #D0 = D[:, L == 0]  # 0类的所有Data
    #D1 = D[:, L == 1]  # 1类的所有Data
    # D.shape:   (10,1600)
    # D0.shape: (10, 491)
    # D1.shape: (10, 1109)
    # True: 0 False: 1

    # D = PCA(D, L, 10)  # Dimensionality reduction  12D -> 10D
    D = LDA(D_Znorm, L, 4)
    # model,minDCF= KFold("MVG", 5, D, L,0.5,0)
    # print("MVG : bestminDCF:{}   ".format(minDCF))

    hyperParList = [{"lam": [ 10 ** -5,10 ** -4,10 ** -3,10 ** -2,10 ** -1,1,10]}]
    hy,model,minDCF= KFoldHyper(hyperParList, 5, D, L,0.5)
    D= [0.11865079365079362, 0.11865079365079362, 0.11865079365079362, 0.11785714285714281, 0.11765873015873013, 0.13293650793650796, 0.2180555555555555]
    D_11 = [0.12202380952380948, 0.12202380952380948, 0.12202380952380948, 0.1226190476190476, 0.12480158730158727, 0.1396825396825397, 0.21964285714285714]
    D_10 = [0.16388888888888886, 0.16388888888888886, 0.16388888888888886, 0.16626984126984123, 0.16686507936507933, 0.17242063492063497, 0.22876984126984126]
    D_Znorm=[ 0.11865079365079362, 0.11865079365079365, 0.11845238095238092, 0.1373015873015873, 0.19999999999999996, 0.33670634920634923, 0.46210317460317457]
    # print("Logic regression : with hyperparamter lambda = {}  bestminDCF:{}   ".format(hy,  minDCF))
    # plt.grid(True)
    # plt.xscale('log')
    line1, = plt.plot(hyperParList[0]["lam"], D,label='Log-Reg(no PCA)')
    line2, = plt.plot(hyperParList[0]["lam"], D_11, label='Log-Reg(PCA = 11)')
    line3, = plt.plot(hyperParList[0]["lam"], D_10, label='Log-Reg(PCA = 10)')

    # line2, = plt.plot(hyperParList[0]["lam"], D_Znorm,label='Log-Reg(z-norm)')
    #
    # plt.legend(handles=[line1, line2, line3])
    # plt.xlabel('lambda')
    # plt.ylabel('minDCF')
    # plt.show()

    # model,minDCF= KFold("SVM", 5, D, L,0.5,0)



if __name__ == '__main__':
    # Hyperparameters
    # hyperparameters = {"m": 10, "l": 0.001}
    main()




    # D = LDA(D,L,2)
    # plot_hist(D, L)
    # (DTR, LTR), (DVAL, LVAL) = split_data(D, L)
    # DTE, LTE = load('./data/Test.txt')
    # # models
    # method = ["MVG", "Bayes"]
    # predict = MVG(DTR, LTR, DVAL,method[0]) # acc:90.0%
    # # predict = MVG(DTR, LTR, DVAL, method[1]) # Bayes method: acc: 90.0%
    #
    # # predict = TiedMVG(DTR, LTR, DVAL, method[0]) # acc: 90.375%
    # # predict = TiedMVG(DTR, LTR, DVAL, method[1]) # acc: 90.375%
    #
    # # predict, LVAL = LOO_Gaussian(D, L, method[0], Tied=False) # MVG acc: 90.66666666666666%
    # # predict, LVAL = LOO_Gaussian(D, L, method[1], Tied=False) # Bayes acc: 90.54166666666667%
    # # predict, LVAL = LOO_Gaussian(D, L, method[0], Tied=True) # TiedMVG acc: 90.54166666666667%
    # # predict, LVAL = LOO_Gaussian(D, L, method[1], Tied=True)  # TiedBayes acc: 90.625%
    #
    # predict = BLR(DTR,LTR,0.001,DVAL) # acc: 92%
    # acc, err = computeAccuracy(predict, LVAL)
    #
    # CM = ConfusionMatrix(predict,LVAL)
    # print(CM)
    # print("-----------test-----------")
    # print(f'|acc:{acc*100}%, err:{err*100}%|')
    # print("--------------------------")

