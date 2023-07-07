import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from scipy import stats
from DataPrepareShow import *
import util
# import sys
# sys.path.append('GenderIdentification/Models/')
from Models import MVG
from Models import LogisticRegression
from Models import SVM
from Models import GMM
import seaborn
import matplotlib.style as style


def PCA(D, L, m):
    mu = mcol(D.mean(1))  # mean of each feature
    # centralized
    DC = D - mu
    C = np.dot(DC, DC.T)
    C = C / L.size  # 协方差矩阵
    U, s, Vh = np.linalg.svd(C)
    P = U[:, 0:m]
    DP = np.dot(P.T, D)

    #draw PCA graph
    draw_PCA(DP,L,False,False,False)
    return DP

def draw_explain_variance(s):
    explain_variance = (s**2)/(np.sum(s**2))
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Explained Variance')
    plt.title('Explained Variance by Principal Components')
    x_data = np.arange(1, len(explain_variance) + 1)
    #adjust the x-axis ticks and show all dimension as x
    plt.xticks(x_data)
    plt.plot(x_data, np.cumsum(explain_variance))
    plt.grid(True)
    plt.show()


def draw_PCA(DP,L,hist1 = True, hist2 = True, scatter = True ):
    D1male = DP[0, L == 0]
    D1female = DP[0, L == 1]
    D2male = DP[1, L == 0]
    D2female = DP[1, L == 1]
    if hist1:
        plt.figure(1)
        plt.hist(D1male, bins=30, density=True, alpha=0.4, label='male')
        plt.hist(D1female, bins=30, density=True, alpha=0.4, label='female')
        plt.legend()
        plt.plot()
    if hist2:
        plt.figure(2)
        plt.hist(D2male, bins=30, density=True, alpha=0.4, label='male')
        plt.hist(D2female, bins=30, density=True, alpha=0.4, label='female')
        plt.legend()
        plt.plot()
    if scatter:
        plt.figure(3)
        plt.scatter(D2male, D1male, alpha=0.4,label='male')
        plt.scatter(D2female, D1female, alpha=0.4, label='female')
        plt.tight_layout()
        plt.plot()

    plt.show()
    # print(f'DP: {DP.shape}')
    # print(f'L: {L.shape}')



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
    W = U[:, ::-1][:, 0:m]  # biggest eigenvector 2分类。can only reduced to 1D
    Dp = np.dot(W.T, D)

    D1x = Dp[0, L == 0]
    D2x = Dp[0, L == 1]
    # print(Dp.shape)
    # plt.hist(Dp[0, L == 0], bins=30, density=True, alpha=0.4, label='male')
    # plt.hist(Dp[0, L == 1], bins=30, density=True, alpha=0.4, label='female')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
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
def KFold(modelName, K, D, L, piTilde, hyperPar):
    ## D0: 12 * 720
    D0 = D[:, L == 0]
    L0 = L[L == 0]
    ## D1: 12 * 1680
    D1 = D[:, L == 1]
    L1 = L[L == 1]
    ## error since will return to list
    # L1 = [x for x in L if x == 1]

    ## shuffle the sample
    np.random.seed(seed=0)
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
        # piT = D0.shape[1] / D1.shape[1]

        if modelName == "MVG":
            model = MVG.MVG(DTR, LTR, DVAL, LVAL)
            model.train(tied = False, bayes = True)
            score.append(model.score())
            label.append(LVAL)
            ##model.estimate(llr)
            # Cfn = 1
            # Cfp = ((piT * Cfn) / 0.1 - (piT * Cfn)) / (1 - piT)
            # minDCF = model.minDcfPi(score, label, Cfn,Cfp,piT)
        elif modelName == "LR":
            model = LogisticRegression.LR(DTR, LTR, DVAL, LVAL, hyperPar["lam"])
            model.train()
            score.append(model.score())
            label.append(LVAL)
            # Cfn = 1
            # Cfp = ((piT * Cfn) / 0.99 - (piT) * Cfn) / (1 - piT)
            # minDCF = model.minDcf(score, label,piTilde)
        elif modelName == "SVM":
            model = SVM.SVM(DTR, LTR, DVAL, LVAL, hyperPar) # {"C":1, "K":0, "gamma":1, "d":2, "c":0}
            #wStar = model.train_linear()
            #hyper C=1 gamma=1 K=0
            alphaStar = model.train_nolinear(util.svm_kernel_type.rbf)
            # print(alphaStar)
            #score.append(model.score(wStar))
            score.append(model.score_nolinear(alphaStar,util.svm_kernel_type.rbf))
            label.append(LVAL)
        elif modelName == "GMM":
            model = GMM.GMM(DTR, LTR, DVAL, LVAL, hyperPar)
            model.train()
            score.append(model.score())
            label.append(LVAL)
    # print("piT is {}".format(piT))
    # print(f'score[0]={score[0].mean()}')

    minDCF = model.minDcf(score, label, piTilde)
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


# def LOO_Gaussian(D, L, method="MVG", Tied=False):
#     predict = []
#     LVAL = []
#     for i in range(D.shape[1]):  # 不需要使用split函数划分验证集训练集了，使用k-fold( k=1 leave one out)
#         DTR = np.delete(D.copy(), i, axis=1)
#         LTR = np.delete(L.copy(), i, axis=0)
#         DVAL = D[:, i:i + 1].copy()
#         LVAL.append(L[i].copy())
#         if Tied:
#             pre = TiedMVG(DTR, LTR, DVAL, method)
#         else:
#             pre = MVG(DTR, LTR, DVAL, method)
#         predict.append(pre)
#
#     predict = np.array(predict).flatten().tolist()
#     return predict, LVAL


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


def KFoldHyper(modelName, hyperParList, K, D, L, piTilde):


    bestminDCF = 1
    bestHyper = 0
    y = []
    w0y = []
    if modelName == "LR":
        x = hyperParList["lam"]
        for i in hyperParList["lam"]:
            hyperPar = {"lam": i}
            model, minDCF = KFold("LR", K, D, L, piTilde, hyperPar)
            y.append(minDCF)
            print("lambda = {}:  minDCF:{} ".format(i, minDCF))
            if minDCF < bestminDCF:
                bestminDCF = minDCF
                bestHyper = {"lam": i}

    if modelName == "GMM":
        for n0 in hyperParList["n0"]:
            y.clear()
            for n1 in hyperParList["n1"]:
                hyperPar = {'n0': n0, 'n1': n1}
                model, minDCF = KFold("GMM", K, D, L, piTilde, hyperPar)
                y.append(minDCF)
                print("n0 = {} n1={}:  minDCF:{} ".format(n0, n1, minDCF))
                if minDCF < bestminDCF:
                    bestminDCF = minDCF
                    bestHyper = {'n0': n0, 'n1': n1}
            w0y.append(y)

    if modelName == "SVM":
        for C in hyperParList["C"]:
            model, minDCF = KFold("SVM", K, D, L, piTilde, {"C":C, "K":0, "loggamma":1, "d":2, "c":1})
            y.append(minDCF)
            print("C = {}:  minDCF:{} ".format(C, minDCF))
            if minDCF < bestminDCF:
                bestminDCF = minDCF
                bestHyper = {"C": C}



    # plt.grid(True)
    # plt.xscale('log')
    # plt.plot(x, y)
    # plt.xlabel('lambda')
    # plt.ylabel('minDCF')
    # plt.title('Line Chart')
    # plt.show()
    return bestHyper, model, bestminDCF


def ConfusionMatrix(predictList, L):
    CM = np.zeros((2, 2))  # 两个类
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

    #plot heatmap
    #corrlationAnalysis(D)

    # Dimensionality reduction
    left_dim = 9
    #D = PCA(D, L, left_dim)
    # D = LDA(D_Znorm, L, 1)


    #Model choosen list=["MVG","LR","SVM","GMM"]
    model = "GMM"
    if model == "MVG":
        model,minDCF= KFold("MVG", 5, D, L,0.5,None)
        print("MVG : bestminDCF:{} ".format(minDCF))
    elif model == "LR":
        hyperParListLR = {"lam": [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 1, 10]}
        hy,model,minDCF= KFoldHyper("LR", hyperParListLR, 5, D, L,0.5)
        print("Logic regression : with hyper_paramter lambda = {}  bestminDCF:{}   ".format(hy["lam"],  minDCF))
    elif model == "GMM":
        hyperParListGMM = {"n0": [0,1,2,3], "n1": [0,1,2,3]}
        hy, model, minDCF = KFoldHyper("GMM", hyperParListGMM, 5, D_Znorm, L, 0.5)
        print("GMM : with hyperparamter n0 ={}, n1={}, bestminDCF:{}  ".format(hy["n0"], hy["n1"], minDCF))
    elif model == "SVM":
        hyperParListSVM = {"C":[ 2 * 10 ** -5, 5 * 10 ** -5, 10 ** -4, 2 * 10 ** -4, 5 * 10 ** -4, 10 ** -3, 2 * 10 ** -3, 5 * 10 ** -3, 10**-2],"K":0, "loggamma":1,"d":2,"c":1}
        hy, model, minDCF = KFoldHyper("SVM", hyperParListSVM, 5, D, L, 0.5)
        print("SVM : bestminDCF:{} ".format(minDCF))
    else:
        print("no corresponding model")


if __name__ == '__main__':
    main()
