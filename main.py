from matplotlib import pylab
import scipy.linalg
from DataPrepareShow import *
import util
from Models import MVG
from Models import LogisticRegression
from Models import SVM
from Models import GMM

from ScoreCalibration import ScoreCalibration


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
    # return DP
    return P

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

##kfold for hyperparameter
## hyper:  dict{hypername:val}
## model: string
## K: int
## D: 12*2400
## L : array([1,0,0,0...]) 2400

def FusionKFold(K, D, L, piTilde, hyperPar, modelList, calibration=False):
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
    modelDict = {}
    actDCFs= {}
    minDCFs = {}
    scoreDict = {
        'MVG': [],
        'LR': [],
        'SVM_Linear': [],
        'SVM_nonlinear': [],
        'GMM': []
    }
    label=[]
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
        if "MVG" in modelList:
            modelDict["MVG"] = MVG.MVG(DTR, LTR, DVAL, LVAL)
            modelDict["MVG"].train(tied=True, bayes=True) # train 的时候就得到了训练的参数， 只需要在score的函数里面传进入测试集，就相当于用训练好的模型跑测试集，即可实现evaluation
            scoreDict["MVG"].append(modelDict["MVG"].score())
            # labelDict["MVG"].append(LVAL)
            ##model.estimate(llr)
            # Cfn = 1
            # Cfp = ((piT * Cfn) / 0.1 - (piT * Cfn)) / (1 - piT)
            # minDCF = model.minDcfPi(score, label, Cfn,Cfp,piT)
        if "LR" in modelList:
            modelDict["LR"] = LogisticRegression.LR(DTR, LTR, DVAL, LVAL, hyperPar["LR"]["lam"])
            modelDict["LR"].train()
            scoreDict["LR"].append(modelDict["LR"].score())
            # labelDict["LR"].append(LVAL)
            # Cfn = 1
            # Cfp = ((piT * Cfn) / 0.99 - (piT) * Cfn) / (1 - piT)
            # minDCF = model.minDcf(score, label,piTilde)
        if "SVM_Linear" in modelList:
            modelDict["SVM_Linear"] = SVM.SVM(DTR, LTR, DVAL, LVAL, hyperPar["SVM_Linear"], "SVM_Linear")  # {"C":1, "K":0, "gamma":1, "d":2, "c":0}
            modelDict["SVM_Linear"].train_linear()
            scoreDict["SVM_Linear"].append(modelDict["SVM_Linear"].score())
            # labelDict["SVM_Linear"].append(LVAL)
        if "SVM_nonlinear" in modelList:
            modelDict["SVM_nonlinear"] = SVM.SVM(DTR, LTR, DVAL, LVAL, hyperPar["SVM_nonlinear"], "SVM_nonlinear")  # {"C":1, "K":0, "gamma":1, "d":2, "c":0}
            # hyper C=1 gamma=1 K=0
            modelDict["SVM_nonlinear"].train_nonlinear(util.svm_kernel_type.poly)
            scoreDict["SVM_nonlinear"].append(modelDict["SVM_nonlinear"].score_nonlinear(util.svm_kernel_type.poly))
            # labelDict["SVM_nonlinear"].append(LVAL)
        if "GMM" in modelList:
            modelDict["GMM"] = GMM.GMM(DTR, LTR, DVAL, LVAL, hyperPar["GMM"])
            modelDict["GMM"].train()
            scoreDict["GMM"].append(modelDict["GMM"].score())
            # labelDict["GMM"].append(LVAL)
        label.append(LVAL)
    # print("piT is {}".format(piT))
    # print(f'score[0]={score[0].mean()}')
    for m in modelList: # 将k折的score拼成一个： (5,480) -> (1,2400)
        scoreDict[m] = util.vrow(np.hstack(scoreDict[m]))
    label = np.hstack(label)

    if len(modelList)>1:
        non_empty_arrays = [v for v in scoreDict.values() if len(v)!=0]
        data_fusion = np.vstack(non_empty_arrays)
        scoreDict["fusion"] = ScoreCalibration(data_fusion, label).KFoldCalibration()
        minDCFs["fusion"] = util.minDcf("[fusion]", scoreDict["fusion"], label, piTilde, False)
        actDCFs["fusion"] = util.normalizedDCF("[fusion]", scoreDict["fusion"], label, piTilde, 1, 1, False)

    for m in modelList:
        if calibration:
            scoreDict[m] = ScoreCalibration(scoreDict[m], label).KFoldCalibration()
        # if len(scoreDict[m]) != 0:
        minDCFs[m] = util.minDcf(f"[{modelDict[m].name}]", scoreDict[m], label, piTilde, False)
        actDCFs[m] = util.normalizedDCF(f"[{modelDict[m].name}]", scoreDict[m], label, piTilde, 1, 1, False)

    return modelDict, actDCFs, minDCFs
def KFold(modelName, K, D, L, piTilde, hyperPar,fusion,calibration):
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
            model.train(tied = True, bayes = True)
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
        elif modelName == "SVM_Linear":
            model = SVM.SVM(DTR, LTR, DVAL, LVAL, hyperPar) # {"C":1, "K":0, "gamma":1, "d":2, "c":0}
            model.train_linear()
            score.append(model.score())
            label.append(LVAL)
        elif modelName == "SVM_nonlinear":
            model = SVM.SVM(DTR, LTR, DVAL, LVAL, hyperPar)  # {"C":1, "K":0, "gamma":1, "d":2, "c":0}
            # hyper C=1 gamma=1 K=0
            model.train_nonlinear(util.svm_kernel_type.poly)
            score.append(model.score_nonlinear(util.svm_kernel_type.poly))
            label.append(LVAL)
        elif modelName == "GMM":
            model = GMM.GMM(DTR, LTR, DVAL, LVAL, hyperPar)
            model.train()
            score.append(model.score())
            label.append(LVAL)
    # print("piT is {}".format(piT))
    # print(f'score[0]={score[0].mean()}')
    score = util.vrow(np.hstack(score))
    label = np.hstack(label)
    if calibration:
        score = ScoreCalibration(score, label).KFoldCalibration()
    if fusion:

        minDCF, FNR, FPR = util.minDcf(modelName, score, label, piTilde, fusion)
        actDCF, FNR, FPR = util.normalizedDCF(modelName, score, label, piTilde, 1, 1, fusion)
        # pdb.set_trace()
        # return model, score, FNR, FPR
        return model, score, label, actDCF, minDCF
    else:
        minDCF = util.minDcf(modelName, score, label, piTilde,fusion)
        actDCF = util.normalizedDCF(modelName, score, label, piTilde, 1, 1, fusion)
        return model, actDCF, minDCF


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
            model, minDCF = KFold("LR", K, D, L, piTilde, hyperPar,False)
            y.append(minDCF)
            print("lambda = {}:  minDCF:{} ".format(i, minDCF))
            if minDCF < bestminDCF:
                bestminDCF = minDCF
                bestModel = model
                bestHyper = {"lam": i}

    elif modelName == "GMM":
        for n0 in hyperParList["n0"]:
            y.clear()
            for n1 in hyperParList["n1"]:
                hyperPar = {'n0': n0, 'n1': n1}
                model, minDCF = KFold("GMM", K, D, L, piTilde, hyperPar,False)
                y.append(minDCF)
                print("n0 = {} n1={}:  minDCF:{} ".format(n0, n1, minDCF))
                if minDCF < bestminDCF:
                    bestminDCF = minDCF
                    bestModel = model
                    bestHyper = {'n0': n0, 'n1': n1}
            w0y.append(y)

    elif modelName == "SVM_Linear":
        for C in hyperParList["C"]:
            model, minDCF = KFold("SVM_Linear", K, D, L, piTilde, {"C":C},False)
            y.append(minDCF)
            print("C = {}:  minDCF:{} ".format(C, minDCF))
            if minDCF < bestminDCF:
                bestminDCF = minDCF
                bestModel = model
                bestHyper = {"C": C}
    elif modelName == "SVM_nonlinear":
        for C in hyperParList["C"]:
            model, minDCF = KFold("SVM_nonlinear", K, D, L, piTilde, {"C":C, "K":0, "loggamma": hyperParList["loggamma"], "d":2, "c":1},False)
            y.append(minDCF)
            print("C = {}:  minDCF:{} ".format(C, minDCF))
            if minDCF < bestminDCF:
                bestminDCF = minDCF
                bestModel = model
                bestHyper = {"C": C}


    # plt.grid(True)
    # plt.xscale('log')
    # plt.plot(x, y)
    # plt.xlabel('lambda')
    # plt.ylabel('minDCF')
    # plt.title('Line Chart')
    # plt.show()
    return bestHyper, bestModel, bestminDCF


def ConfusionMatrix(predictList, L):
    CM = np.zeros((2, 2))  # 两个类
    # real class:    0   1
    # predict   : 0  TN  FN
    #             1  FP  TP
    #
    for i in range(predictList.size):
        CM[predictList[i], L[i]] += 1
    return CM

def DET(D, Dz, L, piT):
    plt.title('DET')
    # -1- GMM 正类：4个高斯+Tied  负类：4个高斯+Tied
    hyperPar = {'n0': 2, 'n1': 2}
    _, score_GMM,FNR, FPR = KFold("GMM", 5, Dz, L, piT, hyperPar,True,calibration=False)
    plt.plot(FPR, FNR, color='red', label='GMM')
    # -2- MVG Tied Daigonal non-Znorm
    _, score_MVG,FNR, FPR = KFold("MVG", 5, D, L, piT, None, True,calibration=False)
    plt.plot(FPR, FNR, color='green', label='MVG')
    # -3- SVM - 线性：C=0.01 ,
    C=0.01
    _, score_SVM_l,FNR, FPR = KFold("SVM_Linear", 5, D, L, piT,{"C": C, "K": 0}, True,calibration=False)
    plt.plot(FPR, FNR, color='skyblue', label='linear SVM')
    # -4- SVM - poly C=0.1
    # C = 0.1
    # hyperPar = {"K":0, "loggamma":1,"d":2,"c":1}
    # _, score_SVM_nl,FNR, FPR = KFold("SVM_nonlinear", 5, D, L, piT,{"C": C, "K": hyperPar["K"], "loggamma": hyperPar["loggamma"], "d": 2, "c": 1}, True,calibration=False)
    # plt.plot(FPR, FNR, color='blue', label='non linear SVM')
    # -5- LR里lambda = 0.001
    hyperPar = {"lam": 0.001}
    _, score_LR,FNR, FPR = KFold("LR", 5, D, L, piT, hyperPar, True,calibration=False)
    plt.plot(FPR, FNR, color='black', label='LR')


    # -6- funsion  GMM + SVM_l
    # score_GMM = np.hstack(score_GMM)
    # score_SVM_l = np.hstack(score_SVM_l)
    # score_data = np.vstack((score_GMM,score_SVM_l))
    # _, score_GMM_SVMl, FNR, FPR = KFold("LR", 5, score_GMM, L, piT, {"lam": 0}, True, calibration=False)
    # plt.plot(FPR, FNR, color='orangered', label='fusion')

    plt.xscale('log')  # 设置横轴为对数尺度
    plt.yscale('log')  # 设置纵轴为对数尺度
    plt.grid(True)
    plt.legend()  # 显示图例
    plt.xlabel('False Positive Rate')
    plt.ylabel('False Negative Rate')
    # plt.savefig('./images/actual_DET_GMM_MVG_LSVM_NLSVM_LR.jpg') # act DCF只有一个threshold 所以FPR FNR就是一个值，没有DET图
    plt.show()


def BayesErrorPlot(D, Dz, L):
    plt.title('Bayes Error Plot')
    effPriorLogOdds = np.linspace(-4, 4, 21)
    hyperPar_GMM = {'n0': 2, 'n1': 2}
    hyperPar_SVM_Linear = {"C": 0.01, "K": 0}
    hyperPar_SVM_nonlinear = {"C": 0.1, "K": 0, "loggamma": 1, "d": 2, "c": 1}
    hyperPar_LR = {"lam": 0.001}
    hyperPar = {"GMM":hyperPar_GMM,
                "SVM_Linear":hyperPar_SVM_Linear,
                "SVM_nonlinear":hyperPar_SVM_nonlinear,
                "LR":hyperPar_LR}

    modelList = ["GMM","SVM_Linear","MVG"]
    colorList = ["r", "g", "b"]

    effP = np.zeros(effPriorLogOdds.size)
    dcfDict={}
    mindcfDict = {}
    for m in modelList:
        dcfDict[m] = np.zeros(effPriorLogOdds.size)
        mindcfDict[m] = np.zeros(effPriorLogOdds.size)
    if len(modelList) > 1:
        dcfDict["fusion"] = np.zeros(effPriorLogOdds.size)
        mindcfDict["fusion"] = np.zeros(effPriorLogOdds.size)

    for idx, p in enumerate(effPriorLogOdds):
        effP[idx] = (1 + np.exp(-p)) ** (-1)
        _, actDcfs, minDcfs= FusionKFold(5,D,L,effP[idx], hyperPar,modelList,calibration=True)
        for m in modelList:
            dcfDict[m][idx] = actDcfs[m]
            mindcfDict[m][idx] = minDcfs[m]
        if len(modelList)>1:
            dcfDict["fusion"][idx] = actDcfs["fusion"]
            mindcfDict["fusion"][idx] = minDcfs["fusion"]
        # _,s, l, dcf[idx], mindcf[idx] = KFold("GMM", 5, Dz, L, effP[idx], hyperPar, True, calibration=True)

    for i, m in enumerate(modelList):

        plt.plot(effPriorLogOdds, dcfDict[m],label=f'{m} DCF',color=colorList[i])
        plt.plot(effPriorLogOdds, mindcfDict[m],label=f'{m} min DCF',color=colorList[i], linestyle="--" )
    plt.plot(effPriorLogOdds, dcfDict["fusion"], label='fusion DCF', color='black')
    plt.plot(effPriorLogOdds, mindcfDict["fusion"], label='fusion min DCF', color='black', linestyle="--")

    plt.grid(True)
    plt.legend()  # 显示图例
    plt.ylim([0,0.5])
    plt.xlim([-4,4])
    plt.xlabel(r'$\log(\frac{\pi}{1-\pi})$')
    plt.ylabel("DCF")
    # plt.savefig('./images/bayes_error_plot_GMM_SVM_l_MVG_calibration.jpg')
    pylab.show()

def Evaluation(DTR, LTR, DTE, LTE):
    hyperPar_GMM = {'n0': 2, 'n1': 2}
    hyperPar_SVM_Linear = {"C": 0.01, "K": 0}
    hyperPar_SVM_nonlinear = {"C": 0.1, "K": 0, "loggamma": 1, "d": 2, "c": 1}
    hyperPar_LR = {"lam": 0.001}
    hyperPar = {"GMM": hyperPar_GMM,
                "SVM_Linear": hyperPar_SVM_Linear,
                "SVM_nonlinear": hyperPar_SVM_nonlinear,
                "LR": hyperPar_LR}

    effP = 0.5
    # ["GMM", "SVM_Linear", "MVG"]
    # ["r", "g", "b"]
    modelList = ["GMM"]
    dcfDict = {}
    mindcfDict = {}
    scoreDict = {
        'MVG': [],
        'LR': [],
        'SVM_Linear': [],
        'SVM_nonlinear': [],
        'GMM': []
    }
    for m in modelList:
        dcfDict[m] = 0
        mindcfDict[m] = 0
    if len(modelList) > 1:
        dcfDict["fusion"] = 0
        mindcfDict["fusion"] = 0
    # Train
    models, _, _= FusionKFold(5,DTR,LTR,effP, hyperPar,modelList,calibration=True)
    print("____________*EVALUATION*____________")


    for model in models.values():
        if model.name == "SVM_nonlinear":
            scoreDict[model.name]= model.evaluation_nonlinear(DTE,util.svm_kernel_type.poly)
        else:
            scoreDict[model.name]= model.evaluation(DTE)

    if len(modelList)>1:
        non_empty_arrays = [v for v in scoreDict.values() if len(v)!=0]
        data_fusion = np.vstack(non_empty_arrays)
        scoreDict["fusion"] = ScoreCalibration(data_fusion, LTE).KFoldCalibration()


    plt.title('Bayes Error Plot')
    effPriorLogOdds = np.linspace(-4, 4, 21)
    effP = np.zeros(effPriorLogOdds.size)
    colorList = ["r", "g", "b"]

    if len(modelList) > 1:
        dcfDict["fusion"] = np.zeros(effPriorLogOdds.size)
        mindcfDict["fusion"] = np.zeros(effPriorLogOdds.size)
    for m in modelList:
        dcfDict[m] = np.zeros(effPriorLogOdds.size)
        mindcfDict[m] = np.zeros(effPriorLogOdds.size)
    for idx, p in enumerate(effPriorLogOdds):
        effP[idx] = (1 + np.exp(-p)) ** (-1)
        if len(modelList) > 1:
            dcfDict["fusion"][idx] = util.normalizedDCF("[fusion]", scoreDict["fusion"], LTE, effP[idx], 1, 1, False)
            mindcfDict["fusion"][idx] = util.minDcf("[fusion]", scoreDict["fusion"], LTE, effP[idx], False)
        for m in modelList:

            mindcfDict[m][idx], _, _ = util.minDcf(f"[{m}]", scoreDict[m], LTE, effP[idx], True)
            dcfDict[m][idx], _, _ = util.normalizedDCF(f"[{m}]", scoreDict[m], LTE, effP[idx], 1, 1, True)
    for i, m in enumerate(modelList):

        plt.plot(effPriorLogOdds, dcfDict[m],label=f'{m} DCF',color=colorList[i])
        plt.plot(effPriorLogOdds, mindcfDict[m],label=f'{m} min DCF',color=colorList[i], linestyle="--" )

    plt.grid(True)
    plt.legend()  # 显示图例
    plt.ylim([0,0.5])
    plt.xlim([-4,4])
    plt.xlabel(r'$\log(\frac{\pi}{1-\pi})$')
    plt.ylabel("DCF")
    pylab.show()
def main(modelName):
    # D [ x0, x1, x2, x3, ...]  xi是列向量，每行都是一个feature
    D, L = load('./data/Train.txt')
    DTE, LTE = load('./data/Test.txt')
    ## plot_hist(D,L)

    ## gaussianize the training data
    D_gaussian = gaussianize(D)
    D_Znorm = Z_norm(D)

    #plot heatmap
    #corrlationAnalysis(D)

    # Dimensionality reduction
    left_dim = 12 # 12 ,11, 10, 9, 8
    # D = LDA(D_Znorm, L, 1)
    P = PCA(D, L, left_dim)
    Dz = np.dot(P.T,D_Znorm)
    D = np.dot(P.T, D)
    DTE = np.dot(P.T, DTE)

    #Model choosen list=["MVG","LR","SVM","GMM"]
    # DET(D,Dz,L,0.5)
    # BayesErrorPlot(D,Dz,L)
    # x = [1, 2, 4, 8, 16]
    # width = 0.2
    # # y1 = [0.114, 0,101,0.103,0.112]
    # # y2 = [0.093, 0,080,0.079,0.085]
    # # y3 = [0.117, 0.094,0.072,0.073]
    # # y3 = [0.161, 0.116, 0.101, 0.098]
    # y1 = [0.114, 0.119, 0.110, 0.118, 0.125]
    # y2 = [0.121, 0.125, 0.097, 0.117, 0.116]
    # # y4 = [0.103, 0.079, 0.072, 0.101, 0.153]
    # # y8 = [0.112, 0.085, 0.073, 0.098, 0.148]
    # # y16 = [0.133, 0.099, 0.085, 0.098, 0.157]
    # X_axis = np.arange(len(x))
    #
    # def addvalue(y, n):
    #     for index, value in enumerate(y):
    #         plt.text(index + width * n, value + 0.005, str(value), rotation=90)
    #
    # plt.bar(X_axis, y1, width, label="Raw(Val)", color="seagreen")
    # plt.bar(X_axis + width, y2, width, label="Raw(Eval)", color="mediumseagreen")
    # # plt.bar(X_axis + 2 * width, y4, width, label="Female K = 4", color="mediumaquamarine")
    # # plt.bar(X_axis + 3 * width, y8, width, label="Female K = 8", color="mediumspringgreen")
    # # plt.bar(X_axis + 4 * width, y16, width, label="Female K = 16", color="lightcyan")
    #
    # addvalue(y1, 0)
    # addvalue(y2, 1)
    # # addvalue(y4, 2)
    # # addvalue(y8, 3)
    # # addvalue(y16, 4)
    #
    # sns.set_context("paper")
    # plt.xticks(X_axis + width /2, x)
    # plt.ylim([0, 0.15])
    # plt.xlabel("GMM Components")
    # plt.ylabel("minDCF")
    # plt.title("GMM")
    # plt.legend(loc="upper center")
    # plt.savefig('./images/gmm_diag.png', dpi=300)
    # plt.show()
    Evaluation(Dz,L, DTE, LTE)
    # model = modelName
    # if model == "MVG":
    #     model,minDCF= KFold("MVG", 5, D, L,0.5,None)
    #     print("MVG : bestminDCF:{} ".format(minDCF))
    # elif model == "LR":
    #     hyperParListLR = {"lam": [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 1, 10]}
    #     hy,model,minDCF= KFoldHyper("LR", hyperParListLR, 5, D_Znorm, L,0.5)
    #     print("Logic regression : with hyper_paramter lambda = {}  bestminDCF:{}   ".format(hy["lam"],  minDCF))
    # elif model == "GMM":
    #     hyperParListGMM = {"n0": [0,1,2,3], "n1": [0,1,2,3]}
    #     hy, model, minDCF = KFoldHyper("GMM", hyperParListGMM, 5, D_Znorm, L, 0.5)
    #     print("GMM : with hyperparamter n0 ={}, n1={}, bestminDCF:{}  ".format(hy["n0"], hy["n1"], minDCF))
    # elif model == "SVM":
    #     #hyperParListSVM = {"C":[ 2 * 10 ** -5, 5 * 10 ** -5, 10 ** -4, 2 * 10 ** -4, 5 * 10 ** -4, 10 ** -3, 2 * 10 ** -3, 5 * 10 ** -3, 10**-2],"K":0, "loggamma":1,"d":2,"c":1}
    #     hyperParListSVM = {
    #         "C": [ 2 * 10 ** -5, 10 ** -2], "K": 0, "loggamma": -5, "d": 2, "c": 1}
    #     hy, model, minDCF = KFoldHyper("SVM", hyperParListSVM, 5, D, L, 0.5)
    #     print("SVM : bestminDCF:{} ".format(minDCF))
    # else:
    #     print("no corresponding model")




if __name__ == '__main__':
    main("MVG")
