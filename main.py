import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

def mcol(v):
    return v.reshape((v.size,1)) # 变成列向量

def mrow(v):
    return v.reshape((1, v.size))

    # print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。

def load(filename):
    DList = []
    labelList = []
    with open(filename) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:12]
                attrs = mcol(np.array([float(i) for i in attrs]))
                label = line.split(',')[-1].strip()

                DList.append(attrs)
                labelList.append(label)
            except:
                pass
    return np.hstack(DList), np.array(labelList, dtype=np.int32)

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
def PCA(D, m):
    mu = mcol(D.mean(1)) # mean of each feature
    # centralized
    DC = D - mu
    C = np.dot(DC,DC.T)
    C = C/L.size # 协方差矩阵
    U, s, Vh = np.linalg.svd(C)
    # print(f'U: {U.shape}')

    P = U[:,0:m]
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
        Dc = D[:, L==i]
        nc = Dc.shape[1] # 样本数量
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
    W = U[:, ::-1][:,0:1] # biggest eigenvector 2分类。can only reduced to 1D
    Dp = np.dot(W.T, D)
    # print(Dp.shape)
    return Dp

def logpdf_GAU_ND(x, mu, C):  # 概率密度、likelihood，x是未去中心化的原数据
    M = x.shape[0]
    a = M * np.log(2 * np.pi)
    _, b = np.linalg.slogdet(C)  #log|C| 矩阵C的绝对值的log  返回值第一个是符号，第二个是log|C|
    xc = (x - mu)
    print(mu.shape)
    # xc应该每行循环列数次
    c = np.dot(xc.T, np.linalg.inv(C)) #np.linalg.inv求矩阵的逆

    c = np.dot(c, xc)

    c = np.diagonal(c)  # 点乘完了取对角线就ok
    return (-1.0 / 2.0) * (a + b + c) # 密度函数的log
def MVG(DTR, LTR, DTE):
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
    # likelihood
    tll0 = np.exp(logpdf_GAU_ND(DTE, mu0, C0))
    tll1 = np.exp(logpdf_GAU_ND(DTE, mu1, C1))
    S = np.vstack((tll0, tll1)) # score
    Priori = 1/3
    SJoint = S * Priori
    SMarginal = mrow(SJoint.sum(0))
    SPost = SJoint / SMarginal
    predict = np.argmax(SPost, axis=0)
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
        if(pre == L[i]):
            res.append(True) # 预测正确
        else:
            res.append(False)
    corr = res.count(True)
    wrong = res.count(False)
    acc = corr / len(res)
    err = wrong / len(res)
    return acc, err

if __name__ == '__main__':
    # Hyperparameters
    m = 10 # 12D -> 10D 降维后的维度
    # D [ x0, x1, x2, x3, ...]  xi是列向量，每行都是一个feature
    D, L = load('./data/Train.txt')
    D = PCA(D,m) # Dimensionality reduction  12D -> 10D
    # DTR = LDA(DTR,LTR,m)

    (DTR, LTR), (DVAL, LVAL) = split_data(D, L)
    DTE, LTE = load('./data/Test.txt')

    predict = MVG(DTR, LTR, DVAL)

    acc, err = computeAccuracy(predict, LVAL) # acc: 92.5%

