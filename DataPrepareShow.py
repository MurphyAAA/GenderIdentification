import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.linalg
from scipy import stats

def mcol(v):
    return v.reshape((v.size, 1))  # 变成列向量


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



def gaussianize(D):
    # 一行一个feature
    ylist=[]
    for ind in range(D.shape[0]):
        # rank(D[ind,:]) 第ind个feature的rank的数组
        res = stats.norm.ppf(rank(D[ind,:]),loc=0,scale=1)
        ylist.append(res)
        # print(res.mean(),res.std())
    # y: 11 * 1839
    y = np.vstack(ylist)
    return y
def rank(x):
    ranks = stats.rankdata(x,method='min') # 一定要是min才是均匀分布！ 不是均匀分布没办法转回高斯，对于一个feature 共N个samples，计算每个sample这个feature的值有多少比他小的，（将N个样本的feature按从小到大排序）
    return (ranks+1) / (len(x) + 2)

def corrlationAnalysis(D):
    data = {}
    for i in range(D.shape[0]):
        data[i] = D[i]

    df = pd.DataFrame(data)
    corr_matrix = df.corr();
    sns.heatmap(corr_matrix,cmap="YlGnBu")
    print(corr_matrix)
    plt.savefig('images/heatmap_D')
    plt.show()

def plot_hist(D,L):
    ## D1 (12, N(num of sample which L == 0))
    D1 = D[:, L == 0]
    D0 = D[:, L == 1]

    for ind in range(D.shape[0]):
        plt.figure()
        plt.hist(D0[ind, :], bins=30, density=True, alpha=0.4, label='male')
        plt.hist(D1[ind, :], bins=30, density=True, alpha=0.4, label='female')
        plt.legend()
        plt.tight_layout()
       # plt.savefig('images/GAU_hist_%d.pdf' % ind)
    plt.show()


