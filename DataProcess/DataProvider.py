import numpy as np


class DataProvider:
    def __init__(self, D, L, seed=0):
        # read data
        # shuffle data
        np.random.seed(seed)
        self.idx = np.random.permutation(D.shape[1])
        # counter
        counter = 0

        self.Data = D
        self.Label = L

        self.pi = (self.Label == 1).sum() / self.Label.size

    def getAll(self):
        return self.Data, self.Label

    def get(self, K, counter):
        nTrain = int(self.Data.shape[1] * 1.0 / K)

        idxTrain = self.idx[counter * nTrain: (counter + 1) * nTrain]
        idxTest = list(set(np.arange(self.Data.shape[1])) - (set(idxTrain)))
        DTR = self.Data[:, idxTrain]
        DTE = self.Data[:, idxTest]
        LTR = self.Label[idxTrain]
        LTE = self.Label[idxTest]

        return DTR, DTE, LTR, LTE