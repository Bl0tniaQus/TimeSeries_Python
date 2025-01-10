import numpy as np
from fastdtw import fastdtw

class DTW_KNN:
    def __init__(self):
        self.X = None
        self.Y = None
    def fit(self, X, Y):
        self.X = X.copy()
        self.Y = Y.copy()
    def dtw(self, a,b):
        dist, _ = fastdtw(a,b)
        return float(dist)
    def predict(self, X, k = 3):
        if len(X[0].shape) == 1:
            X = np.array([X])
        n_train = len(self.X)
        n_test = len(X)
        Y_kind = np.unique(self.Y)
        Pred_Y = np.zeros(n_test)
        for index_test in range(n_test):
            Distance = np.zeros(n_train)
            for index_train in range(n_train):
                Dist = self.dtw(self.X[index_train], X[index_test])
                Distance[index_train] = Dist
            Inds = np.argsort(Distance,stable=True)
            
            counts = np.zeros(len(Y_kind))
            for j in range(k):
                counts[np.nonzero(Y_kind == self.Y[Inds[j]])] += 1
            ids = np.argwhere(counts == np.amax(counts))
            if len(ids) == 1:
                Pred_Y[index_test] = Y_kind[np.argmax(counts)]
            else:
                Pred_Y[index_test] = self.Y[Inds[0]]
        if len(Pred_Y) == 1:
            Pred_Y = Pred_Y[0]
        return Pred_Y

