import numpy as np
from scipy.linalg import svd
import time
#from fastdtw import fastdtw

class LDMLT:
    def __init__(self, triplets_factor = 20, cycles = 3, alpha_factor = 5):
        self.triplets_factor = triplets_factor
        self.cycles = cycles
        self.alpha_factor = alpha_factor
        self.M = None
        self.X = None
        self.Y = None
    def DTW(self, MTS_1, MTS_2, M, distOnly = False):
        MTS_1 = MTS_1.T
        MTS_2 = MTS_2.T
        _, col_1 = MTS_1.shape
        row, col_2 = MTS_2.shape
        d = np.zeros((col_1, col_2))
        D1 = MTS_1.T @ M @ MTS_1
        D2 = MTS_2.T @ M @ MTS_2
        D3 = MTS_1.T @ M @ MTS_2
        d = np.array([[D1[i, i] + D2[j, j] - 2 * D3[i, j] for j in range(col_2)] for i in range(col_1)])
        D = np.zeros_like(d)
        for m in range(col_1):
            for n in range(col_2):
                if m == 0 and n == 0:
                    D[m, n] = d[m, n]
                elif m == 0 and n > 0:
                    D[m, n] = d[m, n] + D[m, n-1]
                elif n == 0 and m > 0:
                    D[m, n] = d[m, n] + D[m-1, n]
                else:
                    D[m, n] = d[m, n] + min(D[m-1, n], min(D[m-1, n-1], D[m, n-1]))
        
        Dist = D[col_1-1, col_2-1]
        if distOnly:
            return Dist, None, None
        n = col_2 - 1
        m = col_1 - 1
        k = 1
        w = np.array([col_1-1, col_2-1])
        while (n + m) != 0:
            if n == 0:
                m -= 1
            elif m == 0:
                n -= 1
            else:
                number = np.argmin((D[m-1, n], D[m, n-1], D[m-1, n-1]))
                if number == 0:
                    m -= 1
                elif number == 1:
                    n -= 1
                else:
                    m -= 1
                    n -= 1
            k += 1
            w = np.vstack((np.array([m, n]), w))
        
        MTS_E1 = np.zeros((row, k))
        MTS_E2 = np.zeros((row, k))
        for i in range(row):
            MTS_E1[i, :] = MTS_1[i, w[:, 0].astype(int)]
            MTS_E2[i, :] = MTS_2[i, w[:, 1].astype(int)]
        return Dist, MTS_E1, MTS_E2
    def predict(self, X, k = 3):
        if len(X[0].shape) == 1:
            X = np.array([X])
        n_train = self.X.shape[0]
        n_test = X.shape[0]
        Y_kind = np.unique(self.Y)
        Pred_Y = np.zeros(n_test)
        for index_test in range(n_test):
            Distance = np.zeros(n_train)
            for index_train in range(n_train):
                #Dist, MTS_E1, MTS_E2 = dtw_metric(X_train[:, index_train], X_test[:, index_test], M)
                Dist, _, _= self.DTW(self.X[index_train], X[index_test], self.M, distOnly = True)
                Distance[index_train] = Dist
            Inds = np.argsort(Distance,stable=True)
            
            counts = np.zeros(len(Y_kind))
            for j in range(k):
                place = np.nonzero(Y_kind == self.Y[Inds[j]])
                counts[place] += 1 / self.Y[Inds[j]]
            Pred_Y[index_test] = Y_kind[np.argmax(counts)]
        if len(Pred_Y) == 1:
            Pred_Y = Pred_Y[0]
        return Pred_Y
        
    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        num_candidates = len(X)
        num_features = len(X[0][0])
        triplets_factor = self.triplets_factor
        # The Mahalanobis matrix M starts from identity matrix
        M = np.eye(num_features, num_features)
        # Get all the labels of the data
        Y_kind = np.unique(Y)
        X_n, Y_n = self.dataRank(X, Y, Y_kind)
        # S record whether dissimilar or not
        S = np.zeros((num_candidates, num_candidates))
        for i in range(num_candidates):
            for j in range(num_candidates):
                if Y_n[i] == Y_n[j]:
                    S[i, j] = 1
        
        Triplet, rho, Error_old = self.selectTriplets(X_n, triplets_factor, M, Y_n, S)
        
        iter_count = len(Triplet)
        total_iter = iter_count
        for i in range(self.cycles):
            alpha = self.alpha_factor / iter_count
            rho = 0
            M = self.updateM(M, X_n, Triplet, alpha, rho)
            Triplet, rho, Error_new = self.selectTriplets(X_n, triplets_factor, M, Y_n, S)
            iter_count = len(Triplet)
            total_iter += iter_count
            self.triplet_factor = Error_new / Error_old * triplets_factor
            cov = (Error_old - Error_new) / Error_old
            if abs(cov) < 10e-5:
                break
            Error_old = Error_new
            print('finished cycle: ', i)
        self.M = M
    def updateM(self, M, X, triplet, gamma, rho):
        M = M / np.trace(M)
        i = 0
        options = np.zeros(5)
        options[4] = 1
        while i < len(triplet):
            i1 = triplet[i, 0]
            i2 = triplet[i, 1]
            i3 = triplet[i, 2]
            Dist1, swi1, swi2 = self.DTW(X[i1], X[i2], M)
            P = swi1 - swi2
            Dist2, swi1, swi3 = self.DTW(X[i1], X[i3], M)
            Q = swi1 - swi3
            IP = np.eye(P.shape[1])
            IQ = np.eye(Q.shape[1])
        
            if Dist2 - Dist1 < rho:
                alpha = gamma / np.trace(np.linalg.inv(np.eye(M.shape[0]) - M) @ M @ Q @ Q.T)
                M_temp = M - alpha * M @ P @ np.linalg.inv(IP + alpha * P.T @ M @ P) @ P.T @ M
                M = M_temp + alpha * M_temp @ Q @ np.linalg.inv(IQ - alpha * Q.T @ M_temp @ Q) @ Q.T @ M_temp
                L, S, R = svd(M)
                M = M / np.sum(np.diag(S))
                M = M / np.trace(M)

            i += 1
        return M * M.shape[0]
    def dataRank(self, X, Y, Y_kind):
        X_data = []
        Y_data = []
        for l in range(len(Y_kind)):
            index = np.nonzero(Y == Y_kind[l])[0]
            X_data.extend([X[i] for i in index])
            Y_data.extend([Y[i] for i in index])
        return X_data, Y_data
    def orderCheck(self, X, M, Y):
        numberCandidate = len(X)
        compactfactor = 2
        Y_kind = np.sort(np.unique(Y))
        index = 0
        j = 0
        map_vector = np.zeros(numberCandidate, dtype=int)
        for i in range(numberCandidate):
            if Y[i] == Y[index] and j < compactfactor:
                map_vector[i] = index
                j += 1
            else:
                index += j
                map_vector[i] = index
                j = 1
        
        map_vector_kind = np.unique(map_vector)
        map_vector_kind_length = len(map_vector_kind)
        S = np.zeros((map_vector_kind_length, map_vector_kind_length))
        for i in range(map_vector_kind_length):
            for j in range(i):
                if Y[map_vector_kind[i]] == Y[map_vector_kind[j]]:
                    S[i, j] = 1
                    S[j, i] = 1
            S[i,i] = 1
        
        Distance = np.zeros((map_vector_kind_length,map_vector_kind_length))
        #TODO możliwa optymalizacja
        for i in range(len(map_vector_kind)):
            for j in range(i, len(map_vector_kind)):
                Dist, _, _ = self.DTW(X[map_vector_kind[i]], X[map_vector_kind[j]], M, distOnly = True)
                Distance[i, j] = Dist
                Distance[j, i] = Dist
        Disorder = np.zeros(numberCandidate)
        for i in range(len(map_vector_kind)):
            Distance_i = Distance[i, :]
            S_i = S[i, :]
            index_ascend = np.argsort(Distance_i, stable=True)
            S_new = S_i[(index_ascend)]
            sum_in = np.sum(S_new == 1)
            rs1 = sum_in
            rs2 = 0
            for j in range(len(map_vector_kind)):
                if S_new[j] == 0:
                    rs2 += rs1
                else:
                    rs1 -= 1
            index = np.nonzero(map_vector == map_vector_kind[i])[0]
            Disorder[index] = rs2
        Distance_Extended = np.zeros((numberCandidate,numberCandidate))
        for i in range(len(map_vector_kind)):
            index_i = np.where(map_vector == map_vector_kind[i])[0]
            for j in range(i, len(map_vector_kind)):
                index_j = np.where(map_vector == map_vector_kind[j])[0]
                Distance_Extended[np.ix_(index_i,index_j)] = Distance[i, j]
                Distance_Extended[np.ix_(index_j,index_i)] = Distance[j, i]
        return Distance_Extended, Disorder
    def selectTriplets(self, X, factor, Mt, Y, S):
        bias = 3
        numberCandidate = len(X)
        triplet = []

        Distance, Disorder = self.orderCheck(X, Mt, Y)
        # Compute the parameter rho
        f, c = np.histogram(Distance, bins=100)
        #??
        l = c[20]
        u = c[80]
        rho = u - l
        error = np.sum(Disorder)
        Disorder = Disorder / (np.sum(Disorder) + np.finfo(float).eps)
        Triplet_N = factor * numberCandidate
        for l in range(numberCandidate):
            Sample_Length = round(np.sqrt(Disorder[l] * Triplet_N))
            if Sample_Length < 1:
                continue
            S_l = S[l, :]
            Distance_l = Distance[l, :]
            index_in = np.nonzero(S_l == 1)[0]
            index_out = np.nonzero(S_l == 0)[0]
            index_descend = np.argsort(-Distance_l[index_in], stable=True)
            index_ascend = np.argsort(Distance_l[index_out], stable=True)
            triplet_itemi = l
            triplet_itemj = index_in[index_descend[bias:min(bias + Sample_Length, len(index_in))]]
            triplet_itemk = index_out[index_ascend[bias:min(bias + Sample_Length, len(index_out))]]
            itemi, itemj, itemk = np.meshgrid(triplet_itemi, triplet_itemj, triplet_itemk)
            new_triplet = np.column_stack((itemi.flatten(order="F"), itemj.flatten(order="F"), itemk.flatten(order="F")))
            if len(triplet)==0:
                triplet = new_triplet
            else:
                triplet = np.concatenate((triplet, new_triplet), axis=0)
        return triplet, rho, error


