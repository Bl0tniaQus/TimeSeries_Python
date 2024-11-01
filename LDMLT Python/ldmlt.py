import numpy as np
from scipy.linalg import svd
import time
#from fastdtw import fastdtw

def dtw_metric(MTS_1, MTS_2, M):
    MTS_1 = MTS_1.T

    MTS_2 = MTS_2.T
    _, col_1 = MTS_1.shape
    row, col_2 = MTS_2.shape
    d = np.zeros((col_1, col_2))
    D1 = MTS_1.T @ M @ MTS_1
    D2 = MTS_2.T @ M @ MTS_2
    D3 = MTS_1.T @ M @ MTS_2
    for i in range(col_1):
        for j in range(col_2):
            d[i, j] = D1[i, i] + D2[j, j] - 2 * D3[i, j]
    
    D = np.zeros_like(d)
    D[0, 0] = d[0, 0]
    for m in range(1, col_1):
        D[m, 0] = d[m, 0] + D[m-1, 0]
    for n in range(1, col_2):
        D[0, n] = d[0, n] + D[0, n-1]
    for m in range(1, col_1):
        for n in range(1, col_2):
            D[m, n] = d[m, n] + min(D[m-1, n], min(D[m-1, n-1], D[m, n-1]))
    
    Dist = D[col_1-1, col_2-1]
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

def GetTriplets_Random(y, factor, Y_kind):
    Triplet = []
    for k in Y_kind:
        k_In = np.where(y == k)[0]
        k_Out = np.where(y != k)[0]
        l_In = len(k_In)
        l_Out = len(k_Out)
        j = 1
        while j < np.ceil(l_In * factor):
            rand_value = np.random.rand(3)
            index_i = np.ceil(rand_value[0] * l_In).astype(int)
            index_j = np.ceil(rand_value[1] * l_In).astype(int)
            index_k = np.ceil(rand_value[2] * l_Out).astype(int)
            if index_i == index_j:
                continue
            select_i = k_In[index_i]
            select_j = k_In[index_j]
            select_k = k_Out[index_k]
            dist_ik = (k_Out[index_k] - k_In[index_i]) @ (k_Out[index_k] - k_In[index_i])
            dist_jk = (k_Out[index_k] - k_In[index_j]) @ (k_Out[index_k] - k_In[index_j])
            if dist_jk > dist_ik:
                select_j = k_In[index_i]
                select_i = k_In[index_j]
            Triplet.append([select_i, select_j, select_k])
            j += 1
    return Triplet

def KNN_TS(X_train, Y_train, X_test, M, K_vector):
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    Y_kind = np.unique(Y_train)
    n = len(Y_train)
    
    Pred_Y = np.zeros((len(K_vector), n_test))
    for index_test in range(n_test):
        Distance = np.zeros(n_train)
        for index_train in range(n_train):
            #Dist, MTS_E1, MTS_E2 = dtw_metric(X_train[:, index_train], X_test[:, index_test], M)
            Dist, MTS_E1, MTS_E2 = dtw_metric(X_train[index_train], X_test[index_test], M)
            Distance[index_train] = Dist
        Inds = np.argsort(Distance,stable=True)
        
        for K_index, K in enumerate(K_vector):
            counts = np.zeros(len(Y_kind))
            for j in range(K):
                place = np.nonzero(Y_kind == Y_train[Inds[j]])
                counts[place] += 1 / Y_train[Inds[j]]
            Pred_Y[K_index, index_test] = Y_kind[np.argmax(counts)]
    return Pred_Y

def LDMLT_TS(X, Y, params=None):
    if params is None:
        params = {}
        params = SetDefaultParams(params)

    num_candidates = len(X)
    num_features = len(X[0][0])
    # The Mahalanobis matrix M starts from identity matrix
    M = np.eye(num_features, num_features)
    # Get all the labels of the data
    Y_kind = np.unique(Y)
    X_n, Y_n = data_rank(X, Y, Y_kind)
    # S record whether dissimilar or not
    S = np.zeros((num_candidates, num_candidates))
    for i in range(num_candidates):
        for j in range(num_candidates):
            if Y_n[i] == Y_n[j]:
                S[i, j] = 1
    Triplet, rho, Error_old = Select_Triplets(X_n, params['tripletsfactor'], M, Y_n, S)
    iter_count = len(Triplet)
    total_iter = iter_count
    for i in range(params['cycle']):
        alpha = params['alphafactor'] / iter_count
        rho = 0
        M = update_M(M, X_n, Triplet, alpha, rho)
        Triplet, rho, Error_new = Select_Triplets(X_n, params['tripletsfactor'], M, Y_n, S)
        iter_count = len(Triplet)
        total_iter += iter_count
        params['tripletsfactor'] = Error_new / Error_old * params['tripletsfactor']
        cov = (Error_old - Error_new) / Error_old
        if abs(cov) < 10e-5:
            break
        Error_old = Error_new
        print('finished cycle: ', i)
    return M

def update_M(M, X, triplet, gamma, rho):
    M = M / np.trace(M)
    i = 0
    options = np.zeros(5)
    options[4] = 1
    while i < len(triplet):
        i1 = triplet[i, 0]
        i2 = triplet[i, 1]
        i3 = triplet[i, 2]
        Dist1, swi1, swi2 = dtw_metric(X[i1], X[i2], M)
        P = swi1 - swi2
        Dist2, swi1, swi3 = dtw_metric(X[i1], X[i3], M)
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

def data_rank(X, Y, Y_kind):
    X_data = []
    Y_data = []
    for l in range(len(Y_kind)):
        index = np.nonzero(Y == Y_kind[l])[0]
        X_data.extend([X[i] for i in index])
        Y_data.extend([Y[i] for i in index])
    return X_data, Y_data

def SetDefaultParams(params):
    if 'cycle' not in params:
        params['cycle'] = 10
    if 'tripletsfactor' not in params:
        params['tripletsfactor'] = 1
    if 'alphafactor' not in params:
        params['alphafactor'] = 0.1
    return params

def Order_Check(X, M, Y):
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
        for j in range(map_vector_kind_length):
            if Y[map_vector_kind[i]] == Y[map_vector_kind[j]]:
                S[i, j] = 1
    
    Distance = np.zeros((map_vector_kind_length,map_vector_kind_length))
    #TODO możliwa optymalizacja
    for i in range(len(map_vector_kind)):
        for j in range(i, len(map_vector_kind)):
            Dist, _, _ = dtw_metric(X[map_vector_kind[i]], X[map_vector_kind[j]], M)
            Distance[i, j] = Dist
    for i in range(len(map_vector_kind)):
        for j in range(i):
            Distance[i, j] = Distance[j, i]
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
    Distance_Low = Distance.copy()
    Distance_Extended = np.zeros((numberCandidate,numberCandidate))
    for i in range(len(map_vector_kind)):
        index_i = np.where(map_vector == map_vector_kind[i])[0]
        for j in range(len(map_vector_kind)):
            index_j = np.where(map_vector == map_vector_kind[j])[0]
            Distance_Extended[np.ix_(index_i,index_j)] = Distance_Low[i, j]
    return Distance_Extended, Disorder
	
def Select_Triplets(X, factor, Mt, Y, S):
    bias = 3
    numberCandidate = len(X)
    triplet = []

    Distance, Disorder = Order_Check(X, Mt, Y)
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

def SetDefaultParams(s):
    if 'tripletsfactor' not in s:
        s['tripletsfactor'] = 10  # using the dynamic triplets selection strategy
    
    if 'cycle' not in s:
        s['cycle'] = 10
    
    if 'alphafactor' not in s:
        s['alphafactor'] = 2
    
    return s


