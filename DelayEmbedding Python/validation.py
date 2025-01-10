import numpy as np
import DelayEmbedding as DE
from scipy.io import loadmat
from sklearn.metrics import accuracy_score
import scipy.io as sio
import time


TRAIN_X = loadmat('../data/MSRA_I_TRAIN_X.mat')
TRAIN_Y = loadmat('../data/MSRA_I_TRAIN_Y.mat')
TEST_X = loadmat('../data/MSRA_I_TEST_X.mat')
TEST_Y = loadmat('../data/MSRA_I_TEST_Y.mat')
TRAIN_X = np.array(TRAIN_X['TRAIN_X'].flat)
TRAIN_Y = np.array(TRAIN_Y['TRAIN_Y'].flat)
TEST_X = np.array(TEST_X['TEST_X'].flat)
TEST_Y = np.array(TEST_Y['TEST_Y'].flat)


# ~ model = DE.DelayEmbedding(DE_step = 3, DE_dim = 6, DE_slid = 2, alpha = 2, beta = 3, grid_size = 0, filter_param = 0.5)
# ~ start = time.time()
# ~ model.fit(TRAIN_X, TRAIN_Y)
# ~ end = time.time()
# ~ Y_pred = model.predict(TEST_X)  # classification
# ~ accuracy = accuracy_score(TEST_Y, Y_pred)
# ~ print(f" acc: {accuracy:.3f} time: {end-start:.4f}")

DE_step = 3
DE_dim = 2
DE_slid = 2
alpha = 2
beta = 3

ds = [1,2,3,4,5,6,7,8,9,10]
res = ""
for d in ds:
    model = DE.DelayEmbedding(d, DE_dim, DE_slid, alpha, beta, 0.1, 0.5)
    model.fit(TRAIN_X,TRAIN_Y)
    Y_pred = model.predict(TEST_X)
    acc = accuracy_score(TEST_Y, Y_pred)
    res = res + f"{d} {acc:.3f}\n"
    print(res)
resf = open("results_de_step.txt", "w")
resf.write(res)
resf.close()
