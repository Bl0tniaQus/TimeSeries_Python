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


model = DE.DelayEmbedding()
model.fit(TRAIN_X, TRAIN_Y)

Y_pred = model.predict(TEST_X)  # classification
accuracy = accuracy_score(TEST_Y, Y_pred)
print(accuracy)
