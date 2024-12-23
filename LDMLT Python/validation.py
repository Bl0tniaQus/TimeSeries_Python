import numpy as np
import ldmlt
import decimal
import time
import sysconfig
from sklearn.metrics import accuracy_score
from scipy.io import loadmat
TRAIN_X = loadmat('../MSRA_I_TRAIN_X.mat')
TRAIN_Y = loadmat('../MSRA_I_TRAIN_Y.mat')
TEST_X = loadmat('../MSRA_I_TEST_X.mat')
TEST_Y = loadmat('../MSRA_I_TEST_Y.mat')
TRAIN_X = np.array(TRAIN_X['TRAIN_X'].flat)
TRAIN_Y = np.array(TRAIN_Y['TRAIN_Y'].flat)
TEST_X = np.array(TEST_X['TEST_X'].flat)
TEST_Y = np.array(TEST_Y['TEST_Y'].flat)

print('training...')
model = ldmlt.LDMLT(20,5,5)
start = time.time()
M = model.fit(TRAIN_X, TRAIN_Y)  # training
end = time.time()
print('training done')
print(end - start)

print('testing...')
Y_pred = model.predict(TEST_X, 3)  # classification
print('testing done')

accuracy = accuracy_score(TEST_Y, Y_pred)
print(accuracy)


#error_new, disorder, knn
