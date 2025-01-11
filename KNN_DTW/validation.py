import numpy as np
import time
from sklearn.metrics import accuracy_score
from scipy.io import loadmat
from knn_dtw import DTW_KNN
import warnings
warnings.filterwarnings('ignore') 

TRAIN_X = loadmat('../data/MSRA_I_TRAIN_X.mat')
TRAIN_Y = loadmat('../data/MSRA_I_TRAIN_Y.mat')
TEST_X = loadmat('../data/MSRA_I_TEST_X.mat')
TEST_Y = loadmat('../data/MSRA_I_TEST_Y.mat')
# ~ TRAIN_X = loadmat('../data/KARD_TRAIN_X.mat')
# ~ TRAIN_Y = loadmat('../data/KARD_TRAIN_Y.mat')
# ~ TEST_X = loadmat('../data/KARD_TEST_X.mat')
# ~ TEST_Y = loadmat('../data/KARD_TEST_Y.mat')
# ~ TRAIN_X = loadmat('../data/FLORENCE_TRAIN_X.mat')
# ~ TRAIN_Y = loadmat('../data/FLORENCE_TRAIN_Y.mat')
# ~ TEST_X = loadmat('../data/FLORENCE_TEST_X.mat')
# ~ TEST_Y = loadmat('../data/FLORENCE_TEST_Y.mat')
TRAIN_X = np.array(TRAIN_X['TRAIN_X'].flat)
TRAIN_Y = np.array(TRAIN_Y['TRAIN_Y'].flat)
TEST_X = np.array(TEST_X['TEST_X'].flat)
TEST_Y = np.array(TEST_Y['TEST_Y'].flat)
n_test = len(TEST_X)
k = 14

model = DTW_KNN()
model.fit(TRAIN_X, TRAIN_Y)
start = time.time()
Y_pred = model.predict(TEST_X, k)
end = time.time()
test_time = (end - start) / n_test
accuracy = accuracy_score(TEST_Y, Y_pred)
print(f"Acc: {accuracy:.4f}; Train time: {0}; Test time: {test_time}")

# ~ model = DTW_KNN()
# ~ model.fit(TRAIN_X, TRAIN_Y)
# ~ ks = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
# ~ res = ""
# ~ for k in ks:
    # ~ print(f"k: {k}")
    # ~ Y_pred = model.predict(TEST_X, k)
    # ~ acc = accuracy_score(TEST_Y, Y_pred)
    # ~ res = res + f"{k} {acc:.3f}\n"
    # ~ print(res)
    # ~ print(TEST_X[0])
# ~ resf = open("results_k.txt", "w")
# ~ resf.write(res)
# ~ resf.close()
