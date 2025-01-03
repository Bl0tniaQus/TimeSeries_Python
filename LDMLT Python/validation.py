import numpy as np
import ldmlt as ld
import ldmlt_optimised as ld_o
import ldmlt as ld
import decimal
import time
import sysconfig
from sklearn.metrics import accuracy_score
from scipy.io import loadmat
import warnings
warnings.filterwarnings('ignore') 



TRAIN_X = loadmat('../data/MSRA_I_TRAIN_X.mat')
TRAIN_Y = loadmat('../data/MSRA_I_TRAIN_Y.mat')
TEST_X = loadmat('../data/MSRA_I_TEST_X.mat')
TEST_Y = loadmat('../data/MSRA_I_TEST_Y.mat')
# ~ TRAIN_X = loadmat('../data/FLORENCE_TRAIN_X.mat')
# ~ TRAIN_Y = loadmat('../data/FLORENCE_TRAIN_Y.mat')
# ~ TEST_X = loadmat('../data/FLORENCE_TEST_X.mat')
# ~ TEST_Y = loadmat('../data/FLORENCE_TEST_Y.mat')
TRAIN_X = np.array(TRAIN_X['TRAIN_X'].flat)
TRAIN_Y = np.array(TRAIN_Y['TRAIN_Y'].flat)
TEST_X = np.array(TEST_X['TEST_X'].flat)
TEST_Y = np.array(TEST_Y['TEST_Y'].flat)

t = []
t_o = []
n = 20
cycles = 1
for i in range(n):
    model = ld.LDMLT(20,cycles,5)
    start = time.time()
    model.fit(TRAIN_X,TRAIN_Y)
    end = time.time()
    t.append(end-start)

for i in range(n):
    model = ld_o.LDMLT(20,cycles,5)
    start = time.time()
    model.fit(TRAIN_X,TRAIN_Y)
    end = time.time()
    t_o.append(end-start)

print(f"1 - min: {min(t):.4f} max: {max(t):.4f} avg: {(sum(t)/n):.4f}")
print(f"2 - min: {min(t_o):.4f} max: {max(t_o):.4f} avg: {(sum(t_o)/n):.4f}")


for k in range(1, 2):
    Y_pred = model.predict(TEST_X, k)  # classification
    accuracy = accuracy_score(TEST_Y, Y_pred)
    print(f"k = {k}; acc: {accuracy:.3f}")

