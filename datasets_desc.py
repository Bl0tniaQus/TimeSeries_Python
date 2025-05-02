import numpy as np
import decimal
import time
import sysconfig
from scipy.io import loadmat
import warnings
warnings.filterwarnings('ignore') 




# ~ TRAIN_X = loadmat('./data/MSRA_I_TRAIN_X.mat')
# ~ TRAIN_Y = loadmat('./data/MSRA_I_TRAIN_Y.mat')
# ~ TEST_X = loadmat('./data/MSRA_I_TEST_X.mat')
# ~ TEST_Y = loadmat('./data/MSRA_I_TEST_Y.mat')
# ~ TRAIN_X = loadmat('./data/KARD_TRAIN_X.mat')
# ~ TRAIN_Y = loadmat('./data/KARD_TRAIN_Y.mat')
# ~ TEST_X = loadmat('./data/KARD_TEST_X.mat')
# ~ TEST_Y = loadmat('./data/KARD_TEST_Y.mat')
# ~ TRAIN_X = loadmat('./data/FLORENCE_TRAIN_X.mat')
# ~ TRAIN_Y = loadmat('./data/FLORENCE_TRAIN_Y.mat')
# ~ TEST_X = loadmat('./data/FLORENCE_TEST_X.mat')
# ~ TEST_Y = loadmat('./data/FLORENCE_TEST_Y.mat')
# ~ TRAIN_X = loadmat('./data/MSRA_II_TRAIN_X.mat')
# ~ TRAIN_Y = loadmat('./data/MSRA_II_TRAIN_Y.mat')
# ~ TEST_X = loadmat('./data/MSRA_II_TEST_X.mat')
# ~ TEST_Y = loadmat('./data/MSRA_II_TEST_Y.mat')
# ~ TRAIN_X = loadmat('./data/MSRA_III_TRAIN_X.mat')
# ~ TRAIN_Y = loadmat('./data/MSRA_III_TRAIN_Y.mat')
# ~ TEST_X = loadmat('./data/MSRA_III_TEST_X.mat')
# ~ TEST_Y = loadmat('./data/MSRA_III_TEST_Y.mat')
# ~ TRAIN_X = loadmat('./data/UTD_TRAIN_X.mat')
# ~ TRAIN_Y = loadmat('./data/UTD_TRAIN_Y.mat')
# ~ TEST_X = loadmat('./data/UTD_TEST_X.mat')
# ~ TEST_Y = loadmat('./data/UTD_TEST_Y.mat')
# ~ TRAIN_X = loadmat('./data/UTK_TRAIN_X.mat')
# ~ TRAIN_Y = loadmat('./data/UTK_TRAIN_Y.mat')
# ~ TEST_X = loadmat('./data/UTK_TEST_X.mat')
# ~ TEST_Y = loadmat('./data/UTK_TEST_Y.mat')
TRAIN_X = loadmat('./data/SYSU_TRAIN_X.mat')
TRAIN_Y = loadmat('./data/SYSU_TRAIN_Y.mat')
TEST_X = loadmat('./data/SYSU_TEST_X.mat')
TEST_Y = loadmat('./data/SYSU_TEST_Y.mat')
TRAIN_X = np.array(TRAIN_X['TRAIN_X'].flat)
TRAIN_Y = np.array(TRAIN_Y['TRAIN_Y'].flat)
TEST_X = np.array(TEST_X['TEST_X'].flat)
TEST_Y = np.array(TEST_Y['TEST_Y'].flat)

lengths = []
n_features = TRAIN_X[0].shape[1]
n_train = len(TRAIN_X)
n_test = len(TEST_X)
n_classes = len(np.unique(TRAIN_Y))

for x in TRAIN_X:
    lengths.append(x.shape[0])
for x in TEST_X:
    lengths.append(x.shape[0])

minl = min(lengths)
maxl = max(lengths)
avgl = sum(lengths) / len(lengths)
print(f"n_features = {n_features}")
print(f"n_train = {n_train}")
print(f"n_test = {n_test}")
print(f"n_classes = {n_classes}")
print(f"min_length = {minl}")
print(f"max_length = {maxl}")
print(f"avg_length = {avgl}")



