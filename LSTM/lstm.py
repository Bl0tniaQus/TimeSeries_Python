import torch
from scipy.io import loadmat
import numpy as np
class LSTM(torch.nn.Module):
    def __init__(self):
        pass


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
# ~ TRAIN_X = loadmat('../data/MSRA_II_TRAIN_X.mat')
# ~ TRAIN_Y = loadmat('../data/MSRA_II_TRAIN_Y.mat')
# ~ TEST_X = loadmat('../data/MSRA_II_TEST_X.mat')
# ~ TEST_Y = loadmat('../data/MSRA_II_TEST_Y.mat')
# ~ TRAIN_X = loadmat('../data/MSRA_III_TRAIN_X.mat')
# ~ TRAIN_Y = loadmat('../data/MSRA_III_TRAIN_Y.mat')
# ~ TEST_X = loadmat('../data/MSRA_III_TEST_X.mat')
# ~ TEST_Y = loadmat('../data/MSRA_III_TEST_Y.mat')
# ~ TRAIN_X = loadmat('../data/UTD_TRAIN_X.mat')
# ~ TRAIN_Y = loadmat('../data/UTD_TRAIN_Y.mat')
# ~ TEST_X = loadmat('../data/UTD_TEST_X.mat')
# ~ TEST_Y = loadmat('../data/UTD_TEST_Y.mat')
# ~ TRAIN_X = loadmat('../data/UTK_TRAIN_X.mat')
# ~ TRAIN_Y = loadmat('../data/UTK_TRAIN_Y.mat')
# ~ TEST_X = loadmat('../data/UTK_TEST_X.mat')
# ~ TEST_Y = loadmat('../data/UTK_TEST_Y.mat')
# ~ TRAIN_X = loadmat('../data/SYSU_TRAIN_X.mat')
# ~ TRAIN_Y = loadmat('../data/SYSU_TRAIN_Y.mat')
# ~ TEST_X = loadmat('../data/SYSU_TEST_X.mat')
# ~ TEST_Y = loadmat('../data/SYSU_TEST_Y.mat')
TRAIN_X = np.array(TRAIN_X['TRAIN_X'].flat)
TRAIN_Y = np.array(TRAIN_Y['TRAIN_Y'].flat)
TEST_X = np.array(TEST_X['TEST_X'].flat)
TEST_Y = np.array(TEST_Y['TEST_Y'].flat)


TRAIN_X_tensor = torch.from_numpy(TRAIN_X)
