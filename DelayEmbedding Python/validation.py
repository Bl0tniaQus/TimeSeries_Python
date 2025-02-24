import numpy as np
import DelayEmbedding as DE
from scipy.io import loadmat
from sklearn.metrics import accuracy_score
import scipy.io as sio
import time


# ~ TRAIN_X = loadmat('../data/MSRA_I_TRAIN_X.mat')
# ~ TRAIN_Y = loadmat('../data/MSRA_I_TRAIN_Y.mat')
# ~ TEST_X = loadmat('../data/MSRA_I_TEST_X.mat')
# ~ TEST_Y = loadmat('../data/MSRA_I_TEST_Y.mat')
# ~ TRAIN_X = loadmat('../data/KARD_TRAIN_X.mat')
# ~ TRAIN_Y = loadmat('../data/KARD_TRAIN_Y.mat')
# ~ TEST_X = loadmat('../data/KARD_TEST_X.mat')
# ~ TEST_Y = loadmat('../data/KARD_TEST_Y.mat')
# ~ TRAIN_X = loadmat('../data/FLORENCE_TRAIN_X.mat')
# ~ TRAIN_Y = loadmat('../data/FLORENCE_TRAIN_Y.mat')
# ~ TEST_X = loadmat('../data/FLORENCE_TEST_X.mat')
# ~ TEST_Y = loadmat('../data/FLORENCE_TEST_Y.mat')
TRAIN_X = loadmat('../data/MSRA_II_TRAIN_X.mat')
TRAIN_Y = loadmat('../data/MSRA_II_TRAIN_Y.mat')
TEST_X = loadmat('../data/MSRA_II_TEST_X.mat')
TEST_Y = loadmat('../data/MSRA_II_TEST_Y.mat')
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
n_test = len(TEST_X)


# ~ model = DE.DelayEmbedding(DE_step = 5, DE_dim = 2, DE_slid = 2, alpha = 2, beta = 2, grid_size = 0.1, filter_param = 0.5)
# ~ start = time.time()
# ~ model.fit(TRAIN_X, TRAIN_Y)
# ~ end = time.time()
# ~ Y_pred = model.predict(TEST_X)  # classification
# ~ accuracy = accuracy_score(TEST_Y, Y_pred)
# ~ print(f" acc: {accuracy:.3f} time: {end-start:.4f}")

# ~ DE_step = 3
# ~ DE_dim = 2
# ~ DE_slid = 2
# ~ alpha = 2
# ~ beta = 3

# ~ ds = [1,2,3,4,5,6,7,8,9,10]
# ~ res = ""
# ~ for d in ds:
    # ~ model = DE.DelayEmbedding(d, DE_dim, DE_slid, alpha, beta, 0.1, 0.5)
    # ~ model.fit(TRAIN_X,TRAIN_Y)
    # ~ Y_pred = model.predict(TEST_X)
    # ~ acc = accuracy_score(TEST_Y, Y_pred)
    # ~ res = res + f"{d} {acc:.3f}\n"
    # ~ print(res)
# ~ resf = open("results_de_step.txt", "w")
# ~ resf.write(res)
# ~ resf.close()
# ~ DE_step = 5
# ~ dd = [1,2,3,4,5,6,7,8]
# ~ res = ""
# ~ for d in dd:
    # ~ model = DE.DelayEmbedding(DE_step, d, DE_slid, alpha, beta, 0.1, 0.5)
    # ~ model.fit(TRAIN_X,TRAIN_Y)
    # ~ Y_pred = model.predict(TEST_X)
    # ~ acc = accuracy_score(TEST_Y, Y_pred)
    # ~ res = res + f"{d} {acc:.3f}\n"
    # ~ print(res)
# ~ resf = open("results_de_dim.txt", "w")
# ~ resf.write(res)
# ~ resf.close()

# ~ DE_step = 5
# ~ DE_dim = 2
# ~ ds = [1,2,3,4,5,6,7,8,9,10]
# ~ res = ""
# ~ for d in ds:
    # ~ model = DE.DelayEmbedding(DE_step, DE_dim, d, alpha, beta, 0.1, 0.5)
    # ~ model.fit(TRAIN_X,TRAIN_Y)
    # ~ Y_pred = model.predict(TEST_X)
    # ~ acc = accuracy_score(TEST_Y, Y_pred)
    # ~ res = res + f"{d} {acc:.3f}\n"
    # ~ print(res)
# ~ resf = open("results_de_slid.txt", "w")
# ~ resf.write(res)
# ~ resf.close()

# ~ DE_step = 5
# ~ DE_dim = 2
# ~ DE_slid = 2
# ~ alphas = [1,2,3,4,5,6,7,8,9,10]
# ~ res = ""
# ~ for a in alphas:
    # ~ model = DE.DelayEmbedding(DE_step, DE_dim, DE_slid, a, beta, 0.1, 0.5)
    # ~ model.fit(TRAIN_X,TRAIN_Y)
    # ~ Y_pred = model.predict(TEST_X)
    # ~ acc = accuracy_score(TEST_Y, Y_pred)
    # ~ res = res + f"{a} {acc:.3f}\n"
    # ~ print(res)
# ~ resf = open("results_alpha.txt", "w")
# ~ resf.write(res)
# ~ resf.close()

# ~ DE_step = 5
# ~ DE_dim = 2
# ~ DE_slid = 2
# ~ alpha = 2
# ~ betas = [1,2,3,4,5,6,7,8,9,10]
# ~ res = ""
# ~ for b in betas:
    # ~ model = DE.DelayEmbedding(DE_step, DE_dim, DE_slid, alpha, b, 0.1, 0.5)
    # ~ model.fit(TRAIN_X,TRAIN_Y)
    # ~ Y_pred = model.predict(TEST_X)
    # ~ acc = accuracy_score(TEST_Y, Y_pred)
    # ~ res = res + f"{b} {acc:.3f}\n"
    # ~ print(res)
# ~ resf = open("results_beta.txt", "w")
# ~ resf.write(res)
# ~ resf.close()

# ~ DE_step = 5
# ~ DE_dim = 2
# ~ DE_slid = 2
# ~ alpha = 2
# ~ beta = 2
# ~ gs = [0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 10]
# ~ res = ""
# ~ for g in gs:
    # ~ model = DE.DelayEmbedding(DE_step, DE_dim, DE_slid, alpha, beta, g, 0.5)
    # ~ model.fit(TRAIN_X,TRAIN_Y)
    # ~ Y_pred = model.predict(TEST_X)
    # ~ acc = accuracy_score(TEST_Y, Y_pred)
    # ~ res = res + f"{g} {acc:.3f}\n"
    # ~ print(res)
# ~ resf = open("results_grid_size.txt", "w")
# ~ resf.write(res)
# ~ resf.close()

# ~ DE_step = 5
# ~ DE_dim = 2
# ~ DE_slid = 2
# ~ alpha = 2
# ~ beta = 2
# ~ gs = 0.1
# ~ fs = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
# ~ res = ""
# ~ for f in fs:
    # ~ model = DE.DelayEmbedding(DE_step, DE_dim, DE_slid, alpha, beta, gs, f)
    # ~ model.fit(TRAIN_X,TRAIN_Y)
    # ~ Y_pred = model.predict(TEST_X)
    # ~ acc = accuracy_score(TEST_Y, Y_pred)
    # ~ res = res + f"{f} {acc:.3f}\n"
    # ~ print(res)
# ~ resf = open("results_filter.txt", "w")
# ~ resf.write(res)
# ~ resf.close()


model = DE.DelayEmbedding(DE_step = 5, DE_dim = 2, DE_slid = 2, alpha = 2, beta = 2, grid_size = 0.1, filter_param = 0.5)
start = time.time()
model.fit(TRAIN_X, TRAIN_Y)
end = time.time()
train_time = end - start
start = time.time()
Y_pred = model.predict(TEST_X)
end = time.time()
test_time = (end - start) / n_test
accuracy = accuracy_score(TEST_Y, Y_pred)
print(f"Acc: {accuracy:.4f}; Train time: {train_time}; Test time: {test_time}")
