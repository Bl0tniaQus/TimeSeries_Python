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


# ~ t = []
# ~ t_o = []
# ~ n = 1
# ~ for i in range(n):
    # ~ model = ld.LDMLT(20,cycles,5)
    # ~ start = time.time()
    # ~ model.fit(TRAIN_X,TRAIN_Y)
    # ~ end = time.time()
    # ~ t.append(end-start)

# ~ for i in range(n):
    # ~ model = ld_o.LDMLT(20,1,5)
    # ~ start = time.time()
    # ~ model.fit(TRAIN_X,TRAIN_Y)
    # ~ end = time.time()
    # ~ t_o.append(end-start)

# ~ print(f"1 - min: {min(t):.4f} max: {max(t):.4f} avg: {(sum(t)/n):.4f}")
# ~ print(f"2 - min: {min(t_o):.4f} max: {max(t_o):.4f} avg: {(sum(t_o)/n):.4f}")


# ~ for k in range(1, 2):
    # ~ Y_pred = model.predict(TEST_X, k)  # classification
    # ~ accuracy = accuracy_score(TEST_Y, Y_pred)
    # ~ print(f"k = {k}; acc: {accuracy:.3f}")
    
    
tripletsfactor = 20
cycle = 15 
alphafactor = 5
k = 1
res = ""
tfs = [10,12,14,16,18,20,22,24,26,28,30]
for tf in tfs:
    print(f"TF {tf}" )
    model = ld_o.LDMLT(tf, cycle, alphafactor)
    model.fit(TRAIN_X,TRAIN_Y)
    Y_pred = model.predict(TEST_X, k)
    acc = accuracy_score(TEST_Y, Y_pred)
    res = res + f"{tf} {acc:.3f}\n"
    print(res)
resf = open("results_tf.txt", "w")
resf.write(res)
resf.close()
# ~ tripletsfactor = 10
# ~ alphas = [2,3,4,5,6,7,8,9,10]
# ~ res = ""
# ~ for a in alphas:
    # ~ print(f"ALPHA {a}" )
    # ~ model = ld_o.LDMLT(tripletsfactor, cycle, a)
    # ~ model.fit(TRAIN_X,TRAIN_Y)
    # ~ Y_pred = model.predict(TEST_X, k)
    # ~ acc = accuracy_score(TEST_Y, Y_pred)
    # ~ res = res + f"{a} {acc:.3f}\n"
    # ~ print(res)
# ~ resf = open("results_alpha.txt", "w")
# ~ resf.write(res)
# ~ resf.close()

# ~ tripletsfactor = 10
# ~ alphafactor = 9
# ~ cycles = [10,12,14,16,18,20,22,24,26,28,30]
# ~ res = ""
# ~ for c in cycles:
    # ~ print(f"cycle {c}" )
    # ~ model = ld_o.LDMLT(tripletsfactor, c, alphafactor)
    # ~ model.fit(TRAIN_X,TRAIN_Y)
    # ~ Y_pred = model.predict(TEST_X, k)
    # ~ acc = accuracy_score(TEST_Y, Y_pred)
    # ~ res = res + f"{c} {acc:.3f}\n"
    # ~ print(res)
# ~ resf = open("results_cycle.txt", "w")
# ~ resf.write(res)
# ~ resf.close()

# ~ tripletsfactor = 20
# ~ alphafactor = 9
# ~ cycles = 18
# ~ model = ld_o.LDMLT(tripletsfactor, cycles, alphafactor)
# ~ model.fit(TRAIN_X, TRAIN_Y)
# ~ ks = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
# ~ res = ""
# ~ for k in ks:
    # ~ print(f"k: {k}")
    # ~ Y_pred = model.predict(TEST_X, k)
    # ~ acc = accuracy_score(TEST_Y, Y_pred)
    # ~ res = res + f"{k} {acc:.3f}\n"
    # ~ print(res)
# ~ resf = open("results_k.txt", "w")
# ~ resf.write(res)
# ~ resf.close()
    
tripletsfactor = 10
alphafactor = 9
cycles = 18
k = 10
n_test = len(TEST_X)
n_train = len(TRAIN_X)
print(f"n_test: {n_test}; n_train: {n_train}")
# ~ model = ld.LDMLT(tripletsfactor, cycles, alphafactor)
# ~ start = time.time()
# ~ model.fit(TRAIN_X, TRAIN_Y)
# ~ end = time.time()
# ~ train_time = end - start
# ~ start = time.time()
# ~ Y_pred = model.predict(TEST_X, k)
# ~ end = time.time()
# ~ test_time = (end - start) / n_test
# ~ accuracy = accuracy_score(TEST_Y, Y_pred)
# ~ print(f"UNOPTIMISED - Acc: {accuracy:.4f}; Train time: {train_time}; Test time: {test_time}")

# ~ model = ld_o.LDMLT(tripletsfactor, cycles, alphafactor)
# ~ start = time.time()
# ~ model.fit(TRAIN_X, TRAIN_Y)
# ~ end = time.time()
# ~ train_time = end - start
# ~ start = time.time()
# ~ Y_pred = model.predict(TEST_X, k)
# ~ end = time.time()
# ~ test_time = (end - start) / n_test
# ~ accuracy = accuracy_score(TEST_Y, Y_pred)
# ~ print(f"OPTIMISED - Acc: {accuracy:.4f}; Train time: {train_time}; Test time: {test_time}")

# ~ model = ld_o.LDMLT()
# ~ M = ld_o.LDMLT.loadM("M_kard.txt")
# ~ start = time.time()
# ~ model.fit(TRAIN_X, TRAIN_Y, M)
# ~ end = time.time()
# ~ train_time = end - start
# ~ start = time.time()
# ~ Y_pred = model.predict(TEST_X, k)
# ~ end = time.time()
# ~ test_time = (end - start) / n_test
# ~ accuracy = accuracy_score(TEST_Y, Y_pred)
# ~ print(f"OPTIMISED - Acc: {accuracy:.4f}; Train time: {train_time}; Test time: {test_time}")

