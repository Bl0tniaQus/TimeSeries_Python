import numpy as np
import ldmlt
import decimal
from scipy.io import loadmat
from sktime.datasets import load_from_tsfile

TRAIN_X = loadmat('../MSRA_I_TRAIN_X.mat')
TRAIN_Y = loadmat('../MSRA_I_TRAIN_Y.mat')
TEST_X = loadmat('../MSRA_I_TEST_X.mat')
TEST_Y = loadmat('../MSRA_I_TEST_Y.mat')
TRAIN_X = np.array(TRAIN_X['TRAIN_X'].flat)
TRAIN_Y = np.array(TRAIN_Y['TRAIN_Y'].flat)
TEST_X = np.array(TEST_X['TEST_X'].flat)
TEST_Y = np.array(TEST_Y['TEST_Y'].flat)
parameters = {
    'tripletsfactor': 20,   # (quantity of triplets in each cycle) = params.tripletsfactor x (quantity of training instances)
    'cycle': 1,            # the maximum cycle
    'alphafactor': 5  # alpha = params.alphafactor/(quantity of triplets in each cycle)
}

max_knn = 3

print('training...')
M = ldmlt.LDMLT_TS(TRAIN_X, TRAIN_Y, parameters)  # training
print('training done')
k_vector = list(range(1, max_knn + 1))

print('testing...')
recognizedLabels = ldmlt.KNN_TS(TRAIN_X, TRAIN_Y, TEST_X, M, k_vector)  # classification
print('testing done')

recognitionRates = []
for k in range(max_knn):
	recognizedSamplesCount = 0
	for i in range(len(TEST_Y)):
		if recognizedLabels[k, i] == TEST_Y[i]:
			recognizedSamplesCount += 1
	recognitionRates.append(recognizedSamplesCount / len(TEST_Y))
print(recognizedSamplesCount)
print(recognitionRates)

#error_new, disorder, knn
