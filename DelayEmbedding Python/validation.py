import numpy as np
import DelayEmbedding as DE
from scipy.io import loadmat
import scipy.io as sio
import time

TRAIN_X = loadmat('../MSRA_I_TRAIN_X.mat')
TRAIN_Y = loadmat('../MSRA_I_TRAIN_Y.mat')
TEST_X = loadmat('../MSRA_I_TEST_X.mat')
TEST_Y = loadmat('../MSRA_I_TEST_Y.mat')
TRAIN_X = np.array(TRAIN_X['TRAIN_X'].flat)
TRAIN_Y = np.array(TRAIN_Y['TRAIN_Y'].flat)
TEST_X = np.array(TEST_X['TEST_X'].flat)
TEST_Y = np.array(TEST_Y['TEST_Y'].flat)


DE_step = 3; 
DE_dim = 2;
DE_slid = 2; 
gridSize = 2/20; 
isGrid = False; 
alpha = 2;
beta = 3; 
print_period = 50;
filter_param = 0.5;
# Create grid for each class
classLabels = np.unique(TRAIN_Y)
n_class = len(classLabels)
n_dimSignal = TRAIN_X[0].shape[0]
Trans = {}
Grid = {}
for y in classLabels:
    Trans[y] = []
    Grid[y] = DE.create_grid(gridSize, np.zeros(DE_dim * n_dimSignal))  # Assuming createGrid is defined elsewhere
startTime_train = time.time()
for loop in range(len(TRAIN_X)):
    if loop % print_period == 0:
        print(f'Trained {loop} / {len(TRAIN_X)}')
    
    # extract data and label
    x = TRAIN_X[loop].T
    # low-pass filter
    for i in range(x.shape[0]):
        x[i, :], _ = DE.lowpass_filter(x[i, :], filter_param)
    # data normalization
    #if 'MSR_Action3D' in dataset[datasetInd]:
    #    x = x - np.tile(x[:, 0].reshape(-1, 1), (1, x.shape[1]))
    y = TRAIN_Y[loop]
    
    # multi-dimensional delay embedding
    point_cloud = DE.delay_embedding_nd(x.T, DE_dim, DE_step, DE_slid)
    # update transition list
    Trans[y] = DE.add2Trans(point_cloud, Trans[y], Grid[y], isGrid)

# refine transition list and compute transition probability
for i in classLabels:
    Trans[i] = DE.Trans_Prob(Trans[i])
endTime_train = time.time() - startTime_train

# testing
startTime_test = time.time()
dist = {}
for i in classLabels:
    dist[i] = 0
prediction = np.zeros(len(TEST_X), dtype=int)
for loop in range(len(TEST_X)):
    if loop % print_period == 0:
        print(f'tested {loop} / {len(TEST_X)}')
    
    # extract data and label
    x = TEST_X[loop].T
    
    # low-pass filter
    for i in range(x.shape[0]):
        x[i, :], _ = DE.lowpass_filter(x[i, :], filter_param)
    
    # data normalization
    #if 'MSR_Action3D' in dataset[datasetInd]:
    #    x = x - np.tile(x[:, 0].reshape(-1, 1), (1, x.shape[1]))
    y = TEST_Y[loop]
    # multi-dimensional delay embedding
    point_cloud = DE.delay_embedding_nd(x.T, DE_dim, DE_step, DE_slid)
    # model matching
    for i in classLabels:
        dist[i] = DE.HDist(point_cloud, Trans[i], Grid[i], i, alpha, beta, isGrid)
    dists = list(dist.values())
    loc = np.argmin(dists)
    prediction[loop] = classLabels[loc]

endTime_test = time.time() - startTime_test

# print training and testing time
print(f'Training time: {endTime_train:.3f}sec, {endTime_train/len(TRAIN_X):.3f}sec per sample')
print(f'Testing time: {endTime_test:.3f}sec, {endTime_test/len(TEST_X):.3f}sec per sample')
# plot confusion matrix
#CM, fig_handle = confusionMatrix(trueLabel[testInd], prediction, categories)
Accuracy = np.mean(TEST_Y == prediction)
print(f'Accuracy = {Accuracy * 100:.2f}%')
