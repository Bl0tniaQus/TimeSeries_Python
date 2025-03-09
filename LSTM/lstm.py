import torch
from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import time
def tensorify(array):
    array_tensors = [torch.Tensor.float(torch.from_numpy(element)) for element in array]
    return array_tensors
def class2probability(array, num_classes):
    arr = np.zeros((len(array), num_classes))
    for i in range(len(array)):
        arr[i][array[i]] = 1
    return arr
class LSTM(torch.nn.Module):
    def __init__(self, n_features, hidden_size, n_classes, lr, bilstm = False):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.bilstm = bilstm
        self.lr = lr
        self.lstm = torch.nn.LSTM(n_features, hidden_size, bidirectional=bilstm, batch_first = True)
        self.linear = torch.nn.Linear(hidden_size * (1+self.bilstm), n_classes)
        self.softmax = torch.nn.Softmax(dim = 0)
    def forward(self, x, h0 = None, c0 = None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(1+self.bilstm,self.hidden_size)
            c0 = torch.zeros(1+self.bilstm,self.hidden_size)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.linear(out[-1, :])
        out = self.softmax(out)
        return out, hn, cn
    def predict(self, x):
        out, _, _ = self.forward(x)
        pred = np.argmax(out.detach().numpy())
        return pred
    def fit(self, x_train, y_train, n_epoch):
        self.losses = []
        self.x = x_train
        self.y = y_train
        h0, c0 = None, None
        loss_function = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        for n in range(n_epoch):
            epoch_losses = []
            ids = np.arange(len(self.x))
            np.random.shuffle(ids)
            for i in range(len(self.x)):
                idx = ids[i]
                x = self.x[idx]
                y = self.y[idx]
                outputs, h0, c0 = self.forward(x, h0, c0)
                loss = loss_function(outputs, y)
                epoch_losses.append(loss.detach().numpy())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                h0 = h0.detach()
                c0 = c0.detach()
            self.losses.append(sum(epoch_losses) / len(epoch_losses))
            

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


LE = LabelEncoder().fit(TRAIN_Y)
TRAIN_Y = LE.transform(TRAIN_Y)
TEST_Y = LE.transform(TEST_Y)

classes = np.unique(TRAIN_Y)
n_classes = len(classes)
TRAIN_Y = class2probability(TRAIN_Y, n_classes)
n_features = TRAIN_X[0].shape[1]
n_train = len(TRAIN_X)
n_test = len(TEST_X)
TRAIN_X_tensor = tensorify(TRAIN_X)
TEST_X_tensor = tensorify(TEST_X)
TRAIN_Y_tensor = tensorify(TRAIN_Y)

learning_rate = 0.0002
n_epoch = 175

start = time.time()
model = LSTM(n_features, n_train*3, n_classes, learning_rate, False)
model.fit(TRAIN_X_tensor, TRAIN_Y_tensor, n_epoch)
end = time.time()
train_time = end - start
start = time.time()
predictions = [model.predict(TEST_X_tensor[i]) for i in range(n_test)]
end = time.time()
test_time = (end - start) / n_test
accuracy = accuracy_score(TEST_Y, predictions)
print(f"LSTM - Acc: {accuracy:.4f}; Train time: {train_time}; Test time: {test_time}")

start = time.time()
model = LSTM(n_features, n_train*3, n_classes, learning_rate, True)
model.fit(TRAIN_X_tensor, TRAIN_Y_tensor, n_epoch)
end = time.time()
train_time = end - start
start = time.time()
predictions = [model.predict(TEST_X_tensor[i]) for i in range(n_test)]
end = time.time()
test_time = (end - start) / n_test
accuracy = accuracy_score(TEST_Y, predictions)
print(f"BILSTM - Acc: {accuracy:.4f}; Train time: {train_time}; Test time: {test_time}")

