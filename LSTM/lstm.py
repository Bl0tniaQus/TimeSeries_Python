import torch
from scipy.io import loadmat
import numpy as np
def tensorify(array):
    array_tensors = [torch.Tensor.float(torch.from_numpy(element)) for element in array]
    return array_tensors
class LSTM(torch.nn.Module):
    def __init__(self, n_features, hidden_size, n_classes):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.lstm = torch.nn.LSTM(n_features, hidden_size)
        self.linear = torch.nn.Linear(hidden_size, n_classes)
        self.softmax = torch.nn.Softmax()
    def forward(self, x, h0 = None, c0 = None):
        if h0 is None or c0 is None:
            print(x.shape)
            h0 = torch.zeros(1,self.hidden_size).to(x.device)
            c0 = torch.zeros(1,self.hidden_size).to(x.device)
            print(h0.shape)
            print(c0.shape)
        # Forward pass through LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.linear(out[-1, :])  # Selecting the last output
        out = self.softmax(out)
        return out, hn, cn
    def fit(self, x_train, y_train, n_epoch):
        self.x = x_train
        self.y = y_train
        h0, c0 = None, None
        loss_function = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters())
        for n in range(n_epoch):
            for i in range(len(self.x)):
                x = self.x[i]
                y = self.y[i]
                outputs, h0, c0 = self.forward(x, h0, c0)
                loss = torch.Tensor.float(loss_function(outputs, y))
                loss.backward()
                optimizer.zero_grad()
                
                optimizer.step()
                h0 = h0.detach()
                c0 = c0.detach()
            

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

classes = np.unique(TRAIN_Y)
n_classes = len(classes)
n_features = TRAIN_X[0].shape[1]
TRAIN_X_tensor = tensorify(TRAIN_X)
TEST_X_tensor = tensorify(TEST_X)
TRAIN_Y_tensor = torch.from_numpy(TRAIN_Y)
TEST_Y_tensor = torch.from_numpy(TEST_Y)
model = LSTM(n_features, 20, n_classes)
model.fit(TRAIN_X_tensor, TRAIN_Y_tensor, 2)
