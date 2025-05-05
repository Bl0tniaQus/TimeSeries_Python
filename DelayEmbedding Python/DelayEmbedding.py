import numpy as np
import warnings
warnings.filterwarnings('ignore')

class DelayEmbedding:
    def __init__(self, DE_step = 3, DE_dim = 2, DE_slid = 2, alpha = 2, beta = 3, grid_size = 0, filter_param = 0.5):
        self.DE_step = DE_step
        self.DE_dim = DE_dim
        self.DE_slid = DE_slid
        self.alpha = alpha
        self.beta = beta
        self.filter_param = filter_param
        self.grid_size = grid_size
        self.Trans = None
        self.classLabels = None
        self.Grid = None
        self.X = None
        self.Y = None
    def fit(self, X, Y):
        self.X = [np.array(X[i].copy()) for i in range(len(X))]
        self.Y = Y.copy()
        self.classLabels = np.unique(self.Y)
        n_class = len(self.classLabels)
        n_dimSignal = self.X[0].shape[1]
        self.Trans = {}
        if self.grid_size >0:
            self.Grid = {'size' : self.grid_size, 'center' : np.zeros(self.DE_dim * n_dimSignal)}
        for y in self.classLabels:
            self.Trans[y] = []
        for loop in range(len(self.X)):
            x = self.X[loop].T
        # low-pass filter
            for i in range(x.shape[0]):
                x[i, :], _ = self.lowpass_filter(x[i, :], self.filter_param)
            y = self.Y[loop]
            
            # multi-dimensional delay embedding
            point_cloud = self.delay_embedding_nd(x.T, self.DE_dim, self.DE_step, self.DE_slid)
            # update transition list
            self.Trans[y] = self.add2Trans(point_cloud, self.Trans[y])
        for i in self.classLabels:
            self.Trans[i] = self.Trans_Prob(self.Trans[i])
        
    def predict(self, X):
        X_test = [np.array(X[i].copy()) for i in range(len(X))]
        if len(X_test[0].shape) == 1:
            X_test = np.array([X_test])
        predictions = [0 for i in range(len(X_test))]
        dist = {}
        for i in self.classLabels:
            dist[i] = 0
        for loop in range(len(X_test)):
            x = X_test[loop].T
            for i in range(x.shape[0]):
                x[i, :], _ = self.lowpass_filter(x[i, :], self.filter_param)
            point_cloud = self.delay_embedding_nd(x.T, self.DE_dim, self.DE_step, self.DE_slid)
            for i in self.classLabels:
                dist[i] = self.HDist(point_cloud, self.Trans[i], i, self.alpha, self.beta)
            dists = list(dist.values())
            loc = np.argmin(dists)
            predictions[loop] = self.classLabels[loc]
        if len(X_test) == 0:
            return predictions[0]
        return predictions
            
            
    def HDist(self, points, Trans, i, alpha=1.0, beta=1.0):
        
        p = Trans.shape[0]
        if p == 0:
            raise ValueError('Transition list is empty. Probably method arguments are invalid.')
        
        m, n = points.shape
        if m ==0:
            return np.nan
        
        if self.Grid is not None:
            gridCenter = self.Grid['center']
            gridSize = self.Grid['size']
            points = np.round((points - np.tile(gridCenter, (m, 1))) / np.tile(gridSize, (m, 1))) * np.tile(gridSize, (m, 1)) + np.tile(gridCenter, (m, 1))
        # direction, location and length of transitions in the embedding space
        vec_Trans = Trans[:, n:2*n] - Trans[:, :n]
        loc_Trans = (Trans[:, n:2*n] + Trans[:, :n]) / 2
        len_Trans = np.sqrt(np.sum(vec_Trans**2, axis=1))

        # direction, location and length of the given trajectory
        vec_points = points[1:, :] - points[:-1, :]
        loc_points = (points[1:, :] + points[:-1, :]) / 2
        len_points = np.sqrt(np.sum(vec_points**2, axis=1))
        # normalized angle between learned transitions and given trajectory
        norm_angle = np.exp(np.real(np.arccos(vec_points @ vec_Trans.T / (np.outer(len_points, len_Trans)))))
        if np.sum(len_points == 0) > 0 and  np.sum(len_Trans == 0) > 0:
            ag_lp = np.argwhere(len_points == 0)
            ag_T = np.argwhere(len_Trans == 0)
            for i in ag_lp:
                for j in ag_T:
                    norm_angle[i, j] = 0
        # normalized length difference
        norm_length = np.exp((np.tile(len_points[:, None], (1, p)) - 
                               np.tile(len_Trans[None, :], (m-1, 1)))**2 / 
                              (np.tile(len_points[:, None], (1, p))**2))
        norm_length[np.isnan(norm_length)] = 0

        # modified Hausdorff distance
        norm_distance = np.zeros((m-1, p))
        for i in range(m-1):
            if len_points[i] > 0:
                norm_distance[i, :] = np.sqrt(np.sum((np.tile(loc_points[i, :], (p, 1)) - loc_Trans)**2, axis=1)) / len_points[i]
            else:
                norm_distance[i, :] = np.sqrt(np.sum((np.tile(loc_points[i, :], (p, 1)) - loc_Trans)**2, axis=1))

        norm_dist = norm_distance + alpha * norm_length + beta * norm_angle
        dist = np.nanmin(norm_dist, axis=1)
        return np.nanmean(dist[len_points > 0])
    def Trans_Prob(self, Trans):
        C, ic = np.unique(Trans, axis = 0, return_inverse=True)
        l = C.shape[0]
        counts = np.bincount(ic, minlength=l)
        prob = counts / counts.sum()
        if len(Trans) != 0:
            Trans = np.hstack((C, prob[:, np.newaxis]))
        else:
            return np.array(Trans)
        return Trans
    def add2Trans(self, points, Trans):
        if self.Grid is not None:
            m, n = points.shape
            gridCenter = self.Grid['center']
            gridSize = self.Grid['size']
            if len(self.Grid['center']) != n:
                gridCenter = np.tile(self.Grid['center'][0], (1, n))
            if self.Grid['size'] != n:
                gridSize = np.tile(self.Grid['size'], (1, n))
            temp = np.round((points - np.tile(gridCenter, (m, 1))) / np.tile(gridSize, (m, 1))) * np.tile(gridSize, (m, 1)) + np.tile(gridCenter, (m, 1))
        else:
            temp = points
        temp = np.hstack((temp[:-1, :], temp[1:, :]))
        if temp.shape[0] > 0:
            if len(Trans)!=0:
                Trans = np.vstack((Trans, temp))
            else:
                Trans = temp
        return Trans

    def delay_embedding(self, x, dim=2, step=1, w=1):
        """
        Delay embedding

        Input:
            x       input signal, a vector
            dim     dimension
            step    delay step 
            w       slide step

        Output:
            y       point cloud, each row is a point
        """

        # Check input
        if len(x) < 1:
            raise ValueError('Not enough input arguments')
        if dim < 1:
            raise ValueError('Too large dimension')

        # Init output
        n = len(x)
        if n < dim:
            raise ValueError('Too large dimension')
        d = round(((n - step * (dim - 1)) / w)+0.001)
        if d<0:
            d = 0
        y = np.full((d, dim), np.nan)
        # Delay embedding
        ind = np.arange(0, n, w)
        for i in range(y.shape[0]):
            temp = x[ind[i]:ind[i] + step * dim:step]
            y[i, :] = np.reshape(temp, (1, len(temp)))
        
        return y




    def delay_embedding_nd(self, x, dim=2, step=1, w=1):
        """
        N-Dimensional delay embedding 

        Input:
            x       input signal, a matrix, each column is a dimension
            dim     dimension
            step    delay step 
            w       slide step

        Output:
            y       point cloud, each row is a point
        """

        # check input
        if x is None:
            raise ValueError('Not enough input arguments')
        n, n_dim = x.shape
        
        # init output
        if n < dim:
            raise ValueError('Too large dimension')
        
        y = []
        # delay embedding
        for i in range(n_dim):
            if len(y) == 0:
                y = self.delay_embedding(x[:, i], dim, step, w)
            else:
                y = np.hstack((y, self.delay_embedding(x[:, i], dim, step, w)))
        return np.array(y)

    def lowpass_filter(self, input_data, param, tol=100):
        output = input_data.copy()
        
        cnt = 0
        tag = True
        for i in range(1, len(input_data)):
            if abs(output[i-1] - input_data[i]) > tol:
                cnt += 1
                output[i] = output[i-1]
            else:
                output[i] = param * output[i-1] + (1 - param) * input_data[i]
        
        if cnt / len(input_data) > 0.5:
            raise ValueError("Invalid data")
            tag = False
        
        return output, tag
