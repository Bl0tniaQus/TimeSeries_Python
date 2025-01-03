import cupy as np
import warnings
warnings.filterwarnings('ignore')

class DelayEmbedding:
    def __init__(self, DE_step = 3, DE_dim = 2, DE_slid = 2, alpha = 2, beta = 3, filter_param = 0.5):
        self.DE_step = DE_step
        self.DE_dim = DE_dim
        self.DE_slid = DE_slid
        self.alpha = alpha
        self.beta = beta
        self.filter_param = filter_param
        self.Trans = None
        self.Grid = None
        self.classLabels = None
    def fit(self, X, Y):
        self.classLabels = np.unique(Y)
        n_class = len(self.classLabels)
        n_dimSignal = X[0].shape[0]
        self.Trans = {}
        for y in self.classLabels:
            self.Trans[y.item()] = []
        for loop in range(len(X)):
            x = X[loop].T
        # low-pass filter
            for i in range(x.shape[0]):
                x[i, :], _ = self.lowpass_filter(x[i, :], self.filter_param)
            # data normalization
            #if 'MSR_Action3D' in dataset[datasetInd]:
            #    x = x - np.tile(x[:, 0].reshape(-1, 1), (1, x.shape[1]))
            y = Y[loop]
            
            # multi-dimensional delay embedding
            point_cloud = self.delay_embedding_nd(x.T, self.DE_dim, self.DE_step, self.DE_slid)
            # update transition list
            self.Trans[y] = self.add2Trans(point_cloud, self.Trans[y])
        for i in self.classLabels:
            self.Trans[i.item()] = self.Trans_Prob(self.Trans[i.item()])
    def predict(self, X):
        if len(X[0].shape) == 1:
            X = np.array([X])
        predictions = [0 for i in range(len(X))]
        dist = {}
        for i in self.classLabels:
            dist[i.item()] = 0
        for loop in range(len(X)):
            x = X[loop].T
            for i in range(x.shape[0]):
                x[i, :], _ = self.lowpass_filter(x[i, :], self.filter_param)
            point_cloud = self.delay_embedding_nd(x.T, self.DE_dim, self.DE_step, self.DE_slid)
            for i in self.classLabels:
                dist[i.item()] = self.HDist(point_cloud, self.Trans[i.item()], i, self.alpha, self.beta)
            dists = np.array(list(dist.values()))
            loc = np.argmin(dists)
            predictions[loop] = self.classLabels[loc]
        if len(X) == 0:
            return predictions[0]
        return predictions
            
            
    def HDist(self, points, Trans, i, alpha=1.0, beta=1.0):
        # compute the modified Hausdorff distance between the given trajectory 
        # (points) and a learned model (Trans).
        if Trans is None:
            raise ValueError('Transition list is empty!')

        m, n = points.shape
        p, _ = Trans.shape

        
        # direction, location and length of transitions in the embedding space
        vec_Trans = Trans[:, n:2*n] - Trans[:, :n]
        loc_Trans = (Trans[:, n:2*n] + Trans[:, :n]) / 2
        len_Trans = np.sqrt(np.sum(vec_Trans**2, axis=1))

        # direction, location and length of the given trajectory
        vec_points = points[1:, :] - points[:-1, :]
        loc_points = (points[1:, :] + points[:-1, :]) / 2
        len_points = np.sqrt(np.sum(vec_points**2, axis=1))
        # normalized angle between learned transitions and given trajectory
        norm_angle = np.exp(np.real(np.arccos(vec_points @ vec_Trans.T / 
                                               (len_points[:, None] * len_Trans[None, :])))
                              )
        
        norm_angle[len_points == 0, len_Trans == 0] = 0
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
        # combine the same transitions in the transition list
        # and compute transition probability
        # Input:
        #   Trans       an existing transition list 
        #               each row records a transition formatted as following
        #               [ start point, end point ]

        # find unique transitions
        C, ic = np.unique(Trans, axis = 0, return_inverse=True)
        l = C.shape[0]

        # compute probabilities of unique transitions
        counts = np.bincount(ic, minlength=l)
        prob = counts / counts.sum()

        # format the new transition list 
        # [start point, end point, transition probability]
        Trans = np.hstack((C, prob[:, np.newaxis]))
        return Trans
    def add2Trans(self, points, Trans):
        # add points to an existing transition list
        # Input:
        #   points      a matrix, each row is a point 
        #   Trans       an existing transition list 
        #               each row records a transition formatted as following
        #               [ start point, end point ]
        #   Grid        an existing Grid created by the function createGrid()
        #   isGrid      a boolean value to indicate whether discretizing 
        #               the embedding space




        temp = points

        # append the new transitions to the end of the transition list
        temp = np.hstack((temp[:-1, :], temp[1:, :]))
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
        y = np.full((round((n - step * (dim - 1)) / w), dim), np.nan)

        # Delay embedding
        ind = np.arange(0, n, w)
        for i in range(y.shape[0]):
            temp = x[ind[i].item():ind[i].item() + step * dim:step]
            y[i, :] = np.asarray(np.reshape(temp, (1, len(temp))))

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
