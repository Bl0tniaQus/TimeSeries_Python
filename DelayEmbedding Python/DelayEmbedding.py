import numpy as np

def HDist(points, Trans, Grid, alpha=1.0, beta=1.0, isGrid=True):
    # compute the modified Hausdorff distance between the given trajectory 
    # (points) and a learned model (Trans).
    
    if Trans is None or Grid is None:
        raise ValueError('Transition list or Grid is empty!')

    m, n = points.shape
    p, _ = Trans.shape

    # approximate the points to the nearest grid cell
    if isGrid:
        gridCenter = Grid['center']
        gridSize = Grid['size']
        points = np.round((points - np.tile(gridCenter, (m, 1))) / 
                          np.tile(gridSize, (m, 1))) * np.tile(gridSize, (m, 1)) + \
                          np.tile(gridCenter, (m, 1))

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
    dist = np.min(norm_dist, axis=1)
    return np.nanmean(dist[len_points > 0])

def Trans_Prob(Trans):
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
    
def add2Trans(points, Trans, Grid, isGrid=False):
    # add points to an existing transition list
    # Input:
    #   points      a matrix, each row is a point 
    #   Trans       an existing transition list 
    #               each row records a transition formatted as following
    #               [ start point, end point ]
    #   Grid        an existing Grid created by the function createGrid()
    #   isGrid      a boolean value to indicate whether discretizing 
    #               the embedding space

    if len(locals()) < 3:
        raise ValueError('Not enough input arguments!')

    # approximate the points to the nearest grid cell
    if isGrid:
        gridCenter = Grid['center']
        gridSize = Grid['size']
        m, n = points.shape
        if len(gridCenter) != n:
            gridCenter = np.tile(gridCenter[0], (1, n))
        if len(gridSize) != n:
            gridSize = np.tile(gridSize[0], (1, n))
        temp = np.round((points - np.tile(gridCenter, (m, 1))) / 
                        np.tile(gridSize, (m, 1))) * np.tile(gridSize, (m, 1)) + \
                        np.tile(gridCenter, (m, 1))
    else:
        temp = points

    # append the new transitions to the end of the transition list
    temp = np.hstack((temp[:-1, :], temp[1:, :]))
    if len(Trans)!=0:
        Trans = np.vstack((Trans, temp))
    else:
        Trans = temp
    
    return Trans

def create_grid(grid_size, grid_center):
    """
    Create an N-dimensional grid.
    
    Input:
        grid_size    a list of N elements, each of which denotes the cell 
                     size of the corresponding dimension
        grid_center   a list of N elements, denoting the center of grid
    """
    if len(locals()) < 2:
        raise ValueError('Not enough input arguments!')

    Grid = {
        'size': grid_size,
        'center': grid_center,
        'coord': [],
        'value': []
    }
    
    return Grid
def delay_embedding(x, dim=2, step=1, w=1):
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
        temp = x[ind[i]:ind[i] + step * dim:step]
        y[i, :] = np.reshape(temp, (1, len(temp)))

    return y

import numpy as np

def delay_embedding_nd(x, dim=2, step=1, w=1):
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
        y.extend(delay_embedding(x[:, i], dim, step, w))
    
    return np.array(y)

def lowpass_filter(input, param, tol=100):
    output = input.copy()
    
    cnt = 0
    tag = True
    for i in range(1, len(input)):
        if abs(output[i-1] - input[i]) > tol:
            cnt += 1
            output[i] = output[i-1]
        else:
            output[i] = param * output[i-1] + (1 - param) * input[i]
    
    if cnt / len(input) > 0.5:
        print('Bad data!!!')
        input("Press Enter to continue...")
        tag = False
    
    return output, tag
