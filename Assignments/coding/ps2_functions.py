# ps2_functions.py
# Jay S. Stanley III, Yale University, Fall 2018
# CPSC 453 -- Problem Set 2
#
# This script contains functions for implementing graph clustering and signal processing.
#

import numpy as np
import codecs
import json
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.linalg import eigh
from sklearn.cluster import KMeans

def load_json_files(file_path):
    '''
    Loads data from a json file

    Inputs:
        file_path   the path of the .json file that you want to read in

    Outputs:
        my_array    this is a numpy array if data is numeric, it's a list if it's a string

    '''

    #  load data from json file
    with codecs.open(file_path, 'r', encoding='utf-8') as handle:
        json_data = json.loads(handle.read())

    # if a string, then returns list of strings
    if not isinstance(json_data[0], str):
        # otherwise, it's assumed to be numeric and returns numpy array
        json_data = np.array(json_data)

    return json_data


def gaussian_kernel(X, kernel_type="gaussian", sigma=3.0, k=5):
    """gaussian_kernel: Build an adjacency matrix for data using a Gaussian kernel
    Args:
        X (N x d np.ndarray): Input data
        kernel_type: "gaussian" or "adaptive". Controls bandwidth
        sigma (float): Scalar kernel bandwidth
        k (integer): nearest neighbor kernel bandwidth
    Returns:
        W (N x N np.ndarray): Weight/adjacency matrix induced from X
    """
    _g = "gaussian"
    _a = "adaptive"

    kernel_type = kernel_type.lower()
    D = squareform(pdist(X))
    if kernel_type == "gaussian":  # gaussian bandwidth checking
        print("fixed bandwidth specified")

        if not all([type(sigma) is float, sigma > 0]):  # [float, positive]
            print("invalid gaussian bandwidth, using sigma = max(min(D)) as bandwidth")
            D_find = D + np.eye(np.size(D, 1)) * 1e15
            sigma = np.max(np.min(D_find, 1))
            del D_find
        sigma = np.ones(np.size(D, 1)) * sigma
    elif kernel_type == "adaptive":  # adaptive bandwidth
        print("adaptive bandwidth specified")

        # [integer, positive, less than the total samples]
        if not all([type(k) is int, k > 0, k < np.size(D, 1)]):
            print("invalid adaptive bandwidth, using k=5 as bandwidth")
            k = 5

        knnDST = np.sort(D, axis=1)  # sorted neighbor distances
        sigma = knnDST[:, k]  # k-nn neighbor. 0 is self.
        del knnDST
    else:
        raise ValueError

    W = ((D**2) / sigma[:, np.newaxis]**2).T
    W = np.exp(-1 * (W))
    W = (W + W.T) / 2  # symmetrize
    W = W - np.eye(W.shape[0])  # remove the diagonal
    return W


# BEGIN PS2 FUNCTIONS


def sbm(N, k, pij, pii, sigma):
    """sbm: Construct a stochastic block model

    Args:
        N (integer): Graph size
        k (integer): Number of clusters
        pij (float): Probability of intercluster edges
        pii (float): probability of intracluster edges

    Returns:
        A (numpy.array): Adjacency Matrix
        gt (numpy.array): Ground truth cluster labels
        coords(numpy.array): plotting coordinates for the sbm
    """
    # create ground truth `gt`
    val = 0
    gt = []
    for i in range(N):
        gt.append(val)
        val += 1
        if val == k: 
            val = 0
    gt.sort() 
    gt = np.array(gt)
    
    # create adjacency matrix `A`
    A = np.zeros((N, N)) 
    
    # randomly generate probabilities for each node pairs from uniform distribution
    temp_p = np.random.uniform(0, 1, int(N * (N + 1) // 2))
    
    # get upper triangular indices of `A`
    up_tri_id = np.triu_indices(N) 
    
    A[up_tri_id] = temp_p 
    
    
    # create coordinates `coords`
    # and initialize values by trigonometric properties of sin and cos
    
    # Return evenly spaced numbers over [0, 2πk/(k+1)]
    partition = np.linspace(0, 2 * np.pi * k / (k + 1), num=k) 
    
    coords = np.random.normal(loc=0, scale=sigma, size=(N, 2))
    
    x, y = np.sin(partition), np.cos(partition)
    
    x_y = np.column_stack((x, y)) 
    
    
    # set start points and end points of cluster
    cluster_partition = np.ceil(np.linspace(0, N, k+1)).astype(int)
    
    cluster_end = zip(cluster_partition[:-1], cluster_partition[1:])
    
    
    # set values by iterating over cluster index points
    for i, (start, end) in enumerate(cluster_end):
        coords[start:end] += x_y[i] 
        A[start:end, start:end] = A[start:end, start:end] < pii 
        A[start:end, end:] = A[start:end, end:] < pij 
    
    
    # set the lower triangular indices of `A`
    low_tri_id = np.tril_indices(N, -1)
    A[low_tri_id] = A.T[low_tri_id] 
    A = A.astype(int) 
    
    
    return A, gt, coords

def L(A, normalized=True):
    """L: compute a graph laplacian

    Args:
        A (N x N np.ndarray): Adjacency matrix of graph
        normalized (bool, optional): Normalized or combinatorial Laplacian

    Returns:
        L (N x N np.ndarray): graph Laplacian
    """
    # calculate degree matrix `D`
    D = np.diag(A.sum(axis=-1))     
    
    # calculate D^{-1/2}
    square_D = np.diag((A.sum(axis=-1)) ** (-1/2)) 
    
    # calculate combinatorial graph Laplacian
    Lc = D - A 
    
    if normalized == True:
        # calculate normalized graph Laplacian
        L = square_D @ Lc @ square_D 
    elif normalized == False: 
        L = Lc 
           
    return L


def compute_fourier_basis(L):
    """compute_fourier_basis: Laplacian Diagonalization

    Args:
        L (N x N np.ndarray): graph Laplacian

    Returns:
        e (N x 1 np.ndarray): graph Laplacian eigenvalues
        psi (N x N np.ndarray): graph Laplacian eigenvectors
    """
    e, psi = np.linalg.eigh(L) 
    
    return e, psi


def gft(s, psi):
    """gft: Graph Fourier Transform (GFT)

    Args:
        s (N x d np.ndarray): Matrix of graph signals.  Each column is a signal.
        psi (N x N np.ndarray): graph Laplacian eigenvectors
    Returns:
        s_hat (N x d np.ndarray): GFT of the data
    """
    s_hat = psi.T @ s 
    
    return s_hat


def filterbank_matrix(psi, e, h):
    """filterbank_matrix: build a filter matrix using the input filter h

    Args:
        psi (N x N np.ndarray): graph Laplacian eigenvectors
        e (N x 1 np.ndarray): graph Laplacian eigenvalues
        h (function handle): A function that takes in eigenvalues
        and returns values in the interval (0,1)

    Returns:
        H (N x N np.ndarray): Filter matrix that can be used in the form
        filtered_s = H@s
    """
    
    # create graph filter
    filter_e = np.zeros_like(e) 
    
    # threshold
    c = 0.5 
    
    if h == "low pass":
        filter_e[filter_e < c] = 1
    
    elif h == "high pass":
        filter_e[filter_e > c] = 1
    
    elif h == "gaussian":
        mu, sigma = (float(i) for i in (input("Please enter mean and sigma with a space: ").split()) )
        
        print(f"mean = {mu}, sigma = {float(sigma)}")
        
        filter_e = np.exp(- (filter_e - mu) ** 2 / (2 * sigma ** 2))
    
    H = psi @ np.diagflat(filter_e) @ psi.T
    
    
    return H

def kmeans(X, k, nrep=5, itermax=300):
    """kmeans: cluster data into k partitions

    Args:
        X (n x d np.ndarray): input data, rows = points, cols = dimensions
        k (int): Number of clusters to partition
        nrep (int): Number of repetitions to average for final clustering 
        itermax (int): Number of iterations to perform before terminating
    Returns:
        labels (n x 1 np.ndarray): Cluster labels assigned by kmeans
    """
    
    # perform kmeans
    n_samples = X.shape[0]
    
    labels_list = []
    
    sum_min_dist = np.zeros((nrep,1), dtype=float)
    
    for i in range(nrep): 
        # find your initial centroids: shape (n_clusters, n_features)
        init = kmeans_plusplus(X, k)  
    
    
        labels = np.zeros((n_samples,1), dtype=int) 
        
        count = 0 
        while count < itermax:
            count += 1
            
            for j in range(n_samples):
                
                min_dist = np.inf
                
                for q, centroid in enumerate(init):
                    
                    dist = np.linalg.norm(X[j] - centroid)
       
                    if dist < min_dist:
                        centroid_id = q 
                        min_dist = dist 
                        
                labels[j] = centroid_id
                
                sum_min_dist[i] += min_dist
                
        labels_list.append(labels)
    
    labels = labels_list[np.argmin(sum_min_dist/n_samples)]
    
    return labels

def kmeans_plusplus(X, k):
    """kmeans_plusplus: initialization algorithm for kmeans
    Args:
        X (n x d np.ndarray): input data, rows = points, cols = dimensions
        k (int): Number of clusters to partition

    Returns:
        centroids (k x d np.ndarray): centroids for initializing k-means
    """
    n_samples, n_features = X.shape
    
    centroids = np.empty((k, n_features), dtype=X.dtype)
    
    mask = np.ones(shape=(n_samples,n_samples), dtype=bool)
     
    # Pick first centroid randomly and track index of point
    centroid_id = np.random.randint(n_samples)
    
    # condensed distance matrix of X with shape (n-1,)
    D = squareform(pdist(X))
    
    # Pick the remaining k-1 points
    for c in range(1, k):

        centroids[c-1] = X[centroid_id]
        
        squared_dist = (D[centroid_id] )** 2
        
        p = squared_dist / squared_dist.sum()
        
        # randomly sample a new centroid weighted by probability distribution
        centroid_id = np.random.choice(n_samples, p=p)

    centroids[k-1] = X[centroid_id]

    return centroids

def SC(L, k, psi=None, nrep=5, itermax=300, sklearn=False):
    """SC: Perform spectral clustering 
            via the Ng method
    Args:
        L (np.ndarray): Normalized graph Laplacian
        k (integer): number of clusters to compute
        nrep (int): Number of repetitions to average for final clustering
        itermax (int): Number of iterations to perform before terminating
        sklearn (boolean): Flag to use sklearn kmeans to test your algorithm
    Returns:
        labels (N x 1 np.array): Learned cluster labels
    """
    if psi is None:
        # compute the first k elements of the Fourier basis
        # use scipy.linalg.eigh
        psi_k = (eigh(L)[1])[:, :k] 
    else:  # just grab the first k eigenvectors
        psi_k = psi[:, :k]

    # normalize your eigenvector rows
    psi_norm = psi_k / np.linalg.norm(psi_k, axis=-1, ord=2, keepdims=True)

    if sklearn: 
        labels = KMeans(n_clusters=k, n_init=nrep,
                        max_iter=itermax).fit_predict(psi_norm)
    else:
        # your algorithm here
        labels = kmeans(X=psi_norm, k=k, itermax=itermax, nrep=nrep)

    return labels
