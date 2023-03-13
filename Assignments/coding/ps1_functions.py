# ps1_functions.py
# CPSC 553 -- Problem Set 1
#
# This script contains uncompleted functions for implementing diffusion maps.
#
# NOTE: please keep the variable names that I have put here, as it makes grading easier.

# import required libraries
import numpy as np
import codecs, json

##############################
# Predefined functions
##############################

def load_json_files(file_path):
    '''
    Loads data from a json file

    Inputs:
        file_path   the path of the .json file that you want to read in

    Outputs:
        json_data    this is a numpy array if data is numeric, it's a list if it's a string

    '''

    #  load data from json file
    with codecs.open(file_path, 'r', encoding='utf-8') as handle:
        json_data = json.loads(handle.read())

    # if a string, then returns list of strings
    if not isinstance(json_data[0], str):
        # otherwise, it's assumed to be numeric and returns numpy array
        json_data = np.array(json_data)

    return json_data


##############################
# Skeleton code (fill these in)
##############################


def compute_distances(X):
    '''
    Constructs a distance matrix from data set, assumes Euclidean distance

    Inputs:
        X       a numpy array of size n x p holding the data set (n observations, p features)

    Outputs:
        D       a numpy array of size n x n containing the euclidean distances between points

    '''
    n = X.shape[0]
    D = np.zeros([n, n])
    
    for i in range(n):
        for j in range(n):
            D[i,j] = np.linalg.norm(X[i] - X[j])

    # return distance matrix
    return D


def compute_affinity_matrix(D, kernel_type, sigma=None, k=None):
    '''
    Construct an affinity matrix from a distance matrix via gaussian kernel.

    Inputs:
        D               a numpy array of size n x n containing the distances between points
        kernel_type     a string, either "gaussian" or "adaptive".
                            If kernel_type = "gaussian", then sigma must be a positive number
                            If kernel_type = "adaptive", then k must be a positive integer
        sigma           the non-adaptive gaussian kernel parameter
        k               the adaptive kernel parameter

    Outputs:
        W       a numpy array of size n x n that is the affinity matrix

    '''
    if kernel_type == 'gaussian':
        if sigma > 0:
            W = np.exp(-D*D/(sigma**2))
        else:
            raise ValueError("Kernel type is gaussian, sigma must be a positive number!")
    
    if kernel_type == 'adaptive':
        if k > 0 and type(k) == int:
            W = np.zeros(shape=D.shape)
            for i in tqdm(range(D.shape[0])):
                sigmak_xi = np.sort(D[i])[k]
                for j in range(D.shape[1]):
                    euclidean_dist = D[i,j]**2 
                    sigmak_xj = np.sort(D[j])[k]
                    W[i,j] = 0.5*(np.exp((-euclidean_dist)/(sigmak_xi**2))+np.exp((-euclidean_dist)/(sigmak_xj**2)))
        else:
            raise ValueError("Kernel type is adaptive, k must be a positive integer!")

    # return the affinity matrix
    return W


def diff_map_info(W):
    '''
    Construct the information necessary to easily construct diffusion map for any t

    Inputs:
        W           a numpy array of size n x n containing the affinities between points

    Outputs:

        diff_vec    a numpy array of size n x n-1 containing the n-1 nontrivial eigenvectors of Markov matrix as columns
        diff_eig    a numpy array of size n-1 containing the n-1 nontrivial eigenvalues of Markov matrix

        We assume the convention that the coordinates in the diffusion vectors are in descending order
        according to eigenvalues.
    '''
    # compute negative square root of diagonal matrix of row sums of affinity matrix
    neg_sqrt_D = np.diag(W.sum(axis=1)**(-0.5))
    
    # create symmetric matrix M_s
    M_s = neg_sqrt_D @ W @ neg_sqrt_D
    
    # compute eigenpairs of symmetric matrix M_s
    # Note eigenvalues are in ascending order
    tilde_eig, tilde_vec = np.linalg.eigh(M_s)
    
    # compute normalized eigenvectors of Markov matrix M
    diff_vec = (neg_sqrt_D @ tilde_vec) / np.linalg.norm(neg_sqrt_D @ tilde_vec, axis=0)
    
    # discard first eigenvalue and eigenvector and re-arrange them in descending order
    diff_vec = np.flip(diff_vec, axis=1)[:,1:]
    diff_eig = tilde_eig[::-1][:-1]


    # return the info for diffusion maps
    return diff_vec, diff_eig


def get_diff_map(diff_vec, diff_eig, t):
    '''
    Construct a diffusion map at t from eigenvalues and eigenvectors of Markov matrix

    Inputs:
        diff_vec    a numpy array of size n x n-1 containing the n-1 nontrivial eigenvectors of Markov matrix as columns
        diff_eig    a numpy array of size n-1 containing the n-1 nontrivial eigenvalues of Markov matrix
        t           diffusion time parameter t

    Outputs:
        diff_map    a numpy array of size n x n-1, the diffusion map defined for t
    '''
    # # first broadcasting diff_eig to size n x n-1, then do Hadamard Product
    diff_map = (diff_eig**t) * diff_vec 

    return diff_map


def diff_map(X, kernel_type, t, sigma=None, k=None):
    """
    Compute diffusion maps of data, return diffusion maps, nontrivial eigenvectors and eigenvalues of Markov matrix
    
    Inputs:
        X           a numpy array of size n x p holding the dataset (n observations, p features)
        kernel_type a string, eigher "gaussian" or "adaptive"
        sigma       non-adaptive gaussian kernel parameter
        k           adaptive kernel parameter
        t           diffusion time parameter t 

    Outputs:
        diff_map    a numpy array of size n x n-1, the diffusion map defined for t 
        diff_vec    a numpy array of size n x n-1 containing the n-1 nontrivial eigenvectors of Markov matrix as columns
        diff_eig    a numpy array of size n-1 containing the n-1 nontrivial eigenvalues of Markov matrix
    """

    # compute distance matrix
    D = compute_distances(X)

    # compute affinity matrix
    W = compute_affinity_matrix(D, kernel_type, sigma, k)

    # compute eigenvectors and eigenvalues of Markov matrix
    diff_vec, diff_eig = diff_map_info(W)

    # compute diffusion maps
    diff_map = get_diff_map(diff_vec, diff_eig, t)

    return diff_map, diff_vec, diff_eig