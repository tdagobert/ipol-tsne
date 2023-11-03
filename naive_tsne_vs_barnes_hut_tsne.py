#!/usr/bin/env python
# coding: utf-8

# # Naive t-SNE & Barnes-Hut t-SNE

# ### Import packages

# In[54]:

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml

import time

# ## Load Data

# In[42]:

# ## Naive t-SNE

# In[56]:


# com def Hbeta(D=np.array([]), beta=1.0):
# com     """
# com         Compute the perplexity and the P-row for a specific value of the
# com         precision of a Gaussian distribution.
# com     """
# com 
# com     # com pute P-row and corresponding perplexity
# com     P = np.exp(-D.copy() * beta) # D : Pairwise distances
# com     sumP = sum(P)
# com     H = np.log(sumP) + beta * np.sum(D * P) / sumP 
# com     # Shannon entropy of Pi when perplexity is result of beta
# com     P = P / sumP # P-row(Pi) : Pairwise similarities
# com     return H, P
# com 
# com def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
# com     """
# com         Performs a binary search to get P-values in such a way that each
# com         conditional Gaussian has the same perplexity.
# com     """
# com 
# com     # Initialize some variables
# com     print("Computing pairwise distances...")
# com     (n, d) = X.shape # (e.g. 2500, 50)
# com     sum_X = np.sum(np.square(X), 1) # sum_X[0] = X[0:0]**2+...+X[0:2499]**2
# com     D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X) 
# com     #Pairwise distances : D[0,1] = (||X[0:0]-X[1:0]||**2+...+||X[0:49]-X[1:49]||**2)
# com     P = np.zeros((n, n)) # final output : P.shape == (2500, 2500)
# com     beta = np.ones((n, 1)) # beta : 1/(2*sigma**2)
# com     logU = np.log(perplexity)
# com 
# com     # Loop over all datapoints
# com     for i in range(n): # com pute P-values from datapoint 1 to datapoint 2500
# com 
# com         # Print progress
# com         if i % 500 == 0:
# com             print("Computing P-values for point %d of %d..." % (i, n))
# com 
# com         # com pute the Gaussian kernel and entropy for the current precision
# com         betamin = -np.inf
# com         betamax = np.inf
# com         Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] 
# com         # if i == 10 -> np.concatenate((np.r_[0:10], np.r_[11:2500])) : [0,1,...,9,11,12,...,2499]        
# com         (H, thisP) = Hbeta(Di, beta[i])
# com 
# com         # Evaluate whether the perplexity is within tolerance
# com         Hdiff = H - logU # Difference between current H and objective log(perplexity)
# com         tries = 0
# com         while np.abs(Hdiff) > tol and tries < 50: # 50 loops or convergence --> out
# com 
# com             # If not, increase or decrease precision
# com             # Update beta values through binary search until entropy matches logU
# com             if Hdiff > 0:
# com                 betamin = beta[i].copy()
# com                 if betamax == np.inf or betamax == -np.inf:
# com                     beta[i] = beta[i] * 2.
# com                 else:
# com                     beta[i] = (beta[i] + betamax) / 2.
# com             else:
# com                 betamax = beta[i].copy()
# com                 if betamin == np.inf or betamin == -np.inf:
# com                     beta[i] = beta[i] / 2.
# com                 else:
# com                     beta[i] = (beta[i] + betamin) / 2.
# com 
# com             # Recompute the values
# com             (H, thisP) = Hbeta(Di, beta[i]) # updated beta and its entropy and P-values
# com             Hdiff = H - logU # Evaluation
# com             tries += 1
# com 
# com         # Set the final row of P
# com         P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP
# com 
# com     # Return final P-matrix
# com     print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
# com     return P # P.shape(2500, 2500)
# com 
# com def pca(X=np.array([]), no_dims=50):
# com     """
# com         Runs PCA on the NxD array X in order to reduce its dimensionality to
# com         no_dims dimensions.
# com     """
# com 
# com     print("Preprocessing the data using PCA...")
# com     (n, d) = X.shape
# com     X = X - np.tile(np.mean(X, 0), (n, 1))
# com     (l, M) = np.linalg.eig(np.dot(X.T, X))
# com     Y = np.dot(X, M[:, 0:no_dims])
# com     return Y
# com 
# com def tsne(X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0):
# com     """
# com         Runs t-SNE on the dataset in the NxD array X to reduce its
# com         dimensionality to no_dims dimensions. The syntaxis of the function is
# com         `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
# com     """
# com 
# com     # Check inputs
# com     if isinstance(no_dims, float):
# com         print("Error: array X should have type float.")
# com         return -1
# com     if round(no_dims) != no_dims:
# com         print("Error: number of dimensions should be an integer.")
# com         return -1
# com 
# com     # Initialize variables
# com     X = pca(X, initial_dims).real # X.shape = (2500, 784) -> (2500, 50)
# com     (n, d) = X.shape # n : 2500 / d : 50
# com     max_iter = 1000 # total iteration
# com     initial_momentum = 0.5
# com     final_momentum = 0.8
# com     eta = 1000 #500 # learning rate
# com     min_gain = 0.01
# com     Y = np.random.randn(n, no_dims) 
# com     # Y.shape == (2500, 2)
# com     # t-SNE output -> initialize through gaussian distribution
# com     dY = np.zeros((n, no_dims)) # gradient
# com     iY = np.zeros((n, no_dims)) # Y(i)-Y(i-1)
# com     gains = np.ones((n, no_dims))
# com 
# com     # com pute P-values
# com     P = x2p(X, 1e-5, perplexity)
# com     P = P + np.transpose(P)
# com     P = P / np.sum(P) # Pij : (p j|i + p i|j) / 2n
# com     P = P * 4. # early exaggeration : multiply all of the pij's in the initial stage of the optimization
# com     # modeling pij's by fairly large qij's
# com     # effect : form tight wide separated clusters, create lots of relatively empty spaces in the map
# com     P = np.maximum(P, 1e-12) 
# com 
# com     # Run iterations
# com     for iter in range(max_iter):
# com 
# com         # com pute pairwise affinities
# com         sum_Y = np.sum(np.square(Y), 1) # sum_Y[0] == Y[0:0]**2 + Y[0:1]**2
# com         num = -2. * np.dot(Y, Y.T)
# com         num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
# com         # num : "Student-t distribution"
# com         num[range(n), range(n)] = 0. 
# com         Q = num / np.sum(num) # pairwise affinities
# com         Q = np.maximum(Q, 1e-12)
# com 
# com         # com pute gradient
# com         PQ = P - Q
# com        
# com         for i in range(n):
# com             dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)
# com             # gradient of Kullback-Leibler divergence between P and Q
# com 
# com         # Perform the update
# com 
# com         # iteration ~20 : momentum == 0.5 / iteration ~1000 : momentum == 0.8
# com         if iter < 20:
# com             momentum = initial_momentum
# com         else:
# com             momentum = final_momentum
# com 
# com         # I think this is the adaptive learning rate 
# com         gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + (gains * 0.8) * ((dY > 0.) == (iY > 0.)) # if direction of the attractive force and Y(i)-Y(i-1) is same direction
# com          # if direction of the attractive force and Y(i)-Y(i-1) is opposite direction
# com         gains[gains < min_gain] = min_gain
# com         iY = momentum * iY - eta * (gains * dY)
# com         Y = Y + iY
# com         Y = Y - np.tile(np.mean(Y, 0), (n, 1))
# com 
# com         # com pute current value of cost function
# com         if (iter + 1) % 100 == 0:
# com             C = np.sum(P * np.log(P / Q))
# com             print("Iteration %d: error is %f" % (iter + 1, C))
# com 
# com         # Stop lying about P-values -> early exaggeration stop
# com         if iter == 100:
# com             P = P / 4.
# com 
# com     # Return solution
# com     return Y # Y.shape == (2500, 2)
# com 
# com 
# com # In[58]:
# com 
# com 
# com print("Running example on 10000 MNIST digits...")
# com start_time = time.time()
# com Y = tsne(data_subset, 2, 50, 40.0)
# com time_taken = time.time()-time_start
# com print('t-SNE done! Time elapsed: {} seconds'.format(time_taken))
# com 
# com 
# com # In[59]:
# com 
# com 
# com df_subset['tsne-naive-2d-x'] = Y[:, 0]
# com df_subset['tsne-naive-2d-y'] = Y[:, 1]
# com 
# com 
# com # In[60]:
# com 
# com 
# com plt.figure(figsize=(16,10))
# com sns.scatterplot(
# com     x="tsne-naive-2d-x", y="tsne-naive-2d-y",
# com     hue="y",
# com     palette=sns.color_palette("hls", 10),
# com     data=df_subset,
# com     legend="full",
# com     alpha=0.3
# com ).set(title='MNIST subset(10000) naive t-SNE iter 1000 ({})'.format(time_taken))
# com 
# com 
# com # ## Barnes-Hut t-SNE
# com 
# com # In[27]:
# com 
# com 
# com tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000)
# com 
# com 
# com # Process t-SNE algorithm on data
# com 
# com # In[50]:
# com 
# com 
# com pca_50 = PCA(n_components=50)
# com time_start = time.time()
# com pca_result_50 = pca_50.fit_transform(data_subset)
# com 
# com print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))
# com tsne_bh_10k = tsne.fit_transform(pca_result_50)
# com time_taken = time.time()-time_start
# com print('t-SNE done! Time elapsed: {} seconds'.format(time_taken))
# com 
# com 
# com # In[61]:
# com 
# com 
# com df_subset['tsne-bh-2d-x'] = tsne_bh_10k[:,0]
# com df_subset['tsne-bh-2d-y'] = tsne_bh_10k[:,1]
# com 
# com 
# com # In[62]:
# com 
# com 


# In[ ]:


def cifar_data_loading(cfg):
    """
    ...
    """
    cifar = fetch_openml("CIFAR_10_small")

    # data preprocessing
    X = cifar.data
    y = cifar.target

    print(X)
    print(y)
    print(f"nombre de Nan : {np.count_nonzero(np.isnan(X))}")

    print(X.shape, y.shape)
    feat_cols = ["pixel" + str(i) for i in range(X.shape[1])]
    df = pd.DataFrame(X.values, columns=feat_cols)
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))
    df['pixel0'] = 0.0

    print('Size of the dataframe: {}'.format(df.shape))

    # we randomly sample 10000 CIFAR data
    np.random.seed(42)  # For reproducability of the results
    rndperm = np.random.permutation(df.shape[0])

    N = cfg.ndata
    df_subset = df.loc[rndperm[:N], :].copy()
    data_subset = df_subset[feat_cols].values
    return data_subset, df_subset


def mnist_data_loading(cfg):
    """
    ...
    """
    mnist = fetch_openml("mnist_784")

    # data preprocessing
    X = mnist.data / 255.0
    y = mnist.target
    print(X.shape, y.shape)
    print(type(X), type(y))

    feat_cols = ["pixel" + str(i) for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feat_cols)
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))
    df['pixel0'] = 0.0

    print('Size of the dataframe: {}'.format(df.shape))

    # we randomly sample 10000 MNIST data
    np.random.seed(42)  # For reproducability of the results
    rndperm = np.random.permutation(df.shape[0])

    N = cfg.ndata
    df_subset = df.loc[rndperm[:N], :].copy()
    data_subset = df_subset[feat_cols].values
    return data_subset, df_subset


def apply_tsne(cfg):
    """
    ...
    """

    # loading the MNIST dataset
    if cfg.dataname == "mnist":
        data, df_subset = mnist_data_loading(cfg)
    elif cfg.dataname == "cifar10":
        data, df_subset = cifar_data_loading(cfg)
        
    if cfg.pca != 0:
        pca_transform = PCA(n_components=cfg.pca)
        print(data)
        data = pca_transform.fit_transform(data)
        
    # initialization of the TSNE
    if cfg.method == "barnes":
        tsne = TSNE(n_components=2, verbose=1, perplexity=cfg.perplexity,
                    n_iter=cfg.iterations)
    elif cfg.method == "naive":
        tsne = TSNE(n_components=2, verbose=1, method='exact',
                    perplexity=cfg.perplexity,
                    n_iter=cfg.iterations)


    # computation of the TSNE
    time_start = time.time()
    tsne_points = tsne.fit_transform(data)
    duration = time.time() - time_start
    print(f"t-SNE done! Time elapsed: {duration} seconds")

    # data managing
    df_subset['tsne-2d-x'] = tsne_points[:, 0]
    df_subset['tsne-2d-y'] = tsne_points[:, 1]

    # plotting the data
    if cfg.method == "barnes":
        title = "Barnes-Hut"
    elif cfg.method == "naive":
        title = "Naive"
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-x", y="tsne-2d-y",
        hue="y",
        palette=sns.color_palette("hls", 10),
        data=df_subset,
        legend="full",
        alpha=1.0
    ).set(
        title=f"{title} t-SNE over {cfg.ndata} data ({duration:5.0f} sec)"
    )
    plt.savefig(cfg.figure, bbox_inches='tight')
    plt.show()

    return 0


def main():
    """
    ...
    """
    d = "Compute the TSNE over a part of the MNIST dataset."
    parser = argparse.ArgumentParser(description=d)
    h = "authorized actions"
    subparsers = parser.add_subparsers(dest="action", help=h)

    a_parser = subparsers.add_parser("tsne", help="Naive TSNE.")
    a_parser.add_argument("--ndata", type=int, required=True,
                          help="Size of the MNIST subset.")
    a_parser.add_argument(
        "--dataname", type=str, required=True, choices=["mnist", "cifar10"],
        help="Type of dataset among CIFAR 10 and MNIST."
    )
    a_parser.add_argument("--method", type=str, required=True,
                          help="Figure name.", choices=["barnes", "naive"])
    a_parser.add_argument("--figure", type=str, required=True,
                          help="Figure name.")
    a_parser.add_argument("--iterations", type=int, required=False,
                          help="Number of iteratons.", default=1000)
    a_parser.add_argument("--perplexity", type=int, required=False,
                          help="Perplexity.", default=40)
    a_parser.add_argument("--pca", type=int, required=False,
                          help="PCA components.", default=0)
    
    cfg = parser.parse_args()
    if cfg.action == "tsne":
        apply_tsne(cfg)
    return 0


if __name__ == "__main__":
    main()
