#!/usr/bin/env python
# coding: utf-8
# BSD 3-Clause License
# 
# Copyright (c) 2023 Sangwon Jung      mrswjung@gmail.com,
#                    Tristan Dagobert  tristan.dagobert@ens-paris-saclay.fr
#
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# 
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# 
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
"""
This program computes the naive t-SNE and the Barnes-Hut t-SNE by calling the 
corresponding scikit-learn functions. It accepts several parameters and plots
the results.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import iio
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml

import time


def write_image_with_colors(cfg, img):
    """
    …
    """
    
    palette = sns.color_palette("hls", 10)
    # select the class
    y = img[:, :, 0]
    class_values = np.unique(y)
    colors = sns.color_palette("hls", 10)
    mappe = np.zeros((y.shape[0], y.shape[1], 3))

    # write the map
    for i in np.arange(class_values.shape[0]):
        mappe[y == class_values[i]] = colors[i]
    mappe = np.array(255.0 * mappe, dtype=np.uint8)
    iio.write(cfg.map, mappe)
    
    # select the assumed RGB channels
    rgb = img[:, :, 1:4]
    # normalisation
    mini = np.min(rgb)
    maxi = np.max(rgb)
    rgb = 255 * (rgb - mini) / (maxi - mini)
    rgb = np.array(rgb, dtype=np.uint8)
    iio.write(cfg.rgb, rgb)
    return


def tif_data_loading(cfg):
    """
    Loads data stored as a TIF image denoted im, where the class is the first
    channel : im.shape = (nrow, ncol, ncan). Could be useful for multi-spectral
    data.
    """
    img = iio.read(cfg.tif)
    datum = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
    X = datum[:, 1:]
    y = datum[:, 0]

    feat_cols = ["pixel" + str(i) for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feat_cols)
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))
    df['pixel0'] = 0.0    
    
    np.random.seed(42)  # For reproducability of the results
    rndperm = np.random.permutation(df.shape[0])

    N = cfg.ndata
    df_subset = df.loc[rndperm[:N], :].copy()
    data_subset = df_subset[feat_cols].values
    
    return data_subset, df_subset, img

    
def csv_data_loading(cfg):
    """
    Loads data provided in CSV format.
    """
    X = pd.read_csv(cfg.csv, sep=',', header=None)
    y = np.array(X.values[:, 0], dtype=np.int16)
    X = X.values[:, 1:]

    feat_cols = ["pixel" + str(i) for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feat_cols)
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))
    df['pixel0'] = 0.0    
    
    np.random.seed(42)  # For reproducability of the results
    rndperm = np.random.permutation(df.shape[0])

    N = cfg.ndata
    df_subset = df.loc[rndperm[:N], :].copy()
    data_subset = df_subset[feat_cols].values
    return data_subset, df_subset


def cifar_data_loading(cfg):
    """
    Loads CIFAR data provided by scikit-learn.
    """
    cifar = fetch_openml("CIFAR_10_small")

    # data preprocessing
    X = cifar.data
    y = cifar.target

    print(f"Number of NaN : {np.count_nonzero(np.isnan(X))}")

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
    Loads MNIST data provided by scikit-learn.
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
    Load data, compute the TSNE clustering and plot the results
    """

    # loading the dataset
    if cfg.dataname == "mnist":
        data, df_subset = mnist_data_loading(cfg)
    elif cfg.dataname == "cifar10":
        data, df_subset = cifar_data_loading(cfg)
    elif cfg.dataname == "csv":
        data, df_subset = csv_data_loading(cfg)
    elif cfg.dataname == "tif":
        data, df_subset, img = tif_data_loading(cfg)
        # plot the image with the encoded colors
        write_image_with_colors(cfg, img)
        exit()
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
                          help="Size of the sample.")
    a_parser.add_argument(
        "--dataname", type=str, required=True,
        choices=["mnist", "cifar10", "csv", "tif"],
        help="Type of dataset among CIFAR 10, MNIST or CSV, TIF format."
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
    a_parser.add_argument("--csv", type=str, required=False,
                          help="CSV filename.")
    a_parser.add_argument("--tif", type=str, required=False,
                          help="TIF filename.")
    a_parser.add_argument("--map", type=str, required=False,
                          help="Output class filename.")
    a_parser.add_argument("--rgb", type=str, required=False,
                          help="RGB output image.")
    
    
    cfg = parser.parse_args()
    if cfg.action == "tsne":
        apply_tsne(cfg)
    return 0


if __name__ == "__main__":
    main()

# Example of usage in bash command :
# 
# $ i=50;
# $ ~/python3.10/bin/python3.10 naive_tsne_vs_barnes_hut_tsne.py tsne \
#    --ndata 50 --figure cifar_${i}.pdf --method barnes --pca $i \
#    --dataname cifar10
