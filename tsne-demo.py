
# # t-SNE demo

import sys

print(sys.argv)

bindir=sys.argv[1]
dataset=sys.argv[2]
data_sample_size=int(sys.argv[3])
preprocess_data=sys.argv[4]
n_pca_components=int(sys.argv[5])
tsne_perplexity=float(sys.argv[6])
tsne_iter=int(sys.argv[7])

preprocess_data_with_pca=True if preprocess_data=='pca' else False

# # Import packages




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import time

# # Load Data and copy it into a dataframe




# faster alternative to fetch_openml
#mnist = fetch_openml(dataset)
d = np.load('%s/%s.npz'%(bindir,dataset), allow_pickle=True)
X = d['data'] / 255.0
y = d['target']
Xshape = X.shape

print(X.shape, y.shape)
print(type(X), type(y))


df = pd.DataFrame(X)



df['y'] = y
df['label'] = df['y'].apply(lambda i: str(i))

X, y = None, None
print('Size of the dataframe: {}'.format(df.shape))

# # Sample random 10000 MNIST data



# For reproducability of the results
np.random.seed(42)
rndperm = np.random.permutation(df.shape[0])



N = data_sample_size
df_subset = df.loc[rndperm[:N],:].copy()
data_subset = df_subset.iloc[:,:Xshape[1]].values


data_subset.shape


# # PCA and tsne



pca2 = PCA(n_components=n_pca_components)


tsne = TSNE(n_components=2, perplexity=tsne_perplexity, n_iter=tsne_iter)

# run pca

time_start = time.time()
pca2_result = pca2.fit_transform(data_subset)
time_taken_PCA = time.time()-time_start
print('PCA done! Time elapsed: {} seconds'.format(time_taken_PCA))


df_subset['PCA2-x'] = pca2_result[:,0]
df_subset['PCA2-y'] = pca2_result[:,1]




plt.figure(figsize=(8,6))
sns.scatterplot(
    x="PCA2-x", y="PCA2-y",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=df_subset,
    legend="full",
    alpha=0.3
).set(title='PCA({})'.format(time_taken_PCA))



plt.savefig('pca.png')

# # run now run tsne on top


if preprocess_data_with_pca:
    pass
else:
    pca2_result=data_subset

# run tsne

time_start = time.time()
tsne_results = tsne.fit_transform(pca2_result)
time_taken_tsne = time.time()-time_start
print('t-SNE done! Time elapsed: {} seconds'.format(time_taken_tsne))



df_subset['tsne-x'] = tsne_results[:,0]
df_subset['tsne-y'] = tsne_results[:,1]



plt.figure(figsize=(8,6))
sns.scatterplot(
    x="tsne-x", y="tsne-y",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=df_subset,
    legend="full",
    alpha=0.3
).set(title='t-SNE ({})'.format(time_taken_tsne))



plt.savefig('tsne.png')



