
#%% Markdown [ ]:

# # t-SNE

import sys


#%% In [ ]:
print(sys.argv)

bindir=sys.argv[1]
dataset=sys.argv[2]
data_sample_size=int(sys.argv[3])
preprocess_data=sys.argv[4]
n_pca_components=int(sys.argv[5])
tsne_perplexity=float(sys.argv[6])
tsne_iter=int(sys.argv[7])

preprocess_data_with_pca=True if preprocess_data=='pca' else False
#%% Markdown [ ]:

# # Import packages



#%% In [ ]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import time
#%% Markdown [ ]:

# # Load Data and copy it into a dataframe



#%% In [ ]:

#mnist = fetch_openml(dataset)
d = np.load('%s/%s.npz'%(bindir,dataset), allow_pickle=True)
X = d['data'] / 255.0
y = d['target']
Xshape = X.shape

print(X.shape, y.shape)
print(type(X), type(y))


#%% In [ ]:

df = pd.DataFrame(X)


#%% In [ ]:

df['y'] = y
df['label'] = df['y'].apply(lambda i: str(i))

X, y = None, None
print('Size of the dataframe: {}'.format(df.shape))
#%% Markdown [ ]:

# # Sample random 10000 MNIST data



#%% In [ ]:

# For reproducability of the results
np.random.seed(42)
rndperm = np.random.permutation(df.shape[0])



#%% In [ ]:

N = data_sample_size
df_subset = df.loc[rndperm[:N],:].copy()
data_subset = df_subset.iloc[:,:Xshape[1]].values


#%% In [ ]:

data_subset.shape
#%% Markdown [ ]:

# # PCA



#%% In [ ]:

pca2 = PCA(n_components=n_pca_components)
#%% Markdown [ ]:

# # tsne



#%% In [ ]:

tsne = TSNE(n_components=2, perplexity=tsne_perplexity, n_iter=tsne_iter)
#%% Markdown [ ]:

# # Process data



#%% In [ ]:

time_start = time.time()
pca_2_result = pca2.fit_transform(data_subset)
time_taken_PCA2 = time.time()-time_start
print('PCA done! Time elapsed: {} seconds'.format(time_taken_PCA2))


#%% In [ ]:

df_subset['tsne-PCA2-x'] = pca_2_result[:,0]
df_subset['tsne-PCA2-y'] = pca_2_result[:,1]


#%% In [ ]:

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-PCA2-x", y="tsne-PCA2-y",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=df_subset,
    legend="full",
    alpha=0.3
).set(title='MNIST PCA 2 ({})'.format(time_taken_PCA2))


#%% In [ ]:

plt.savefig('pca.png')
#%% Markdown [ ]:

# # run now run tsne on top


if preprocess_data_with_pca:
    pass
else:
    pca_2_result=data_subset

#%% In [ ]:

time_start = time.time()
tsne_results = tsne.fit_transform(pca_2_result)
time_taken_PCA2 = time.time()-time_start
print('t-SNE done! Time elapsed: {} seconds'.format(time_taken_PCA2))


#%% In [ ]:

df_subset['tsne-PCA10-x'] = tsne_results[:,0]
df_subset['tsne-PCA10-y'] = tsne_results[:,1]


#%% In [ ]:

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-PCA10-x", y="tsne-PCA10-y",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=df_subset,
    legend="full",
    alpha=0.3
).set(title='PCA 10 t-SNE ({})'.format(time_taken_PCA2))


#%% In [ ]:

plt.savefig('tsne.png')


#%% In [ ]:

