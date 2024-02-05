# ipol-tsne

This program computes the naive t-SNE and the Barnes-Hut t-SNE by calling the 
corresponding scikit-learn functions. It accepts several parameters and plots
the results.

Example of call :

$ python naive_tsne_vs_barnes_hut_tsne.py tsne --ndata 50 --figure cifar.pdf --method barnes --pca 10 --dataname cifar10;

