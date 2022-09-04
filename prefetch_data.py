import sys
import numpy as np
from sklearn.datasets import fetch_openml
dset=sys.argv[1]
outdir=sys.argv[2]
mnist = fetch_openml(dset)
np.savez_compressed('%s/%s.npz'%(outdir,dset),data=np.array(mnist.data),target=np.array(mnist.target))

#mnist = fetch_openml("mnist_784")
#mnist = fetch_openml("CIFAR_10_small")
#mnist = fetch_openml("Fashion-MNIST")

