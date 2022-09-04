import sys
from sklearn.datasets import fetch_openml
mnist = fetch_openml(sys.argv[1])
#mnist = fetch_openml("mnist_784")
#mnist = fetch_openml("CIFAR_10_small")
#mnist = fetch_openml("Fashion-MNIST")

