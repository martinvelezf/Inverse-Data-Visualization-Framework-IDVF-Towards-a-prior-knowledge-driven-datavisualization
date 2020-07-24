import sys
sys.path.insert(1,"../")
import matplotlib.pyplot as plt
import kpca.kpca as kpca
from Kernels.LLE import LocallyLinearEmbedding 
# This import is needed to modify the way figure behaves
from mpl_toolkits.mplot3d import Axes3D
Axes3D

#----------------------------------------------------------------------
# Locally linear embedding of the swiss roll

from sklearn import manifold, datasets
X, color = datasets.make_swiss_roll(n_samples=1500)

print("Computing LLE embedding")
X_r, err = manifold.locally_linear_embedding(X, n_neighbors=6,
                                             n_components=2)
print("Done. Reconstruction error: %g" % err)

#----------------------------------------------------------------------
# Plot result
LLE=LocallyLinearEmbedding(n_neighbors=6)
K=LLE.K(X)

KPCA=kpca.KernelPCA(2)
X_LLE=KPCA.fit(X, K)
X_KLLE=KPCA.transform(X,K)
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)

ax.set_title("Original data")
ax = fig.add_subplot(221)
ax.scatter(X_r[:, 0], X_r[:, 1], c=color, cmap=plt.cm.Spectral)
plt.axis('tight')
ax = fig.add_subplot(222)
ax.scatter(X_LLE[:, 0], X_LLE[:, 1], c=color, cmap=plt.cm.Spectral)

ax.set_title("KLle")
ax = fig.add_subplot(223)
ax.scatter(X_KLLE[:, 0], X_KLLE[:, 1], c=color, cmap=plt.cm.Spectral)

plt.xticks([]), plt.yticks([])
plt.title('Projected data')
plt.show()