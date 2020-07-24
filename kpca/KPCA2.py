import numpy as np
from scipy import linalg
from scipy.sparse.linalg import eigsh
from sklearn.utils import check_random_state
from sklearn.utils.extmath import svd_flip
from sklearn.utils.validation import (check_is_fitted, check_array,
                                _check_psd_eigenvalues)

from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KernelCenterer
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import normalize
#Insert own modules
import sys
sys.path.insert(1,"../")

#Kernel from other linear methods
from Kernels.LLE import LocallyLinearEmbedding
from Kernels.laplacian import SpectralEmbedding
from Kernels.Isomap import Isomap
from Kernels.CMDS import KCMDS
class KernelPCA(TransformerMixin, BaseEstimator):
    def __init__(self, degree=3, coef0=1, kernel_params=None,
                 alpha=1.0, eigen_solver='auto', neigh=8,
                 tol=0, max_iter=None, remove_zero_eig=True, n_components=2,
                 random_state=None, n_jobs=None,coeficient=None,nkernel=10):
        self.kernel_params = kernel_params
        self.gamma = 0.0001
        self.neigh=neigh
        self.nkernel=nkernel
        self.n_components=n_components
        self.degree = degree
        self.coef0 = coef0
        self.alpha = alpha
        self.eigen_solver = eigen_solver
        self.remove_zero_eig = remove_zero_eig
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._centerer=KernelCenterer()
        self.coeficient=coeficient
    def KLE(self,X,neigh):
        LE=SpectralEmbedding(n_neighbors=neigh)
        return LE.K(X)

    def KIsomap(self,X,neigh):
        Iso=Isomap(n_neighbors=neigh)
        return Iso.K(X)

    def KLLE(self,X,neigh):
        LLE=LocallyLinearEmbedding(n_neighbors=neigh) 
        return LLE.K(X)

    def kernels(self,X):
        kern=[ 'linear', 'poly', 'polynomial', 'rbf', 'laplacian', 'sigmoid', 'cosine']
        if self.neigh>len(X):
            self.neigh=len(X)-3
        tkernel=len(kern)+3*self.neigh-1
        K=[]
        K.append((KCMDS(X)))
        for i in kern:
            if i=='rbf':
                for j in range(0,self.nkernel-tkernel):
                    self.gamma=(j)*0.01
                    K.append((self._get_kernel(X,i)))
            else:
                K.append((self._get_kernel(X,i)))
        
        for i in range(2,2+self.neigh):
            K.append((self.KLE(X,i)))
            K.append((self.KLLE(X,i)))
            K.append((self.KIsomap(X,i)))
        
        #IN case that coeffcients weren't set
        if not (self.coeficient):
            self.coeficient=np.zeros(len(K))
            self.coeficient[0]=1 #Linear Kernel

        self.SuperK=self.createSuperK(K)
        return K
    
    def _get_kernel(self, X,kernel):
        params = {"gamma": self.gamma,
                      "degree": self.degree,
                      "coef0": self.coef0}
        return pairwise_kernels(X, None, metric=kernel,filter_params=True, n_jobs=self.n_jobs,**params)
    def Solve(self, K):
        
        #GET EIGENVALUES AND EIGENVECTOR THE CENTER KERNEL
        self.lambdas_, self.vectors_ = linalg.eigh(K, eigvals=(K.shape[0] - self.n_components, K.shape[0] - 1))
        
        # make sure that the eigenvalues are ok and fix numerical issues
        self.lambdas_ = _check_psd_eigenvalues(self.lambdas_,
                                               enable_warnings=False)

        # flip eigenvectors' sign to enforce deterministic output
        self.vectors_, _ = svd_flip(self.vectors_,
                                   np.empty_like(self.vectors_).T)

        # sort eigenvectors in descending order
        indices = self.lambdas_.argsort()[::-1]
        self.lambdas_ = self.lambdas_[indices]
        self.vectors_ = self.vectors_[:, indices]

        # remove eigenvectors with a zero eigenvalue (null space) if required
        if self.remove_zero_eig:
            self.vectors_ = self.vectors_[:, self.lambdas_ > 0]
            self.lambdas_ = self.lambdas_[self.lambdas_ > 0]
        
        return K

    
    def fit_transform(self, X, y=None):
        #X=normalize(X)
        X = check_array(X, accept_sparse='csr', copy=True)
        self.K = self.kernels(X)
        return self.KPCA(self.coeficient)

    def KPCA(self,alpha):
        ##GET THE KERNEL WITH ALPHAS
        self.coeficient=alpha
        self.Kernel=self.Kf(self.K)
        #CENTER THE KERNEL
        self.Center_Kernel = self._centerer.fit_transform(self.Kernel) 
        #GET THE EIGENVALUES AND EIGENVECTORS
        self.Solve(self.Center_Kernel)
        #GET THE DIMENSIONAL REDUCTION
        X_Transform=self.vectors_ * np.sqrt(self.lambdas_)
        #X_Transform=np.matmul(self.Kernel,self.vectors_)
        return X_Transform,self.SuperK

    def Kf(self,K):
        Kf=np.zeros(K[0].shape)
        for i in range(0,len(self.coeficient)):
            Kf+=self.coeficient[i]*K[i]
        return Kf
    
    def createSuperK(self,K):
        data=[]
        for i in range(len(K)):
            data.append(np.ravel(K[i])[np.newaxis])
        return np.concatenate(tuple(data),axis=0).T


"""
HOW TO USE IT
KPCA=KernelPCA(n_components=2,gamma=15,nkernel=100)
X=np.random.random([10,10])
X_kpca,SuperK=KPCA.fit_transform(X)
"""
KPCA=KernelPCA(n_components=2,nkernel=100)
X=np.random.random([10,10])
X_kpca,SuperK=KPCA.fit_transform(X)
