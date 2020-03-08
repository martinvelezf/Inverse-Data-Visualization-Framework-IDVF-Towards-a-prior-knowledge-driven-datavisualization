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

#Kernel from other linear methods
from Kernels.LLE import LLE
from Kernels.laplacian import LE
from Kernels.Isomap import Iso

from scipy.spatial import procrustes

class KernelPCA(TransformerMixin, BaseEstimator):
    def __init__(self, kernel="linear",
                 gamma=None, degree=3, coef0=1, kernel_params=None,
                 alpha=1.0, fit_inverse_transform=False, eigen_solver='auto',
                 tol=0, max_iter=None, remove_zero_eig=False, n_components=2,
                 random_state=None, copy_X=True, n_jobs=None,coeficient=None,nkernel=10):
        self.kernel_params = kernel_params
        self.gamma = gamma
        self.nkernel=nkernel
        self.n_components=n_components
        self.degree = degree
        self.coef0 = coef0
        self.alpha = alpha
        self.fit_inverse_transform = fit_inverse_transform
        self.eigen_solver = eigen_solver
        self.remove_zero_eig = remove_zero_eig
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.copy_X = copy_X
        self._centerer=KernelCenterer()
        self.coeficient=coeficient
    def kernels(self,X):
        kern=[ 'linear', 'poly', 'polynomial', 'rbf', 'laplacian', 'sigmoid', 'cosine']
        tkernel=len(kern)+2
        K=[]
        self.gamma=0.00001
        gamma_jump=1.02
        K.append(normalize(self._get_kernel(X,'rbf')))
        #K.append(procrustes(K[0],normalize(LLE.K(X)))[1])
        #K.append(procrustes(K[0],normalize(LE.K(X)))[1])
        #K.append(procrustes(K[0],normalize(Iso.K(X)))[1])
        K.append((LLE.K(X)))
        K.append((LE.K(X)))
        K.append((Iso.K(X)))
        for i in kern:
            if i=='rbf':
                for j in range(1,self.nkernel-tkernel):
                    #self.gamma=gamma_jump*self.gamma
                    self.gamma=0.3*j
                    K.append(self._get_kernel(X,i))
                    #K.append(procrustes(K[0],normalize(self._get_kernel(X,i)))[1])
            else:
                K.append(self._get_kernel(X,i))
                #K.append(procrustes(K[0],normalize(self._get_kernel(X,i)))[1])
        if not (self.coeficient):
            self.coeficient=np.zeros(len(K))
            self.coeficient[0]=1
        self.SuperK=self.createSuperK(K)
        return K
    
    def _get_kernel(self, X,kernel):
        params = {"gamma": self.gamma,"degree": self.degree,"coef0": self.coef0}
        return pairwise_kernels(X, None, metric=kernel,filter_params=True, n_jobs=self.n_jobs,**params)
    def normalize(self,v):
        return v/max(v)
    def Solve(self, K):
        
        # SELECT THE BEST METHOD TO CALCULATE THE EIGENVALUES
        if self.eigen_solver == 'auto':
            if K.shape[0] > 200 and self.n_components < 10:
                eigen_solver = 'arpack'
            else:
                eigen_solver = 'dense'
        else:
            eigen_solver = self.eigen_solver

        #GET EIGENVALUES AND EIGENVECTOR THE CENTER KERNEL
        if eigen_solver == 'dense':
            self.lambdas_, self.vectors_ = linalg.eigh(K, eigvals=(K.shape[0] - self.n_components, K.shape[0] - 1))
        elif eigen_solver == 'arpack':
            random_state = check_random_state(self.random_state)
            # initialize with [-1,1] as in ARPACK
            v0 = random_state.uniform(-1, 1, K.shape[0])
            self.lambdas_, self.vectors_ = eigsh(K, self.n_components,
                                                which="LA",
                                                tol=self.tol,
                                                maxiter=self.max_iter,
                                                v0=v0)

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

    def fit(self, X):
        
        return self

    def fit_transform(self, X, y=None):
        #X=normalize(X)
        X = check_array(X, accept_sparse='csr', copy=self.copy_X)
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