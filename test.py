from kpca.KPCA2 import KernelPCA
from interfaz.matplot import MvPoints
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.datasets import load_iris
import numpy as np
from scipy import linalg
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from sklearn.preprocessing import normalize
    
class ShowKernel():
    def __init__(self,X,y=None):
        self.X=X
        self.KPCA=KernelPCA(n_components=2,nkernel=100)
        X_kpca,self.SuperK=self.KPCA.fit_transform(self.X)
        self.e1,self.e2=self.KPCA.lambdas_
        self.v1,self.v2=self.norm(X_kpca[:, 0]),self.norm(X_kpca[:, 1])

        ###Interactive Plot
        fig, ax = plt.subplots(1,2)
        self.mv=MvPoints(self.v1,self.v2,fig,ax,alpha=0.5,color=y)

        ###Button
        axnext = plt.axes([0.81, 0.01, 0.09, 0.05])
        #plt.subplots_adjust(bottom=0.2)
        bnext = Button(axnext, 'Kernel')
        bnext.on_clicked(self.next)
        plt.show()
    def norm(self,X):
        return X/X.max()
    def orientation(self,v1,v2):
        if np.matmul(v1[np.newaxis],v2[np.newaxis].T)<0:
            return -v1
        else:
            return v1    

    def find_M(self,x,y):
        return self.e1*np.matmul(x[np.newaxis].T,x[np.newaxis])+self.e2*np.matmul(y[np.newaxis].T,y[np.newaxis])

    def next(self,event):
        M1=self.find_M(np.asarray(self.mv.x),np.asarray(self.mv.y))
        M=np.asmatrix(np.ravel(M1)[:,np.newaxis])
        #Finde new alpha
        alpha=np.matmul(np.linalg.pinv(self.SuperK),M)
        print(np.linalg.norm(np.matmul(self.SuperK,alpha)-M))
        #alpha is tin the form of a matrix
        alphas=np.asarray(alpha.T)[0]
        #Find new kernel
        X_kpca,self.SuperK=self.KPCA.KPCA(np.array(alphas))
        self.e1,self.e2=np.sqrt(self.KPCA.lambdas_)
        self.v1,self.v2=X_kpca[:, 0],X_kpca[:, 1]
        #Check orientation
        self.v1=self.orientation(self.v1,np.asarray(self.mv.x))
        self.v2=self.orientation(self.v2,np.asarray(self.mv.y))
        self.mv.change(self.norm(self.v1),self.norm(self.v2))




def Database(db,i=0,s=None):
    if db=='iris':
        iris=load_iris()
        return iris.data[i:s, :],iris.target[i:s]  # we only take the first two features.
    if db=='moons':
       return make_moons(n_samples = s, noise = 0.02, random_state = 417) 

DB=['iris','moons']
K=[]
for i in DB:
    X,y=Database(i,i=0,s=100)
    K.append(ShowKernel(X,y))
