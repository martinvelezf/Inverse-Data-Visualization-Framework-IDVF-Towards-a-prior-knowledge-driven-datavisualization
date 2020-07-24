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
from scipy.io import savemat


class ShowKernel():
    def __init__(self,X,y,name,c=0):
        self.c=c # Count the number of aproximation made by the user
        self.X=X
        self.y=y

        self.title=name
        self.KPCA=KernelPCA(n_components=2,nkernel=100)
        X_kpca,self.SuperK=self.KPCA.fit_transform(self.X)
        
        
        #Save values to matlab
        savemat("results/"+self.title+'_KCMDS.mat',dict(x=X_kpca))

        self.v1,self.v2=self.normx(X_kpca[:, 0]),self.normy(X_kpca[:, 1])
        """
        fg,axis= plt.subplots(1)
        axis.set_title("KCMDS")
        axis.scatter(X_kpca[:, 0],X_kpca[:, 1],c=y,cmap=plt.cm.Spectral)
        
        """
        ###Interactive Plot
        fig, ax = plt.subplots(1,2)
        ax[0].set_title("DR from KCMDS")
        
        self.mv=MvPoints(self.v1,self.v2,fig,ax,alpha=0.5,color=y)

        ###Button
        axnext = plt.axes([0.81, 0.01, 0.09, 0.05])
        #plt.subplots_adjust(bottom=0.2)
        bnext = Button(axnext, 'Kernel')
        bnext.on_clicked(self.next)
        plt.show()
        

        
    def normx(self,X):
        self.e1=X.max()
        return X/self.e1
    def normy(self,X):
        self.e2=X.max()
        return X/self.e2    
    def orientation(self,v1,v2):
        if np.matmul(v1[np.newaxis],v2[np.newaxis].T)<0:
            return -v1
        else:
            return v1    

    def find_M(self,x,y):
        return (self.e1**2)*np.matmul(x[np.newaxis].T,x[np.newaxis])+(self.e2**2)*np.matmul(y[np.newaxis].T,y[np.newaxis])

    def next(self,event):
        #Save expected values
        Expected=np.array([np.asarray(self.mv.x),np.asarray(self.mv.y)]).T
        
        #Get Matrix M
        M1=self.find_M(np.asarray(self.mv.x),np.asarray(self.mv.y))
        M=np.asmatrix(np.ravel(M1)[:,np.newaxis])
        #Finde new alpha
        alpha=np.matmul(np.linalg.pinv(self.SuperK),M)
        
        #error
        #print(np.linalg.norm(np.matmul(self.SuperK,alpha)-M))
        
        #alpha is tin the form of a matrix
        alphas=np.asarray(alpha.T)[0]
        #Find new kernel
        X_kpca,self.SuperK=self.KPCA.KPCA(np.array(alphas))
        """
        figure,axi= plt.subplots(1)
        axi.set_title("Expected")
        axi.scatter(np.asarray(self.mv.x),np.asarray(self.mv.y),c=self.y,cmap=plt.cm.Spectral)
        fg,axis= plt.subplots(1)
        axis.set_title("Mixed Kernel")
        axis.scatter(X_kpca[:, 0],X_kpca[:, 1],c=self.y,cmap=plt.cm.Spectral)
        plt.show()
        """
        #Save values to matlab
        savemat("results/"+self.title+'_RESULT_'+str(self.c)+'.mat',dict(mix=X_kpca,exp=Expected))
        self.c+=1 #increase the count
        
        self.v1,self.v2=self.normx(X_kpca[:, 0]),self.normy(X_kpca[:, 1])
        #Check orientation
        self.v1=self.orientation(self.v1,np.asarray(self.mv.x))
        self.v2=self.orientation(self.v2,np.asarray(self.mv.y))
        self.mv.change(self.v1,self.v2)
        #plt.show()
        
    


        



