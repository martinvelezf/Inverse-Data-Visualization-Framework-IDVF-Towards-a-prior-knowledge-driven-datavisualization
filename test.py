
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.utils import check_random_state
from sklearn.datasets import *
from scipy.io import savemat
from show import ShowKernel    

from sklearn.decomposition import PCA
from sklearn.manifold import MDS
def sphere(n=100):
	
	# Create our sphere.
	random_state = check_random_state(0)
	p = random_state.rand(n) * (2 * np.pi)
	t = random_state.rand(n) * np.pi

	# Sever the poles from the sphere.
	#indices = ((t < (np.pi - (np.pi / 8))) & (t > ((np.pi / 8))))
	#colors = p[indices]
	x, y, z = np.sin(t) * np.cos(p),np.sin(t) * np.sin(p),np.cos(t)
	sphere_data = np.array([x, y, z]).T
	return sphere_data,p

def Database(db,name,i=0,n=800):
    if name!='iris':
        X,y=db[name](n)
        return X,y  # we only take the first two features.
   
    else:
        return iris.data[i:n, :],iris.target[i:n]

    
DB={'S_Curve':make_s_curve,'Swiss_Roll':make_swiss_roll,'IRIS':load_iris}#'Sphere':sphere,'moons':'','boston':load_boston,'wine':load_wine,'cancer':load_breast_cancer}
K=[]


c=0
cmds = MDS(n_components=2)
pca = PCA(n_components=2)
for i in DB:
    #Get Values from database
    X,y=Database(DB,i,i=30,n=200)
    
    pca.fit(X)
    PCA=pca.transform(X)
    
    CMDS=cmds.fit_transform(X)
    
    savemat("results/"+i+".mat",dict(x=X,cmds=CMDS,pca=PCA,y=y))
    K.append(ShowKernel(X,y,i))
    
    """
    fig = plt.figure()
    fig_cmds= plt.figure()
    
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.Spectral)
    ax.set_title(i)
    
    ax_cmds = fig_cmds.add_subplot(111)
    ax_cmds.scatter(X_r[:, 0], X_r[:, 1],c=y, cmap=plt.cm.Spectral)
    ax_cmds.set_title("CMDS")
    plt.show()
    
    """
    
    
    
   