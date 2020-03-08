from sklearn.datasets import make_moons
from sklearn.datasets import load_iris
from show import ShowKernel    

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
