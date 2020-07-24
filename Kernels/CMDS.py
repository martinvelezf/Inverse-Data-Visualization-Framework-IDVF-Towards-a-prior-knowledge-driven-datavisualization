"""
Multi-dimensional Scaling (MDS)
"""

# author: Nelle Varoquaux <nelle.varoquaux@gmail.com>
# License: BSD

import numpy as np

from sklearn.metrics import euclidean_distances
from sklearn.utils import check_random_state, check_array, check_symmetric
from sklearn.isotonic import IsotonicRegression




def KCMDS(X):
   
        X = check_array(X)
        D = euclidean_distances(X)
        n=len(X)
        e=np.ones(4)[np.newaxis]
        H=np.eye(n)-np.matmul(e,e.T)
        return (-0.5)*np.matmul(H,D,H)
    