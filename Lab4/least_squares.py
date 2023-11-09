# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np
from costs import *


def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    # ***************************************************
    # COPY YOUR CODE FROM EX03 HERE
    # least squares: TODO
    # returns optimal weights, MSE
    
    'The optimal weights can be calculated as w = (tx.T*tx)^(-1)*tx.T*y'
    A = np.dot(tx.T, tx)
    B =  np.linalg.inv(A)
    w = np.dot(np.dot(B, tx.T), y)
    
    mse = compute_mse(y, tx, w)
    
    return w, mse
    
    # ***************************************************
    
