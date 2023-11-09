# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np


def compute_loss_mse(y, tx, w):
    """Calculate the loss using MSE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss by MSE
    
    e = y - tx.dot(w)
    mse = e.dot(e) / (2*len(e))
    return mse
    
    # ***************************************************

    
def compute_rmse(y, tx, w):
    # Calculate the RMSE
    mse = compute_loss_mse(y, tx, w)
    rmse = np.sqrt(2*mse)
    return rmse    


def compute_loss_mae(y, tx, w):
    """Calculate the loss using MAE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss by MAE
    
    e = y - np.dot(tx,w)
    loss = 0
    for i in range(len(y)):
        if e[i] > 0: loss += e[i]
        elif e[i] < 0: loss += -e[i]
    
    loss = loss/(len(y))
    return loss
    # **************************************************  
