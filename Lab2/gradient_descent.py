# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Gradient Descent
"""
import numpy as np
import costs
from costs import *

def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute gradient vector
    
    e = y - np.dot(tx,w)
    grad = -np.dot(tx.T, e)/len(e)
    
    return grad
    
    # ***************************************************


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        loss: the final loss after the hole algorithm
        w: a numpy array of shape=(2, ) of the optimal parameters w0, w1
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: compute gradient and loss
        
        grad = compute_gradient(y, tx, w)
        loss = compute_loss_mse(y, tx, w)
        
        # ***************************************************

        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: update w by gradient
        
        w = w - gamma*grad
        
        # ***************************************************

        # store w and loss
        ws.append(w)
        losses.append(loss)
        '''
        To print all the iterations of the algorithm
        print(
            "GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )
        '''
        
    loss = losses[max_iters - 1]
    w = ws[max_iters - 1]

    return loss, w
       
    # ***************************************************



