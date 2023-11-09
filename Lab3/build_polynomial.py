# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # polynomial basis function: TODO
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    
    poly = np.ones((0, len(x)))
    for i in range(degree + 1):
        poly = np.vstack((poly, np.power(x, i)))
    
    return poly.T
    
    # ***************************************************
    
