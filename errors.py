""" 
Module errors. Contains:
error_prop Calculates the error range caused by the uncertainty of the fit
    parameters. Covariances are taken into account.
cover_to_corr: Converts covariance matrix into correlation matrix.
"""


import numpy as np


def error_prop(x, func, parameter, covar):
    """
    Calculates 1 sigma error ranges for number or array. It uses error
    propagation with variances and covariances taken from the covar matrix.
    Derivatives are calculated numerically. 
    
    """
    
    # initiate sigma the same shape as parameter

    var = np.zeros_like(x)   # initialise variance vektor
    # Nested loop over all combinations of the parameters
    for i in range(len(parameter)):
        # derivative with respect to the ith parameter
        deriv1 = deriv(x, func, parameter, i)

        for j in range(len(parameter)):
            # derivative with respct to the jth parameter
            deriv2 = deriv(x, func, parameter, j)
            
            
                
            # multiplied with the i-jth covariance
            # variance vektor 
            var = var + deriv1*deriv2*covar[i, j]

    sigma = np.sqrt(var)
    return sigma


def deriv(x, func, parameter, ip):
    """
    Calculates numerical derivatives from function
    values at parameter +/- delta.  Parameter is the vector with parameter
    values. ip is the index of the parameter to derive the derivative.

    """

    # print("in", ip, parameter[ip])
    # create vector with zeros and insert delta value for relevant parameter
    # delta is calculated as a small fraction of the parameter value
    scale = 1e-6   # scale factor to calculate the derivative
    delta = np.zeros_like(parameter, dtype=float)
    val = scale * np.abs(parameter[ip])
    delta[ip] = val  #scale * np.abs(parameter[ip])
    
    diff = 0.5 * (func(x, *parameter+delta) - func(x, *parameter-delta))
    dfdx = diff / val

    return dfdx


def covar_to_corr(covar):
    """ Converts the covariance matrix into a correlation matrix """

    # extract variances from the diagonal and calculate std. dev.
    sigma = np.sqrt(np.diag(covar))
    # construct matrix containing the sigma values
    matrix = np.outer(sigma, sigma)
    # and divide by it
    corr = covar/matrix
    
    return corr
                       
