import numpy as np

def calculate_hypothesis(X, theta, i):
    """
        :param X            : 2D array of our dataset
        :param theta        : 1D array of the trainable parameters
        :param i            : scalar, index of current training sample's row
    """
    
    #########################################
    # Write your code here
    # You must calculate the hypothesis for the i-th sample of X, given X, theta and i.
    hypothesis = 0.0
    
    for j in range(X.shape[1]):
        if j > 1:
            hypothesis += theta[j] * X[i][j] ** j  
        else:
            hypothesis += theta[j] * X[i][j]
    
    ########################################/
    
    return hypothesis
