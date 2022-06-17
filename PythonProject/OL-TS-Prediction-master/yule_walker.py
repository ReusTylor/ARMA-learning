# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 11:03:34 2018

@author: mgdjohossou
"""


#We import the necessary packages
import numpy as np
from scipy.linalg import toeplitz
from scipy.linalg import inv
#from statsmodels.tsa.arima_model import ARIMA as arima



def lag(xs, n):
    if n > 0:
        return np.r_[np.full(n, np.nan), xs[:-n]]
    elif n < 0:
        return np.r_[xs[-n:], np.full(-n, np.nan)]
    else:
        return xs



def computeCov(X, h):
    mean = X.mean()
    size = len(X) - abs(h)
    
    lagged = lag(X, -abs(h))
    cov = ((X - mean)[0:size] * (lagged - mean)[0:size]).sum() / (size)
    
    return cov


def computeCor(X, h):
    return computeCov(X, h) / computeCov(X, 0)
    
    

def computeYWMatrix(X, p):
    corrs = []
    for i in range(0, p + 1):
        corrs.append(computeCor(X, i))
    
    ywMatrix = toeplitz(np.array(corrs[0 : -1]))
    return corrs, ywMatrix


def computeARCoefs(X, p):
    corrs, ywMatrix = computeYWMatrix(X, p)
    corrsReshaped = np.array(corrs[1:]).reshape((p, 1))
    
    coefs = inv(ywMatrix).dot(corrsReshaped)
    return coefs

def generateAR(coefs):
    x0 = np.random.norm()
    x = [x0]
    for i 
    return 0

def testAlgorithm(p):
    
    return 0


def predict(h):
    
    return 0