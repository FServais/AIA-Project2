# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 14:07:50 2015

"""

import numpy as np

from utils import get_random_state
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

random_state = get_random_state()


def make_data(x,n_samples = 1):

    y = np.zeros(n_samples)

    for i in range(n_samples):
        eps = random_state.normal(0, 1)
        y[i] = (np.sin(x) + np.cos(x)) * x**2 + eps

    return y


def bayes_model(x):
            
    bayes = (np.sin(x)+ np.cos(x))*x**2

    return bayes

def variance_bias(n_samples, x0, n_fit):
        
    y_estimate = np.zeros((n_samples,2))
    bias_squared = np.zeros((2))
    variance = np.zeros((2))
    x = np.zeros((n_samples))
    y = np.zeros((n_samples))
    linear_regression = LinearRegression()
    knn_regressor = KNeighborsRegressor()
    for n in range(n_fit):
        x = np.zeros((n_samples))
        y = np.zeros((n_samples))
        for i in range (n_samples):
            x[i] = random_state.uniform(low=-9.0, high=9.0)
            y[i] = make_data(x[i])
        x = np.array([x]).transpose()
        j = 0
        linear_regression.fit(x,y)
        knn_regressor.fit(x,y)
        y_estimate[i,j] = linear_regression.predict(x0)
        j +=1
        y_estimate[i,j] = knn_regressor.predict(x0)
        
    for i in range(2):    
        bias_squared[i] = (bayes_model(x0)- np.mean(y_estimate[:,i]))**2
        variance[i] = np.var(y_estimate[:,i])
    
    return variance, bias_squared
    

def residual_error(x0, n_samples):
    y = make_data(x0, n_samples)
    return np.var(y)



if __name__ == "__main__":

    x0 = np.linspace(-9.0, 9.0, 40)
    n_samples = 2000
    
    #Residual    
    
    res_errors = np.zeros(len(x0))
    for i in range(len(x0)):
        res_errors[i] = residual_error(x0[i], n_samples)

    print("Residual error : {}".format(np.mean(res_errors)))

    #Variance and Squared bias    
    
    name_regression = ["Linear regression", "knn regression"]
    
    variance = np.zeros((len(x0),2))
    bias_squared = np.zeros((len(x0),2))
    
    
    for i in range(len(x0)):
        variance[i,:], bias_squared[i,:] = variance_bias(n_samples, x0[i],500)
        
    for i in range(2):
        print(name_regression[i])
        print("Variance = {}".format(np.mean(variance[i,:])))
        print("Bias Squared = {}".format(np.mean(bias_squared[i,:])))
        
        
    
    #PLOT
