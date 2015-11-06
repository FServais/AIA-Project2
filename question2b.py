# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 14:07:50 2015

"""

import numpy as np
from sklearn.utils import check_random_state
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

random_state = 0
random_state = check_random_state(random_state) 



def make_data(x, n):
    

    
    for i in range(n_samples):
           
        x[i] = random_state.uniform(low=-9.0, high=9.0)
        noise[i] = random_state.normal(0,1)
        y[i] = bayes[i] + noise[i]
        
    return y,x,bayes
        
def bayes_model(x):
            
    bayes = (np.sin(x)+ np.cos(x))*x**2
        
    return bayes

def variance_bias(n_sample,x0, x, y, n_fit):
        
    y_estimate = np.zeros((n_fit,2))
    bias_squared = np.zeros(2)
    variance = np.zeros(2)
    linear_regression = LinearRegression()
    knn_regressor = KNeighborsRegressor()
    for i in range(n_fit):
        j = 0
        linear_regression.fit(x,y)
        knn_regressor.fit(x,y)
        y_estimate[i,j] = linear_regression.predict(x0)
        j +=1
        y_estimate[i,j] = knn_regressor.predict(x0)
    
    for i in range(2):    
        bias_squared = (bayes_model(x0)- np.mean(y_estimate))**2
        variance = np.var(y_estimate)
    
    return variance, bias_squared
    



#MAIN

x0 = np.linspace(-9,9,90)

n_samples = 2000

x = np.zeros((n_samples))
y = np.zeros((n_samples))

for i in range(n_samples):
    x[i] = random_state.uniform(low=-9.0, high=9.0)
    y[i] = make_data(x[i])
    
for i in range(len(x0)):
    
    



