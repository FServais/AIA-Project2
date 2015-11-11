# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 14:07:50 2015

"""

import numpy as np
from matplotlib import pyplot as plt

from utils import get_random_state
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

random_state = get_random_state()


def make_data(x,n_samples = 1):

    y = np.zeros(n_samples)

    for i in range(n_samples):
        eps = random_state.normal(0, 1)
        y[i] = bayes_model(x) + eps

    return y


def bayes_model(x):
            
    bayes = (np.sin(x)+ np.cos(x))*x**2

    return bayes

def variance_bias(regressors, n_samples, x0, n_fit):
        
    y_estimate = np.zeros((n_samples,2))
    bias_squared = np.zeros((2))
    variance = np.zeros((2))
    x = np.zeros((n_samples))
    y = np.zeros((n_samples))
    for n in range(n_fit):
        x = np.zeros((n_samples))
        y = np.zeros((n_samples))
        for i in range (n_samples):
            x[i] = random_state.uniform(low=-9.0, high=9.0)
            y[i] = make_data(x[i])
        x = np.array([x]).transpose()
        j = 0
        regressors[0].fit(x,y)
        regressors[1].fit(x,y)
        y_estimate[i,j] = regressors[0].predict(x0)
        j +=1
        y_estimate[i,j] = regressors[1].predict(x0)
        
        
    for i in range(2):    
        bias_squared[i] = (bayes_model(x0)- np.mean(y_estimate[:,i]))**2
        variance[i] = np.var(y_estimate[:,i])
    
    return variance, bias_squared
    

def residual_error(x0, n_samples):
    y = make_data(x0, n_samples)
    return np.var(y)



if __name__ == "__main__":

    x0 = np.linspace(-9.0, 9.0, 15)
    n_samples = 2000
    
        
    
    linear_regression = LinearRegression()
    knn_regressor = KNeighborsRegressor()    
    
    regressors = ([linear_regression, knn_regressor])
    
    #Residual    
    
    res_errors = np.zeros(len(x0))
    for i in range(len(x0)):
        res_errors[i] = residual_error(x0[i], n_samples)

    

    #Variance and Squared bias    
    
    name_regression = ["Linear regression", "knn regression"]
    
    variance = np.zeros((len(x0),2))
    bias_squared = np.zeros((len(x0),2))
    
    
    for i in range(len(x0)):
        variance[i,:], bias_squared[i,:] = variance_bias(regressors,n_samples, x0[i],500)
    
    #PLOT
    pred = np.zeros((len(x0),2))
    for i in range(len(x0)):
        for j in range(2):
            pred[i,j] = regressors[j].predict(x0[i])
            
    #Prediction with Linear regression
    
    plt.figure()
    plt.plot(x0, (np.sin(x0)+ np.cos(x0))*x0**2)
    plt.plot(x0, pred[:,0])
    
    plt.title( "Residual errors")
    plt.xlabel("x")
    plt.xlim((-9.0, 9.0))
    plt.savefig("Residual_errors.pdf")
    
    plt.figure()
    plt.plot(x0, (np.sin(x0)+ np.cos(x0))*x0**2)
    plt.plot(x0, pred[:,1])
    
    plt.title( "Residual errors")
    plt.xlabel("x")
    plt.xlim((-9.0, 9.0))
    plt.savefig("Residual_errors.pdf")
    
    
    
    print("Residual error : {}".format(np.mean(res_errors)))
    
    plt.figure()
    plt.plot(x0, res_errors)
    plt.title( "Residual errors")
    plt.xlabel("x")
    plt.xlim((-9.0, 9.0))
    plt.savefig("Residual_errors.pdf")
    
    
    for i in range(2):
        print(name_regression[i])
        print("Variance = {}".format(np.mean(variance[:,i])))
        plt.figure()
        plt.plot(x0, variance[:,i])
        plt.title( name_regression[i] +" : Variance")
        plt.xlabel("x")
        plt.xlim((-9.0, 9.0))
        plt.savefig(name_regression[i]+" variance.pdf")
        print("Bias Squared = {}".format(np.mean(bias_squared[:,i])))
        plt.figure()
        plt.plot(x0, bias_squared[:,i])
        plt.title( name_regression[i]+" : Squared Bias")
        plt.xlabel("x")
        plt.xlim((-9.0, 9.0))
        plt.savefig(name_regression[i]+" squared_bias.pdf")
        print("Total Error = {}".format(np.mean(bias_squared[:,i])+ np.mean(variance[:,i])))
        plt.figure()
        plt.plot(x0, (bias_squared[:,i]+variance[:,i]))
        plt.title( name_regression[i]+" : Total error")
        plt.xlabel("x")
        plt.xlim((-9.0, 9.0))
        plt.savefig(name_regression[i]+" total_error.pdf")
        
        
        
        
        
        
        
        
        
        
        
        
    
