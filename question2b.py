# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 14:07:50 2015

"""

import numpy as np
from matplotlib import pyplot as plt
from utils import get_random_state
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor



random_state = get_random_state()

'''
This function is used to create a sample of data thanks
to the given function

Input : - x : The input value
        - n_samples : number of samples made with this input
        
Output : - y : The data
'''
def make_data(x,n_samples = 1, noise = 1):

    y = np.zeros(n_samples)

    for i in range(n_samples):
        eps = random_state.normal(0, noise)
        y[i] = bayes_model(x) + eps

    return y

'''
This function is used to create to bayes model of a given function

Input : - x : The input value

Output : - bayes : Bayes model create from x
'''
def bayes_model(x):
            
    bayes = (np.sin(x)+ np.cos(x))*x**2

    return bayes

'''
This function is used to create the squared bias and the variance at a given
x0 for given supervised learning algorithm

Input : 

Output : 

'''
def variance_bias(regressors, n_samples, x0, n_fit, noise = 1):
        
    y_estimate = np.zeros((n_fit,len(regressors)))
    bias_squared = np.zeros(len(regressors))
    variance = np.zeros(len(regressors))
    x = np.zeros((n_samples))
    y = np.zeros((n_samples))
    for n in range(n_fit):
        x = np.zeros((n_samples))
        y = np.zeros((n_samples))
        for i in range (n_samples):
            x[i] = random_state.uniform(low=-9.0, high=9.0)
            y[i] = make_data(x[i],1,1)
        x = np.array([x]).transpose()
        for j in range(len(regressors)):
            regressors[j].fit(x,y)
            y_estimate[n,j] = regressors[j].predict(x0)
            
      
    for i in range(len(regressors)):    
        bias_squared[i] = (bayes_model(x0)- np.mean(y_estimate[:,i]))**2
        variance[i] = np.var(y_estimate[:,i])
        print bayes_model(x0)
        print np.mean(y_estimate[:,i])
    
    return variance, bias_squared
    

def residual_error(x0, n_samples):
    y = make_data(x0, n_samples)
    return np.var(y)
    
def plot_var_bias_size_LS(regressors, x0,name_regression):
    
    size_LS = [50, 100, 250, 500,750, 1000, 1500, 2500]
    
    mean_var = np.zeros((len(size_LS),len(regressors)))
    mean_bias = np.zeros((len(size_LS),len(regressors)))
    error = np.zeros((len(size_LS),len(regressors)))
    
    for s in range(len(size_LS)):
        variance = np.zeros((len(x0),len(regressors)))
        bias_squared = np.zeros((len(x0),len(regressors)))
        for i in range(len(x0)):
            variance[i,:], bias_squared[i,:] = variance_bias(regressors,size_LS[s], x0[i],500)
        
        for j in range(len(regressors)):
            mean_var[s,j] = np.mean(variance[:,j])
            mean_bias[s,j] = np.mean(bias_squared[:,j])
            error[s,j] = mean_var[s,j] + mean_bias[s,j]
            
    for j in range(len(regressors)):     
        plt.figure()
        plt.plot(size_LS,mean_var[:,j], label="Mean variance")
        plt.plot(size_LS,mean_bias[:,j], label="Mean squared bias")
        plt.plot(size_LS,error[:,j], label="Error")
        plt.title( name_regression[j]+" :Effect of the change of size of the LS")
        plt.xlabel("Size of the LS")
        plt.legend(loc = "lower center")
        plt.savefig(name_regression[j]+ "change of LS.pdf")
        

def plot_var_bias_complexity(x0, n_samples):
    
    n_neighbors = [1,2, 3,4, 5 ,7, 10, 15, 25,100] #
    
    mean_var = np.zeros(len(n_neighbors))
    mean_bias = np.zeros(len(n_neighbors))
    error = np.zeros(len(n_neighbors))    
    
    for k in range(len(n_neighbors)):
        knn_regressor = KNeighborsRegressor(n_neighbors[k]) 
        regressors = ([knn_regressor])
        variance = np.zeros(len(x0))
        bias_squared = np.zeros(len(x0))
        for i in range(len(x0)):
            variance[i], bias_squared[i] = variance_bias(regressors,n_samples, x0[i],500)
        
    
        mean_var[k] = np.mean(variance)
        mean_bias[k] = np.mean(bias_squared)
        error[k] = mean_var[k] + mean_bias[k] 
            
    
    plt.figure()
    plt.plot(n_neighbors,mean_var, label="Mean variance")
    plt.plot(n_neighbors,mean_bias, label="Mean squared bias")
    plt.plot(n_neighbors,error, label="Error")
    plt.title("Effect of the change of complexity")
    plt.xlabel("Number of neighbors")
    plt.legend(loc = "lower center")
    plt.savefig("Change of complexity.pdf")
    

    return variance , bias_squared,mean_var, mean_bias 
    
def plot_var_bias_over_noise(regressors, x0,name_regression):
    
    noise = [0,0.25,0.50,0.75,1,1.5,2,3]
    
    mean_var = np.zeros((len(noise),len(regressors)))
    mean_bias = np.zeros((len(noise),len(regressors)))
    error = np.zeros((len(noise),len(regressors)))
    
    for n in range(len(noise)):
        print 1
        variance = np.zeros(len(x0))
        bias_squared = np.zeros(len(x0))
        for i in range(len(x0)):
            variance[i,:], bias_squared[i,:] = variance_bias(regressors,n_samples, x0[i],50,noise[n])
            
        for j in range(len(regressors)):
            mean_var[n,j] = np.mean(variance[:,j])
            mean_bias[n,j] = np.mean(bias_squared[:,j])
            error[n,j] = mean_var[n,j] + mean_bias[n,j]

            
    for j in range(len(regressors)):     
        plt.figure()
        plt.plot(noise,mean_var[:,j], label="Mean variance")
        plt.plot(noise,mean_bias[:,j], label="Mean squared bias")
        plt.plot(noise,error[:,j], label="Error")
        plt.title( name_regression[j]+" :Effect of the change of noise")
        plt.xlabel("Size of the LS")
        plt.legend(loc = "lower center")
        plt.savefig(name_regression[j]+ "change of noise.pdf")
    
    



if __name__ == "__main__":

    x0 = np.linspace(-9.0,9.0,90)
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
    

    variance = np.zeros((len(x0),len(regressors)))
    bias_squared = np.zeros((len(x0),len(regressors)))
    
    for i in range(len(x0)):
        variance[i,:], bias_squared[i,:] = variance_bias(regressors,n_samples, x0[i],500)
    
    #PLOT
    pred = np.zeros((len(x0),len(regressors)))
    for i in range(len(x0)):
        for j in range(2):
            pred[i,j] = regressors[j].predict(x0[i])
            
    #Prediction with Linear regression
    
    plt.figure()
    plt.plot(x0, (np.sin(x0)+ np.cos(x0))*x0**2, linewidth = 2.0, label="Real function")
    plt.plot(x0, pred[:,0], label = "Linear prediction") 
    plt.plot(x0, pred[:,1], label = "knn prediction")
    plt.title( "Prediction against real function")
    plt.xlabel("x")
    plt.legend(loc = "upper center")
    plt.xlim((-9.0, 9.0))
    plt.savefig("pred.pdf")

    
    
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

    
    #QUESTION 2 D    
    
    
    #plot_var_bias_size_LS(regressors, x0, name_regression)
    
    #v, b,mean_var, mean_bias = plot_var_bias_complexity(x0, n_samples)
    
    #plot_var_bias_over_noise(regressors,x0, name_regression)
    
    

          
        
        
        
        
        
        
        
        
        
        
        
    
