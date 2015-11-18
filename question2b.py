# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 14:07:50 2015

"""

import numpy as np
from matplotlib import pyplot as plt
from utils import get_random_state
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import SGDRegressor



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
def variance_bias(regressors, n_samples, x0, n_fit, noise = 1, nb_irrelevant = 0):
        
    y_estimate = np.zeros((n_fit,len(regressors)))
    bias_squared = np.zeros(len(regressors))
    variance = np.zeros(len(regressors))
    #if nb_irrelevant != 0:
    #   x0_extended = np.ones(nb_irrelevant+1)*x0

    
    for n in range(n_fit):
        if nb_irrelevant == 0:
            #print 5
            x = np.zeros((n_samples))
            y = np.zeros((n_samples))
            for i in range (n_samples):
                x[i] = random_state.uniform(low=-9.0, high=9.0)
                y[i] = make_data(x[i],1,noise)
            x = np.array([x]).transpose()
            for j in range(len(regressors)):
                regressors[j].fit(x,y)
                y_estimate[n,j] = regressors[j].predict(x0)
        '''        
        else:
            print 6
            X = np.zeros((n_samples,nb_irrelevant+1))
            y = np.zeros((n_samples,nb_irrelevant+1))
            for i in range(n_samples):
                for r in range(nb_irrelevant+1):
                    X[i,r] = random_state.uniform(low=-9.0, high=9.0)
                    y[i] = make_data(X[i,0],1,noise)
            for j in range(len(regressors)):
                regressors[j].fit(X,y)
                print regressors[j].predict(x0_extended)
                y_estimate[n,j] = regressors[j].predict(x0_extended)
        '''
    for i in range(len(regressors)):    
        bias_squared[i] = (bayes_model(x0)- np.mean(y_estimate[:,i]))**2
        variance[i] = np.var(y_estimate[:,i])
        

    
    return variance, bias_squared
    

def residual_error(x0, n_samples):
    y = make_data(x0, n_samples)
    return np.var(y)
    
def plot_var_bias_size_LS(regressors, x0,name_regression):
    
    size_LS = [5,25,50,75, 100, 250, 500,750, 1000]# 1500, 2500]
    
    mean_var = np.zeros((len(size_LS),len(regressors)))
    mean_bias = np.zeros((len(size_LS),len(regressors)))
    mean_res_error = np.ones(len(size_LS))
    error = np.zeros((len(size_LS),len(regressors)))
    
    for s in range(len(size_LS)):
        print 1
        variance = np.zeros((len(x0),len(regressors)))
        bias_squared = np.zeros((len(x0),len(regressors)))
        for i in range(len(x0)):
            variance[i,:], bias_squared[i,:] = variance_bias(regressors,size_LS[s], x0[i],10)
        
        for j in range(len(regressors)):
            mean_var[s,j] = np.mean(variance[:,j])
            mean_bias[s,j] = np.mean(bias_squared[:,j])
            
            error[s,j] = mean_var[s,j] + mean_bias[s,j] + mean_res_error[s]
            
    for j in range(len(regressors)):     
        plt.figure()
        plt.plot(size_LS,mean_var[:,j], label="Mean variance")
        plt.plot(size_LS,mean_bias[:,j], label="Mean squared bias")
        plt.plot(size_LS,mean_res_error, label="Residual error")
        plt.plot(size_LS,error[:,j], label="Error")
        plt.title( name_regression[j]+" :Effect of the change of size of the LS")
        plt.xlabel("Size of the LS")
        plt.legend(loc = "upper right")
        plt.savefig(name_regression[j]+ "change of LS.pdf")
        

        

def plot_var_bias_complexity(x0, n_samples,name_regression):
    
    
    parameters =np.array([[5,25,75,150,250,500,1000,1500,2000],[1,2, 3,4, 5 ,7, 10, 15, 30]])
    
    xlabel = ["Number of iterations", "Number of neighbors"]    
    
    name_regression[0] ="SGD regression"
    
    mean_var = np.zeros((np.shape(parameters)[1],2))
    mean_bias = np.zeros((np.shape(parameters)[1],2))
    mean_res_error = np.ones(np.shape(parameters)[1])
    error = np.zeros((np.shape(parameters)[1],2))   
    
    for k in range(np.shape(parameters)[1]):
        print 1
        knn_regressor = KNeighborsRegressor(parameters[1,k])
        SGD_regressor = SGDRegressor(n_iter = parameters[0,k])
        regressors = [SGD_regressor,knn_regressor]
        variance = np.zeros((len(x0),len(regressors)))
        bias_squared = np.zeros((len(x0),len(regressors)))
        for i in range(len(x0)):
            variance[i,:], bias_squared[i,:] = variance_bias(regressors,n_samples, x0[i],50)
        for j in range(len(regressors)):
            mean_var[k,j] = np.mean(variance[:,j])
            mean_bias[k,j] = np.mean(bias_squared[:,j])
            error[k,j] = mean_var[k,j] + mean_bias[k,j] + mean_res_error[k]
            
    for j in range(len(regressors)): 
        plt.figure()
        plt.plot(parameters[j,:],mean_var[:,j], label="Mean variance")
        plt.plot(parameters[j,:],mean_bias[:,j], label="Mean squared bias")
        plt.plot(parameters[j,:],mean_res_error, label="Residual error")
        plt.plot(parameters[j,:],error[:,j], label="Error")
        plt.title(name_regression[j]+" :Effect of the change of complexity")
        plt.xlabel(xlabel[j])
        plt.legend(loc = "upper center")
        plt.savefig(name_regression[j]+"Change of complexity.pdf")
        
    plt.figure()
    plt.plot(parameters[0,:], mean_var[:,0])
    plt.xlabel(xlabel[0])
    plt.title(name_regression[0]+" : Mean variance")
    plt.savefig(name_regression[0]+"Variance.pdf")
    
     
    
def plot_var_bias_over_noise(regressors, x0, name_regression,n_samples):
    
    noise = [0.1,0.25,0.50,0.75,1,1.5,2,3,5,10,20]
    
    mean_var = np.zeros((len(noise),len(regressors)))
    mean_bias = np.zeros((len(noise),len(regressors)))
    error = np.zeros((len(noise),len(regressors)))
    
    for n in range(len(noise)):
        print 1
        variance = np.zeros((len(x0),len(regressors)))
        bias_squared = np.zeros((len(x0),len(regressors)))
        for i in range(len(x0)):
            variance[i,:], bias_squared[i,:] = variance_bias(regressors,n_samples, x0[i],100,noise[n])
            
        for j in range(len(regressors)):
            mean_var[n,j] = np.mean(variance[:,j])
            mean_bias[n,j] = np.mean(bias_squared[:,j])
            error[n,j] = mean_var[n,j] + mean_bias[n,j] + noise[n]

            
    for j in range(len(regressors)):     
        plt.figure()
        plt.plot(noise,mean_var[:,j], label="Mean variance")
        plt.plot(noise,mean_bias[:,j], label="Mean squared bias")
        plt.plot(noise, noise, label="Residual error")
        plt.plot(noise,error[:,j], label="Error")
        plt.title( name_regression[j]+" :Effect of the change of noise")
        plt.xlabel("Variance of the noise")
        plt.legend(loc = "center left")
        plt.savefig(name_regression[j]+ "change of noise.pdf")
        
    plt.figure()
    plt.plot(noise, mean_var[:,0])
    plt.xlabel("Variance of the noise")
    plt.title(name_regression[0]+" : Mean variance")
    plt.savefig(name_regression[0]+"Variance_noise.pdf")
    

def plot_var_bias_over_irrelevant_variables(regressors, x0, name_regression, n_samples):
    
    
    
    nb_irrelevant = [2]
    
    mean_var = np.zeros((len(nb_irrelevant),len(regressors)))
    mean_bias = np.zeros((len(nb_irrelevant),len(regressors)))
    error = np.zeros((len(nb_irrelevant),len(regressors)))
    
    for n in range(len(nb_irrelevant)):
        print 1
        variance = np.zeros((len(x0),len(regressors)))
        bias_squared = np.zeros((len(x0),len(regressors)))
        for i in range(len(x0)):
            variance[i,:], bias_squared[i,:] = variance_bias(regressors,n_samples, x0[i],50,1,nb_irrelevant[i])
            
        for j in range(len(regressors)):
            mean_var[n,j] = np.mean(variance[:,j])
            mean_bias[n,j] = np.mean(bias_squared[:,j])
            error[n,j] = mean_var[n,j] + mean_bias[n,j]

            
    for j in range(len(regressors)):     
        plt.figure()
        plt.plot(noise,mean_var[:,j], label="Mean variance")
        plt.plot(noise,mean_bias[:,j], label="Mean squared bias")
        plt.plot(noise,error[:,j], label="Error")
        plt.title( name_regression[j]+" :Effect of the addition of irrelevant variables")
        plt.xlabel("Number of irrelevant variables")
        plt.legend(loc = "lower center")
        plt.savefig(name_regression[j]+ "with irrelevant variables.pdf")
    
    





if __name__ == "__main__":

    x0 = np.linspace(-9.0,9.0,90)
    n_samples = 1000
    
    linear_regression = LinearRegression()
    knn_regressor = KNeighborsRegressor()    
    
    regressors = [linear_regression, knn_regressor]
    
    
    
    
    #Residual    
    
    res_errors = np.zeros(len(x0))
    for i in range(len(x0)):
        res_errors[i] = residual_error(x0[i], n_samples)

    

    #Variance and Squared bias    
    
    name_regression = ["Linear regression", "knn regression"]
    
    '''
    variance = np.zeros((len(x0),len(regressors)))
    bias_squared = np.zeros((len(x0),len(regressors)))
    
    for i in range(len(x0)):
        print 1
        variance[i,:], bias_squared[i,:] = variance_bias(regressors,n_samples, x0[i],500)
    
    #PLOT
    pred = np.zeros((len(x0),len(regressors)))
    for i in range(len(x0)):
        for j in range(len(regressors)):
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

    '''
    
    #QUESTION 2 D    
    
    
    #plot_var_bias_size_LS(regressors, x0, name_regression)
    #print 2
    #plot_var_bias_complexity(x0, n_samples,name_regression)
    #print 2
    plot_var_bias_over_noise(regressors,x0, name_regression, n_samples)
    #print 2
    #plot_var_bias_over_irrelevant_variables(regressors, x0, name_regression, n_samples)
    '''
    n_fit = 50
    noise = 1
    x0 = -9.0
    nb_irrelevant = 2
    
    y_estimate = np.zeros((n_fit,len(regressors)))
    bias_squared = np.zeros(len(regressors))
    variance = np.zeros(len(regressors))
    if nb_irrelevant != 0:
        x0_extended = np.ones(nb_irrelevant+1)*x0

    
    for n in range(n_fit):
        if nb_irrelevant == 0:
            x = np.zeros((n_samples))
            y = np.zeros((n_samples))
            for i in range (n_samples):
                x[i] = random_state.uniform(low=-9.0, high=9.0)
                y[i] = make_data(x[i],1,noise)
            x = np.array([x]).transpose()
            for j in range(len(regressors)):
                regressors[j].fit(x,y)
                y_estimate[n,j] = regressors[j].predict(x0)
        else:
            X = np.zeros((n_samples,nb_irrelevant+1))
            y = np.zeros((n_samples,nb_irrelevant+1))
            for i in range(n_samples):
                for r in range(nb_irrelevant+1):
                    X[i,r] = random_state.uniform(low=-9.0, high=9.0)
                    y[i] = make_data(X[i,0],1,noise)
            for j in range(len(regressors)):
                regressors[j].fit(X,y)
                print regressors[j].predict(x0_extended)
                #y_estimate[n,j] = regressors[j].predict(x0_extended)
      
    #for i in range(len(regressors)):    
    #    bias_squared[i] = (bayes_model(x0)- np.mean(y_estimate[:,i]))**2
    #    variance[i] = np.var(y_estimate[:,i])

    
    '''
          
        
        
        
        
        
        
        
        
        
        
        
    
