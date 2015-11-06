# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 14:07:50 2015

"""

import numpy as np
from utils import get_random_state

random_state = get_random_state()

def make_data(n_samples):

    x = np.zeros((n_samples,1))
    noise = np.zeros((n_samples,1))
    y = np.zeros((n_samples,1))
    
    bayes = bayes_model()
    
    for i in range(n_samples):
           
        x[i] = random_state.uniform(low=-9.0, high=9.0)
        noise[i] = random_state.normal(0,1)
        y[i] = bayes[i] + noise[i]
        
    return y,x,bayes


def bayes_model():

    x = np.zeros((n_samples,1))
    bayes = np.zeros((n_samples,1))    
    
    for i in range(n_samples):
        x[i] = random_state.uniform(low=-9.0, high=9.0)
        bayes[i] = (np.sin(x[i])+ np.cos(x[i]))*x[i]**2
        
    return bayes



if __name__ == "__main__":

    n_samples = 2000

    y,x,bayes = make_data(n_samples)

