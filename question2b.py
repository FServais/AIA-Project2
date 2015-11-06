# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 14:07:50 2015

"""

import numpy as np
from utils import get_random_state

random_state = get_random_state()


def var(l):
    return np.var(l)


def make_data(x, n_samples=1):

    y = np.zeros(n_samples)

    for i in range(n_samples):
        eps = random_state.normal(0, 1)
        y[i] = (np.sin(x) + np.cos(x)) * x**2 + eps

    return y


def bayes_model():

    x = np.zeros((n_samples,1))
    bayes = np.zeros((n_samples,1))    
    
    for i in range(n_samples):
        x[i] = random_state.uniform(low=-9.0, high=9.0)
        bayes[i] = (np.sin(x[i])+ np.cos(x[i]))*x[i]**2
        
    return bayes


def residual_error(x0, n_samples):
    y = make_data(x0, 2000)
    return var(y)

if __name__ == "__main__":

    x0 = np.linspace(-9.0, 9.0, 90)
    n_samples = 2000

    y = make_data(n_samples)

    res_errors = np.zeros(len(x0))
    for i in range(len(x0)):
        res_errors[i] = residual_error(x0[i], n_samples)

    print(np.mean(res_errors))

