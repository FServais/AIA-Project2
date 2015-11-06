# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 14:07:50 2015

@author: Sacha_2407
"""

import numpy as np
from sklearn.utils import check_random_state

n_samples = 2000
random_state=None
#def make_data(n_samples):
random_state = check_random_state(random_state) 
x = np.zeros((n_samples,1))
noise = np.zeros((n_samples,1))
y = np.zeros((n_samples,1))



for i in range(n_samples):
       
    x[i] = random_state.uniform(low=-9.0, high=9.0)
    noise[i] = random_state.normal(0,1)
    y[i] = (np.sin(x[i])+ np.cos(x[i]))*x[i]**2 + noise[i]