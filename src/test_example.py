# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 17:51:03 2020

@author: saksh

driver function to test nn_funcs library
Classification done on data, can be used with regression prblems too (adjust activation function accordingly)
"""
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

import nn_funcs as nn

# number of samples in the data set
N_SAMPLES = 1000
# fraction of total observations used in test_set
TEST_SIZE = 0.1

"""
Model architecture
"""
nn_arch = [
    {"input_dim": 2, "output": 25, "activation": "relu"},
    {"input_dim": 25, "output": 50, "activation": "relu"},
    {"input_dim": 50, "output": 50, "activation": "relu"},
    {"input_dim": 50, "output": 25, "activation": "relu"},
    {"input_dim": 25, "output": 1, "activation": "sigmoid"}
]

"""
Making training and test datasets
"""
X, y = make_moons(n_samples = N_SAMPLES, noise=0.2, random_state=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

"""
Estimating model parameters on the training set
"""
#np.seterr(divide='ignore', invalid = 'ignore')
params_cache = nn.train_nn(nn_arch, np.transpose(X_train), np.transpose(y_train.reshape((y_train.shape[0], 1))), 0.02, 3000)

"""
Estimating target values for test input using estimated model parameters
"""
output, _ = nn.full_forward_prop(np.transpose(X_test), params_cache, nn_arch)

"""
Target value accuracy when compared to true output
"""
accuracy = nn.nn_accuracy(output, np.transpose(y_test.reshape((y_test.shape[0], 1))))
print(accuracy)