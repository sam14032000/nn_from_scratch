# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 17:38:37 2020

@author: saksh

Functions for neural net

future scope: incorporate optimisation functions
"""

#init
#as promised, no imports other than numpy :)
import numpy as np

"""
Initiate model based on pre-specified architecture 
"""
def init_layers(nn_arch, seed = 83):
    param_cache = {}
    np.random.seed(seed)
    
    for layer_id,layer in enumerate(nn_arch):
        input_dim = layer['input_dim']
        output_dim = layer['output']
        
        param_cache["W" + str(layer_id)] = np.random.randn(output_dim,input_dim)*0.1 #normal(0,0.1) Low variance normal distribution, with 0 mean
        param_cache["B" + str(layer_id)] = np.random.randn(output_dim, 1)*0.1
    return param_cache

"""
Activation functions and first derivatives
"""

def sigmoid(X):
    return 1/(1+np.exp(-X))
    
def relu(X):
    return np.maximum(0,X)
    
def sigmoid_backward(dZ_curr, A_curr):
    return sigmoid(A_curr)*(1-sigmoid(A_curr))*dZ_curr
    
def relu_backward(dZ_curr, A_curr):
    X = np.copy(dZ_curr)
    X[A_curr <= 0] = 0
    return X

"""
Scoring and accuracy metrics
"""
def error_cost(output,target):
    m = target.shape[1]
    #print(m)
    if m != 0:
        x = -1/m * (np.dot(target,np.log(output).T) + np.dot((1-target),np.log(1-output).T))
    #print(output)
    return np.squeeze(x)
    
def nn_accuracy(output,target):
    op = prob_to_class(output)
    return (op == target).all(axis=0).mean()
    
def prob_to_class(output):
    op = np.copy(output)
    op[op>0.5] = 1
    op[op<=0.5] = 0
    return op

"""
Single step of forward propagation
"""
def single_forward_prop(W_curr, B_curr, Z_prev, activation):
    if activation == 'sigmoid':
        forward_activation = sigmoid
    elif activation == 'relu':
        forward_activation = relu
    A_curr = np.dot(W_curr, Z_prev) + B_curr
    
    return A_curr, forward_activation(A_curr)

"""
Complete forward propagation
"""
def full_forward_prop(nn_input, param_cache, nn_arch):
    memory_cache = {}
    A_curr = nn_input
    
    for layer_id, layer in enumerate(nn_arch):
        Z_prev = A_curr
        W_curr = param_cache["W" + str(layer_id)]
        B_curr = param_cache["B" + str(layer_id)]
        activation = layer['activation']
        A_curr, Z_curr = single_forward_prop(W_curr, B_curr, Z_prev, activation)
        
        memory_cache["A" + str(layer_id)] = A_curr
        memory_cache["Z" + str(layer_id)] = Z_prev
        
    return Z_curr, memory_cache

"""
Single step of backward propagation
"""
def single_back_prop(back_input, A_curr, W_curr, B_curr, Z_prev, activation):
    m = back_input.shape[1]
    
    if activation == 'sigmoid':
        backward_activation = sigmoid_backward
    elif activation == 'relu':
        backward_activation = relu_backward
    
    delta_A = backward_activation(back_input, A_curr)
    
    W_err = np.dot(delta_A, Z_prev.T)/m
    B_err = np.sum(delta_A, axis = 1, keepdims = True)/m
    back_input_ = np.dot(W_curr.T, delta_A)
    
    return W_err, B_err, back_input_

"""
Complete backward propagation
"""
def full_back_prop(param_cache, memory_cache, nn_arch, output, target):
    back_input = -1*(np.divide(target,output) - np.divide((1-target),(1-output)))
    
    for layer_id, layer in reversed(list(enumerate(nn_arch))):
        A_curr = memory_cache["A" + str(layer_id)]
        Z_prev = memory_cache["Z" + str(layer_id)]
        
        W_curr = param_cache["W" + str(layer_id)]
        B_curr = param_cache["B" + str(layer_id)]
        
        activation = layer['activation']
        
        W_err, B_err, back_input_ = single_back_prop(back_input, A_curr, W_curr, B_curr, Z_prev, activation)
        back_input = back_input_
        
        param_cache["dW" + str(layer_id)] = W_err
        param_cache["dB" + str(layer_id)] = B_err
        
    return param_cache

"""
Modifying parameters based on learning rate; end of one iteration
"""
def update_params(nn_arch, param_cache, learning_rate):
    #m = target.shape[1]
    for layer_id, layer in enumerate(nn_arch):
        param_cache["W" + str(layer_id)] -= learning_rate*param_cache["dW" + str(layer_id)]
        param_cache["B" + str(layer_id)] -= learning_rate*param_cache["dB" + str(layer_id)]
        
    return param_cache

"""
Main neural net training wrapper function
"""
def train_nn(nn_arch, X_train, y_train, learning_rate, epochs, score_logs = True):
    error_history = []
    accuracy_history = []
    param_cache = init_layers(nn_arch)
    
    for i in range(epochs):
        output, memory_cache = full_forward_prop(X_train, param_cache, nn_arch)
        param_cache = full_back_prop(param_cache, memory_cache, nn_arch, output, y_train)
        param_cache = update_params(nn_arch, param_cache, learning_rate)
        #output, memory_cache = full_forward_prop(X_train, param_cache, nn_arch)
        err = error_cost(output, y_train)
        accuracy = nn_accuracy(output, y_train)
        error_history.append(err)
        accuracy_history.append(accuracy)
        
        if (i%50 == 0):
            if(score_logs):
                print("Iteration: {0:05} - err: {1:.5f} - accuracy: {2:.5f}".format(i, err, accuracy))
    
    return param_cache