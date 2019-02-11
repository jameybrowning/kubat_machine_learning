# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 08:28:37 2017

@author: James Browning
Implements backpropogation of error for a predefined number of output and 
single layer of hidden neurons using fixed learning rate to perform digit handwritten digit
classification.
Based on An introduction to Machine Learning, Miroslav Kubat, 2nd edition table 5.2

"""
import numpy as np
# Load characters 1-9 dataset
from sklearn import datasets
digits = datasets.load_digits()
image = digits.images
target = digits.target
#define network
learn_rate = 0.1
n_char = 64
n_hid_neur = 64
n_out_neur = 10
#number of epochs
n_epochs = 10
#number of training sets
n_train = 1000
# initialize hidden neuron array
hid_neur = np.zeros(n_hid_neur)
# initialize output neuron array
out_neur = np.zeros(n_out_neur)
# initialize k_j weights (char/hidden weights) range [-1,1]
w_ch = np.random.uniform(low=-1, high=1, size=(n_hid_neur,n_char))
# initialize j_i weights (hidden/output weights) range [-1,1]
w_ho = np.random.uniform(low=-1, high=1, size=(n_out_neur,n_hid_neur))
for m in range(0,n_epochs):
    print("epoch ",m)
    for n in range (0,n_train):
        # initialize first 1x64 characteristic vector for first
        # image in dataset
        x = np.reshape(image[n, :,:], 64)
        
        #initialize target vector
        t = np.array([0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2])
        t[target[n]]=0.8
        # calculate hidden neuron value
        for i in range(0,n_hid_neur):
            neur_sum = 0    
            for j in range(0,n_char):
                neur_sum = neur_sum + x[j]*w_ch[i,j]
            hid_neur[i] = 1/(1+np.exp(-neur_sum))
        # calculate output neuron value
        for i in range(0,n_out_neur):
            neur_sum = 0    
            for j in range(0,n_hid_neur):
                neur_sum = neur_sum + hid_neur[j]*w_ho[i,j]
            out_neur[i] = 1/(1+np.exp(-neur_sum))
        #begin backpropogation of error
        #initialialize output and hidden meoron error arrays
        out_err = np.zeros(n_out_neur)
        hid_err = np.zeros(n_hid_neur)
        #calculate each output neuron's responsibility for the network's error
        for i in range(0,n_out_neur):
            out_err[i] = out_neur[i]*(1-out_neur[i])*(t[i]-out_neur[i])
        #calculate each hidden neuron's responsibility for the network's error
        for i in range(0,n_hid_neur):
            hid_err[i] = hid_neur[i]*(1-hid_neur[i])*np.sum(out_err*w_ho[:,i])
        #update hidden/output weights
        for i in range(0,n_out_neur):
            for j in range(0,n_hid_neur):
                w_ho[i,j]=w_ho[i,j]+learn_rate*out_err[i]*hid_neur[j]
        #update char/hidden weights
        for i in range(0,n_hid_neur):
            for j in range(0,n_char):
                w_ch[i,j]=w_ch[i,j]+learn_rate*hid_err[i]*x[j]
    
    
    #begin test
    test_num = 1008
    test_class = target[test_num]
    x = np.reshape(image[test_num, :,:], 64)
    for i in range(0,n_hid_neur):
        neur_sum = 0    
        for j in range(0,n_char):
            neur_sum = neur_sum + x[j]*w_ch[i,j]
        hid_neur[i] = 1/(1+np.exp(-neur_sum))
    # calculate output neuron value
    for i in range(0,n_out_neur):
        neur_sum = 0    
        for j in range(0,n_hid_neur):
            neur_sum = neur_sum + hid_neur[j]*w_ho[i,j]
        out_neur[i] = 1/(1+np.exp(-neur_sum))
    test_array = np.array([0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2])
    test_array[target[test_num]]=0.8
    test_error = np.sum((test_array-out_neur)*(test_array-out_neur))
    print("test class = ",test_class)
    print("output neuron array = ",out_neur)
    print("predicted = ", np.argmax(out_neur))
    #print("hidden neuron array = ",hid_neur)
    #print("ch weights = ",w_ho[:,8])
    print("test error = ",test_error)
    



