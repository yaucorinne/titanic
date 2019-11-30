#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 21:23:00 2019

@author: luischavesrodriguez
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#original datasets, do not modify
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

#replace_map should contain the same heading name as the data frame for it to 
# do anything
replace_map = {'Embarked':{'S': 1, 'C': 2,'Q':3},
               'Sex':{'male':0,'female':1}}

traincopy = train.copy()
testcopy = test.copy()

traincopy = traincopy.dropna()
traincopy.replace(replace_map,inplace = True)

Xtrain = traincopy.loc[:,['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
ytrain = traincopy.loc[:,'Survived']



def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoidGradient(z):
    return sigmoid(z)*(1-sigmoid(z))


# we for now will only be using 7 features to predict the survival status
input_layer_size = 7
hidden_layer_size = 10
num_labels = 2
m = np.size(Xtrain,axis = 0)

#random initialisation of NN weights
theta1 = np.random.rand(hidden_layer_size,input_layer_size+1)
theta2 = np.random.rand(num_labels, hidden_layer_size+1)

nnparams = np.concatenate((theta1.flatten(),theta2.flatten()), axis = 0)

reg = 0

def NNCostFunction(nnparams, input_layer_size, hidden_layer_size, num_labels,
                   X,y, reg):
    theta1 = np.reshape(nnparams[0:hidden_layer_size*(input_layer_size+1)],
                                 (hidden_layer_size,input_layer_size+1))
    theta2  = np.reshape(nnparams[1+hidden_layer_size*(input_layer_size+1):],
                                 (hidden_layer_size,input_layer_size+1))
    m = np.size(X,axis = 0)
    theta1grad = np.zeros(np.size(theta1))
    theta2grad = np.zeros(np.size(theta2))
    J = 0
    
    #FeedForward
    X = np.concatenate(np.ones(m,1),X, axis = 1)
    z2 = theta1*np.transpose(X);
    z2 = np.transpose(z2);
    # take sigmoid of z(2) to make activity level of hidden layer 1 aka layer 2
    a2 = sigmoid(z2);
    #add bias to hidden layer
    a2 = np.oncatenate((np.ones(np.size(a2,1), 1),a2), axis = 1);

    #calculate
    z3 = theta2*np.transpose(a2);
    a3 = sigmoid(z3);

    a3 = np.transpose(a3);
    
    all_combos = np.eye(num_labels)
    ylarge = all_combos[y,:]
    
    J = (1/m)*np.sum(np.sum(-ylarge*np.log(a3)-(1-ylarge)*np.log(1-a3)));
    



