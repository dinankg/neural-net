"""
---
Starting with basic code for Deep Learning
This file will build some function to implement logistic regression.
---

--- 
Logistic Regression functional code (Binary classification)
---

"""

#importing packages
import numpy as np
#import matplotlib.pyplot as plt

#from getdata import import_data,import_some_data  #Data from my computer

def sigmoid(x):
    #Finds element wise sigmoid activation of input
    sig = 1/ (1+np.exp(-1*x))
    return sig

def forward_prop(X,Y,w,b):
    #Forward Propagation for Logistic Regression
    #X - n by m matrix with m samples
    # A is the activations that are found by applying
    # sigmoid activation
    # Using Cross entropy loss 
    
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X)+b)
    cost = -1/m * np.sum(np.log(A)*Y + np.log(1-A)*(1-Y))
    return A,cost

def back_prop(A,cost,X,Y):
    m = X.shape[1]
    dw = 1/m * np.dot(X,(A-Y).T)
    db = 1/m * np.sum((A-Y))
    return dw,db

def optimize(w,b,X,Y,iteration,learning_rate = 0.01):
   #Optimizing using gradient descent
    
    costs = []
    for i in range(iteration):
        A,cost = forward_prop(X,Y,w,b)
        dw,db = back_prop(A,cost,X,Y)
        #print(max(A))
        w -= dw*learning_rate
        b -= db*learning_rate
        if i %100 ==0:
            costs.append(cost)
            print("Cost after " +str(i) + 
                  "th iteration:" +str(cost))
    return w,dw,b,db,costs


def predict(w,b,X):
    m = X.shape[1]
    Y_pred = np.zeros([1,m])
    w = w.reshape(X.shape[0],1)
    A = sigmoid(np.dot(w.T,X)+b)
    Y_pred = np.squeeze(np.round(A))
    return Y_pred

def init_param(m):
    #initialize w,b 
    w = np.random.randn(m,1)*0.001
    b = 0
    return w,b

'''Implemented as a separate function
def testing():
    
    #importing dataset
    #Cat vs dog classification for Kaggle competition
    X_train,Y_train,X_test,Y_test = import_some_data()
    
    X_flat = X_train.reshape(X_train.shape[0],-1).T /255
    X_test_flat = X_test.reshape(X_test.shape[0],-1).T/255
    
    m = X_flat.shape[0]
    w,b = init_param(m)
    A,cost= forward_prop(X=X_flat,Y=Y_train,w=w,b=b)
    dw,db = back_prop(A,cost,X_flat,Y_train)
    w_test,dw,b_test,db,cost = optimize(w,b,X_flat,Y_train,iteration = 1000,learning_rate = 0.1)
    
    Y_pred = predict(w_test,b_test,X_test_flat)
    np.sum(Y_pred == Y_test)
'''

    
    
    
    
    