'''
Functions to implement a N layer Neural Network
X - n by m where n is the feature size and m is sample size
Layers can be ReLu, sigmoid or tanh with output layer being sigmoid

'''
import numpy as np

def init_params(layer_dims):
    #Outputs the initialization for each layer with random init
    params = {}
    for l in range(1,len(layer_dims)):
        params["W"+str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1]) * 0.01
        params["b"+str(l)] = np.zeros((layer_dims[l],1))
    return params

def linear_forward(A,W,b):
    Z = np.dot(W,A) + b
    cache = (A,W,b)
    return Z,cache


#Forward Activation Functions
def sigmoid(Z):
    A = 1/(1+np.exp(-1*Z))
    cache = Z
    return A,cache

def relu(Z):
    A = np.maximum(0,Z)
    assert(A.shape == Z.shape)
    cache = Z
    return A,cache

def tanh(Z):
    A = np.tanh(Z)
    assert(A.shape == Z.shape)
    cache = Z
    return A,cache
#End of Forward Activations
   
    
def linear_activ_forward(A_prev,W,b,activation):
    
    Z,linear_cache = linear_forward(A_prev,W,b)
    if activation == "sigmoid":
        A,activ_cache = sigmoid(Z)
    elif activation == "relu":
        A,activ_cache == relu(Z)
    elif activation == "tanh":
        A,activ_cache = tanh(Z)
    cache = (linear_cache,activ_cache)
    return cache

def forward_prop(X,params,activations):
    '''
    X : input matrix.
    params: W,b for each layer of the network.
    activations: list of activation function names for each layer.
    '''
    L = len(params)//2
    assert(L == (len(activations)+1))#No. of activations+1 = no. of layers
    assert(activations[L] == "sigmoid") #Ensure sigmoid activation #Remove if not.
    A = X
    caches=[]
    
    
    for l in range(1,L):
        A_prev = A
        A,cache = linear_activ_forward(A_prev,params['W' + str(l)],params['b' + str(l)],activations[l-1])
        caches.append(cache)
        
    assert(A.shape == (1,X.shape[1]))  
    return(A,caches)
   
def compute_cost(A,Y):
    #Cross Entropy Loss
    m = Y.shape[1]
    cost = 1/m * (np.dot(Y,np.log(A).T) + np.dot(1-Y,np.log(1-A).T))
    cost = np.squeeze(cost)
    return cost

def linear_back(dZ,cache):
    #cache is output of linear_forwward
    A_prev,W,b = cache
    m = A_prev.shape[1]
    dW = 1/m *np.dot(dZ,A_prev.T)
    db = 1/m * np.sum(dZ,keepdims = True)
    dA_prev = np.dot(W.T,dZ)
    return dW,db,dA_prev

#Backprop for activation functions:

def relu_back(dA,cache):
    
    Z = cache
    dZ = np.array(dA,copy = True)
    dZ[Z<=0] = 0
    
    assert(dZ.shape == Z.shape)
    return dZ

def sigmoid_back(dA,cache):
    
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA*s*(1-s)
    
    assert (dZ.shape == Z.shape)
    return dZ

def tanh_back(dA,cache):
    
    Z = cache
    t = np.tanh(Z)
    dZ = dA * (1-t**2)
    
    assert (dZ.shape == Z.shape)
    return dZ

def linear_activation_backward(dA, cache, activation):
    
    
    linear_cache,activation_cache = cache
    
    if activation == "relu":
        dZ = relu_back(dA, activation_cache)
        dA_prev, dW, db = linear_back(dZ,linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_back(dA,activation_cache)
        dA_prev, dW, db = linear_back(dZ,linear_cache)
    elif activation == "tanh":
        dZ = tanh_back(dA,activation_cache)
        dA_prev, dW, db = linear_back(dZ,linear_cache)
    
    return dA_prev,dW,db


def back_prop(A,Y,caches,activations):
    
    grads = {}
    L = len(caches) # the number of layers
    m = A.shape[1]
    Y = Y.reshape(A.shape)
    dA = -(np.divide(Y,A) - np.divide(1-Y,1-A)) #backprop from loss function
    grads["dA"+str(L)] = dA
    for l in reversed(range(L)):
        
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, activations[l])
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        
    return grads


def update_params(params,grads,learning_rate = 0.1):
    L = len(params)//2
    for l in range(L):
        params["W" + str(l+1)] -= grads["dW" + str(l + 1)]*learning_rate
        params["b" + str(l+1)] -= grads["db" + str(l + 1)]*learning_rate
    return params

def predict(X,params,activations):
    
    A,cache = forward_prop(X,params,activations)
    p=np.zeros((1,X.shape[1]))
    
    for i in range(0, A.shape[1]):
        if A[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    return p

    
        

    
    
    
