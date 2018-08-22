'''
Func to implement 2 Layer NN
1st Layer: tanh activation of user specified size 
2nd Layer: Output Sigmoid activation

Generate data code also generates sample data 

'''
import numpy as np



def generate_data(m):
    #Generate FLower looking dataset
    #Code idea from Coursera DL course
    #m is the number of samples
    N = int(m/2)
    D = 2
    X = np.zeros((m,D))
    Y = np.zeros((m,1))
    a = 4 
    
    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j
        
    X = X.T
    Y = Y.T
    return X,Y

def sigmoid(x):
    s = 1/(1 + np.exp(-1*x))
    return s

'''
2 Layer NN with
X: 2 dimensional input data
Y: Binary label
1st hidden layer: size 4 with tanh activation
ouput layer: sigmoid
cost func: cross entropy
'''
def init_param(X,Y,n_h):
    #initializing different layers' weight matrices and bias vectors
    np.random.seed(2)
    #Setting size of each layer
    n_x = X.shape[0]
    n_y = Y.shape[0]
    
    W1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros([n_h,1])
    W2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros([n_y,1])
    
    params = { "W1" : W1,"b1" : b1, "W2" : W2, "b2" : b2}
    return params


def forward_prop(X,params):
    #Forward Propagation with tanh and Sigmoid
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]
    
    Z1 = np.dot(W1,X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)

    cache = {"Z1":Z1, "A1":A1, "Z2":Z2, "A2":A2}
    return A2,cache

def compute_cost(A2,Y,params):
    #Cross-Entropy loss
    m = Y.shape[1]
    cost = -1/m * (np.dot(np.log(A2),Y.T) + np.dot(np.log(1-A2),1-Y.T))
    cost = np.squeeze(cost)
    return cost


def back_prop(params,cache,X,Y):
    #Back Propagating to find gradients
    m = X.shape[1]
    
    #W1 = params["W1"]
    W2 = params["W2"]
    #Z1 = cache["Z1"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    dZ2 = A2-Y
    dW2 = 1/m* np.dot(dZ2,A1.T)
    db2 = 1/m * np.sum(dZ2,axis = 1,keepdims = True)
    
    dZ1 = np.multiply(np.dot(W2.T,dZ2),(1-np.power(A1,2)))
    dW1 = 1/m * np.dot(dZ1,X.T)
    db1 = 1/m * np.sum(dZ1,axis = 1,keepdims = True)
    
    grads = {"dW1" : dW1,"db1" : db1, "dW2" : dW2, "db2" : db2}
    return grads

def update_params(params,grads,learning_rate = 1.2):
    #Updating weight and bias for gradient descent
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]
    
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    W1 -=dW1*learning_rate
    b1 -=db1*learning_rate
    W2 -=dW2*learning_rate
    b2 -=db2*learning_rate
    params = { "W1" : W1,"b1" : b1, "W2" : W2, "b2" : b2}
    
    return params

def nn_model(X,Y,n_h,iteration = 10000):
    #Implementing the model
    params = init_param(X=X,Y=Y,n_h=n_h)

    for i in range(0,iteration):

        A2,cache = forward_prop(X,params)
        cost = compute_cost(A2=A2,Y=Y,params=params)
        grads = back_prop(params=params,cache = cache,X=X,Y=Y)
        params = update_params(params=params,grads=grads,learning_rate = 0.9)
        
        if i%1000==0:
            print("Cost after iteration %i: %f"  %(i,cost))
            print(A2.shape)
    return params

def predict(params,X):
    #Prediction using Weights and an input matrix
    A2,cache = forward_prop(X=X,params=params)
    preds = np.round(A2)
    return preds




#X,Y = generate_data(400)
