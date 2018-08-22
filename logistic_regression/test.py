#Code for implementing regression
#Imports RGB pics for binary classification
#prints num of misclassifications


#from getdata import import_some_data
from regression import*
X_train,Y_train,X_test,Y_test = import_some_data()
    
X_flat = X_train.reshape(X_train.shape[0],-1).T /255
X_test_flat = X_test.reshape(X_test.shape[0],-1).T/255

m = X_flat.shape[0]
w,b = init_param(m)
w_test,dw,b_test,db,cost = optimize(w,b,X_flat,Y_train,iteration = 1000,
                                        learning_rate = 0.1)
Y_pred = predict(w_test,b_test,X_test_flat)
print("Number of misclassified images = " +str(np.sum(Y_pred == Y_test)))
