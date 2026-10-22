import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from keras.datasets import mnist #import the class contianing the data set


# (train_X, train_y), (test_X, test_y) = mnist.load_data() #load dataset from keras database

def write(data, file_name = ""):
    """writes to the local 'Dataset' folder"""
    np.savetxt(f"Dataset/{file_name}.csv", data, delimiter=",")

def read(file_name = ""):
    """reads from the local 'Dataset' folder"""
    return  np.loadtxt(f"Dataset/{file_name}.csv", delimiter=",")    


train_X = read("train_X")
train_y = read("train_y")
test_X = read("test_X")
test_y = read("test_y")

def shuffle_dataset(D):
    np.random.shuffle(D)

#compose all into dataset, D and shuffle

D_X = np.vstack((train_X, test_X))
D_y = np.hstack((train_y, test_y)).reshape(-1,1)

D = np.hstack((D_y, D_X))
shuffle_dataset(D)

#now re-sepperate and manipulate to correct form:
m , n  = D.shape
train_size = round(0.9 * m)
D_train = D[0:train_size].T
y_train = D_train[0].astype(np.int8)
X_train = D_train[1:n] /255

D_test = D[train_size:m].T
y_test = D_test[0].astype(np.int8)
X_test = D_test[1:n] /255

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    """returns |Z|"""
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return 100* (np.sum(predictions == Y) / Y.size)

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 500 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(X_train, y_train, 0.10, 1000)


def test(params, X_test, y_test):
    W1, b1, W2, b2 = params
    nodes = forward_prop(W1, b1, W2, b2, X_test)[-1]
    y_hat = get_predictions(nodes)
    print(f"final accuracy on test set: {get_accuracy(y_hat, y_test)}")

test((W1, b1, W2, b2 ), X_test, y_test)
