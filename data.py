from keras.datasets import mnist
import numpy as np


def load_mnist(input_preprocessing = "uniform"):
    """loads and returns mangled MNIST labled digit data set"""
    (X_train, y_train), (X_test, y_test) = mnist.load_data() #load dataset from keras database

    if input_preprocessing not in ("uniform", "standardised"):
        raise ValueError("input_preprocessing must be either 'uniform' or 'standardised'")
    if input_preprocessing == "uniform":
        X_train = X_train.reshape(60000, -1).T / 255
        X_test = X_test.reshape(10000, -1).T / 255
    elif input_preprocessing == "standardised":
        X_train = X_train.reshape(60000, -1).T / 255
        X_test = X_test.reshape(10000, -1).T / 255

        sample_mean = np.mean(X_train, axis=1, keepdims=True)
        sample_std = np.std(X_train, axis=1, keepdims=True)
        X_train = (X_train - sample_mean) / (sample_std + 1e-8)
        X_test = (X_test - sample_mean) / (sample_std + 1e-8)

    return X_train, y_train, X_test, y_test