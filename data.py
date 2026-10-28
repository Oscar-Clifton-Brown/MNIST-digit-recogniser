from keras.datasets import mnist


def load_mnist():
    """loads and returns mangled MNIST labled digit data set"""
    (X_train, y_train), (X_test, y_test) = mnist.load_data() #load dataset from keras database

    X_train = X_train.reshape(60000, -1).T / 255
    X_test = X_test.reshape(10000, -1).T / 255

    return X_train, y_train, X_test, y_test