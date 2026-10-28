import numpy as np


#static functions:


def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    Y = Y.astype(int)  # Ensure Y is of integer type
    one_hot_Y = np.zeros((Y.size, 10))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

#network class:

class NeuralNetwork:
    def _init_params(self, layer_sizes):
        return [
            (
                np.random.rand(layer_sizes[i + 1], layer_sizes[i]) - 0.5,
                np.random.rand(layer_sizes[i + 1], 1) - 0.5,
            )
            for i in range(len(layer_sizes) - 1)
        ]

    def __init__(self, layer_sizes, learning_rate=0.1):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.params = self._init_params(layer_sizes)

    #training:

    def forward_prop(self, X):
        """return output layer activations and store all node states for backpropagation"""
        Node_states = [(np.array([]), X)]
        for i in range(0, len(self.params)):
            W, b = self.params[i]
            Z = W.dot(Node_states[-1][1]) + b
            Node_states.append((Z, ReLU(Z) if i < len(self.params) - 1 else softmax(Z)))
            self.Node_states = Node_states
        return Node_states[-1][1]

    def backward_prop(self, X, Y):
        """computes gradients and stores them in self.grads"""
        grads = []
        for i in range(len(self.params) - 1, -1, -1):
            W, b = self.params[i]
            Z, A = self.Node_states[i + 1]
            if i == len(self.params) - 1:
                dZ = A - one_hot(Y)
            else:
                dZ = self.params[i + 1][0].T.dot(dZ) * ReLU_deriv(Z)
            dW = 1 / X.shape[1] * dZ.dot(self.Node_states[i][1].T)
            db = 1 / X.shape[1] * np.sum(dZ, axis=1, keepdims=True)
            grads.append((dW, db))
        self.grads = grads[::-1]
    
    def update(self):
        """Updates the parameters using the computed gradients and the learning rate."""
        for i in range(len(self.params)):
            W, b = self.params[i]
            dW, db = self.grads[i]
            self.params[i] = (W - self.learning_rate * dW, b - self.learning_rate * db) 

    def predict(self, X):
        """Predicts the classes for the given input data."""
        predictions = self.forward_prop(X)
        return np.argmax(predictions, axis=0)
    
    def accuracy(self, X, y):
        """Computes the accuracy of the model on the given dataset."""
        predicted_classes = self.predict(X)
        return 100* (np.sum(predicted_classes == y) / y.size)
    
    def fit(self, X, y, epochs=10, batch_size=64, X_val=None, y_val=None):
        """Trains the neural network using mini-batch gradient descent."""
        m = X.shape[1]
        for epoch in range(epochs):
            permutation = np.random.permutation(m)
            X_shuffled = X[:, permutation]
            y_shuffled = y[permutation]
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[:, i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                self.forward_prop(X_batch)
                self.backward_prop(X_batch, y_batch)
                self.update()
            if X_val is not None and y_val is not None:
                val_accuracy = self.accuracy(X_val, y_val)
                print(f"Epoch {epoch + 1}/{epochs}, Validation Accuracy: {val_accuracy:.2f}%")


    def predict_proba(self, X):
        """returns the relative propbability of X"""
        predictions = self.forward_prop(X)
        return np.max(predictions, axis=0)

    def evaluate(self, X, y):
        """Returns (cross-entropy loss, accuracy in %) on the given dataset."""
        A = self.forward_prop(X)            
        y = y.astype(int)
        n = y.size
        loss = -np.mean(np.log(A[y, np.arange(n)] + 1e-12))
        accuracy = 100 * np.mean(np.argmax(A, axis=0) == y)
        return loss, accuracy

    def save(self, path):
        """Saves the architecture, learning rate and parameters to a .npz file."""
        arrays = {}
        for i, (W, b) in enumerate(self.params):
            arrays[f"W{i}"] = W
            arrays[f"b{i}"] = b
        np.savez(
            path,
            layer_sizes=np.array(self.layer_sizes),
            learning_rate=self.learning_rate,
            **arrays,
        )

    @classmethod
    def load(cls, path):
        """Rebuilds a network from a file written by save()."""
        data = np.load(path)
        layer_sizes = data["layer_sizes"].tolist()
        net = cls(layer_sizes, learning_rate=float(data["learning_rate"]))
        net.params = [
            (data[f"W{i}"], data[f"b{i}"])
            for i in range(len(layer_sizes) - 1)
        ]
        return net

