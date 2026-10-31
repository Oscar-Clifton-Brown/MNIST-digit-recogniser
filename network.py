import numpy as np
import matplotlib.pyplot as plt


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
    one_hot_Y = np.zeros((Y.size, 10), dtype=np.float32)
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


#network class:

class NeuralNetwork:
    def _init_params(self, layer_sizes, weight_init):
        params = []
        output_layer = len(layer_sizes) - 2
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            if i == output_layer:
                limit = np.sqrt(6 / (fan_in + fan_out))
                weights = np.random.uniform(
                    -limit, limit, size=(fan_out, fan_in)
                ).astype(np.float32)
            elif weight_init == "norm":
                weights = np.random.normal(
                    0, np.sqrt(2 / fan_in), size=(fan_out, fan_in)
                ).astype(np.float32)
            elif weight_init == "unif":
                limit = np.sqrt(6 / fan_in)
                weights = np.random.uniform(
                    -limit, limit, size=(fan_out, fan_in)
                ).astype(np.float32)
            else:
                raise ValueError("weight_init must be 'norm' or 'unif'")
            biases = np.zeros((fan_out, 1), dtype=np.float32)
            params.append((weights, biases))
        return params

    def __init__(self, layer_sizes, learning_rate=0.1, weight_init="norm"):
        """Create a network using He normal or uniform hidden-layer weights."""
        if weight_init not in ("norm", "unif"):
            raise ValueError("weight_init must be 'norm' or 'unif'")
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weight_init = weight_init
        self.params = self._init_params(layer_sizes, weight_init)
        self.history = {}
        self.learning_rate_history = {}

    #training:

    def forward_prop(self, X):
        """return output layer activations and store all node states for backpropagation"""
        X = np.asarray(X, dtype=np.float32)
        Node_states = [(np.array([], dtype=np.float32), X)]
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
    
    def update(self, learning_rate):
        """Updates the parameters using the computed gradients and learning rate."""
        for i in range(len(self.params)):
            W, b = self.params[i]
            dW, db = self.grads[i]
            self.params[i] = (W - learning_rate * dW, b - learning_rate * db)
    
    def fit(self, X, y, epochs=10, batch_size=64, schedule=None, X_val=None, y_val=None):
        """Train with mini-batch gradient descent and an optional step-based schedule."""
        if schedule is not None and not callable(schedule):
            raise TypeError("schedule must be callable or None")

        X = np.asarray(X, dtype=np.float32)
        m = X.shape[1]
        step = 0
        for epoch in range(epochs):
            permutation = np.random.permutation(m)
            epoch_learning_rate_sum = 0.0
            batches = 0
            for i in range(0, m, batch_size):
                idx = permutation[i:i + batch_size]
                X_batch = np.ascontiguousarray(X[:, idx])
                y_batch = y[idx]
                self.forward_prop(X_batch)
                self.backward_prop(X_batch, y_batch)
                learning_rate = self.learning_rate if schedule is None else schedule(step)
                if (
                    isinstance(learning_rate, (bool, np.bool_))
                    or not isinstance(learning_rate, (int, float, np.integer, np.floating))
                    or not np.isfinite(learning_rate)
                    or learning_rate < 0
                ):
                    raise ValueError("schedule must return a finite, non-negative scalar learning rate")
                self.update(float(learning_rate))
                epoch_learning_rate_sum += learning_rate
                batches += 1
                step += 1
            if batches:
                self.learning_rate_history[epoch + 1] = epoch_learning_rate_sum / batches
            if X_val is not None and y_val is not None:
                loss, accuracy = self.evaluate(X_val, y_val)
                self.history[epoch + 1] = (loss, accuracy)
                print(f"Epoch {epoch + 1}/{epochs} loss = {loss:.4f}, accuracy = {accuracy:.2f}%")


    def predict_proba(self, X):
        """returns the relative propbability of X"""
        predictions = self.forward_prop(X)
        return np.max(predictions, axis=0)

    def predict(self, X):
        """Predicts the classes for the given input data."""
        predictions = self.forward_prop(X)
        return np.argmax(predictions, axis=0)
    
    def accuracy(self, X, y):
        """Computes the accuracy of the model on the given dataset."""
        predicted_classes = self.predict(X)
        return 100* (np.sum(predicted_classes == y) / y.size)

    def evaluate(self, X, y):
        """Returns (cross-entropy loss, accuracy in %) on the given dataset."""
        A = self.forward_prop(X)            
        y = y.astype(int)
        n = y.size
        loss = -np.mean(np.log(A[y, np.arange(n)] + 1e-12))
        accuracy = 100 * np.mean(np.argmax(A, axis=0) == y)
        return loss, accuracy

    def plot_learn(
        self,
        title="Metrics over Epochs",
        plot_loss=True,
        plot_accuracy=True,
        plot_learning_rate=False,
    ):
        """Plot up to two validation metrics and/or epoch-average learning rate."""
        selected = [
            ("loss", plot_loss),
            ("accuracy", plot_accuracy),
            ("learning rate", plot_learning_rate),
        ]
        selected = [name for name, enabled in selected if enabled]
        if len(selected) > 2:
            raise ValueError("At most two of loss, accuracy, and learning rate can be plotted")
        if not selected:
            raise ValueError("Select at least one metric to plot")
        if any(name in ("loss", "accuracy") for name in selected) and not self.history:
            raise ValueError("No validation history available; train with validation data first")
        if "learning rate" in selected and not self.learning_rate_history:
            raise ValueError("No learning-rate history available; train the model first")

        fig, ax1 = plt.subplots()
        axes = [ax1] if len(selected) == 1 else [ax1, ax1.twinx()]
        styles = {
            "loss": ("tab:red", "Test Loss", lambda epoch: self.history[epoch][0], self.history),
            "accuracy": (
                "tab:blue",
                "Test Accuracy (%)",
                lambda epoch: self.history[epoch][1],
                self.history,
            ),
            "learning rate": (
                "tab:green",
                "Learning Rate (epoch average)",
                lambda epoch: self.learning_rate_history[epoch],
                self.learning_rate_history,
            ),
        }
        for name, axis in zip(selected, axes):
            color, label, value_at_epoch, history = styles[name]
            epochs = list(history.keys())
            axis.set_xlabel("Epochs")
            axis.set_ylabel(label, color=color)
            axis.plot(epochs, [value_at_epoch(epoch) for epoch in epochs], color=color)
            axis.tick_params(axis="y", labelcolor=color)

        fig.tight_layout()
        plt.title(title)
        plt.show()

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
            weight_init=self.weight_init,
            **arrays,
        )

    @classmethod
    def load(cls, path):
        """Rebuilds a network from a file written by save()."""
        data = np.load(path)
        layer_sizes = data["layer_sizes"].tolist()
        weight_init = str(data["weight_init"]) if "weight_init" in data else "norm"
        net = cls(
            layer_sizes,
            learning_rate=float(data["learning_rate"]),
            weight_init=weight_init,
        )
        net.params = [
            (data[f"W{i}"], data[f"b{i}"])
            for i in range(len(layer_sizes) - 1)
        ]
        return net

    #some schedules

    @staticmethod
    def constant(lr0):
        return lambda t: lr0

    @staticmethod
    def step_decay(lr0, gamma=0.1, every=10_000):
        return lambda t: lr0 * gamma ** (t // every)

    @staticmethod
    def exp_decay(lr0, lam):
        return lambda t: lr0 * np.exp(-lam * t)

    @staticmethod
    def inverse_time(lr0, lam):
        return lambda t: lr0 / (1 + lam * t)

    @staticmethod
    def cosine(lr0, total, lr_min=0.0, warmup=0):
        def f(t):
            if t < warmup:
                return lr0 * (t + 1) / warmup
            p = (t - warmup) / max(1, total - warmup)
            return lr_min + 0.5 * (lr0 - lr_min) * (1 + np.cos(np.pi * min(p, 1.0)))
        return f

    @staticmethod
    def one_cycle(lr_max, total, pct_up=0.3, lr_start=None, lr_end=None):
        lr_start = lr_start if lr_start is not None else lr_max / 25
        lr_end = lr_end if lr_end is not None else lr_max / 1e4
        up = int(total * pct_up)
        def f(t):
            if t < up:
                return lr_start + (lr_max - lr_start) * t / up
            p = (t - up) / max(1, total - up)
            return lr_end + 0.5 * (lr_max - lr_end) * (1 + np.cos(np.pi * min(p, 1.0)))
        return f
