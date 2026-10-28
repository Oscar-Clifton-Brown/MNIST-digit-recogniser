from network import NeuralNetwork
from data import load_mnist

X_train, y_train, X_test, y_test = load_mnist()
Network = NeuralNetwork([784, 128, 64, 10], learning_rate=0.05)
Network.fit(X_train, y_train, epochs=10, X_val=X_test, y_val=y_test)

loss, acc = Network.evaluate(X_test, y_test)
print(f"loss = {loss:.4f}, accuracy = {acc:.2f}%")

Network.save("Saved_paramaters/784_128_64_10.npz")