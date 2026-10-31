This project implements a fully connected feed-forward neural network for supervised classification. The model takes a feature vector as input, applies a sequence of linear transformations and nonlinear activations, and produces a probability distribution over class labels. In the case of MNIST, each input image is flattened into a vector and processed through a series of hidden layers before the final output layer assigns probabilities to the ten digit classes. The network is defined by a set of weight matrices and bias vectors, where each layer computes z = W·a + b and then applies a nonlinearity. Hidden layers use the ReLU activation function, which retains positive values and suppresses negative ones, while the final layer uses softmax so that the outputs form a valid probability distribution. This architecture is standard for multi-class classification problems where each sample belongs to exactly one class.

Training is performed by minimising cross-entropy loss, which measures the difference between the predicted class probabilities and the true labels. The labels are encoded using one-hot vectors, allowing the error signal to be computed efficiently and propagated backwards through the network. Backpropagation applies the chain rule to compute gradients with respect to each weight and bias, and the parameters are updated using mini-batch gradient descent scaled by some learning rate paramater. In practice, the training loop shuffles the dataset, splits it into small batches, performs forward and backward passes on each batch, and updates the network parameters iteratively over multiple epochs. This allows the network to learn increasingly useful feature representations from the training data.

Training also incorperates the posibilitly of using a learning rate schedle, some of which are already incorperated into NeuralNetwork as static methods.

The scope of this network is broad within the domain of structured vector inputs and standard classification tasks. Because the architecture is defined by layer sizes rather than a fixed number of neurons, it can be adapted to different input dimensions and hidden-layer widths. It supports training, prediction, probability estimation, accuracy measurement, and saving/loading model parameters. However, it has clear limitations. It is a dense, fully connected network rather than a convolutional or transformer-based model, so it does not exploit spatial structure in image data in the way modern vision architectures do. It also uses a fixed set of mathematical choices: ReLU hidden activations, softmax output, cross-entropy loss, and mini-batch gradient descent without advanced optimisers or regularisation. As a result, while the implementation is mathematically sound and educationally clear, it is relatively limited in complexity, optimisation capability, and generality compared with production-grade deep learning systems.


## Mathematical Description

### Notation

| Symbol | Meaning |
|---|---|
| $L$ | Number of layers (excluding the input) |
| $n_\ell$ | Number of neurons in layer $\ell$, with $n_0$ the input dimension and $n_L = K$ the number of classes |
| $\mathbf{x} \in \mathbb{R}^{n_0}$ | Input feature vector (for MNIST, a flattened image, so $n_0 = 784$) |
| $\mathbf{y} \in \{0,1\}^K$ | One-hot encoded true label |
| $W^{(\ell)} \in \mathbb{R}^{n_\ell \times n_{\ell-1}}$ | Weight matrix of layer $\ell$ |
| $\mathbf{b}^{(\ell)} \in \mathbb{R}^{n_\ell}$ | Bias vector of layer $\ell$ |
| $\mathbf{z}^{(\ell)} \in \mathbb{R}^{n_\ell}$ | Pre-activation of layer $\ell$ |
| $\mathbf{a}^{(\ell)} \in \mathbb{R}^{n_\ell}$ | Activation of layer $\ell$, with $\mathbf{a}^{(0)} = \mathbf{x}$ |
| $\hat{\mathbf{y}} \in [0,1]^K$ | Predicted probability vector, $\hat{\mathbf{y}} = \mathbf{a}^{(L)}$ |
| $\eta$ | Learning rate |
| $B$ | Mini-batch size |
| $\odot$ | Elementwise (Hadamard) product |

### Forward Propagation

Each layer applies an affine transformation followed by a nonlinearity:

$$
\mathbf{z}^{(\ell)} = W^{(\ell)} \mathbf{a}^{(\ell-1)} + \mathbf{b}^{(\ell)}, \qquad \ell = 1, \dots, L.
$$

**Hidden layers** ($\ell = 1, \dots, L-1$) use the ReLU activation, applied elementwise:

$$
\mathbf{a}^{(\ell)} = \operatorname{ReLU}\!\left(\mathbf{z}^{(\ell)}\right), \qquad \operatorname{ReLU}(z) = \max(0, z).
$$

**The output layer** ($\ell = L$) uses the softmax function:

$$
\hat{y}_k = a^{(L)}_k = \operatorname{softmax}\!\left(\mathbf{z}^{(L)}\right)_k = \frac{e^{z^{(L)}_k}}{\sum_{j=1}^{K} e^{z^{(L)}_j}}, \qquad k = 1, \dots, K.
$$

This guarantees $\hat{y}_k > 0$ and $\sum_{k=1}^{K} \hat{y}_k = 1$, so $\hat{\mathbf{y}}$ is a valid probability distribution over the classes. In practice, softmax is computed in a numerically stable form by subtracting $\max_j z^{(L)}_j$ from each entry before exponentiating, which leaves the result unchanged.

### Loss Function

For a single sample with one-hot label $\mathbf{y}$, the cross-entropy loss is

$$
\mathcal{L}(\hat{\mathbf{y}}, \mathbf{y}) = -\sum_{k=1}^{K} y_k \log \hat{y}_k.
$$

Since $\mathbf{y}$ is one-hot with the true class $c$ (so $y_c = 1$), this reduces to $\mathcal{L} = -\log \hat{y}_c$.

For a mini-batch of $B$ samples $\{(\mathbf{x}_i, \mathbf{y}_i)\}_{i=1}^{B}$, the loss is the average

$$
J = \frac{1}{B} \sum_{i=1}^{B} \mathcal{L}(\hat{\mathbf{y}}_i, \mathbf{y}_i).
$$

### Backpropagation

Define the error signal of layer $\ell$ as $\boldsymbol{\delta}^{(\ell)} = \dfrac{\partial \mathcal{L}}{\partial \mathbf{z}^{(\ell)}}$.

**Output layer.** For softmax combined with cross-entropy, the chain rule simplifies to

$$
\boldsymbol{\delta}^{(L)} = \hat{\mathbf{y}} - \mathbf{y}.
$$

**Hidden layers.** The error is propagated backwards using

$$
\boldsymbol{\delta}^{(\ell)} = \left( W^{(\ell+1)} \right)^{\!\top} \boldsymbol{\delta}^{(\ell+1)} \odot \operatorname{ReLU}'\!\left(\mathbf{z}^{(\ell)}\right), \qquad \ell = L-1, \dots, 1,
$$

where the ReLU derivative is applied elementwise:

$$
\operatorname{ReLU}'(z) =
\begin{cases}
1 & z > 0, \\
0 & z \le 0.
\end{cases}
$$

**Parameter gradients.** For a single sample,

$$
\frac{\partial \mathcal{L}}{\partial W^{(\ell)}} = \boldsymbol{\delta}^{(\ell)} \left( \mathbf{a}^{(\ell-1)} \right)^{\!\top}, \qquad
\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(\ell)}} = \boldsymbol{\delta}^{(\ell)}.
$$

Averaging over a mini-batch gives

$$
\frac{\partial J}{\partial W^{(\ell)}} = \frac{1}{B} \sum_{i=1}^{B} \boldsymbol{\delta}_i^{(\ell)} \left( \mathbf{a}_i^{(\ell-1)} \right)^{\!\top}, \qquad
\frac{\partial J}{\partial \mathbf{b}^{(\ell)}} = \frac{1}{B} \sum_{i=1}^{B} \boldsymbol{\delta}_i^{(\ell)}.
$$

In matrix form, if the columns of $\Delta^{(\ell)} \in \mathbb{R}^{n_\ell \times B}$ and $A^{(\ell-1)} \in \mathbb{R}^{n_{\ell-1} \times B}$ hold the per-sample quantities, then
$\dfrac{\partial J}{\partial W^{(\ell)}} = \dfrac{1}{B}\, \Delta^{(\ell)} \left( A^{(\ell-1)} \right)^{\!\top}$.

### Mini-Batch Gradient Descent

After each forward and backward pass on a mini-batch, the parameters are updated by

$$
W^{(\ell)} \leftarrow W^{(\ell)} - \eta \, \frac{\partial J}{\partial W^{(\ell)}}, \qquad
\mathbf{b}^{(\ell)} \leftarrow \mathbf{b}^{(\ell)} - \eta \, \frac{\partial J}{\partial \mathbf{b}^{(\ell)}}.
$$

One **epoch** consists of shuffling the training set, partitioning it into $\lceil N / B \rceil$ mini-batches (where $N$ is the number of training samples), and applying the update above once per mini-batch. Training repeats this for a fixed number of epochs.

### Prediction and Evaluation

The predicted class for an input $\mathbf{x}$ is the class with the highest probability:

$$
\hat{c}(\mathbf{x}) = \underset{k \in \{1, \dots, K\}}{\arg\max} \; \hat{y}_k.
$$

Accuracy over a dataset of $N$ samples with true classes $c_i$ is

$$
\text{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\!\left[ \hat{c}(\mathbf{x}_i) = c_i \right],
$$

where $\mathbf{1}[\cdot]$ is the indicator function, equal to $1$ if the condition holds and $0$ otherwise.

### Weight Initialization

`NeuralNetwork` accepts `weight_init="norm"` or `weight_init="unif"` (default: `"norm"`). These select He normal initialization with standard deviation $\sqrt{2/n_{\text{in}}}$ or He uniform initialization over $[-\sqrt{6/n_{\text{in}}}, \sqrt{6/n_{\text{in}}}]$ for hidden ReLU layers. Biases start at zero. The softmax output layer uses Glorot uniform initialization. The selected option is stored when saving a model and restored when loading it.

```python
Network = NeuralNetwork([784, 128, 64, 10], learning_rate=0.05, weight_init="unif")
```
