import numpy as np
import copy
import math

np.random.seed(42)

class Layer(object):
    def set_input_shape(self, shape):
        self.input_shape = shape

    def layer_name(self):
        return self.__class__.__name__

    def parameters(self):
        return 0

    def forward_pass(self, X, training):
        raise NotImplementedError()

    def backward_pass(self, accum_grad):
        raise NotImplementedError()

    def output_shape(self):
        raise NotImplementedError()

class Dense(Layer):
    def __init__(self, n_units, input_shape=None):
        self.layer_input = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
        self.W = None
        self.w0 = None

    def initialize(self, optimizer):
        limit = 1 / math.sqrt(self.input_shape[0])
        self.W = np.random.uniform(-limit, limit, (self.input_shape[0], self.n_units))
        self.w0 = np.zeros((1, self.n_units))
        self.W_optimizer = copy.copy(optimizer)
        self.w0_optimizer = copy.copy(optimizer)

    def parameters(self):
        return np.prod(self.W.shape) + np.prod(self.w0.shape)

    def forward_pass(self, X, training=True):
        self.layer_input = X
        return X.dot(self.W) + self.w0

    def backward_pass(self, accum_grad):
        W = self.W
        if self.trainable:
            grad_w = self.layer_input.T.dot(accum_grad)
            grad_w0 = np.sum(accum_grad, axis=0, keepdims=True)
            self.W = self.W_optimizer.update(self.W, grad_w)
            self.w0 = self.w0_optimizer.update(self.w0, grad_w0)
        accum_grad = accum_grad.dot(W.T)
        return accum_grad

    def output_shape(self):
        return (self.n_units,)

dense_layer = Dense(n_units=3, input_shape=(2,))

class MockOptimizer:
    def update(self, weights, grad):
        return weights - 0.01 * grad
optimizer = MockOptimizer()

dense_layer.initialize(optimizer)

# The example output is wrong on the website
# And this is the correct
# Forward pass output: [[ 0.10162127 -0.33551992 -0.64490545]]
# Backward pass output: [[ 0.20816524 -0.22928937]]

X = np.array([[1, 2]])
output = dense_layer.forward_pass(X)
print("Forward pass output:", output)

accum_grad = np.array([[0.1, 0.2, 0.3]])
back_output = dense_layer.backward_pass(accum_grad)
print("Backward pass output:", back_output)
