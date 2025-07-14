import numpy as np
from utils.activations import reLU, softmax, reLuDerivative
  
# this is a multi class classification network
class Neural_Network:
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size, learning_rate=.1, optimizer=None):
        self.optimizer = optimizer
        self.learning_rate = learning_rate

        self.input_layer_size = input_layer_size
        self.output_layer_size = output_layer_size
        self.hidden_layer_size = hidden_layer_size

        # Initialize weights with small random values to keep activations stable.
        # He initialization might be better
        self.W1 = np.random.randn(self.hidden_layer_size, self.input_layer_size) * 0.01
        self.W2 = np.random.randn(self.output_layer_size, self.hidden_layer_size) * 0.01

        self.b1 = np.zeros((self.hidden_layer_size, 1))
        self.b2 = np.zeros((self.output_layer_size, 1))


    def forward(self, X):
        """
        X: (batch_size, input_size)
        returns: (batch_size, output_size) probability matrix
        """
        self.Z1 = np.dot(self.W1, X) + self.b1
        self.A1 = reLU(self.Z1)

        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = softmax(self.Z2)

        return self.A2

    # I want to understand this deeper and provide some useful annotations
    def backward(self, X, Y):
        """
        X: (batch_size, input_size)
        Y: (batch_size, output_size) one-hot labels
        """

        m = X.shape[1]

        dZ2 = self.A2 - Y

        dW2 = (1 / m) * np.dot(dZ2, self.A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * reLuDerivative(self.Z1)

        dW1 = (1 / m) * np.dot(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        if self.optimizer:
            self.optimizer.update()
        else:
            self.update_parameters(dW1, db1, dW2, db2)
  

    def update_parameters(self, dW1, db1, dW2, db2):
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2