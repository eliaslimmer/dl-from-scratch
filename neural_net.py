import numpy as np
  

class Neural_Network:
    def __init__(self, inputLayerSize, outputLayerSize, hiddenLayerSize):
      self.inputLayerSize = inputLayerSize
      self.outputLayerSize = outputLayerSize
      self.hiddenLayerSize = hiddenLayerSize

      # Weights and Bias
      self.W1 = np.random.randn(self.hiddenLayerSize, self.inputLayerSize) * 0.01
      self.W2 = np.random.randn(self.outputLayerSize, self.hiddenLayerSize) * 0.01

      self.B1 = np.zeros((self.hiddenLayerSize, 1))
      self.B2 = np.zeros((self.outputLayerSize, 1))


    def forward(self, X):
        self.Z1 = np.dot(self.W1, X) + self.B1
        self.A1 = self.reLU(self.Z1)

        self.Z2 = np.dot(self.W2, self.A1) + self.B2
        self.A2 = self.softmax(self.Z2)

        return self.A2

    def backward(self, X, Y, learning_rate=.1):
        m = X.shape[1]

        dZ2 = self.A2 - Y

        dW2 = (1 / m) * np.dot(dZ2, self.A1.T)
        dB2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * self.reLuDerivative(self.Z1)

        dW1 = (1 / m) * np.dot(dZ1, X.T)
        dB1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        self.W2 -= learning_rate * dW2
        self.B2 -= learning_rate * dB2
        self.W1 -= learning_rate * dW1
        self.B1 -= learning_rate * dB1


    def reLU(self, Z):
        return np.maximum(0, Z)

    def reLuDerivative(self, Z):
        return Z > 0

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return expZ / np.sum(expZ, axis=0, keepdims=True)

    def softmaxDerivative(self, Z):
        return self.softmax(Z) * (1 - self.softmax(Z))
