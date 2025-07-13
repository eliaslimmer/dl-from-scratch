import numpy as np

def reLU(self, Z):
    return np.maximum(0, Z)

def reLuDerivative(self, Z):
    return Z > 0

def softmax(self, Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def softmaxDerivative(self, Z):
    return self.softmax(Z) * (1 - self.softmax(Z))
