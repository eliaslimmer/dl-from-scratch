import numpy as np

def reLU(Z):
    return np.maximum(0, Z)

def reLuDerivative(Z):
    return Z > 0

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def softmaxDerivative(Z):
    return softmax(Z) * (1 - softmax(Z))
