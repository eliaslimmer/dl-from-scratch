import numpy as np


# ==== utils ====




# ==== layers ====

class Conv2D:
    def __init__(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass


class ReLu:
    def forward(self, x):
        self.mask = x>0
        return np.maximum(0, x)
    
    def backward(self, d_out):
        return d_out * self.mask


class MaxPool2D:
    def forward(self):
        pass

    def backward(self):
        pass

class Flatten:
    def forward(self):
        pass

    def backward(self):
        pass


class Dense:
    def __init__(self, in_dim, out_dim):
        self.W = np.random.randn(out_dim, in_dim) * 0.1
        self.b = np.zeroes((out_dim, 1))

    def forward(self):
        pass

    def backward(self):
        pass


# ==== Network ====

class SimpleCNN:
    def __init__(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass


# ==== Trainings loop ====

def train():
    pass