import numpy as np

def cross_entropy_loss(self, Y, A2):
        m = Y.shape[1]
        # Clip A2 to avoid log(0)
        A2 = np.clip(A2, 1e-12, 1.0)
        loss = -np.sum(Y * np.log(A2)) / m
        return loss

def accuracy(self, A2, Y):
    predictions = np.argmax(A2, axis=0) 
    labels = np.argmax(Y, axis=0)
    return np.mean(predictions == labels)