class Neural_Network(object):
  import numpy as np
  

class Neural_Network:
    def __init__(self):
      # hyperparameters
      self.inputLayerSize = 784
      self.outputLayerSize = 10
      self.hiddenLayerSize = 256

      # Weights
      self.W1 = np.random.randn(self.hiddenLayerSize, self.inputLayerSize) * 0.01
      self.W2 = np.random.randn(self.outputLayerSize, self.hiddenLayerSize) * 0.01

      self.B1 = np.zeros((self.hiddenLayerSize, 1))
      self.B2 = np.zeros((self.outputLayerSize, 1))

      # learning rate
      self.alpha = .1

    def forward(self, X):
        Z1 = np.dot(self.W1, X) + self.B1
        A1 = self.reLU(Z1)

        Z2 = np.dot(self.W2, A1) + self.B2
        A2 = self.softmax(Z2)

        return Z1, A1, Z2, A2

    def backward(self, X, Y, Z1, A1, Z2, A2):
        m = X.shape[1]

        # dA2 = A2 - Y
        # dZ2 = dA2 * self.reLuDerivative(Z2)
        dZ2 = A2 - Y

        dW2 = (1 / m) * np.dot(dZ2, A1.T)
        dB2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * self.reLuDerivative(Z1)


        dW1 = (1 / m) * np.dot(dZ1, X.T)
        dB1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        self.W2 -= self.alpha * dW2
        self.B2 -= self.alpha * dB2
        self.W1 -= self.alpha * dW1
        self.B1 -= self.alpha * dB1


    def gradient_descent(self, X_train, Y_train, X_test, Y_test, epochs=10):

        for i in range(epochs):

            Z1, A1, Z2, A2 = self.forward(X_train)
            self.backward(X_train, Y_train, Z1, A1, Z2, A2)


            if i % 10 == 0:
                    train_acc = self.accuracy(A2, Y_train)

                    _, _, _, A2_test = self.forward(X_test)
                    test_acc = self.accuracy(A2_test, Y_test)

                    train_accuracies.append(train_acc)
                    test_accuracies.append(test_acc)

                    loss = self.cross_entropy_loss(Y_train, A2)
                    print(f"Epoch {i}, Loss: {loss:.4f}, Train Acc: {train_acc * 100:.2f}%, Test Acc: {test_acc * 100:.2f}%")


    def reLU(self, Z):
        return np.maximum(0, Z)

    def reLuDerivative(self, Z):
        return Z > 0

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return expZ / np.sum(expZ, axis=0, keepdims=True)

    def softmaxDerivative(self, Z):
        return self.softmax(Z) * (1 - self.softmax(Z))

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