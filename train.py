from neural_net import Neural_Network
from utils import accuracy, cross_entropy_loss


def train(model: Neural_Network, X_train, Y_train, X_test, Y_test, logger=None, epochs=10, log_every=10, logging=True):
    
    for epoch in range(epochs):

        Z1, A1, Z2, A2 = model.forward(X_train)
        model.backward(X_train, Y_train, Z1, A1, Z2, A2)

        if epoch % log_every == 0 or epoch == epoch - 1 and logging == True:
            train_acc = accuracy(A2, Y_train)

            _, _, _, A2_test = model.forward(X_test)
            test_acc = accuracy(A2_test, Y_test)
            loss = cross_entropy_loss(Y_train, A2)

            if logger:
                logger.log(epoch, loss, train_acc, test_acc)