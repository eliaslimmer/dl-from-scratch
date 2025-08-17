from model.neural_net import Neural_Network
from utils.metrics import accuracy, cross_entropy_loss


def train(model: Neural_Network, X_train, Y_train, X_test, Y_test, logger=None, epochs=10, log_every=10, logging=True):
    
    for epoch in range(epochs):

        A2 = model.forward(X_train)
        model.backward(X_train, Y_train)

        # logging accuracy and loss
        if epoch % log_every == 0 or epoch == epoch - 1 and logging == True and logger:
            train_acc = accuracy(A2, Y_train)

            A2_test = model.forward(X_test)
            test_acc = accuracy(A2_test, Y_test)
            training_loss = cross_entropy_loss(Y_train, A2)
            validation_loss = cross_entropy_loss(Y_test, A2_test)


            if logger:
                logger.log(epoch, training_loss, validation_loss, train_acc, test_acc)