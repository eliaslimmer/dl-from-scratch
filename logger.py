from matplotlib import pyplot as plt
import numpy as np

class TrainingLogger:
    def __init__(self):
        self.train_accuracies = []
        self.test_accuracies = []
        self.losses = []

    def log(self, epoch, loss, train_acc, test_acc):
        self.losses.append(loss)
        self.train_accuracies.append(train_acc)
        self.test_accuracies.append(test_acc)

        print(f"[Epoch {epoch}] Loss: {loss:.4f} | Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")

    ## todo needs x coordinate I guess
    def plot_accuracies(self):

        N = len(self.train_accuracies)
        x = np.arange(N) * 10

        plt.plot(x, self.train_accuracies, label="train accuracy")
        plt.plot(x, self.test_accuracies, label="test accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy over Epochs")
        plt.legend()
        plt.grid(True)
        plt.show()