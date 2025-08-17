from matplotlib import pyplot as plt
import numpy as np

class TrainingLogger:
    def __init__(self):
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_losses = []
        self.val_losses = []

    def log(self, epoch, train_loss, val_loss, train_acc, val_acc):
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)

        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

    ## todo needs x coordinate
    def plot_accuracies(self):

        N = len(self.train_accuracies)
        x = np.arange(N) * 10
        plt.plot(x, self.train_accuracies, label="train accuracy")
        plt.plot(x, self.val_accuracies, label="validation accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy over Epochs")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_losses(self):
        N = len(self.val_accuracies)
        x = np.arange(N) * 10
        plt.plot(x, self.train_losses, label="training loss")
        plt.plot(x, self.val_losses, label="validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curves")
        plt.legend()
        plt.show()