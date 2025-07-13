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

    def plot():
        pass