from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class ModelTrainer():
    def __init__(self, X, y, model, plot_residuals, plot_confusion_matrix):
        self.X = X
        self.y = y
        self.model = model
        self.plot_residuals = plot_residuals
        self.plot_confusion_matrix = plot_confusion_matrix
        self.y_pred = None
        self.y_true = None
    def train(self, test_size):
        try:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
            self.model.fit(X_train, y_train)

            self.y_pred = self.model.predict(X_test)
        except ValueError as e:
            pass
    def plot(self):
        if self.plot_confusion_matrix:
            cm = confusion_matrix(y_true=self.y_true, y_pred=self.y_pred)
            disp = ConfusionMatrixDisplay(cm, display_labels=)