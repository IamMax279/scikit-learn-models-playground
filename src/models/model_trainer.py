from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class ModelTrainer():
    def __init__(self, X, y, model, target_name):
        self.X = X
        self.y = y
        self.model = model
        self.X_test = None
        self.y_pred = None
        self.y_true = None
        self.target_name = target_name
    def train(self, test_size=0.25):
        try:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
            self.X_test = X_test

            self.model.fit(X_train, y_train)

            self.y_pred = self.model.predict(X_test)
            self.y_true = y_test
        except ValueError as e:
            print(f"Something went wrong training the model: {e}")
    def plot_confusion_matrix(self, labels=[0, 1]):
        try:
            cm = confusion_matrix(y_true=self.y_true, y_pred=self.y_pred)
            disp = ConfusionMatrixDisplay(cm, display_labels=labels)
            disp.plot(cmap='Purples')

            plt.show()
        except ValueError as e:
            print(f"given labels array does not match the dataset labels: {e}")
    def prediction_vs_actual_scatterplot(self, feature_name):
        try:
            plt.figure(figsize=(10, 6))
            plt.title('Prediction vs Actual Data')

            sns.scatterplot(x=self.X_test[feature_name], y=self.y_true, color='blue')
            sns.scatterplot(x=self.X_test[feature_name], y=self.y_pred, color='red')
            
            plt.xlabel(feature_name.capitalize())
            plt.ylabel(self.target_name.capitalize())
            
            plt.show()
        except ValueError as e:
            print(f"Something went wrong trying to plot the comparison: {e}")
    def get_score(self):
        return self.model.score(self.X_test, self.y_true)