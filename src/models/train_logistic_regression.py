from src.features.feature_extraction import get_patients_features
from src.data.load_data import load_diabetes_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

def train_logistic_regression():
    X = get_patients_features()
    y = load_diabetes_data()['diabetes']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    score = model.score(X_test, y_test)
    print('score:', score)

    # In this case, the graphs look a bit odd since the LR model assigns binary values

    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
    # insert possible targets (0 - healthy, 1 - diabetic)
    disp = ConfusionMatrixDisplay(cm, display_labels=[0, 1])
    # set color map shades for color blue
    disp.plot(cmap='Blues')
    plt.title('Confusion matrix')
    plt.show()

    plt.figure(figsize=(10, 6))

    sns.scatterplot(x=y_test, y=y_pred, color='blue')
    plt.xlabel('Actual diabetes outcome')
    plt.ylabel('Predicted diabetes outcome')
    plt.title('Data vs predictions')
    plt.xticks([0, 1])
    plt.yticks([0, 1])
    plt.grid(True)

    plt.show()

train_logistic_regression()