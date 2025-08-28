from src.features.feature_extraction import get_patients_features
from src.data.load_data import load_diabetes_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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

    plt.figure(figsize=(10, 6))

    sns.scatterplot(x=X_test['blood_glucose_level'], y=y_test, color='blue')
    sns.scatterplot(x=X_test['blood_glucose_level'], y=y_pred, color='red')
    plt.xlabel('Actual diabetes outcome')
    plt.ylabel('Predicted diabetes outcome')
    plt.title('Data vs predictions')

    plt.show()

    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, bins=15, kde=True)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')

    plt.show()

train_logistic_regression()