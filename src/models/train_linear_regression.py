from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from src.data.load_data import load_housing_data
from src.features.feature_extraction import get_housing_features

def train_model():
    df = load_housing_data()

    X = get_housing_features()
    y = df['median_house_value']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # create a scatter plot of the data points
    plt.scatter(X_test['median_income'], y_test, color='blue', label='Actual Data')

    # plot the model's predictions as a line
    plt.plot(X_test['median_income'], y_pred, color='red', label='Model Prediction')

    plt.xlabel('Median Income')
    plt.ylabel('Median House Value')
    plt.title('Linear Regression: Median Income vs. House Value')
    plt.legend()

    plt.show()

train_model()