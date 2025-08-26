from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from src.data.load_data import load_housing_data
from src.features.feature_extraction import get_housing_features

def train_model():
    X = get_housing_features()
    y = load_housing_data()['Housing_price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = RandomForestRegressor(n_estimators=100, bootstrap=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    score = model.score(X_test, y_test)

    print('score:', score)

    # scatter the data points
    plt.scatter(X_test['MedInc'], y_test, color='blue', label='Actual data')
    
    # plot the model's predictions
    plt.scatter(X_test['MedInc'], y_pred, color='red', label='Model prediction')

    plt.xlabel('California dataset')
    plt.ylabel('Average house value')
    plt.title('Data vs prediction')

    plt.legend()
    plt.show()

train_model()