from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from src.data.load_data import load_housing_data
from src.features.feature_extraction import get_housing_features

def train_RFR():
    X = get_housing_features()
    y = load_housing_data()['Housing_price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = RandomForestRegressor(n_estimators=100, bootstrap=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    score = model.score(X_test, y_test)
    print('score:', score)

    # .reshape() takes number of rows and number of cols respectively
    test_data = pd.DataFrame(
        data=np.array([40.0, 6.8, 1.01, 2.5, 8.2, 322.0, 37.6, -122.2]).reshape(1, -1),
        columns=['HouseAge', 'AveRooms', 'AveBedrms', 'AveOccup', 'MedInc', 'Population', 'Latitude', 'Longitude']
    )
    print('prediction:', model.predict(test_data))

    residuals = y_test - y_pred

    # 10 inch width, 6 inch height
    plt.figure(figsize=(10, 6))

    sns.scatterplot(x=X_test['MedInc'], y=y_test, color='blue')
    sns.scatterplot(x=X_test['MedInc'], y=y_pred, color='red')

    plt.xlabel('Actual housing prices')
    plt.ylabel('Predicted housing prices')
    plt.title('Data vs Predictions')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, bins=15, kde=True)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.show()

train_RFR()