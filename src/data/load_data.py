import pandas as pd
from sklearn.datasets._california_housing import fetch_california_housing

def load_housing_data():
    bunch = fetch_california_housing()

    # kolejno: wartości cech, nazwy kolumn (cech)
    df = pd.DataFrame(bunch['data'], columns=bunch['feature_names'])

    df['Housing_price'] = bunch['target']

    return df

def load_diabetes_data():
    df = pd.read_csv('data/diabetes.csv')

    return df