import pandas as pd

def load_housing_data():
    df = pd.read_csv('data/housing.csv')

    df['population_per_house'] = df['population'] / df['households']
    df['rooms_per_house'] = df['total_rooms'] / df['households']

    return df