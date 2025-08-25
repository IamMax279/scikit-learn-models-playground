from src.data.load_data import load_housing_data

def get_housing_features():
    df = load_housing_data()
    
    # Select multiple columns from the data frame - hence the nested list
    feature_columns = df[
        ['housing_median_age',
        'median_income',
        'population_per_house',
        'rooms_per_house']
    ]

    return feature_columns