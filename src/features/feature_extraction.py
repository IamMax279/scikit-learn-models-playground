from src.data.load_data import load_housing_data

def get_housing_features():
    df = load_housing_data()

    feature_data = df[[
        'HouseAge',
        'AveRooms',
        'AveBedrms',
        'AveOccup',
        'MedInc',
        'Population',
        'Latitude',
        'Longitude'
    ]]
    
    return feature_data