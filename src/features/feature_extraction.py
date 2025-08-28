from src.data.load_data import load_housing_data, load_diabetes_data

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

def get_patients_features():
    df = load_diabetes_data()

    df['gender'] = df['gender'].apply(lambda x: 1 if x == 'Male' else 0)

    mapping = {
        'No Info': 0,
        'current': 1,
        'ever': 2,
        'former': 3,
        'never': 4,
        'not current': 5,
    }
    df['smoking_history'] = df['smoking_history'].map(mapping)

    feature_data = df[[
        'gender',
        'age',
        'hypertension',
        'heart_disease',
        'smoking_history',
        'bmi',
        'HbA1c_level',
        'blood_glucose_level'
    ]]

    return feature_data