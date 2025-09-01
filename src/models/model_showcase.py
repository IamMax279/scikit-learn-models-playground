from src.models.model_trainer import ModelTrainer
from src.data.load_data import load_diabetes_data, load_housing_data
from src.features.feature_extraction import get_patients_features, get_housing_features
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def show_model_params(features, target, model_instance, target_name):
    model = ModelTrainer(features, target, model_instance, target_name)
    model.train()
    
    score = model.get_score()
    print('score:', score)

    model.plot_confusion_matrix()
    model.prediction_vs_actual_scatterplot(target_name)

# LOGISTIC REGRESSION
show_model_params(
    get_patients_features(),
    load_diabetes_data()['diabetes'],
    LogisticRegression(max_iter=500),
    'smoking_history'
)

# RANDOM FOREST CLASSIFIER
show_model_params(
    get_patients_features(),
    load_diabetes_data()['diabetes'],
    RandomForestClassifier(n_estimators=100, bootstrap=True),
    'smoking_history'
)

# RANDOM FOREST REGRESSOR
# show_model_params(
#     get_housing_features(),
#     load_housing_data()['Housing_price'],
#     RandomForestRegressor(n_estimators=100, bootstrap=True),
#     'MedInc'
# )