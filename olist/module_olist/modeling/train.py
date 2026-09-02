import pandas as pd
from module_olist.modeling.cross_validation import cross_validate_models

from module_olist.modeling.pipeline import create_gradient_boosting_pipeline, create_xgboost_pipeline, create_lightgbm_pipeline


def train_models(X_train: pd.DataFrame, y_train: pd.Series):
     
    best_model_name = cross_validate_models(X_train, y_train)

    best_model_name.fit(X_train, y_train)
    trained_models = best_model_name

    return trained_models