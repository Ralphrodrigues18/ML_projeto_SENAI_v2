import pandas as pd
from loguru import logger

from sklearn.model_selection import StratifiedKFold, cross_val_score

from module_olist.modeling.pipeline import create_gradient_boosting_pipeline, create_xgboost_pipeline, create_lightgbm_pipeline


def cross_validate_models(X_train: pd.DataFrame, y_train: pd.Series,):

    models = {
        "Gradient Boosting": create_gradient_boosting_pipeline(),
        "XGBoost": create_xgboost_pipeline(),
        "LightGBM": create_lightgbm_pipeline(),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}

    for name, model in models.items():

        scores = cross_val_score(model, X_train, y_train, cv=cv)

        results[name] = {
            "scores": scores,
            "mean": scores.mean(),
            "std": scores.std(),
        }

        logger.info(f"Modelo: {name}")
        logger.info(f"F1 por fold: {scores:.5f}")
        logger.info(f"F1 médio: {scores.mean():.3f}")
        logger.info(f"Desvio padrão: {scores.std():.3f}")

    return results
