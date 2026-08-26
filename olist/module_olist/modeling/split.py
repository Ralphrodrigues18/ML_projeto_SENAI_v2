import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = [
    "purchase_hour",
    "purchase_weekday",
    "promised_days",
    "purchase_month",
    "item_count",
    "seller_count",
    "total_price",
    "total_freight",
    "customer_state",

]

TARGET = "is_late"

def split_data(data:pd.DataFrame):
    """
    Function to split the data into train and test sets.

    Args:
        data (pd.DataFrame): The input dataframe containing features and target.

    Returns:
        tuple: A tuple containing the train and test sets for features and target.
    """
    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    return X_train, X_test, y_train, y_test

