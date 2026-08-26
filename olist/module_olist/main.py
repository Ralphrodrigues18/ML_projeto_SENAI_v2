from module_olist.config import INTERIM_DATA_DIR, ORDERS_PATH, ITEMS_PATH, CUSTOMERS_PATH, CUSTOMERS_PATH
from module_olist.dataset import load_data, create_dataset, save_dataset
from module_olist.features import create_features
from module_olist.modeling.train import train_models
from module_olist.modeling.evaluate import evaluate_models
from module_olist.split import split_data
from loguru import logger


def main():

    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for path in [ORDERS_PATH, ITEMS_PATH, CUSTOMERS_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {path}"
            )

    orders, items, customers = load_data(
        orders_path=ORDERS_PATH,
        items_path=ITEMS_PATH,
        customers_path=CUSTOMERS_PATH,
    )

    data = create_dataset(
        orders=orders,
        itens=items,
        customers=customers,
    )

    data_features = create_features(data)

    save_dataset(data_features, INTERIM_DATA_DIR / "orders_dataset_refined.csv")

    X_train, X_test, y_train, y_test = split_data(data_features)

    models = train_models(X_train, y_train)

    evaluate_models(models, X_test, y_test)




if __name__ == "__main__":
    main()
