from module_olist.config import INTERIM_DATA_DIR, RAW_DATA_DIR, ORDERS_PATH, ITEMS_PATH, CUSTOMERS_PATH, CUSTOMERS_PATH, OUTPUT_PATH
from module_olist.dataset import load_data, create_dataset, save_dataset
from module_olist.features import create_features
from loguru import logger


def main():
    logger.info("Iniciando pipeline de preparação dos dados...")


    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Dados brutos: {RAW_DATA_DIR}")
    logger.info(f"Dados intermediários: {INTERIM_DATA_DIR}")


    for path in [ORDERS_PATH, ITEMS_PATH, CUSTOMERS_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {path}"
            )


    logger.info("Carregando datasets...")

    orders, items, customers = load_data(
        orders_path=ORDERS_PATH,
        items_path=ITEMS_PATH,
        customers_path=CUSTOMERS_PATH,
    )

    logger.success(
        f"Orders carregado: {orders.shape[0]:,} linhas, "
        f"{orders.shape[1]} colunas"
    )

    logger.success(
        f"Items carregado: {items.shape[0]:,} linhas, "
        f"{items.shape[1]} colunas"
    )

    logger.success(
        f"Customers carregado: {customers.shape[0]:,} linhas, "
        f"{customers.shape[1]} colunas"
    )


    logger.info("Criando dataset base...")

    data = create_dataset(
        orders=orders,
        itens=items,
        customers=customers,
    )

    logger.success(
        f"Dataset base criado: {data.shape[0]:,} linhas, "
        f"{data.shape[1]} colunas"
    )

    logger.info("Criando features...")

    data_features = create_features(data)

    logger.success(
        f"Features criadas: {data_features.shape[0]:,} linhas, "
        f"{data_features.shape[1]} colunas"
    )


    logger.info("Salvando dataset processado...")

    save_dataset(
        dataset=data_features,
        output_path=OUTPUT_PATH,
    )

    logger.success("Pipeline executado com sucesso!")
    logger.success(f"Arquivo final: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
