
from module_olist.modeling.inference import predict_delay


def read_int(prompt: str, minimum: int = None, maximum: int = None) -> int:
    """Lê um número inteiro pelo terminal com validação."""
    while True:
        try:
            value = int(input(prompt))

            if minimum is not None and value < minimum:
                print(f"Informe um valor maior ou igual a {minimum}.")
                continue

            if maximum is not None and value > maximum:
                print(f"Informe um valor menor ou igual a {maximum}.")
                continue

            return value

        except ValueError:
            print("Informe um número inteiro válido.")


def read_float(prompt: str, minimum: float = None) -> float:
    """Lê um número decimal pelo terminal com validação."""
    while True:
        try:
            value = float(input(prompt).replace(",", "."))

            if minimum is not None and value < minimum:
                print(f"Informe um valor maior ou igual a {minimum}.")
                continue

            return value

        except ValueError:
            print("Informe um número válido.")


def read_state() -> str:
    """Lê a sigla do estado do cliente."""
    while True:
        state = input("Estado do cliente (ex.: BA): ").strip().upper()

        if len(state) == 2 and state.isalpha():
            return state

        print("Informe a sigla do estado com 2 letras.")


def main():

    
    print("\nPREVISÃO DE ATRASO DE ENTREGA - OLIST")
    
    print("\nInforme os dados do pedido.")
    

    # Features categóricas relacionadas ao momento da compra
    purchase_hour = read_int(
        "Hora da compra (0-23): ",
        minimum=0,
        maximum=23,
    )

    purchase_weekday = read_int(
        "Dia da semana (0=segunda ... 6=domingo): ",
        minimum=0,
        maximum=6,
    )

    purchase_month = read_int(
        "Mês da compra (1-12): ",
        minimum=1,
        maximum=12,
    )

    # Prazo prometido
    promised_days = read_float(
        "Dias prometidos para entrega: ",
        minimum=0,
    )

    # Informações dos itens
    item_count = read_int(
        "Quantidade de itens: ",
        minimum=1,
    )

    seller_count = read_int(
        "Quantidade de vendedores: ",
        minimum=1,
    )

    total_price = read_float(
        "Valor total dos produtos (R$) (50.00): ",
        minimum=0,
    )

    total_freight = read_float(
        "Valor total do frete (R$) (10.00): ",
        minimum=0,
    )

    customer_state = read_state()

    features = {
        "purchase_hour": purchase_hour,
        "purchase_weekday": purchase_weekday,
        "promised_days": promised_days,
        "purchase_month": purchase_month,
        "item_count": item_count,
        "seller_count": seller_count,
        "total_price": total_price,
        "total_freight": total_freight,
        "customer_state": customer_state,
    }

    try:
        result = predict_delay(features)

    except FileNotFoundError as error:
        print()
        print(f"ERRO: {error}")
        return

    except ValueError as error:
        print()
        print(f"ERRO: {error}")
        return

    print()
    
    print("RESULTADO DA PREVISÃO")
    
    print(f"Modelo: {result['model_name']}")
    print(f"Probabilidade de atraso: {result['probability']:.2%}")
    print(f"Threshold utilizado: {result['threshold']:.2f}")
    print(f"Classificação: {result['label']}")
    print(f"is_late: {result['is_late']}")
    


if __name__ == "__main__":
    main()
