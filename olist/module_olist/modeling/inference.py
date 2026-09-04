
import json
from pathlib import Path

import joblib
import pandas as pd

from module_olist.config import MODELS_DIR


MODEL_PATH = MODELS_DIR / "best_model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"

# Features utilizadas pelo split.py e pelo pipeline.py
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


def load_model(
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
):
    """
    Carrega o modelo treinado e os metadados salvos pelo train.py.

    Returns:
        tuple: (model, metadata)
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em: {model_path}. "
            "Execute primeiro o treinamento."
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Arquivo de metadados não encontrado em: {metadata_path}. "
            "Execute primeiro o treinamento."
        )

    model = joblib.load(model_path)

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return model, metadata


def validate_features(features: dict) -> None:
    """
    Verifica se todas as features necessárias para a previsão foram informadas.
    """
    missing_features = [
        feature for feature in FEATURES
        if feature not in features
    ]

    if missing_features:
        raise ValueError(
            "Features obrigatórias não informadas: "
            + ", ".join(missing_features)
        )


def predict_delay(
    features: dict,
    model=None,
    metadata=None,
) -> dict:
    """
    Realiza a previsão de atraso de um pedido.

    Args:
        features: dicionário contendo as 9 features utilizadas pelo modelo.
        model: modelo carregado. Se não informado, será carregado automaticamente.
        metadata: metadados do modelo. Se não informado, serão carregados automaticamente.

    Returns:
        dict contendo:
            - is_late: 1 para atraso, 0 para entrega no prazo
            - label: texto da classificação
            - probability: probabilidade de atraso
            - threshold: threshold utilizado
            - model_name: modelo utilizado
    """
    validate_features(features)

    if model is None or metadata is None:
        model, metadata = load_model()

    # Mantém somente as colunas esperadas pelo modelo e na ordem definida.
    X = pd.DataFrame(
        [[features[feature] for feature in FEATURES]],
        columns=FEATURES,
    )

    probability = float(model.predict_proba(X)[0, 1])

    threshold = float(metadata["threshold"])

    prediction = int(probability >= threshold)

    return {
        "is_late": prediction,
        "label": "ATRASO" if prediction == 1 else "NO PRAZO",
        "probability": probability,
        "threshold": threshold,
        "model_name": metadata.get("model_name", "Desconhecido"),
    }
