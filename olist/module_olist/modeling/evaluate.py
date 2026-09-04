import numpy as np
from loguru import logger
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

"""
Avalia os modelos treinados e imprime métricas de desempenho.
precision, recall, f1 score, roc-auc e matriz de confusão para cada modelo.

precision: A precisão é a proporção de verdadeiros positivos em relação ao total de positivos previstos pelo modelo.
    Em outras palavras, mede a capacidade do modelo de não classificar como positivo um exemplo que é realmente negativo.

recall: O recall é a proporção de verdadeiros positivos em relação ao total de positivos
    reais. Em outras palavras, mede a capacidade do modelo de encontrar todos os exemplos positivos.

f1 score: O F1 score é a média harmônica entre precisão e recall. 
    Ele é útil quando se deseja um equilíbrio entre precisão e recall, especialmente em casos de classes desbalanceadas.

roc-auc: A métrica ROC-AUC (Receiver Operating Characteristic - Area Under the Curve) mede a capacidade do modelo de distinguir entre classes. 
    Quanto maior o valor da AUC, melhor o modelo é em separar as classes.

"""


def evaluate_models(models, X_test, y_test, model_name):

    y_proba = models.predict_proba(X_test)[:, 1]

    best_threshold = None
    best_f1 = -1
    best_precision = None
    best_recall = None
    best_y_pred = None
    for threshold in np.arange(0.05, 0.5, 0.01):
        y_pred = (y_proba >= threshold).astype(int)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall
            best_y_pred = y_pred
    roc_auc = roc_auc_score(y_test, y_proba)


    logger.info(f"Modelo: {model_name}")
    logger.info(f"Melhor Threshold: {best_threshold:.2f}")
    logger.info(f"Precision: {best_precision:.3f}")
    logger.info(f"Recall: {best_recall:.3f}")
    logger.info(f"F1 Score: {best_f1:.3f}")
    logger.info(f"ROC-AUC: {roc_auc:.3f}")
    logger.info(f"Matriz de Confusão:\n{confusion_matrix(y_test, best_y_pred)}")
