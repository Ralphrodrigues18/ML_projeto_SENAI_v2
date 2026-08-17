"""Calcular frequência de pedidos atrasados entre pedidos com prazos menores.

Regra usada:
- prazo estimado = `order_estimated_delivery_date - order_purchase_timestamp` (dias)
- 'prazos menores' = lead time estimado < mediana
- 'atraso' = `order_delivered_customer_date` > `order_estimated_delivery_date`

Gera um CSV em `data/processed/delay_frequency_by_group.csv` com resumo.
"""
from pathlib import Path
import pandas as pd


def compute_delay_frequency(orders_csv: Path, out_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(
        orders_csv,
        parse_dates=[
            "order_purchase_timestamp",
            "order_estimated_delivery_date",
            "order_delivered_customer_date",
        ],
        infer_datetime_format=True,
    )

    # remover linhas sem purchase ou estimated
    df = df.dropna(subset=["order_purchase_timestamp", "order_estimated_delivery_date"])

    # lead time estimado em dias (float)
    df["estimated_lead_days"] = (
        (df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]).dt.total_seconds() / 86400.0
    )

    median_lead = df["estimated_lead_days"].median()

    # considerar apenas pedidos com data de entrega do cliente disponível ao avaliar atraso
    df_with_delivered = df.dropna(subset=["order_delivered_customer_date"]).copy()

    # atraso em dias: positivo se entregue após a data estimada
    df_with_delivered["delay_days"] = (
        (df_with_delivered["order_delivered_customer_date"] - df_with_delivered["order_estimated_delivery_date"]).dt.total_seconds() / 86400.0
    )

    df_with_delivered["is_delayed"] = df_with_delivered["delay_days"] > 0

    # grupos: prazos menores (below median) vs others
    df_with_delivered["is_short_estimated_lead"] = df_with_delivered["estimated_lead_days"] < median_lead

    summary = []
    for group_name, group_df in (
        ("short_estimated_lead", df_with_delivered[df_with_delivered["is_short_estimated_lead"]]),
        ("not_short_estimated_lead", df_with_delivered[~df_with_delivered["is_short_estimated_lead"]]),
        ("all_with_delivered", df_with_delivered),
    ):
        n = len(group_df)
        n_delayed = int(group_df["is_delayed"].sum())
        freq = float(n_delayed) / n if n > 0 else None
        median_lead_group = float(group_df["estimated_lead_days"].median()) if n > 0 else None
        summary.append({
            "group": group_name,
            "n_orders": n,
            "n_delayed": n_delayed,
            "delay_frequency": freq,
            "median_estimated_lead_days": median_lead_group,
        })

    out_df = pd.DataFrame(summary)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    return out_df


def main():
    root = Path(__file__).resolve().parents[1]
    orders_csv = root / "data" / "raw" / "olist_orders_dataset.csv"
    out_csv = root / "data" / "processed" / "delay_frequency_by_group.csv"

    res = compute_delay_frequency(orders_csv, out_csv)
    print(res.to_string(index=False))
    print(f"Saved summary to: {out_csv}")


if __name__ == "__main__":
    main()
