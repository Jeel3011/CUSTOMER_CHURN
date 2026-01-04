import pandas as pd

from src.pipeline.predict_pipeline import PredictPipeline
from src.analytics.kpi import ChurnKPI


if __name__ == "__main__":
    # Load new/unseen customer data
    df = pd.read_csv("Notebook/data/test.csv")  # change path if needed


    # Run prediction
    pipeline = PredictPipeline()
    predictions = pipeline.predict(df)

    # Compute KPIs
    kpi = ChurnKPI(predictions)
    results = kpi.compute_kpis()

    print("\n===== KPI SUMMARY =====")
    print("Total customers:", results["total_customers"])
    print("High risk customers:", results["high_risk_customers"])
    print("Medium risk customers:", results["medium_risk_customers"])
    print("Low risk customers:", results["low_risk_customers"])
    print("Average churn probability:", results["average_churn_probability"])

    print("\n===== TOP 5 RISKY CUSTOMERS =====")
    print(results["top_risky_customers"].head())
