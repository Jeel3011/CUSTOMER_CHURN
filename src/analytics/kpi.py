import pandas as pd
from src.logger import logging
from src.exeption import CustomException



class ChurnKPI:
    def __init__(self, df: pd.DataFrame):
        try:
            if not isinstance(df, pd.DataFrame):
                raise CustomException("Input must be a pandas DataFrame")

            if "churn_probability" not in df.columns:
                raise CustomException("Missing churn_probability column")

            if "risk_level" not in df.columns:
                raise CustomException("Missing risk_level column")

            self.df = df.copy()
            logging.info("KPI module initialized successfully")

        except Exception as e:
            raise CustomException(e)

    def compute_kpis(self):
        try:
            total_customers = len(self.df)

            high_risk = (self.df["risk_level"] == "High").sum()
            medium_risk = (self.df["risk_level"] == "Medium").sum()
            low_risk = (self.df["risk_level"] == "Low").sum()

            avg_churn_prob = round(self.df["churn_probability"].mean(), 4)
            max_churn_prob = round(self.df["churn_probability"].max(), 4)

            churn_distribution = (
                self.df["risk_level"]
                .value_counts(normalize=True)
                .mul(100)
                .round(2)
                .to_dict()
            )

            top_risky_customers = (
                self.df
                .sort_values(by="churn_probability", ascending=False)
                .head(10)
            )

            kpi_result = {
                "total_customers": total_customers,
                "high_risk_customers": high_risk,
                "medium_risk_customers": medium_risk,
                "low_risk_customers": low_risk,
                "average_churn_probability": avg_churn_prob,
                "maximum_churn_probability": max_churn_prob,
                "risk_distribution_percent": churn_distribution,
                "top_risky_customers": top_risky_customers
            }

            logging.info("KPI computation completed successfully")
            return kpi_result

        except Exception as e:
            raise CustomException(e)
