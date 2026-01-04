import os
import sys
import pandas as pd
import numpy as np

from src.utils import load_object
from src.logger import logging
from src.exeption import CustomException



class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame):
        try:
            logging.info("Starting prediction pipeline")

           
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )

            model_path = os.path.join(project_root, "artifacts", "model.pkl")
            preprocessor_path = os.path.join(project_root, "artifacts", "preprocessor.pkl")

            
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            
            if not isinstance(features, pd.DataFrame):
                raise CustomException("Input features must be a pandas DataFrame")

            
            data_scaled = preprocessor.transform(features)

           
            churn_prob = model.predict_proba(data_scaled)[:, 1]

            
            risk_bucket = pd.cut(
                churn_prob,
                bins=[0.0, 0.4, 0.7, 1.0],
                labels=["Low", "Medium", "High"]
            )

            result = features.copy()
            result["churn_probability"] = churn_prob
            result["risk_level"] = risk_bucket

            logging.info("Prediction pipeline completed successfully")

            return result

        except Exception as e:
            logging.error("Exception occurred in prediction pipeline")
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        customer_id: int,
        tenure_months: int,
        monthly_usage: float,
        subscription_plan: str,
        monthly_revenue: float,
        support_tickets: int,
        last_login_days: int,
        payment_delay: int
    ):
        self.customer_id = customer_id
        self.tenure_months = tenure_months
        self.monthly_usage = monthly_usage
        self.subscription_plan = subscription_plan
        self.monthly_revenue = monthly_revenue
        self.support_tickets = support_tickets
        self.last_login_days = last_login_days
        self.payment_delay = payment_delay

    def get_data_as_dataframe(self):
        try:
            data = {
                "customer_id": [self.customer_id],
                "tenure_months": [self.tenure_months],
                "monthly_usage": [self.monthly_usage],
                "subscription_plan": [self.subscription_plan],
                "monthly_revenue": [self.monthly_revenue],
                "support_tickets": [self.support_tickets],
                "last_login_days": [self.last_login_days],
                "payment_delay": [self.payment_delay]
            }

            return pd.DataFrame(data)

        except Exception as e:
            raise CustomException(e, sys)
