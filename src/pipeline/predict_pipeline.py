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
        tenure: int,
        monthly_charges: float,
        total_charges: float,
        contract_type: str,
        payment_method: str,
        internet_service: str,
        tech_support: str
    ):
        self.tenure = tenure
        self.monthly_charges = monthly_charges
        self.total_charges = total_charges
        self.contract_type = contract_type
        self.payment_method = payment_method
        self.internet_service = internet_service
        self.tech_support = tech_support

    def get_data_as_dataframe(self):
        try:
            data = {
                "tenure": [self.tenure],
                "MonthlyCharges": [self.monthly_charges],
                "TotalCharges": [self.total_charges],
                "Contract": [self.contract_type],
                "PaymentMethod": [self.payment_method],
                "InternetService": [self.internet_service],
                "TechSupport": [self.tech_support],
            }

            return pd.DataFrame(data)

        except Exception as e:
            raise CustomException(e, sys)
