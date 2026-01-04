import sys
import os
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

import xgboost as xgb

from src.logger import logging
from src.exeption import CustomException

from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array: np.ndarray, test_array: np.ndarray):
        try:
            logging.info("Starting model training process")

            
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]


            num_negative = np.sum(y_train == 0)
            num_positive = np.sum(y_train == 1)

            scale_pos_weight = num_negative / num_positive
           
            models = {
                "LogisticRegression": LogisticRegression(max_iter=1000),
                "XGBoost": xgb.XGBClassifier(
                    eval_metric="auc",
                    scale_pos_weight=scale_pos_weight,
                    n_estimators=100,
                    random_state=42
                )
            }

            params = {
                "LogisticRegression": {
                    "C": [0.01, 0.1, 1, 10]
                },
                "XGBoost": {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 5],
                    "learning_rate": [0.05, 0.1],
                    "subsample": [0.8],
                    "colsample_bytree": [0.8]
                }
            }

            best_model = None
            best_score = 0.0
            best_model_name = None

           
            
            for model_name, model in models.items():
                logging.info(f"Training {model_name}")

                grid = GridSearchCV(
                    estimator=model,
                    param_grid=params[model_name],
                    scoring="roc_auc",
                    cv=3,
                    
                    n_jobs=-1,
                    verbose=0
                )

                grid.fit(X_train, y_train)

                trained_model = grid.best_estimator_

                y_prob = trained_model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_prob)

                logging.info(
                    f"{model_name} ROC-AUC: {roc_auc} | Best Params: {grid.best_params_}"
                )

                if roc_auc > best_score:
                    best_score = roc_auc
                    best_model = trained_model
                    best_model_name = model_name

           
            if best_score < 0.60:
                raise CustomException("No suitable model found with ROC-AUC >= 0.60")

            logging.info(
                f"Best model selected: {best_model_name} with ROC-AUC: {best_score}"
            )

           
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info("Best model saved successfully")

            return best_model_name, best_score

        except Exception as e:
            raise CustomException(e, sys)