import pandas as pd
import os

from sklearn.ensemble import RandomForestRegressor
from src.datascience.entity.config_entity import ModelTrainerConfig
from src.datascience import logger
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
import joblib
import mlflow
import mlflow.sklearn

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/chaulagainrupesh1/end-to-end-ml-project.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "chaulagainrupesh1"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "c86602b44127e4fdb1576928c0aa2ddbf3118d8d"

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)


        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]


        # lr = ElasticNet(alpha=self.config.alpha, l1_ratio=self.config.l1_ratio, random_state=42)
        # lr.fit(train_x, train_y)

        # joblib.dump(lr, os.path.join(self.config.root_dir, self.config.model_name))
        
        models = {
            "ElasticNet": ElasticNet(alpha=self.config.alpha, l1_ratio=self.config.l1_ratio, random_state=42),
            "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),
            "XGBoost": XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
        }
        
        best_model = None
        best_rmse = float("inf")
        mlflow.set_tracking_uri(self.config.mlflow_uri)