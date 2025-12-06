import pandas as pd
import os
import numpy as np
from sklearn.base import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.datascience.entity.config_entity import ModelTrainerConfig
from src.datascience import logger
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
import joblib
import mlflow
import mlflow.sklearn
from src.datascience import logger

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/chaulagainrupesh1/end-to-end-ml-project.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "chaulagainrupesh1"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "c86602b44127e4fdb1576928c0aa2ddbf3118d8d"

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        
    def eval_metrics(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)


        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]
        
        models = {
            "ElasticNet": ElasticNet(alpha=self.config.alpha, l1_ratio=self.config.l1_ratio, random_state=42),
            "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),
            "XGBoost": XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
        }
        
        best_model = None
        best_rmse = float("inf")
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        
        for name, model in models.items():
            with mlflow.start_run(run_name = name):
                model.fit(train_x, train_y)
                
                # predictions
                preds = model.predict(test_x)
                
                # metrics
                (rmse, mae, r2) = self.eval_metrics(test_y, preds)
                
                # log metrics
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)
                
                # Log params (generic for now)
                mlflow.log_params(self.config.all_params)
                
                # Log model
                mlflow.sklearn.log_model(model, "model", registered_model_name=f"{name}Model")
                
                 # Save locally
                model_path = os.path.join(self.config.root_dir, f"{name}_model.joblib")
                joblib.dump(model, model_path)
                
                # Track best model
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = (name, model, model_path)
        
        logger.info(f"Best model: {best_model[0]} with RMAE={best_rmse}")
        return best_model
                
                