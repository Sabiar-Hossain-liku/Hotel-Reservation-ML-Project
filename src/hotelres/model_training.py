import os
import mlflow.artifacts
import pandas as pd
import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from hotelres.logger import get_logger
from hotelres.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *
from utils.commonfunctions import read_yaml, load_data
import lightgbm as lgb

import mlflow
import mlflow.sklearn

logger = get_logger (__name__)

class ModelTraining:

    def __init__(self, train_path,test_path,model_output_path) -> None:

        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path

        self.params_dist = LIGHTGM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS
    
    def  load_split(self) -> tuple:
        try:
            logger.info(f"Loading data from {self.train_path}")
            train_df = load_data(self.train_path)
            
            logger.info(f"Loading data from {self.test_path}")
            test_df = load_data(self.test_path)


            X_train = train_df.drop(columns=["booking_status"])
            y_train = train_df['booking_status']

            X_test = test_df.drop(columns=["booking_status"])
            y_test = test_df['booking_status']

            logger.info("Data Is splitted Successfully for Trainning")
            return X_train,y_train,X_test,y_test
        
        except Exception as e:
            logger.error(f"Error while location data {e}")
            raise CustomException("Failed to load data",e)
    
    def training_lgbm(self,X_train,y_train):

        try:
            logger.info("Model initialization")

            lgbm_model = lgb.LGBMClassifier(random_state=self.random_search_params['random_state'])

            logger.info("starting Hyperparameter Tuning")

            ransom_search = RandomizedSearchCV(
                estimator=lgbm_model,
                param_distributions=self.params_dist,
                n_iter = self.random_search_params['n_iter'],
                cv = self.random_search_params['cv'],
                n_jobs=self.random_search_params['n_jobs'],
                verbose=self.random_search_params['verbose'],
                random_state=self.random_search_params['random_state'],
                scoring=self.random_search_params['scoring']
            )

            ransom_search.fit(X_train,y_train)

            logger.info("Hyper Parameter Tuning Completed")

            best_params = ransom_search.best_params_
            best_lgbm_model = ransom_search.best_estimator_

            logger.info(f"the best parameters are : {best_params}")

            return best_lgbm_model
        
        except Exception as e:
            logger.error(f"Error while training model {e}")
            raise CustomException("Failed to train model",e)
    
    def evaluate_model (self,model,X_test,y_test):

        logger.info("evaluating the model")

        y_pred = model.predict(X_test)
        print("Length of X_test:", len(X_test))
        print("Length of y_test:", len(y_test))

        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        f1 = f1_score(y_test,y_pred)
        
        logger.info(f"Accuracy Score : {accuracy}")
        logger.info(f"Precision Score : {precision}")
        logger.info(f"Recall Score : {recall}")
        logger.info(f"F1 Score : {f1}")

        evaluators ={
                "accuracy" : accuracy,
                "precison" : precision,
                "recall" : recall,
                "f1" : f1
            } 

        return evaluators
    
    def save_model(self,model):

        try:
            os.makedirs(os.path.dirname(self.model_output_path),exist_ok=True)

            logger.info("saving model..")

            joblib.dump(model,self.model_output_path)
            logger.info(f"Model saved to {self.model_output_path}")
        
        except Exception as e:
            logger.error(f"Error while saving model {e}")
            raise CustomException("Failed to save the model",e)
        
    def run(self):
        try:
            with mlflow.start_run():
                logger.info("_Starting the model training pipeline")

                logger.info("Starting our MLFLOW experimentation---")

                logger.info("Logging the training and testing dataset to MLFLOW")
                mlflow.log_artifact(self.train_path,artifact_path='datasets')
                mlflow.log_artifact(self.test_path,artifact_path='datasets')


                X_train,y_train,X_test,y_test = self.load_split()

                model = self.training_lgbm(X_train,y_train)
                metrics = self.evaluate_model(model,X_test,y_test)
                self.save_model(model)
                mlflow.log_artifact(self.model_output_path)

                mlflow.log_params(model.get_params())
                mlflow.log_metrics(metrics)


                logger.info("_Model Trainning is Successfull")
        except Exception as e:
            logger.error(f"Error while training pipeline {e}")
            raise CustomException("Failed training pipeline",e)
        

if __name__ =="__main__":
    trainer =ModelTraining(PROCESSED_TRAIN,PROCESSED_TEST,MODEL_OUTPUT_PATH)
    trainer.run()
    