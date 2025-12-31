import os
import sys
import pandas as pd
import numpy as np
from networksecurity.constants import training_pipeline
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifact_entity import ModelTrainerArtifact,DataTransformationArtifact,ClassificationMetric
from networksecurity.logging.logger import logging
from networksecurity.exception.exception import Network_Security_Exception 
from networksecurity.utils.main_utils.utils import load_obj,load_numpy_array,save_obj,evaluate_models
from networksecurity.utils.ml_utils.model.classification_metric import get_classification_metric_score
from networksecurity.utils.ml_utils.metric.estimator import NetworkModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,
                 data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise Network_Security_Exception(e,sys)
        
    def train_model(self,x_train,y_train,x_test,y_test):
        try:
            models = {
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(verbose=1),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1),
                "AdaBoost": AdaBoostClassifier(),
            }

            params = {
                        "Decision Tree": {
                            'criterion':['gini', 'entropy', 'log_loss'],
                            # 'splitter':['best','random'],
                            # 'max_features':['sqrt','log2'],
                        },
                        "Random Forest":{
                            # 'criterion':['gini', 'entropy', 'log_loss'],
                            
                            # 'max_features':['sqrt','log2',None],
                            'n_estimators': [8,16,32,128,256]
                        },
                        "Gradient Boosting":{
                            # 'loss':['log_loss', 'exponential'],
                            'learning_rate':[.1,.01,.05,.001],
                            'subsample':[0.6,0.7,0.75,0.85,0.9],
                            # 'criterion':['squared_error', 'friedman_mse'],
                            # 'max_features':['auto','sqrt','log2'],
                            'n_estimators': [8,16,32,64,128,256]
                        },
                        "Logistic Regression":{},
                        "AdaBoost":{
                            'learning_rate':[.1,.01,.001],
                            'n_estimators': [8,16,32,64,128,256]
                        }
                    }
            
            model_report:dict=evaluate_models(X_train=x_train,y_train=y_train,X_test=x_test,y_test=y_test,
                                          models=models,param=params)
            
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            print("Best Model Name: ",best_model_name)
            print("Best Model Score: ",best_model_score)
            best_model = models[best_model_name]

            y_predict_train = best_model.predict(x_train)
            self.classification_train_metric = get_classification_metric_score(y_true=y_train,y_pred=y_predict_train)

            y_predict_test = best_model.predict(x_test)
            self.classification_test_metric = get_classification_metric_score(y_true=y_test,y_pred=y_predict_test)

            save_obj(file_path=self.model_trainer_config.model_trainer_trained_model_object_file_path,obj=best_model)

            save_obj("final_model/model.pkl",best_model)
            
            preprocessor_object = load_obj(self.data_transformation_artifact.transformed_object_filepath)
            Network_Model = NetworkModel(preprocessor = preprocessor_object,model =best_model)
            model_trainer_artifact = ModelTrainerArtifact(trained_model_filepath=self.model_trainer_config.model_trainer_trained_model_object_file_path,
                                                          trained_metric_path=self.classification_train_metric,
                                                          test_metric_path=self.classification_test_metric)
            

            
            
            return model_trainer_artifact
        
        except Exception as e:
            raise Network_Security_Exception(e,sys)
        

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_arr = load_numpy_array(self.data_transformation_artifact.transformed_trained_filepath)
            test_arr = load_numpy_array(self.data_transformation_artifact.transformed_test_filepath)

            x_train = train_arr[:,:-1]
            y_train = train_arr[:,-1]
            x_test = test_arr[:,:-1]
            y_test = test_arr[:,-1]

            model_trainer_artifact = self.train_model(x_train,y_train,x_test,y_test)

            return model_trainer_artifact
        except Exception as e:
            raise Network_Security_Exception(e,sys)