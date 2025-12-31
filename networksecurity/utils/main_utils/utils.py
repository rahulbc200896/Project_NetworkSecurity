import os
import sys
import yaml
import pickle
import numpy as np
from networksecurity.logging.logger import logging
from networksecurity.exception.exception import Network_Security_Exception 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


def read_yaml_file(file_path:str) -> dict:
    try:
        with open(file_path, "rb") as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise Network_Security_Exception(e,sys)
    

def write_yaml_file(file_path:str,content:object,replace:bool = False) -> None:
    try:
        if replace:
           if os.path.exists(file_path):
               os.remove(file_path)
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content,file)
    except Exception as e:
        raise Network_Security_Exception(e,sys)
    
def save_obj(file_path:str,obj:object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file:
            pickle.dump(obj,file)
    except Exception as e:
        raise Network_Security_Exception(e,sys)
    
def save_numpy_array(file_path:str,array:np.array):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file:
            np.save(file,array)
    except Exception as e:
        raise Network_Security_Exception(e,sys)
    
def load_obj(file_path:str) -> object:
    try:
        with open(file_path,"rb") as file:
            return pickle.load(file)
    except Exception as e:
        raise Network_Security_Exception(e,sys)
    
def load_numpy_array(file_path:str) -> np.array:
    try:
        with open(file_path,"rb") as file:
            return np.load(file)
    except Exception as e:
        raise Network_Security_Exception(e,sys)
    
'''
def evaluate_models(X_train,y_train,X_test,y_test,models,param):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = param[list(models.keys())[i]]

            grid = GridSearchCV(model,param_grid=param,cv=3)

            grid.fit(X_train,y_train)
            model.set_params(**grid.best_params_)
            model.fit(X_train,y_train)

            y_pred = model.predict(X_test)

            model_f1_score = f1_score(y_true=y_test,y_pred=y_pred)

            report[list(models.keys())[i]] = model_f1_score

        return report
    except Exception as e:
        raise Network_Security_Exception(e,sys)

'''

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    report = {}

    for model_name, model in models.items():

        if model_name not in param:
            raise ValueError(f"Missing hyperparameters for model: {model_name}")

        model_params = param[model_name]

        gs = GridSearchCV(
            model,
            model_params,
            cv=3,
            n_jobs=-1,
            verbose=1
        )

        gs.fit(X_train, y_train)

        model.set_params(**gs.best_params_)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)

        report[model_name] = score

    return report

