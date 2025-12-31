from sklearn.metrics import precision_score,f1_score,recall_score
from networksecurity.entity.artifact_entity import ClassificationMetric
from networksecurity.exception.exception import Network_Security_Exception
import sys


class NetworkModel:
    def __init__(self,preprocessor,model):
        self.preprocessor = preprocessor
        self.model = model

    def Predict(self,x):
        try:
            scaled_data = self.preprocessor.transform(x)
            y_pred = self.model.predict(scaled_data)
            return y_pred
        except Exception as e:
            raise Network_Security_Exception(e,sys)
