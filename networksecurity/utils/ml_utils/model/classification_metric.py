from sklearn.metrics import precision_score,f1_score,recall_score
from networksecurity.entity.artifact_entity import ClassificationMetric
from networksecurity.exception.exception import Network_Security_Exception
import sys

def get_classification_metric_score(y_true,y_pred) -> ClassificationMetric:
        try:
            model_f1_score = f1_score(y_true,y_pred)
            model_precision_score = precision_score(y_true,y_pred)
            model_recall_score = recall_score(y_true,y_pred)

            score = ClassificationMetric(f1_score=model_f1_score,
                                        precision_score=model_precision_score,
                                        recall_score=model_recall_score)
            
            return score
        except Exception as e :
              raise Network_Security_Exception(e,sys)

