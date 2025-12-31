import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact
from networksecurity.logging.logger import logging
from networksecurity.exception.exception import Network_Security_Exception


class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise Network_Security_Exception(e,sys)
        
    def export_data(self):
        try:
            df = pd.read_csv("Network_Data\phisingData.csv")
            return df
        except Exception as e:
            raise Network_Security_Exception(e,sys)
        
    def export_data_into_feature_store(self,data_frame:pd.DataFrame):
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)

            data_frame.to_csv(feature_store_file_path,index=False,header=True)

            return data_frame
        
        except Exception as e:
            raise Network_Security_Exception(e,sys)
        
    def split_as_train_test(self,data_frame:pd.DataFrame):
        try:
            train_set,test_set = train_test_split(data_frame,test_size=self.data_ingestion_config.train_test_split_ratio,random_state=42)

            dir_path = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dir_path,exist_ok=True)

            train_set.to_csv(self.data_ingestion_config.train_file_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.test_file_path,index=False,header=True)
        
        except Exception as e:
            raise Network_Security_Exception(e,sys)
        
    def initiate_data_ingestion(self):
        try:
            data_frame = self.export_data()
            data_frame = self.export_data_into_feature_store(data_frame=data_frame)
            self.split_as_train_test(data_frame=data_frame)
            dataingestionartifact = DataIngestionArtifact(trained_file_path=self.data_ingestion_config.train_file_path,
                                                          test_file_path=self.data_ingestion_config.test_file_path)
            
            return dataingestionartifact
        
        except Exception as e:
            raise Network_Security_Exception(e,sys)