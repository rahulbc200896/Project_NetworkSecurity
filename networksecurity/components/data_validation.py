import os
import sys
import pandas as pd
from networksecurity.constants import training_pipeline
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from networksecurity.logging.logger import logging
from networksecurity.exception.exception import Network_Security_Exception 
from networksecurity.utils.main_utils.utils import read_yaml_file,write_yaml_file
from scipy.stats import ks_2samp


class DataValidation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_config:DataValidationConfig):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_config = data_validation_config
        self._schema_config = read_yaml_file(training_pipeline.SCHEMA_FILE_PATH)

    @staticmethod
    def read_dataframe(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise Network_Security_Exception(e,sys)
    
    def validate_number_of_columns(self,df:pd.DataFrame) -> bool:
        try:
            num_of_columns = len(self._schema_config)
            if len(df.columns) == num_of_columns:
                return True
            return False
        except Exception as e:
            raise Network_Security_Exception(e,sys)
        
    def detect_data_drift(self,base_df,current_df,threshold = 0.05) -> bool:
        try:
            status = True
            report = {}
            for columns in base_df.columns:
                d1 = base_df[columns]
                d2 = current_df[columns]
                is_same_dist = ks_2samp(d1,d2)
                if threshold <= is_same_dist.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status = False
                report.update({columns:{
                    "pvalue":float(is_same_dist.pvalue),
                    "drift_status": is_found
                }})
            drift_report_filepath = self.data_validation_config.data_validation_drift_report_dir
            dir_path = os.path.dirname(drift_report_filepath)
            os.makedirs(dir_path,exist_ok=True)
            write_yaml_file(file_path=drift_report_filepath,content=report)
        except Exception as e:
            raise Network_Security_Exception(e,sys)
        
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            train_df = DataValidation.read_dataframe(train_file_path)
            test_df = DataValidation.read_dataframe(test_file_path)

            status = self.validate_number_of_columns(train_df)
            if not status:
                error_message = f"Train dataframe does not contain all columns\n"
            status = self.validate_number_of_columns(test_df)
            if not status:
                error_message = f"Test dataframe does not contain all columns\n"

            status = self.detect_data_drift(base_df=train_df,current_df=test_df)
            dir_path = os.path.dirname(self.data_validation_config.valid_train_path)
            os.makedirs(dir_path,exist_ok=True)
            train_df.to_csv(self.data_validation_config.valid_train_path,index=False,header=True)
            test_df.to_csv(self.data_validation_config.valid_test_path,index=False,header=True)

            datavalidationartifact = DataValidationArtifact(validation_status=status,
                                                            valid_trained_filepath=train_file_path,
                                                            valid_test_filepath=test_file_path,
                                                            invalid_trained_filepath=None,
                                                            invalid_test_filepath=None,
                                                            data_drift_filepath=self.data_validation_config.data_validation_drift_report_dir)
            return datavalidationartifact
        except Exception as e:
            raise Network_Security_Exception(e,sys)