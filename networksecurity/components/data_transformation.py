import os
import sys
import pandas as pd
import numpy as np
from networksecurity.constants import training_pipeline
from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.entity.artifact_entity import DataValidationArtifact,DataTransformationArtifact
from networksecurity.logging.logger import logging
from networksecurity.exception.exception import Network_Security_Exception 
from networksecurity.utils.main_utils.utils import save_numpy_array,save_obj
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from networksecurity.constants import training_pipeline

class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,
                 data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise Network_Security_Exception(e,sys)
        
    @staticmethod
    def read_dataframe(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise Network_Security_Exception(e,sys)
        
    def get_data_transformed_object(self) -> Pipeline:
        try:
            imputer:KNNImputer = KNNImputer(**training_pipeline.DATA_TRANSFORMATION_IMPUTE_PARAMS)
            processor:Pipeline = Pipeline([
                ("imputer",imputer)
            ])
            return processor
        except Exception as e:
            raise Network_Security_Exception(e,sys)
        
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            self.train_data = self.data_validation_artifact.valid_trained_filepath
            self.test_data = self.data_validation_artifact.valid_test_filepath

            train_df = DataTransformation.read_dataframe(self.train_data).reset_index(drop=True)
            train_featured_input_data = train_df.drop(columns=[training_pipeline.TARGET_COLUMN],axis=1)
            train_featured_target_data = train_df[training_pipeline.TARGET_COLUMN]
            train_featured_target_data = train_featured_target_data.replace(-1,0)

            test_df = DataTransformation.read_dataframe(self.test_data).reset_index(drop=True)
            test_featured_input_data = test_df.drop(columns=[training_pipeline.TARGET_COLUMN],axis=1)
            test_featured_target_data = test_df[training_pipeline.TARGET_COLUMN]
            test_featured_target_data = test_featured_target_data.replace(-1,0)

            preprocessor = self.get_data_transformed_object()

            preprocessor_object = preprocessor.fit(train_featured_input_data)
            train_input_arr = preprocessor_object.transform(train_featured_input_data)
            test_input_arr = preprocessor_object.transform(test_featured_input_data)

            save_obj(file_path=self.data_transformation_config.data_transformed_object_data,obj=preprocessor_object)

            save_obj("final_model/preprocessor.pkl",preprocessor_object)


            train_arr = np.c_[train_input_arr,np.array(train_featured_target_data)]
            test_arr = np.c_[test_input_arr,np.array(test_featured_target_data)]

            save_numpy_array(file_path=self.data_transformation_config.data_tranformation_tranformed_trained_data,array=train_arr)
            save_numpy_array(file_path=self.data_transformation_config.data_tranformation_tranformed_test_data,array=test_arr)

            self.data_transformation_artifact = DataTransformationArtifact(transformed_object_filepath=self.data_transformation_config.data_transformed_object_data,
                                                                           transformed_trained_filepath=self.data_transformation_config.data_tranformation_tranformed_trained_data,
                                                                           transformed_test_filepath=self.data_transformation_config.data_tranformation_tranformed_test_data)

            return self.data_transformation_artifact


        except Exception as e:
            raise Network_Security_Exception(e,sys)