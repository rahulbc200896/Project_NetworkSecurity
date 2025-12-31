import os
import sys
from networksecurity.entity.config_entity import DataIngestionConfig,TrainingPipelineConfig,DataValidationConfig,DataTransformationConfig,ModelTrainerConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,DataTransformationArtifact
from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer
from networksecurity.logging.logger import logging
from networksecurity.exception.exception import Network_Security_Exception


class TrainingPipeline:
    def __init__(self):
        self.trainingpipelineconfig = TrainingPipelineConfig()

    def start_data_ingestion(self):
        try:
            self.dataingestionconfig = DataIngestionConfig(training_pipeline_config = self.trainingpipelineconfig)
            dataingestion = DataIngestion(data_ingestion_config=self.dataingestionconfig)
            dataingestionartifact = dataingestion.initiate_data_ingestion()
            return dataingestionartifact
        except Exception as e:
            raise Network_Security_Exception(e,sys)
        
    def start_data_validation(self,dataingestionartifact:DataIngestionArtifact):
        try:
            self.datavalidationconfig = DataValidationConfig(training_pipeline_config= self.trainingpipelineconfig)
            data_validation = DataValidation(data_validation_config=self.datavalidationconfig,data_ingestion_artifact=dataingestionartifact)
            datavalidationartifact = data_validation.initiate_data_validation()
            return datavalidationartifact
        except Exception as e:
            raise Network_Security_Exception(e,sys)
        
    def start_data_transformation(self,datavalidationartifact:DataValidationArtifact):
        try:
            self.datatransformationconfig = DataTransformationConfig(training_pipeline_config= self.trainingpipelineconfig)
            data_transformation = DataTransformation(data_transformation_config=self.datatransformationconfig,data_validation_artifact=datavalidationartifact)
            datatransformationartifact = data_transformation.initiate_data_transformation()
            return datatransformationartifact
        except Exception as e:
            raise Network_Security_Exception(e,sys)
        
    def start_model_training(self,datatransformationartifact:DataTransformationArtifact):
        try:
            self.modeltrainerconfig = ModelTrainerConfig(training_pipeline_config= self.trainingpipelineconfig)
            modeltrainer = ModelTrainer(model_trainer_config=self.modeltrainerconfig,data_transformation_artifact=datatransformationartifact)
            modeltrainerartifact = modeltrainer.initiate_model_trainer()
            return modeltrainerartifact
        except Exception as e:
            raise Network_Security_Exception(e,sys)
        
    def run_pipeline(self):
        try:
            dataingestionartifact = self.start_data_ingestion()
            datavalidationartifact = self.start_data_validation(dataingestionartifact=dataingestionartifact)
            datatransformationartifact = self.start_data_transformation(datavalidationartifact=datavalidationartifact)
            modeltrainerartifact = self.start_model_training(datatransformationartifact=datatransformationartifact)
            return modeltrainerartifact
        except Exception as e:
            raise Network_Security_Exception(e,sys)