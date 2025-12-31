import os
import sys
from networksecurity.entity.config_entity import DataIngestionConfig,TrainingPipelineConfig,DataValidationConfig,DataTransformationConfig,ModelTrainerConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact
from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer
from networksecurity.logging.logger import logging
from networksecurity.exception.exception import Network_Security_Exception

if __name__ == '__main__':
    try:
        trainingpipelineconfig = TrainingPipelineConfig()
        dataingestionconfig = DataIngestionConfig(training_pipeline_config = trainingpipelineconfig)
        dataingestion = DataIngestion(dataingestionconfig)
        dataingestionartifact = dataingestion.initiate_data_ingestion()
        print(dataingestionartifact)

        datavalidationconfig = DataValidationConfig(training_pipeline_config= trainingpipelineconfig)
        data_validation = DataValidation(data_validation_config=datavalidationconfig,data_ingestion_artifact=dataingestionartifact)
        datavalidationartifact = data_validation.initiate_data_validation()
        print(datavalidationartifact)

        datatransformationconfig = DataTransformationConfig(training_pipeline_config= trainingpipelineconfig)
        data_transformation = DataTransformation(data_transformation_config=datatransformationconfig,data_validation_artifact=datavalidationartifact)
        datatransformationartifact = data_transformation.initiate_data_transformation()
        print(datatransformationartifact)

        modeltrainerconfig = ModelTrainerConfig(training_pipeline_config= trainingpipelineconfig)
        modeltrainer = ModelTrainer(model_trainer_config=modeltrainerconfig,data_transformation_artifact=datatransformationartifact)
        modeltrainerartifact = modeltrainer.initiate_model_trainer()

        print(modeltrainerartifact)
    except Exception as e:
        raise Network_Security_Exception(e,sys)